"""Proves the new score descends the same landscape as the published one.

`gen_region_success_scores.score_block` is a re-implementation of
`gen_optimization_scores.score_shard` with extra diagnostics. If the two ever
disagree, every number in data/optimization-scores-{v2,betweenset}.jsonl stops
being comparable with the new file -- silently, because both would still look
like plausible scores.

Two comparisons, and the distinction between them matters
---------------------------------------------------------
A. **score_block vs score_shard, both run here, in ONE process on ONE GPU.**
   This is the invariant this repo controls, so it is a hard failure. Running
   both in the same process removes every environmental variable between them:
   any difference is in the code.

B. **both of them vs the stored row.** This is only reproducible to the extent
   that jax / jaxley / optax have not moved since the row was written
   (2026-08-25, commit 6c6fd87). A 100-step optimizer trajectory on a rough
   landscape amplifies a 1e-15 kernel difference into an O(1) difference in
   d_final, so a library upgrade alone can shift these numbers. If A passes and
   B fails, the port is faithful and the environment drifted -- reported loudly,
   but it is not a code defect and it does not fail the test.

   What it DOES mean is that new rows must not be pooled with old ones without
   saying so: `meta.commit` and `meta.loss_set_version` are in every row for
   exactly this reason.

Needs a GPU and ~10 minutes (two runs of 20 ideal x 64 init x 100 steps).
Submit it, do not run it on a login node:

    cd encoding/slurm_scripts && sbatch ./region_crossval.sh
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                       # encoding/
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts", "datagen"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import gen_optimization_scores as REF  # noqa: E402
import losses as L  # noqa: E402
import sim as S  # noqa: E402
from gen_region_success_scores import (  # noqa: E402
    SUCCESS_RADII,
    ladder_score,
    score_block,
    success_table,
)
from helper import get_datagen_config  # noqa: E402
from logger import get_std_logger  # noqa: E402

REF_PATH = "./data/optimization-scores-betweenset.jsonl"

# A: the port must be faithful to the reference implementation.
PORT_TOL = 1e-9
# B: agreement with a row written weeks ago under possibly different libraries.
STORED_TOL = 2e-3

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print("FAIL:", msg)


def load_reference(path, set_key, loss_name):
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("set") == set_key and r.get("loss_name") == loss_name:
            return r
    raise SystemExit(f"no row for set={set_key!r} loss={loss_name!r} in {path}")


def _versions():
    import importlib.metadata as md

    out = {}
    for p in ("jax", "jaxlib", "jaxley", "optax", "equinox", "numpy"):
        try:
            out[p] = md.version(p)
        except Exception:  # noqa: BLE001
            out[p] = "?"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set_key", default="gLeak+eLeak+gNa")
    ap.add_argument("--loss", default="encoder-zscore")
    ap.add_argument("--ref", default=REF_PATH)
    args = ap.parse_args(argv)

    ref = load_reference(args.ref, args.set_key, args.loss)
    m = ref["meta"]
    print(f"reference row: {ref['set']} {ref['loss_name']} "
          f"ideal{ref['ideal_seeds']} init{ref['init_seeds']} "
          f"bounds{ref['init_bounds']} steps={m['steps']} opt={m['optimizer']}")
    print(f"  written by commit {m.get('commit','?')[:10]} on {m.get('device')}")
    print(f"  now running with  {_versions()}")
    print(f"  this device       {jax.devices()[0].device_kind}")

    logger = get_std_logger()
    config = get_datagen_config()
    simulator = S.Sim("Pospischil", config, checkpoint_lengths=m["checkpoint"])
    loss_fn = L.get_loss_fn(L.LOSS_SPECS[ref["loss_name"]], "Pospischil", config)

    common = dict(
        steps=m["steps"], max_lr=m["max_learning_rate"], eps=m["polyak_eps"],
        optimizer=m["optimizer"], chunk=m["chunk"],
        init_bounds=tuple(ref["init_bounds"]), logger=logger,
    )
    seeds = (tuple(ref["ideal_seeds"]), tuple(ref["init_seeds"]))

    print("\n--- running the REFERENCE implementation (score_shard) ---")
    ref_mean, _, _, ref_succ, ref_cap, ref_raw = REF.score_shard(
        simulator, loss_fn, ref["params"], *seeds, **common
    )

    print("\n--- running the NEW implementation (score_block) ---")
    new_raw, d0, diag, f_at_truth = score_block(
        simulator, loss_fn, ref["params"], *seeds, **common
    )
    new_cap = float(diag["cap_frac"].mean())

    # ---- A. port fidelity: a hard failure -------------------------------
    print("\n=== A. score_block vs score_shard (same process, same GPU) ===")
    d = np.abs(new_raw - ref_raw).max()
    print(f"  per-run distance matrix  max|diff| = {d:.3e}  (tol {PORT_TOL:g})")
    if d > PORT_TOL:
        fail(f"the port is NOT faithful: distance matrices differ by {d:.3e}")
    dc = abs(new_cap - ref_cap)
    print(f"  polyak_cap_frac          max|diff| = {dc:.3e}")
    if dc > PORT_TOL:
        fail(f"the port is NOT faithful: cap_frac differs by {dc:.3e}")

    # ---- B. agreement with the stored row: informational ----------------
    print("\n=== B. this run vs the row on disk (library drift shows up here) ===")
    drift = []
    dm = np.abs(ref_mean - np.array(ref["optimization-scores"])).max()
    print(f"  optimization-scores      max|diff| = {dm:.3e}")
    if dm > STORED_TOL:
        drift.append(f"optimization-scores by {dm:.3e}")
    succ = success_table(new_raw, REF.SUCCESS_TOLS)
    for t in REF.SUCCESS_TOLS:
        dd = np.abs(succ[float(t)] - np.array(ref[f"optimization-success@{t:g}"])).max()
        print(f"  optimization-success@{t:<6g} max|diff| = {dd:.3e}")
        if dd > STORED_TOL:
            drift.append(f"success@{t:g} by {dd:.3e}")
    dcap = abs(ref_cap - m["polyak_cap_frac"])
    print(f"  polyak_cap_frac          max|diff| = {dcap:.3e}")
    if dcap > STORED_TOL:
        drift.append(f"cap_frac by {dcap:.3e}")

    # ---- invariants that must hold regardless ---------------------------
    print("\n=== invariants ===")
    print(f"  f_at_truth               {f_at_truth:.3e}")
    if f_at_truth > 1e-12:
        fail(f"loss is not 0 at the truth: {f_at_truth:.3e}")
    # "Better than chance" must be asked of the SUCCESS RATE, not the mean.
    # On these landscapes the mean final distance routinely EXCEEDS the mean
    # start distance -- the published row does it too (mean 2.98 against a d0
    # of 1.76) -- because a handful of starts fly out to distance 5-6 and drag
    # the average past every start that converged. That is the entire reason
    # this score is a bounded success rate, so asserting on the mean would
    # reject correct data. The mean is printed as a diagnostic only.
    _, score = ladder_score(new_raw, SUCCESS_RADII)
    _, baseline = ladder_score(d0, SUCCESS_RADII)
    print(f"  d0 mean                  {d0.mean():.4f}")
    print(f"  mean final distance      {new_raw.mean():.4f} "
          f"(diagnostic only; {d0.mean() - new_raw.mean():+.4f} vs start)")
    print(f"  region score             {score:.4f} "
          f"vs chance {baseline:.4f}  ({score / max(baseline, 1e-9):.1f}x)")
    if score <= baseline:
        fail(f"region score {score:.4f} is no better than chance {baseline:.4f}")

    if drift:
        print("\n" + "=" * 72)
        print("NOTE: this run does not reproduce the stored row:")
        for x in drift:
            print(f"  - {x}")
        print("Part A passed, so the port is faithful and the difference is")
        print("environmental -- jax/jaxley/optax have moved since the row was")
        print(f"written (commit {m.get('commit','?')[:10]}). A 100-step trajectory")
        print("on a rough landscape amplifies a 1e-15 kernel difference into an")
        print("O(1) change in d_final, so this is expected after an upgrade.")
        print("CONSEQUENCE: do not pool new rows with the old score files")
        print("without saying so. meta.commit records which is which.")
        print("=" * 72)

    if FAILURES:
        print(f"\nFAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print("  ", f)
        return 1
    print("\ncross-validation passed: score_block is bit-faithful to score_shard"
          + (" (stored row differs; see the note above)" if drift else
             " and reproduces the stored row"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
