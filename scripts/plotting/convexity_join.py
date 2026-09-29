"""Join the convexity descriptors to the region-of-success optimization score.

The successor to ``score_join.py`` on ``big-loss-scoring``, which joined the same
convexity file to the *mean-final-distance* score family
(``data/optimization-scores*.jsonl``). That score has been retired twice over --
its v1 is invalid (starts drawn from a wider box than the swept landscape) and
its v2 was superseded by the region-of-success score in commit ``fca501f`` -- so
the figures built on it describe a quantity nothing else in the project uses any
more.

What this module joins instead:

``data/convexity-scores.jsonl``        56 sets x 8 losses x 6 descriptors x 100 seeds
``data/optimization-region-scores.jsonl``  7 arms x 2 optimizers x 56 sets

The join is ``(parameter set, loss_name) -> aligned pairs over ideal-seed index``.
Three things make it safe, and each is asserted by :func:`check` rather than
assumed:

* both generators derive the target neuron from ``sim.ideal_params_for_seed``, so
  seed *i* means the same neuron on both sides. ``tests/test_sim.py`` pins that
  PRNG stream bit-for-bit; if it ever drifted every correlation here would
  silently become noise.
* the convexity grid window (``+-1.5``, ``param_groups.GRID_PLAN[3]["bounds"]``)
  is the same window the region score's starts are drawn from. Mismatching these
  two is exactly what made ``optimization-scores.jsonl`` v1 invalid.
* the two stores use identical ``loss_name`` and ``set`` vocabularies, so no name
  mapping is involved -- but a typo upstream would produce an empty join rather
  than a wrong one, so the coverage is reported, not inferred.

**Seed coverage is asymmetric.** Convexity was scored over ideal seeds 0-99, the
region study over 0-9. :func:`paired` intersects, so every cell carries 10 shared
seeds. That is ample pooled (56 x 10 = 560 points for the seed unit) and thin per
set (a Fisher band of about +-0.6 for the line figure) -- the line figure draws
that band rather than hiding it.

**The score is recomputed, never read.** ``ladder_score`` and ``SUCCESS_RADII``
are imported from the generator, so no figure here can disagree with
``gen_region_success_scores.py --check``. The default matrix is ``raw_d_best``,
not ``raw_d_final``: under polyak roughly 60% of runs end farther from the truth
than they started while ``d_best`` p50 is about half of ``d_final`` p50 (see
``claude-tasks/2026-08-28-region-success-score.md``), so the final iterate mostly
measures where a diverging optimizer stopped. A convexity descriptor is a claim
about whether the landscape *lets* the optimizer find the neuron, which is the
best iterate.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "datagen"))

import numpy as np  # noqa: E402

import convexity as cx  # noqa: E402
import jsonl_store as JS  # noqa: E402
import param_groups as pg  # noqa: E402

# The ladder and the score, straight from the generator -- never restated here.
from gen_region_success_scores import (  # noqa: E402,F401
    SUCCESS_RADII,
    ladder_score,
)

CONVEXITY_PATH = Path("./data/convexity-scores.jsonl")
REGION_PATH = Path("./data/optimization-region-scores.jsonl")

# The matrices a row stores, all shaped (n_ideal_seeds, n_init_seeds).
MATRICES = ("raw_d_best", "raw_d_final", "raw_d0")

# ---------------------------------------------------------------------------
# the descriptor vocabulary
# ---------------------------------------------------------------------------
#
# Taken from `convexity.py`, the module that computes these, so the two can never
# drift. Earlier revisions of this file restated both dicts as literals, because
# `convexity.py` was still unported; it is on this branch now, and
# `claude-tasks/2026-08-30-convexity-vs-region-score.md` v1.1 records that the
# literals were checked against it (identical order, identical signs) before
# being deleted.
#
# STAT_LABELS stays local: figure labels are a presentation choice, not part of
# the descriptor contract.

STAT_NAMES = cx.SCORE_NAMES

STAT_LABELS = {
    "locmin": "local minima",
    "basin": "basin fraction",
    "basin_wide": "basin (wide)",
    "rough": "roughness",
    "flat": "flat fraction",
    "argmin_err": "argmin error",
}

# The sign each descriptor is expected to take AGAINST THE REGION SCORE.
#
# `convexity.EXPECTED_SIGN` states these against optimization *error* (lower is
# better): basin/basin_wide negative, the other four positive. The region score is
# a success rate -- higher is better -- so every sign here is that table flipped
# once, at import, as a pure function of it. So: more local minima or more
# roughness -> a lower score; a wider basin -> a higher one; a flat landscape
# gives the optimizer no gradient and a displaced global minimum means the
# optimum is not the truth, so both are negative.
#
# What this must NOT become is the `set_metric_orientation` it replaced, which
# flipped these as a process-global side effect of parsing argv -- which is
# precisely how a figure and its own table end up quoting different expectations.
EXPECTED_SIGN = {k: -v for k, v in cx.EXPECTED_SIGN.items()}

# Stated against error, for the record, so the flip above is auditable.
EXPECTED_SIGN_VS_ERROR = dict(cx.EXPECTED_SIGN)

SCORE_LABEL = {
    "raw_d_best": "region score (best iterate)",
    "raw_d_final": "region score (final iterate)",
    "raw_d0": "region score of the starts (chance)",
}


def _set_of(row) -> str:
    """The parameter-set key for a row, from `set` or rebuilt from `params`."""
    return row["set"] if row.get("set") else pg.set_key(row["params"])


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_convexity(path=CONVEXITY_PATH, losses=None) -> dict:
    """``(set, loss_name, descriptor) -> {seed: score}``."""
    keep = set(losses) if losses else None
    out: dict[tuple, dict[int, float]] = {}
    for row in JS.load_rows(Path(path)):
        scores = row.get("convexity-scores")
        if not scores:
            continue
        loss_name = row.get("loss_name") or row["loss"][0]
        if keep is not None and loss_name not in keep:
            continue
        seeds = row.get("seeds")
        if seeds is None:
            lo, hi = row["seed_range"]
            seeds = list(range(lo, hi))
        if len(seeds) != len(scores):
            continue  # malformed row; skip rather than mis-align
        bucket = out.setdefault((_set_of(row), loss_name, row["convexity"]), {})
        for seed, value in zip(seeds, scores):
            if value is not None:
                bucket[seed] = float(value)
    return out


def _budget(row) -> tuple:
    """The fields that together define a comparable measurement.

    ``optimizer`` and ``steps`` are read from the TOP LEVEL of the row. The
    ancestor of this function read them from ``meta``, where the region store
    does not put them -- both lookups return None, the polyak and adam rows land
    in one bucket, and they overwrite each other seed by seed with no warning.
    """
    meta = row.get("meta") or {}
    return (
        row.get("optimizer"),
        row.get("steps"),
        tuple(row.get("ideal_seeds") or ()),
        tuple(row.get("init_seeds") or ()),
        tuple(float(v) for v in (row.get("init_bounds") or ())),
        meta.get("loss_set_version"),
    )


def load_region_rows(path=REGION_PATH, *, optimizer, losses=None,
                     quiet=False) -> list[dict]:
    """The rows of ONE budget for ONE optimizer.

    ``optimizer`` is required, not defaulted: the store holds a polyak and an
    adam arm for every cell, they are different experiments, and silently
    picking whichever happens to cover more cells is how the two get pooled.
    """
    rows = [r for r in JS.load_rows(Path(path))
            if r.get("optimizer") == optimizer]
    if not rows:
        present = sorted({r.get("optimizer")
                          for r in JS.load_rows(Path(path))})
        raise SystemExit(
            f"no rows with optimizer={optimizer!r} in {path}; present: {present}"
        )
    if losses:
        keep = set(losses)
        missing = keep - {r["loss_name"] for r in rows}
        if missing:
            raise SystemExit(
                f"losses {sorted(missing)} absent from {path} under "
                f"optimizer={optimizer!r}; present: "
                f"{sorted({r['loss_name'] for r in rows})}"
            )
        rows = [r for r in rows if r["loss_name"] in keep]

    by_budget: dict[tuple, list[dict]] = {}
    for row in rows:
        by_budget.setdefault(_budget(row), []).append(row)
    chosen = max(by_budget, key=lambda b: len(by_budget[b]))
    kept = by_budget[chosen]
    if len(by_budget) > 1 and not quiet:
        print(f"  {len(by_budget)} budgets present (optimizer, steps, "
              f"ideal_seeds, init_seeds, init_bounds, loss_set_version); using "
              f"{chosen} ({len(kept)} rows). The rest answer a different "
              "question and are excluded: "
              + ", ".join(f"{b} ({len(v)})" for b, v in by_budget.items()
                          if b != chosen))

    versions = {tuple(sorted(((r.get("meta") or {}).get("versions") or {}).items()))
                for r in kept}
    if len(versions) > 1 and not quiet:
        print(f"  WARNING {len(versions)} distinct library-version sets among "
              "the kept rows. A 100-step trajectory amplifies a kernel "
              "difference into an O(1) change in the distance, so these are not "
              "strictly one experiment.")
    if not quiet:
        print(f"  {len(kept)} rows | optimizer {optimizer} | losses "
              f"{sorted({r['loss_name'] for r in kept})} | "
              f"{len({_set_of(r) for r in kept})} parameter sets | "
              f"ideal_seeds {list(chosen[2])} init_seeds {list(chosen[3])} "
              f"init_bounds {list(chosen[4])}")
    return kept


def score_per_seed(row, *, matrix="raw_d_best", radii=SUCCESS_RADII) -> np.ndarray:
    """The ladder score of one row, one value per ideal seed.

    Recomputed from the stored raw distance matrix with the generator's own
    ``ladder_score``, so this cannot drift from what ``--check`` reports.
    """
    raw = np.asarray(row[matrix], dtype=float)
    per_seed, _ = ladder_score(raw, radii)
    return np.asarray(per_seed, dtype=float)


def load_optimization(path=REGION_PATH, *, optimizer, matrix="raw_d_best",
                      radii=SUCCESS_RADII, losses=None, quiet=False,
                      rows=None) -> dict:
    """``(set, loss_name) -> {seed: score}`` for one optimizer arm."""
    if matrix not in MATRICES:
        raise SystemExit(f"unknown matrix {matrix!r}; want one of {MATRICES}")
    if rows is None:
        rows = load_region_rows(path, optimizer=optimizer, losses=losses,
                                quiet=quiet)
    out: dict[tuple, dict[int, float]] = {}
    for row in rows:
        lo, _hi = row["ideal_seeds"]
        values = score_per_seed(row, matrix=matrix, radii=radii)
        bucket = out.setdefault((_set_of(row), row["loss_name"]), {})
        for offset, value in enumerate(values):
            if np.isfinite(value):
                bucket[lo + offset] = float(value)
    return out


# ---------------------------------------------------------------------------
# the join
# ---------------------------------------------------------------------------


def paired(convexity: dict, optimization: dict, stat: str) -> dict:
    """``(set, loss) -> (convexity values, optimization values)`` over shared seeds.

    Only seeds present on BOTH sides are kept, and they are ordered by seed, so
    the two vectors are aligned by construction rather than by insertion order.
    Reused verbatim from ``score_join.paired``.
    """
    out = {}
    for (set_key, loss, conv), cvx in convexity.items():
        if conv != stat:
            continue
        opt = optimization.get((set_key, loss))
        if not opt:
            continue
        shared = sorted(set(cvx) & set(opt))
        if not shared:
            continue
        out[(set_key, loss)] = ([cvx[s] for s in shared],
                                [opt[s] for s in shared])
    return out


def aggregate_by_set(pairs, how="median") -> dict:
    """Collapse each parameter set to ONE point -- the BETWEEN-set view.

    Both sides are aggregated over the SAME shared seeds, which is what makes the
    result a paired observation. In particular the optimization side is not the
    stored ``region_score`` (a mean over all ten seeds of a different matrix) and
    the convexity side is not a median over all one hundred.

    The two units answer different questions and the choice dominates the
    correlation:

    * **seed** (within-set): for a fixed parameter set, do the ideal traces whose
      landscape is more convex optimize better? The strict question -- it holds
      the problem fixed and varies only the target neuron.
    * **set** (between-set): do parameter sets whose landscapes are more convex
      optimize better? Most of the variance here is between problems of very
      different difficulty, so |rho| is much larger and means less.
    """
    agg = np.median if how == "median" else np.mean
    out = {}
    for key, (x, y) in pairs.items():
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 1:
            continue
        out[key] = ([float(agg(x[ok]))], [float(agg(y[ok]))])
    return out


def regroup_by_loss(pairs) -> dict:
    """Merge every parameter set of a loss into ONE group, keyed by loss."""
    out = {}
    for (_set_key, loss), (x, y) in pairs.items():
        gx, gy = out.setdefault(loss, ([], []))
        gx.extend(x)
        gy.extend(y)
    return out


def coverage(convexity: dict, optimization: dict) -> dict:
    """What is actually available, for a plot subtitle and the JSON sidecar."""
    return {
        "sets": sorted({k[0] for k in convexity} | {k[0] for k in optimization}),
        "losses": sorted({k[1] for k in convexity}
                         | {k[1] for k in optimization}),
        "stats": sorted({k[2] for k in convexity}),
        "n_optimization_cells": sum(len(v) for v in optimization.values()),
        "n_convexity_cells": sum(len(v) for v in convexity.values()),
    }


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


# `--save_raw` writes `np.round(raw, 5)`, so a distance sitting within 5e-6 of a
# radius can land on the other side of it when the ladder is recomputed from the
# stored matrix. One such flip moves one radius' rate for one seed by 1/100,
# hence the row score by 1/100 / 5 radii / 10 seeds = 2e-4 exactly -- which is
# attained in practice, so the comparison below has to admit it. Anything past
# one flip is a real disagreement.
_ROUND_QUANTUM = 2e-4
_C1_TOL = _ROUND_QUANTUM * 1.01


def check(*, convexity_path=CONVEXITY_PATH, score_path=REGION_PATH,
          optimizer="polyak", losses=None, matrix="raw_d_best",
          radii=SUCCESS_RADII, quiet=False) -> list[str]:
    """Assert the join is sound. Returns the list of failures (empty = pass).

    Every figure runs this first: a silently mis-aligned join produces a
    plausible-looking cloud of noise, which is far worse than a crash.
    """
    fails, say = [], (lambda *a: None) if quiet else print

    rows = load_region_rows(score_path, optimizer=optimizer, losses=losses,
                            quiet=quiet)
    cvx = load_convexity(convexity_path, losses=losses)
    opt = load_optimization(score_path, optimizer=optimizer, matrix=matrix,
                            radii=radii, losses=losses, quiet=True, rows=rows)

    # Informational, not a check: which generation of the convexity store these
    # rows came from. `collect_convexity_scores.py` stamps `loss_set_version` and
    # keys a row by `seed_range`, so a store rebuilt over a narrower seed range
    # (say the ten seeds the region study used, rather than all one hundred) adds
    # rows under a NEW key beside the old ones instead of replacing them.
    # `load_convexity` pools them by seed, which is correct -- the overlap carries
    # identical values -- but a reader should be told it happened.
    cvx_rows = [r for r in JS.load_rows(Path(convexity_path))
                if losses is None or r.get("loss_name") in set(losses)]
    say(f"  convexity store: {len(cvx_rows)} rows | loss_set_version "
        f"{sorted({r.get('loss_set_version') for r in cvx_rows})} | seed_range "
        f"{sorted({tuple(r['seed_range']) for r in cvx_rows if 'seed_range' in r})}")

    # C1 -- the ladder is being applied the way the generator applies it.
    worst, n_off = 0.0, 0
    for row in rows:
        _, recomputed = ladder_score(np.asarray(row["raw_d_final"], float),
                                     SUCCESS_RADII)
        delta = abs(recomputed - float(row["region_score"]))
        worst = max(worst, delta)
        if delta > _C1_TOL:
            n_off += 1
    verdict = "PASS" if n_off == 0 else "FAIL"
    say(f"C1 {verdict}  region_score reproduced from raw_d_final on "
        f"{len(rows) - n_off}/{len(rows)} rows (max |diff| {worst:.2e}, "
        f"tolerance {_C1_TOL:.2e} = one np.round(.,5) flip)")
    if n_off:
        fails.append(f"C1: {n_off} rows disagree with their stored region_score")

    # C2 -- the two stores describe the same parameter sets.
    cvx_sets = {k[0] for k in cvx}
    opt_sets = {k[0] for k in opt}
    same = cvx_sets == opt_sets
    say(f"C2 {'PASS' if same else 'FAIL'}  {len(cvx_sets)} convexity sets vs "
        f"{len(opt_sets)} score sets"
        + ("" if same else f"; symmetric difference "
                           f"{sorted(cvx_sets ^ opt_sets)}"))
    if not same:
        fails.append("C2: the two stores cover different parameter sets")

    # C3 -- every joined cell carries the full shared-seed complement.
    counts = {len(v[0]) for v in paired(cvx, opt, "basin_wide").values()}
    n_shared = len(set(range(*rows[0]["ideal_seeds"])))
    ok = counts == {n_shared} if counts else False
    say(f"C3 {'PASS' if ok else 'FAIL'}  every joined cell has "
        f"{n_shared} shared seeds (observed {sorted(counts)})")
    if not ok:
        fails.append(f"C3: joined cells have {sorted(counts)} seeds, "
                     f"expected {n_shared}")

    # C4 -- the starts really are shared across losses, so the comparison is
    # paired. This is invariant C1 of the generator, read back out of the store.
    by_set: dict[str, set] = {}
    for row in rows:
        by_set.setdefault(_set_of(row), set()).add(row.get("d0_hash"))
    bad = {s: h for s, h in by_set.items() if len(h) != 1}
    say(f"C4 {'PASS' if not bad else 'FAIL'}  one d0_hash per parameter set "
        f"across the {len({r['loss_name'] for r in rows})} losses "
        f"({len(by_set)} sets checked)")
    if bad:
        fails.append(f"C4: {len(bad)} sets have losses with different starts")

    # C5 -- the landscape window and the start window agree. A mismatch here is
    # what invalidated optimization-scores.jsonl v1.
    keep = set(losses) if losses else None
    grid = {tuple(float(v) for v in (r.get("grid") or {}).get("bounds", ()))
            for r in JS.load_rows(Path(convexity_path))
            if keep is None or r.get("loss_name") in keep}
    starts = {tuple(float(v) for v in rows[0]["init_bounds"])}
    ok = len(grid) == 1 and grid == starts
    say(f"C5 {'PASS' if ok else 'FAIL'}  convexity grid bounds {sorted(grid)} "
        f"== optimization start bounds {sorted(starts)}")
    if not ok:
        fails.append(f"C5: grid {sorted(grid)} != starts {sorted(starts)}")

    say("")
    say(f"{'PASS' if not fails else 'FAIL'}: "
        f"{5 - len(fails)}/5 checks passed")
    return fails


# ---------------------------------------------------------------------------
# the shared CLI
# ---------------------------------------------------------------------------
#
# All three figure scripts take the same inputs and must resolve them the same
# way, or two figures in one directory end up describing different experiments.
# Defaults come from `configs/convexity_vs_opt.yaml`.


def add_common_args(ap, cfg):
    """The flags every plot_convexity_vs_opt_* script shares."""
    ap.add_argument("--convexity", default=cfg["convexity_path"])
    ap.add_argument("--score_path", default=cfg["score_path"])
    ap.add_argument("--optimizer", default=cfg["optimizer"],
                    help="which optimizer arm to correlate against. The store "
                         "holds polyak and adam for every cell; they are "
                         "different experiments and are never pooled.")
    ap.add_argument("--matrix", default=cfg["matrix"], choices=list(MATRICES),
                    help="which stored distance matrix the score is computed "
                         "from. raw_d_best asks whether the landscape lets the "
                         "optimizer find the neuron; raw_d_final also asks "
                         "whether it stayed there.")
    ap.add_argument("--losses", default=",".join(cfg["losses"]),
                    help="comma-separated subset")
    ap.add_argument("--out", default=cfg["plot_path"])
    ap.add_argument("--format", default="pdf", choices=["pdf", "png"],
                    help="pdf for the report, png for a quick look")
    ap.add_argument("--skip_check", action="store_true",
                    help="skip the join invariants (do not use for a figure "
                         "that will be read)")
    return ap


def resolve(args, *, quiet=False):
    """Run the gate, then load both sides. Returns ``(convexity, optimization)``.

    The gate runs FIRST and hard-stops on failure: a mis-aligned join draws a
    perfectly plausible cloud of noise, which is worse than a crash because
    nothing about the figure looks wrong.
    """
    losses = [s for s in args.losses.split(",") if s] if args.losses else None
    if not args.skip_check:
        fails = check(convexity_path=Path(args.convexity),
                      score_path=Path(args.score_path),
                      optimizer=args.optimizer, losses=losses,
                      matrix=args.matrix, quiet=quiet)
        if fails:
            raise SystemExit("join invariants failed:\n  "
                             + "\n  ".join(fails))
    convexity = load_convexity(Path(args.convexity), losses=losses)
    optimization = load_optimization(Path(args.score_path),
                                     optimizer=args.optimizer,
                                     matrix=args.matrix, losses=losses,
                                     quiet=True)
    return convexity, optimization


def provenance(args) -> str:
    """The one-line footnote every figure carries.

    Two things a reader cannot recover from the axes and will misread without:
    which iterate the score is computed on, and that the join crosses the
    2026-08-26 library reinstall.
    """
    return (f"{SCORE_LABEL.get(args.matrix, args.matrix)} | {args.optimizer} | "
            f"ladder {list(SUCCESS_RADII)} | descriptors from a 91^3 grid on "
            f"+-1.5 (2026-08-21), scores from 2026-08-28/29: this join crosses "
            f"the 2026-08-26 venv reinstall")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--convexity", default=str(CONVEXITY_PATH))
    ap.add_argument("--score_path", default=str(REGION_PATH))
    ap.add_argument("--optimizer", default="polyak")
    ap.add_argument("--matrix", default="raw_d_best", choices=list(MATRICES))
    ap.add_argument("--losses", default=None,
                    help="comma-separated subset (default: every loss present)")
    ap.add_argument("--check", action="store_true",
                    help="run the join's invariants and exit non-zero on failure")
    args = ap.parse_args(argv)

    losses = args.losses.split(",") if args.losses else None
    if not args.check:
        ap.error("nothing to do; pass --check")
    fails = check(convexity_path=Path(args.convexity),
                  score_path=Path(args.score_path), optimizer=args.optimizer,
                  losses=losses, matrix=args.matrix)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
