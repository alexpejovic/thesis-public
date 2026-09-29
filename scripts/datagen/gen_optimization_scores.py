"""Optimization scores: how close gradient descent gets to the true parameters.

The score, per the study definition
-----------------------------------
For each ideal-trace seed: take the fixed set of random initializations, run
``optax.polyak_sgd`` for 100 steps on each, take the **norm of the error on the
optimized parameters** (in transformed space), and the score for that seed is the
**mean over the initializations**.

This replaces the older "first offset from which the optimizer can no longer
converge" metric, which was an escalating-distance search whose result only
reached disk as a line in a log file.

Why polyak, and the one caveat
------------------------------
Polyak's step is ``min((f - f_min) / (||g||^2 + eps), max_lr)``, so the
displacement per step is ``min(f/||g||, max_lr * ||g||)``. On the first branch,
scaling a loss by c scales f and ||g|| alike and the displacement is **exactly
scale-invariant** -- verified: multiplying a loss by 1000 leaves the trajectory
bit-identical. That is what makes it possible to compare eight losses whose
magnitudes differ by ~1000x. It also self-anneals as f -> 0, so no learning-rate
schedule is needed.

``f_min=0.0`` is exact here rather than a guess: every one of the eight losses is
identically zero when the traces match.

The caveat, and exactly when it bites. Invariance holds only while the
``max_learning_rate`` cap does NOT bind. For a locally quadratic loss
``f = a |d|^2``:

    ||g|| = 2a|d|,   step s = f/||g||^2 = 1/(4a),   displacement = s||g|| = |d|/2

so the displacement is scale-free (a cancels) but the *step* is ``1/(4a)`` and
the cap binds whenever ``a < 1/(4 * max_lr)``. On the capped branch the
displacement becomes ``max_lr * 2a|d|``, which does depend on a.

So the cap binds for **flat, low-curvature** losses -- not merely small-valued
ones. Every row records ``polyak_cap_frac``, the fraction of steps where the cap
bound, so the capped fraction can be reported per loss.

**Measured on the real problem, the cap binds most of the time and that is a
good thing.** On a 3-parameter spike-w1 landscape (seed 0, mean initial distance
3.74, 100 steps):

    max_lr=1e0   final error 2.43   cap_frac 0.67
    max_lr=1e3   final error 28.7   cap_frac 0.92
    max_lr=1e6   final error 43.6   cap_frac 0.97

These landscapes are locally flat and noisy, so the raw Polyak step f/||g||^2 is
enormous and raising the cap makes the optimizer diverge far outside the
parameter bounds. The cap is therefore doing the job of a trust region, and
``max_learning_rate=1.0`` is the right default.

The honest consequence for interpretation: when the cap binds, polyak reduces to
plain gradient descent at lr = max_lr, so the scale-invariance above does NOT
hold and a loss with systematically larger gradients will appear to optimize
better. That is a caveat to state alongside the results, not a bug -- and
``polyak_cap_frac`` is what lets you state it quantitatively per loss.

``eps=1e-30``, not optax's default of 0. At the exact optimum f and ||g||^2 are
both 0 and the step is 0/0. Measured, 1e-30 guards that while leaving the
convergence floor untouched (7.2e-9), whereas 1e-12 degrades it to 5.7e-8. The
trailing ``zero_nans()`` in the chain is a second line of defence -- and it is a
real one: it is what keeps the exact-optimum case finite.

Sharding, so partial results are usable
---------------------------------------
``--shard N`` splits the ideal-seed range into blocks of N and appends each block
as its own row the moment it finishes. A shard is a valid row under the same
schema -- just with a narrower ``ideal_seeds`` range -- so the plotting scripts
concatenate shards with no special handling and a heatmap can be drawn from the
first shard onward. ``--merge`` folds complete shard sets into one row.

Usage (from encoding/):
    python scripts/datagen/gen_optimization_scores.py \\
        --params Na_gNa K_gK Km_gKm --ideal_seeds 0 100 --init_seeds 0 1000 \\
        --loss spike-w1 --shard 10
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # encoding/
sys.path.insert(0, _HERE)

import jsonl_store as JS  # noqa: E402
import losses as L  # noqa: E402
import param_groups as pg  # noqa: E402
import sim as S  # noqa: E402
from helper import get_datagen_config  # noqa: E402
from logger import get_std_logger  # noqa: E402

OUT_PATH = Path("./data/optimization-scores.jsonl")

KEY_FIELDS = ("params", "ideal_seeds", "init_seeds", "init_bounds", "loss")

# See the module docstring: guards the 0/0 step at the exact optimum without
# moving the convergence floor.
POLYAK_EPS = 1e-30

OPTIMIZERS = ("polyak", "adam", "rmsprop")

# Success radii for the success-RATE score, in transformed space.
#
# The mean-error score is dominated by the runs that fly off to distance 5-6, so
# a single bad start can swamp ten good ones. A success rate asks the question the
# study actually cares about -- "did the optimizer land on the right neuron" --
# and is bounded, so divergence cannot skew it.
#
# Several radii are computed from the same run because they cost nothing once the
# distance matrix exists, and the right one is not obvious a priori:
#
#   0.100  the STRICT basin radius (convexity.BASIN_STRICT_FLOOR), so
#          `success@0.1` and the `basin` descriptor ask the same question
#   0.375  the WIDE basin radius (0.25 x the 1.5 half-span), matching `basin_wide`
#   0.173  0.10 * sqrt(3): "within 0.1 per parameter", the first study's convention
#   0.433  0.25 * sqrt(3): "within 0.25 per parameter"
#   0.866  0.50 * sqrt(3): a loose "same basin" criterion
SUCCESS_TOLS = (0.1, 0.173, 0.375, 0.433, 0.866)


def _make_optimizer(name, max_lr, eps, steps):
    """The gradient transformation, with NaN guards on both sides.

    ``zero_nans`` comes first to sanitize the incoming gradient (``nan_to_num``
    inside the simulator has a NaN cotangent), and again last because polyak can
    itself produce a NaN from a 0/0 step -- a guard placed only before it cannot
    catch that.
    """
    if name == "polyak":
        core = optax.polyak_sgd(
            max_learning_rate=max_lr, f_min=0.0, eps=eps
        )
    elif name == "adam":
        core = optax.adam(optax.cosine_decay_schedule(max_lr, steps))
    elif name == "rmsprop":
        core = optax.rmsprop(max_lr)
    else:
        raise ValueError(f"unknown optimizer {name!r}; want one of {OPTIMIZERS}")
    return optax.chain(optax.zero_nans(), core, optax.zero_nans())


def build_runner(
    traces_of,
    loss_fn,
    idx,
    *,
    steps=100,
    max_lr=1.0,
    eps=POLYAK_EPS,
    optimizer="polyak",
):
    """A jitted, vmapped optimizer run returning (error norm, cap fraction).

    ``traces_of(x8, states6) -> (n_amps, n_t)`` is injected rather than imported
    so the whole scan/vmap/optax/scatter path can be exercised against an
    analytic simulator in a test, with no simulation at all.

    Vmapped over ``(target8, states6, ideal_traces, x0)``, so one call handles a
    batch of (ideal seed, init seed) pairs.
    """
    opt = _make_optimizer(optimizer, max_lr, eps, steps)
    is_polyak = optimizer == "polyak"

    def run_one(target8, states6, ideal_traces, x0):
        def objective(x_sub):
            # Scatter the optimized subset into the full 8-vector. Gradients
            # reach only x_sub, so the frozen parameters stay pinned at truth for
            # free and the D-dim error norm equals the 8-dim one exactly.
            full = target8.at[idx].set(x_sub)
            return loss_fn(traces_of(full, states6), ideal_traces)

        def body(carry, _):
            x, state, n_capped = carry
            value, grad = jax.value_and_grad(objective)(x)
            if is_polyak:
                # Recompute the uncapped step to see whether the cap bound.
                raw = value / (jnp.sum(grad ** 2) + eps)
                n_capped = n_capped + jnp.where(raw > max_lr, 1.0, 0.0)
                updates, state = opt.update(grad, state, x, value=value)
            else:
                updates, state = opt.update(grad, state, x)
            return (optax.apply_updates(x, updates), state, n_capped), None

        init = (x0, opt.init(x0), jnp.zeros((), dtype=x0.dtype))
        (x_final, _, n_capped), _ = jax.lax.scan(body, init, None, length=steps)
        return jnp.linalg.norm(x_final - target8[idx]), n_capped / steps

    return jax.jit(jax.vmap(run_one, in_axes=(0, 0, 0, 0)))


def score_shard(
    simulator,
    loss_fn,
    params,
    ideal_seeds,
    init_seeds,
    *,
    steps=100,
    max_lr=1.0,
    eps=POLYAK_EPS,
    optimizer="polyak",
    chunk=128,
    init_bounds=(-3.0, 3.0),
    logger=None,
):
    """Aggregates for one block of ideal seeds.

    Returns ``(mean, std, median, success_rates, cap_frac, raw)`` where
    ``success_rates`` maps each radius in ``SUCCESS_TOLS`` to a per-ideal-seed
    fraction of initializations that finished within that radius of the truth,
    and ``raw`` is the full ``(n_ideal, n_init)`` matrix of final error norms.
    """
    idx = S.param_indices(params)
    dim = len(params)

    ideal_list = list(range(*ideal_seeds))
    init_list = list(range(*init_seeds))
    n_i, n_j = len(ideal_list), len(init_list)

    targets = simulator.targets_for_seeds(ideal_list)          # (I, 8)
    states = simulator.init_states_for_seeds(ideal_list)        # (I, 6)
    ideal_traces = jax.jit(jax.vmap(simulator.traces_of, in_axes=(0, 0)))(
        targets, states
    )                                                          # (I, n_amps, n_t)
    lo_b, hi_b = init_bounds
    starts = jnp.stack(
        [S.start_for(j, dim, lo=lo_b, hi=hi_b) for j in init_list]
    )                                                          # (J, D)

    runner = build_runner(
        simulator.traces_of, loss_fn, idx,
        steps=steps, max_lr=max_lr, eps=eps, optimizer=optimizer,
    )

    # Ideal-major flattening, so a chunk usually covers whole ideal seeds.
    ii, jj = np.meshgrid(np.arange(n_i), np.arange(n_j), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    dist = np.empty(n_i * n_j)
    caps = np.empty(n_i * n_j)

    t0 = time.time()
    total = n_i * n_j
    for lo in range(0, total, chunk):
        hi = min(lo + chunk, total)
        sl = slice(lo, hi)
        d, c = runner(
            targets[ii[sl]], states[ii[sl]], ideal_traces[ii[sl]], starts[jj[sl]]
        )
        # The only host sync, once per chunk rather than once per step.
        dist[sl] = np.asarray(d)
        caps[sl] = np.asarray(c)
        if logger:
            logger.info(f"{hi}/{total} cells ({time.time() - t0:.0f}s)")

    raw = dist.reshape(n_i, n_j)
    success = {t: (raw < t).mean(axis=1) for t in SUCCESS_TOLS}
    return (
        raw.mean(axis=1),
        raw.std(axis=1),
        np.median(raw, axis=1),
        success,
        float(caps.mean()),
        raw,
    )


def _commit() -> str:
    try:
        from git import Repo

        return Repo("../").commit().hexsha
    except Exception:  # noqa: BLE001
        return "unknown"


def _device() -> str:
    try:
        return jax.devices()[0].device_kind
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", nargs="+",
                    help="required unless --merge")
    ap.add_argument("--ideal_seeds", nargs=2, type=int,
                    default=list(pg.IDEAL_SEEDS), metavar=("LO", "HI"))
    ap.add_argument("--init_seeds", nargs=2, type=int,
                    default=list(pg.INIT_SEEDS), metavar=("LO", "HI"))
    ap.add_argument("--loss", choices=list(L.LOSS_NAMES),
                    help="required unless --merge")
    ap.add_argument("--channel_type", default="Pospischil")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--optimizer", default="polyak", choices=OPTIMIZERS)
    ap.add_argument("--max_learning_rate", type=float, default=1.0)
    ap.add_argument("--polyak_eps", type=float, default=POLYAK_EPS)
    ap.add_argument("--init_bounds", nargs=2, type=float, default=[-3.0, 3.0],
                    metavar=("LO", "HI"),
                    help="range the optimization starts are drawn from, in "
                         "transformed space. MUST match the landscape window the "
                         "convexity descriptors were computed on (GRID_PLAN "
                         "bounds, currently +-1.5) or the two scores describe "
                         "different regions of parameter space and correlating "
                         "them is meaningless. Recorded in meta and part of the "
                         "budget key, so runs with different bounds are never "
                         "merged.")
    ap.add_argument("--chunk", type=int, default=2048,
                    help="(ideal, init) pairs per jitted call. Batched steps/sec "
                         "is CONSTANT in chunk (the 6001-step scan is "
                         "latency-bound), so throughput scales linearly with it "
                         "and the only limit is memory. Measured on an A100-40GB: "
                         "512 is the ceiling without checkpointing, 2048 with.")
    ap.add_argument("--checkpoint", nargs="*", type=int, default=[78, 78],
                    metavar="LEN",
                    help="jx.integrate recursive checkpointing, e.g. 78 78 "
                         "(78*78 >= 6001). Costs ~1.5x compute but allows 4x the "
                         "chunk, a net 2.7x speedup. Pass with no values to "
                         "disable.")
    ap.add_argument("--shard", type=int, default=10,
                    help="ideal seeds per output row; each is appended as it "
                         "finishes so partial results are plottable")
    ap.add_argument("--save_raw", action="store_true",
                    help="also store the full (ideal, init) error matrix")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="de-duplicate and rewrite the output file, then exit")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    if args.merge:
        # A repair/de-duplicate pass over the whole file; needs no job flags.
        kept = JS.collect(out_path, KEY_FIELDS)
        print(f"merged {kept} rows into {out_path}")
        return 0

    missing = [n for n in ("params", "loss") if getattr(args, n) is None]
    if missing:
        ap.error("the following are required unless --merge is given: "
                 + ", ".join("--" + n for n in missing))

    params = S.canonical(args.params)
    set_key = pg.set_key(params)
    i_lo, i_hi = args.ideal_seeds
    j_lo, j_hi = args.init_seeds
    if i_hi <= i_lo or j_hi <= j_lo:
        ap.error("seed ranges must be non-empty half-open [LO, HI)")

    logger = get_std_logger()
    config = get_datagen_config()
    checkpoint = args.checkpoint if args.checkpoint else None
    simulator = S.Sim(args.channel_type, config, checkpoint_lengths=checkpoint)
    loss_fn = L.get_loss_fn(L.LOSS_SPECS[args.loss], args.channel_type, config)

    meta_common = {
        "steps": args.steps,
        "optimizer": args.optimizer,
        "max_learning_rate": args.max_learning_rate,
        "polyak_eps": args.polyak_eps,
        "f_min": 0.0,
        "init_bounds": [float(args.init_bounds[0]), float(args.init_bounds[1])],
        "n_amps": int(simulator.n_amps),
        "chunk": args.chunk,
        "checkpoint": checkpoint,
        "x64": True,
        "commit": _commit(),
        "device": _device(),
    }

    done = JS.existing_keys(out_path, KEY_FIELDS) if not args.force else set()
    shard = args.shard if args.shard > 0 else (i_hi - i_lo)

    for a in range(i_lo, i_hi, shard):
        b = min(a + shard, i_hi)
        probe = {
            "params": params,
            "ideal_seeds": [a, b],
            "init_seeds": [j_lo, j_hi],
            # Top-level, not only in meta, because it is part of the resume key.
            "init_bounds": [float(args.init_bounds[0]), float(args.init_bounds[1])],
            "loss": list(L.LOSS_SPECS[args.loss]),
        }
        if JS.row_key(probe, KEY_FIELDS) in done:
            print(f"skip {set_key} {args.loss} ideal[{a},{b})  (already done)")
            continue

        t0 = time.time()
        logger.info(f"{set_key} {args.loss} ideal[{a},{b}) x init[{j_lo},{j_hi})")
        mean, std, median, success, cap_frac, raw = score_shard(
            simulator, loss_fn, params, (a, b), (j_lo, j_hi),
            steps=args.steps, max_lr=args.max_learning_rate,
            eps=args.polyak_eps, optimizer=args.optimizer,
            chunk=args.chunk, init_bounds=tuple(args.init_bounds), logger=logger,
        )

        row = {
            **probe,
            "optimization-scores": [float(v) for v in mean],
            "optimization-scores-std": [float(v) for v in std],
            "optimization-scores-median": [float(v) for v in median],
            # Fraction of initializations that landed within each radius of the
            # truth. Bounded, so a diverging start cannot skew it the way the
            # mean error can.
            **{f"optimization-success@{t:g}": [float(v) for v in success[t]]
               for t in SUCCESS_TOLS},
            "loss_name": args.loss,
            "set": set_key,
            "dim": len(params),
            "meta": {**meta_common,
                     "polyak_cap_frac": cap_frac,
                     "wall_s": round(time.time() - t0, 1),
                     "created_utc": datetime.now(timezone.utc).isoformat()},
        }
        JS.append_row(out_path, row)

        if args.save_raw:
            raw_dir = out_path.parent / "optimization-scores-raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{set_key}__{args.loss}__{a}-{b}__{j_lo}-{j_hi}"
            np.savez_compressed(raw_dir / f"{stem}.npz", errors=raw)

        print(f"wrote {set_key} {args.loss} ideal[{a},{b}): "
              f"mean score {mean.mean():.4f}, cap_frac {cap_frac:.3f} "
              f"({time.time() - t0:.0f}s)")
        if cap_frac > 0.5 and args.optimizer == "polyak":
            print("  WARNING: the polyak step cap bound on most steps, so this "
                  "loss's scores are NOT scale-invariant -- see the docstring.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
