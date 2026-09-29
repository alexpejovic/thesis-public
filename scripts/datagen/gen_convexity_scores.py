"""Convexity scores for one (parameter set, loss, descriptor) across ideal seeds.

For a parameter set, a range of ideal-trace seeds, a loss and a convexity
descriptor, this writes one JSON object to ``data/convexity-scores.jsonl``
holding one score per seed.

Two phases, as required: it first makes sure the loss sweep each seed needs
exists, then scores it. **A sweep that already exists is loaded, never
regenerated** -- and because one sweep carries all eight losses scored on all six
descriptors, the 48 (loss, descriptor) invocations after the first are pure
cache reads.

    python scripts/datagen/gen_convexity_scores.py \\
        --params Na_gNa K_gK Km_gKm --seeds 0 100 \\
        --loss encoder-minmax --convexity basin

``--no-generate`` reports what is missing instead of simulating it, which is the
mode to use once a SLURM array has pre-warmed the sweeps.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # encoding/
sys.path.insert(0, _HERE)

import convexity as cx  # noqa: E402
import gen_loss_sweep as G  # noqa: E402
import jsonl_store as JS  # noqa: E402
import losses as L  # noqa: E402
import param_groups as pg  # noqa: E402
import sim as S  # noqa: E402
from helper import get_datagen_config  # noqa: E402
from logger import get_std_logger  # noqa: E402

OUT_PATH = Path("./data/convexity-scores.jsonl")

# Fields that identify a row for resume purposes.
KEY_FIELDS = ("params", "seed_range", "loss", "convexity")


def _commit() -> str:
    try:
        from git import Repo

        return Repo("../").commit().hexsha
    except Exception:  # noqa: BLE001
        return "unknown"


def build_row(
    params,
    seed_range,
    loss_name,
    convexity,
    *,
    channel_type="Pospischil",
    config=None,
    generate=True,
    batch_size=512,
    keep_grid=False,
    bounds=None,
    samples=None,
    loss_names=None,
    logger=None,
) -> dict:
    config = config if config is not None else get_datagen_config()
    params = S.canonical(params)
    key = pg.set_key(params)
    # Every loss by default, because one simulation serves all of them: that is
    # the whole reason the sweep cache is worth having. But it must be
    # overridable. A cached sweep that lacks even one requested loss is a
    # `topup`, and a topup is a FULL re-simulation -- the raw grids are not kept,
    # so a loss that was never scored cannot be scored from cache. The sweeps on
    # disk hold the eight losses of `LOSS_SET_VERSION` 1; asking for all nine on
    # a tree at version 2 would re-simulate every one of them.
    loss_names = list(L.LOSS_NAMES) if loss_names is None else list(loss_names)
    # Fall back to the registry for whatever was not given explicitly, so a
    # smoke test or a pilot can use a smaller grid without editing the plan.
    if bounds is None or samples is None:
        grid = pg.grid_for(key)
        bounds = bounds if bounds is not None else grid["bounds"]
        samples = samples if samples is not None else grid["samples"]

    lo, hi = seed_range
    seeds = list(range(lo, hi))
    scores, skipped, hashes = [], [], []
    n_generated = 0

    for seed in seeds:
        path, status = G.ensure_sweep(
            channel_type, params, seed, bounds, samples, loss_names,
            config=config, generate=generate, batch_size=batch_size,
            keep_grid=keep_grid, logger=logger,
        )
        if status != "hit":
            n_generated += 1
        hashes.append(path.name)

        if not (path / "descriptors.json").exists():
            # --no-generate and the sweep is absent.
            scores.append(None)
            skipped.append(seed)
            continue

        value = G.read_descriptor(path, loss_name, convexity)
        if value is None:
            # Recorded non-spiking seed: no landscape to score.
            scores.append(None)
            skipped.append(seed)
        else:
            scores.append(value)

    if logger:
        logger.info(
            f"{key} {loss_name}/{convexity}: {len(seeds) - len(skipped)}"
            f"/{len(seeds)} scored, {n_generated} sweeps generated"
        )

    return {
        # The five keys the study asks for.
        "params": params,
        "seed_range": [lo, hi],
        "loss": list(L.LOSS_SPECS[loss_name]),
        "convexity": convexity,
        "convexity-scores": scores,
        # Provenance. `convexity-scores` always has one entry per seed, with
        # null for a skipped one, so index <-> seed is never ambiguous.
        "loss_name": loss_name,
        "set": key,
        "seeds": seeds,
        "skipped_seeds": skipped,
        "grid": {"bounds": list(bounds), "samples": samples},
        "dim": len(params),
        "sweep_key_hashes": hashes,
        "loss_set_version": L.LOSS_SET_VERSION,
        "basin_wide_fraction": cx.BASIN_WIDE_FRACTION,
        "channel_type": channel_type,
        "commit": _commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set_key", help="parameter set as a param_groups key, "
                                      "instead of --params")
    ap.add_argument("--params", nargs="+",
                    help="required unless --set_key or --collect is given")
    ap.add_argument("--seeds", nargs=2, type=int, default=list(pg.IDEAL_SEEDS),
                    metavar=("LO", "HI"), help="half-open [LO, HI)")
    ap.add_argument("--loss", choices=list(L.LOSS_NAMES),
                    help="required unless --collect")
    ap.add_argument("--convexity", choices=list(cx.SCORE_NAMES),
                    help="required unless --collect")
    ap.add_argument("--channel_type", default="Pospischil")
    ap.add_argument("--generate", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="simulate missing sweeps (default) or just report gaps")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--keep_grid", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="also store the raw loss grids (48 MB/sweep at 91^3)")
    ap.add_argument("--losses", type=G.parse_losses, default=None,
                    help="comma-separated losses to ensure are scored on each "
                         "sweep; default all of them. Narrow it to the losses a "
                         "cached sweep already holds to avoid forcing a "
                         "re-simulation.")
    ap.add_argument("--opt_bounds", nargs=2, type=float, default=None,
                    help="override the registry grid bounds (smoke tests, pilot)")
    ap.add_argument("--opt_samples", type=int, default=None,
                    help="override the registry samples per axis")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--force", action="store_true",
                    help="recompute even if this exact row already exists")
    ap.add_argument("--collect", action="store_true",
                    help="de-duplicate and rewrite the output file, then exit")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    if args.collect:
        # A repair pass over the whole file; it needs none of the job flags.
        kept = JS.collect(out_path, KEY_FIELDS)
        print(f"collected {kept} rows into {out_path}")
        return 0

    if (args.set_key is None) == (args.params is None):
        ap.error("give exactly one of --set_key or --params")
    missing = [n for n in ("loss", "convexity") if getattr(args, n) is None]
    if missing:
        ap.error("the following are required unless --collect is given: "
                 + ", ".join("--" + n for n in missing))

    params = S.canonical(args.params if args.params else pg.params_for(args.set_key))
    seed_range = [args.seeds[0], args.seeds[1]]
    if seed_range[1] <= seed_range[0]:
        ap.error(f"--seeds must be a non-empty half-open range, got {seed_range}")

    probe = {
        "params": params,
        "seed_range": seed_range,
        "loss": list(L.LOSS_SPECS[args.loss]),
        "convexity": args.convexity,
    }
    key = JS.row_key(probe, KEY_FIELDS)
    if not args.force and key in JS.existing_keys(out_path, KEY_FIELDS):
        print(f"already done: {pg.set_key(params)} {args.loss}/{args.convexity} "
              f"seeds {seed_range}")
        return 0

    # stdout only: the per-sweep detail already goes to each sweep dir's log.
    logger = get_std_logger()
    row = build_row(
        params, seed_range, args.loss, args.convexity,
        channel_type=args.channel_type,
        generate=args.generate,
        batch_size=args.batch_size,
        keep_grid=args.keep_grid,
        bounds=tuple(args.opt_bounds) if args.opt_bounds else None,
        samples=args.opt_samples,
        loss_names=args.losses,
        logger=logger,
    )
    JS.append_row(out_path, row)

    n_ok = sum(1 for v in row["convexity-scores"] if v is not None)
    print(f"wrote {pg.set_key(params)} {args.loss}/{args.convexity}: "
          f"{n_ok}/{len(row['seeds'])} seeds scored -> {out_path}")
    if row["skipped_seeds"]:
        print(f"  skipped seeds: {row['skipped_seeds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
