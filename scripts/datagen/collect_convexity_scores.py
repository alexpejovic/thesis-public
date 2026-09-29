"""Turn the cached sweep descriptors into ``data/convexity-scores.jsonl``.

``gen_convexity_scores.py`` handles one (parameter set, loss, descriptor) cell per
invocation, which is the right granularity for a SLURM array over *sweeps*. But
once the sweeps exist, scoring is pure JSON reading -- 56 sets x 8 losses x 6
descriptors = 2,688 rows -- and paying a Python plus JAX import for each would
cost far more than the work. This does all of them in one process, with the
per-sweep descriptor files read once and reused across the 48 (loss, descriptor)
cells that share them.

By default a row is written only when **every** seed in the range has a score, so
running this while the sweep array is still going is harmless: incomplete cells
are reported and skipped rather than written, which matters because a row is
keyed by (params, seed_range, loss, descriptor) and a partial row would otherwise
block the complete one from ever being written.

    # after the sweep array finishes
    python scripts/datagen/collect_convexity_scores.py

    # look at what is available mid-run, writing partial rows to a scratch file
    python scripts/datagen/collect_convexity_scores.py --allow-partial \\
        --min-seeds 20 --out /tmp/partial-convexity.jsonl
"""

import argparse
import json
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
from gen_convexity_scores import KEY_FIELDS, OUT_PATH, _commit  # noqa: E402
from helper import get_datagen_config  # noqa: E402


def _load_descriptors(path: Path, cache: dict):
    """``descriptors.json`` for one sweep, memoised across the 48 cells."""
    key = str(path)
    if key not in cache:
        f = path / "descriptors.json"
        try:
            cache[key] = json.loads(f.read_text()) if f.exists() else None
        except json.JSONDecodeError:
            cache[key] = None
    return cache[key]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs=2, type=int, default=list(pg.IDEAL_SEEDS),
                    metavar=("LO", "HI"))
    ap.add_argument("--sets", default=None, help="comma-separated subset")
    ap.add_argument("--losses", default=None, help="comma-separated subset")
    ap.add_argument("--convexity", default=None, help="comma-separated subset")
    ap.add_argument("--channel_type", default="Pospischil")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--allow-partial", dest="allow_partial", action="store_true",
                    help="write a row even if some seeds have no sweep yet")
    ap.add_argument("--min-seeds", dest="min_seeds", type=int, default=None,
                    help="with --allow-partial, the minimum scored seeds to write")
    ap.add_argument("--force", action="store_true",
                    help="rewrite rows that already exist")
    args = ap.parse_args(argv)

    config = get_datagen_config()
    out_path = Path(args.out)
    lo, hi = args.seeds
    seeds = list(range(lo, hi))

    sets = args.sets.split(",") if args.sets else list(pg.STUDY_SETS)
    losses = args.losses.split(",") if args.losses else list(L.LOSS_NAMES)
    stats = args.convexity.split(",") if args.convexity else list(cx.SCORE_NAMES)

    done = set() if args.force else JS.existing_keys(out_path, KEY_FIELDS)
    cache: dict = {}
    n_written = n_skipped = n_incomplete = 0
    commit, created = _commit(), datetime.now(timezone.utc).isoformat()

    for set_key in sets:
        params = pg.params_for(set_key)
        grid = pg.grid_for(set_key)
        bounds, samples = grid["bounds"], grid["samples"]

        # Resolve each seed's sweep directory once for the whole set: the 48
        # (loss, descriptor) cells below all read the same 100 files.
        per_seed = []
        for seed in seeds:
            key = G.sweep_key(args.channel_type, params, seed, bounds, samples,
                              config)
            path = G.sweep_dir(key)
            per_seed.append((seed, path, _load_descriptors(path, cache)))

        available = sum(1 for _, _, d in per_seed if d is not None)
        need = len(seeds) if not args.allow_partial else (args.min_seeds or 1)
        if available < need:
            n_incomplete += len(losses) * len(stats)
            print(f"  {set_key}: only {available}/{len(seeds)} sweeps present "
                  f"-- skipping its {len(losses) * len(stats)} cells")
            continue

        for loss_name in losses:
            for stat in stats:
                probe = {
                    "params": params,
                    "seed_range": [lo, hi],
                    "loss": list(L.LOSS_SPECS[loss_name]),
                    "convexity": stat,
                }
                if JS.row_key(probe, KEY_FIELDS) in done:
                    n_skipped += 1
                    continue

                scores, skipped, hashes = [], [], []
                for seed, path, desc in per_seed:
                    hashes.append(path.name)
                    value = None
                    if desc is not None:
                        cell = desc.get("scores", {}).get(loss_name)
                        if cell is not None:
                            value = float(cell[stat])
                    scores.append(value)
                    if value is None:
                        skipped.append(seed)

                # The set-level `available` check above counts SWEEPS; this one
                # counts scores for THIS loss. They differ whenever a sweep was
                # cached without a loss the current loss set contains -- as every
                # sweep on disk was, for `encoder-zscore-2`. Without this a cell
                # like that is written as an all-null row, and because a row is
                # keyed by (params, seed_range, loss, convexity) that null row
                # permanently blocks the real one. Skipping honours the contract
                # this module's docstring already states.
                if len(seeds) - len(skipped) < need:
                    n_incomplete += 1
                    if stat == stats[0]:  # once per loss, not once per cell
                        print(f"  {set_key}: {loss_name} scored on "
                              f"{len(seeds) - len(skipped)}/{len(seeds)} seeds "
                              f"-- skipping its {len(stats)} cells")
                    continue

                JS.append_row(out_path, {
                    **probe,
                    "convexity-scores": scores,
                    "loss_name": loss_name,
                    "set": set_key,
                    "seeds": seeds,
                    "skipped_seeds": skipped,
                    "grid": {"bounds": list(bounds), "samples": samples},
                    "dim": len(params),
                    "sweep_key_hashes": hashes,
                    "loss_set_version": L.LOSS_SET_VERSION,
                    "basin_wide_fraction": cx.BASIN_WIDE_FRACTION,
                    "channel_type": args.channel_type,
                    "commit": commit,
                    "created_utc": created,
                })
                n_written += 1

    print(f"\nwrote {n_written} rows, skipped {n_skipped} already present, "
          f"{n_incomplete} cells not yet scorable -> {out_path}")
    if n_incomplete and not args.allow_partial:
        print("re-run once the sweep array has finished to pick those up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
