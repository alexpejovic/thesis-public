"""Line graph: Pearson correlation per parameter set, one line per loss.

Each x position is its own correlation over that parameter set's shared ideal
seeds -- no pooling, which is the point of a line plot: it shows whether a
descriptor predicts optimization *consistently* across parameter sets, or only
on average.

**Read the bands before the lines.** The region study used 10 ideal seeds, so
every point here is an n=10 correlation and its Fisher-z 95% band is roughly
+-0.6 around zero. Individual wiggles carry essentially no information; what the
figure can support is whether a loss's line sits mostly above or mostly below
zero, and the median annotated per line. That is a real limitation of this unit,
not of the descriptor -- the pooled version in ``plot_convexity_vs_opt_grid.py``
is where the n=560 numbers live.

    cd encoding
    ../.venv/bin/python scripts/plotting/plot_convexity_vs_opt_lines.py --stat all
    ../.venv/bin/python scripts/plotting/plot_convexity_vs_opt_lines.py \\
        --stat basin_wide --sets gNa+gK+gKm,gLeak+eLeak+gNa
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import convexity_join as CJ  # noqa: E402
import plot_style as PS  # noqa: E402
from helper import get_convexity_vs_opt_config  # noqa: E402


def per_set_pearson(convexity, optimization, stat, losses, sets):
    """``{loss: {set: corr dict}}`` for one descriptor."""
    pairs = CJ.paired(convexity, optimization, stat)
    out = {loss: {} for loss in losses}
    for (set_key, loss), (x, y) in pairs.items():
        if loss not in out or set_key not in sets:
            continue
        out[loss][set_key] = PS.corr(x, y)
    return out


def draw(ax, data, sets, losses, stat, *, legend=False):
    xs = np.arange(len(sets))
    medians = {}
    for loss in losses:
        rows = data.get(loss, {})
        y = np.array([rows.get(s, {}).get("pearson", np.nan) for s in sets])
        n = np.array([rows.get(s, {}).get("n", 0) for s in sets])
        if not np.any(np.isfinite(y)):
            continue
        colour = PS.COLORS.get(loss, PS.MUTED)
        ax.plot(xs, y, marker="o", markersize=3.5, linewidth=1.4, color=colour,
                label=PS.LABELS.get(loss, loss))
        lo = np.full_like(y, np.nan)
        hi = np.full_like(y, np.nan)
        for i, (r, nn) in enumerate(zip(y, n)):
            lo[i], hi[i] = PS.fisher_ci(r, nn)
        ok = np.isfinite(lo) & np.isfinite(hi)
        if ok.any():
            ax.fill_between(xs[ok], lo[ok], hi[ok], color=colour, alpha=0.10,
                            linewidth=0)
        med = float(np.nanmedian(y)) if np.any(np.isfinite(y)) else np.nan
        medians[loss] = med
        # The only statistic on this figure that survives n=10: the median of
        # the 56 per-set correlations, drawn as a flat reference line.
        if np.isfinite(med):
            ax.axhline(med, color=colour, linewidth=0.9, linestyle=":",
                       alpha=0.75)

    ax.axhline(0.0, color=PS.MUTED, linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_ylim(-1.05, 1.05)
    # 56 set names never fit; label every `step`-th and let the rest be ticks.
    # The x axis exists to say "these are different problems", not to be read
    # off -- the per-set numbers are all in pearson_lines.json.
    step = max(1, len(sets) // 14)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels(sets[::step], rotation=40, ha="right", fontsize=6)
    ax.set_xlim(-0.5, len(sets) - 0.5)
    sign = "+" if CJ.EXPECTED_SIGN[stat] > 0 else "-"
    med_txt = "  ".join(f"{PS.SHORT_LABELS.get(l, l)} {m:+.2f}"
                        for l, m in medians.items() if np.isfinite(m))
    PS.style(ax, ylabel="Pearson r",
             title=f"{CJ.STAT_LABELS[stat]}  (expected {sign})")
    if med_txt:
        ax.text(0.99, 0.02, f"median r   {med_txt}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5, color=PS.MUTED)
    if legend:
        ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    return medians


def main(argv=None) -> int:
    cfg = get_convexity_vs_opt_config()
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    CJ.add_common_args(ap, cfg)
    ap.add_argument("--stat", default="all",
                    help="one descriptor, or 'all' for a 2x3 panel")
    ap.add_argument("--sets", default=None,
                    help="comma-separated parameter-set keys "
                         "(default: all present in both files)")
    args = ap.parse_args(argv)

    convexity, optimization = CJ.resolve(args)
    cov = CJ.coverage(convexity, optimization)
    if not convexity or not optimization:
        print(f"nothing to plot: {len(convexity)} convexity keys, "
              f"{len(optimization)} optimization keys")
        return 1

    losses = [l for l in args.losses.split(",") if l in set(cov["losses"])]
    if args.sets:
        sets = args.sets.split(",")
    else:
        # Only sets with both sides present, sorted for a stable x-axis.
        sets = sorted({k[0] for k in optimization} & {k[0] for k in convexity})
    if not sets or not losses:
        print("no parameter sets or losses in common between the two files")
        return 1

    stats = (list(CJ.STAT_NAMES) if args.stat == "all" else [args.stat])
    for s in stats:
        if s not in CJ.STAT_NAMES:
            ap.error(f"unknown descriptor {s!r}; want one of "
                     f"{list(CJ.STAT_NAMES)} or 'all'")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.format
    foot = CJ.provenance(args)
    score = CJ.SCORE_LABEL.get(args.matrix, args.matrix)
    band = (f"each point is an n={len(next(iter(CJ.paired(convexity, optimization, stats[0]).values()))[0])} "
            f"correlation; the shaded Fisher-z 95% band is about +-0.6, so read "
            f"the dotted per-loss medians, not the wiggles")

    results, medians = {}, {}
    if len(stats) == 1:
        stat = stats[0]
        data = per_set_pearson(convexity, optimization, stat, losses, set(sets))
        results[stat] = data
        fig, ax = plt.subplots(figsize=(11.0, 4.8))
        medians[stat] = draw(ax, data, sets, losses, stat, legend=True)
        ax.set_xlabel("parameter set", color=PS.TEXT, fontsize=10)
        fig.suptitle(f"Per-set correlation with {score}", color=PS.TEXT,
                     fontsize=11, x=0.01, ha="left")
        fig.text(0.01, 0.005, band + "  |  " + foot, color=PS.MUTED,
                 fontsize=6, ha="left", va="bottom", wrap=True)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        name = f"pearson_lines_{stat}.{ext}"
        PS.save(fig, out_dir / name)
    else:
        ncol, nrow = 2, 3
        fig, axes = plt.subplots(
            nrow, ncol, sharey=True, squeeze=False,
            figsize=(7.6 * ncol, 3.4 * nrow))
        for k, stat in enumerate(stats):
            ax = axes[k // ncol][k % ncol]
            data = per_set_pearson(convexity, optimization, stat, losses,
                                   set(sets))
            results[stat] = data
            medians[stat] = draw(ax, data, sets, losses, stat, legend=(k == 0))
        for k in range(len(stats), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle(f"Per-set correlation with {score}", color=PS.TEXT,
                     fontsize=12, x=0.005, ha="left")
        fig.text(0.005, 0.004, band + "  |  " + foot, color=PS.MUTED,
                 fontsize=6.5, ha="left", va="bottom", wrap=True)
        fig.tight_layout(rect=(0, 0.03, 1, 0.975))
        name = f"pearson_lines_all.{ext}"
        PS.save(fig, out_dir / name)

    (out_dir / "pearson_lines.json").write_text(json.dumps({
        "provenance": foot, "matrix": args.matrix, "optimizer": args.optimizer,
        "sets": sets, "losses": losses, "coverage": cov,
        "medians": medians, "results": results,
    }, indent=1))
    print(f"  wrote {out_dir}/{name} and pearson_lines.json "
          f"({len(sets)} sets, {len(losses)} losses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
