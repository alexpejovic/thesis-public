"""Heatmap: losses (y) x convexity descriptors (x), cells = Spearman rho.

Each cell is the rank correlation between a loss's region-of-success score and
one convexity descriptor of that same loss's landscape. Rebuilt from
``big-loss-scoring``'s version of this figure, which correlated against the
retired mean-final-distance score; the join now goes through
``convexity_join.py`` and the score is recomputed from ``raw_d_best``.

Three poolings are written, and they answer different questions:

``spearman_grid``  (headline)
    over ideal-trace seeds, rank-normalized inside each parameter set before
    pooling. Anything that moves both sides at once -- a set that is simply
    harder to fit -- would otherwise read as correlation that really says
    "harder problems are harder". 56 sets x 10 shared seeds = 560 points.

``spearman_grid_raw``
    the same seeds pooled without rank-normalizing, for reference. Confounded by
    set difficulty.

``spearman_grid_by_set``
    one point per parameter set (median over its shared seeds), n=56. The
    between-landscape view the first study reported, which gives much larger
    |rho| because most of the variance is between problems of different
    difficulty.

Reading the sign: the region score is higher-is-better, so ``basin``/
``basin_wide`` are expected POSITIVE and ``locmin``/``rough``/``flat``/
``argmin_err`` NEGATIVE. ``convexity_join.EXPECTED_SIGN`` records this and
``tables.md`` flags every cell that comes out against expectation.

    cd encoding
    ../.venv/bin/python scripts/plotting/plot_convexity_vs_opt_grid.py --per_set
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


def build_cells(convexity, optimization, losses, stats, *, ranked=True,
                unit="seed"):
    """``{(loss, stat): corr dict}`` plus the per-set breakdown.

    ``unit="seed"``  correlate over ideal-trace seeds (within parameter set)
    ``unit="set"``   collapse each set to its median, then correlate over sets
    """
    cells, per_set = {}, {}
    for stat in stats:
        pairs = CJ.paired(convexity, optimization, stat)
        for loss in losses:
            sub = {k: v for k, v in pairs.items() if k[1] == loss}
            per_set[(loss, stat)] = {k[0]: PS.corr(x, y)
                                     for k, (x, y) in sub.items()}
            if not sub:
                cells[(loss, stat)] = PS.corr([], [])
                continue
            if unit == "set":
                # One point per parameter set, so no within-set ranking applies.
                collapsed = CJ.aggregate_by_set(sub)
                x = np.concatenate([np.asarray(v[0], float)
                                    for v in collapsed.values()])
                y = np.concatenate([np.asarray(v[1], float)
                                    for v in collapsed.values()])
            elif ranked:
                x, y = PS.rank_within_group(sub)
            else:
                x = np.concatenate([np.asarray(v[0], float)
                                    for v in sub.values()])
                y = np.concatenate([np.asarray(v[1], float)
                                    for v in sub.values()])
            cells[(loss, stat)] = PS.corr(x, y)
    return cells, per_set


def plot_heatmap(cells, losses, stats, path, *, title, subtitle, footnote):
    n_cells = sum(1 for v in cells.values() if np.isfinite(v["spearman"]))
    mat = np.full((len(losses), len(stats)), np.nan)
    for i, loss in enumerate(losses):
        for j, stat in enumerate(stats):
            mat[i, j] = cells[(loss, stat)]["spearman"]

    fig, ax = plt.subplots(
        figsize=(1.45 * len(stats) + 3.4, 0.78 * len(losses) + 2.9))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(
        [f"{CJ.STAT_LABELS[s]}\n[{'+' if CJ.EXPECTED_SIGN[s] > 0 else '-'}]"
         for s in stats], rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(losses)))
    ax.set_yticklabels([PS.LABELS.get(l, l) for l in losses], fontsize=9)
    ax.tick_params(colors=PS.MUTED)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    for i, loss in enumerate(losses):
        for j, stat in enumerate(stats):
            c = cells[(loss, stat)]
            rho = c["spearman"]
            if not np.isfinite(rho):
                ax.text(j, i, "-", ha="center", va="center", color=PS.MUTED,
                        fontsize=9)
                continue
            colour = "white" if abs(rho) > 0.55 else PS.TEXT
            mark = PS.stars(c["p_spearman"], n_cells)
            # A ring on the cells that run against expectation: with only three
            # losses the eye reads the table before it reads tables.md.
            if np.sign(rho) != CJ.EXPECTED_SIGN[stat]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="#b23a2f",
                                           linewidth=1.6, zorder=4))
            ax.text(j, i - 0.13, f"{rho:+.2f}{mark}", ha="center", va="center",
                    color=colour, fontsize=9)
            ax.text(j, i + 0.24, f"n={c['n']}", ha="center", va="center",
                    color=colour, fontsize=6.5, alpha=0.75)

    ax.set_title(title, color=PS.TEXT, fontsize=12, loc="left", pad=10)
    # In figure coords with wrapping: in data coords a long subtitle ran off the
    # right edge of the canvas.
    fig.text(0.012, 0.985, subtitle, color=PS.MUTED, fontsize=8, ha="left",
             va="top", wrap=True)
    fig.text(0.012, 0.008, footnote, color=PS.MUTED, fontsize=6.5, ha="left",
             va="bottom", wrap=True)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label("Spearman rho", color=PS.TEXT, fontsize=9)
    cb.ax.tick_params(colors=PS.MUTED, labelsize=8)
    cb.outline.set_visible(False)

    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    PS.save(fig, path, {"losses": list(losses), "stats": list(stats),
                        "cells": {f"{l}|{s}": cells[(l, s)]
                                  for l in losses for s in stats}})


def plot_per_set(per_set, losses, stats, path):
    """One small heatmap per parameter set -- a broken join shows as a dead row."""
    sets = sorted({s for v in per_set.values() for s in v})
    if not sets:
        return
    ncol = min(4, len(sets))
    nrow = int(np.ceil(len(sets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.6 * nrow),
                             squeeze=False)
    for k, set_key in enumerate(sets):
        ax = axes[k // ncol][k % ncol]
        mat = np.full((len(losses), len(stats)), np.nan)
        for i, loss in enumerate(losses):
            for j, stat in enumerate(stats):
                c = per_set.get((loss, stat), {}).get(set_key)
                if c:
                    mat[i, j] = c["spearman"]
        ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(set_key, color=PS.TEXT, fontsize=9, loc="left")
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(stats, rotation=90, fontsize=6)
        ax.set_yticks(range(len(losses)))
        ax.set_yticklabels([PS.SHORT_LABELS.get(l, l) for l in losses],
                           fontsize=6)
        ax.tick_params(colors=PS.MUTED)
    for k in range(len(sets), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout()
    PS.save(fig, path)


def markdown(cells, losses, stats, cov) -> str:
    n_cells = sum(1 for v in cells.values() if np.isfinite(v["spearman"]))
    out = []
    out.append("| loss | " + " | ".join(
        f"{CJ.STAT_LABELS[s]} [{'+' if CJ.EXPECTED_SIGN[s] > 0 else '-'}]"
        for s in stats) + " |")
    out.append("|---|" + "---|" * len(stats))
    for loss in losses:
        row = []
        for stat in stats:
            c = cells[(loss, stat)]
            if not np.isfinite(c["spearman"]):
                row.append("-")
                continue
            mark = PS.stars(c["p_spearman"], n_cells)
            agrees = np.sign(c["spearman"]) == CJ.EXPECTED_SIGN[stat]
            cell = f"{c['spearman']:+.2f}{mark} (n={c['n']})"
            if agrees and mark:
                cell = f"**{cell}**"
            elif not agrees:
                cell = f"{cell} !"
            row.append(cell)
        out.append(f"| {PS.LABELS.get(loss, loss)} | " + " | ".join(row) + " |")

    out.append("\n**Best predictor per loss** (largest |rho|)\n")
    out.append("| loss | descriptor | rho | agrees with expectation | n |")
    out.append("|---|---|---|---|---|")
    for loss in losses:
        best, best_c = None, None
        for stat in stats:
            c = cells[(loss, stat)]
            if np.isfinite(c["spearman"]) and (
                    best_c is None
                    or abs(c["spearman"]) > abs(best_c["spearman"])):
                best, best_c = stat, c
        if best is None:
            out.append(f"| {PS.LABELS.get(loss, loss)} | - | - | - | - |")
        else:
            agrees = np.sign(best_c["spearman"]) == CJ.EXPECTED_SIGN[best]
            out.append(f"| {PS.LABELS.get(loss, loss)} | {CJ.STAT_LABELS[best]} "
                       f"| {best_c['spearman']:+.2f} "
                       f"| {'yes' if agrees else 'NO'} | {best_c['n']} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    cfg = get_convexity_vs_opt_config()
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    CJ.add_common_args(ap, cfg)
    ap.add_argument("--stats", default=None, help="comma-separated subset")
    ap.add_argument("--unit", default="both", choices=["seed", "set", "both"])
    ap.add_argument("--per_set", action="store_true",
                    help="also write a facet grid, one heatmap per parameter set")
    args = ap.parse_args(argv)

    convexity, optimization = CJ.resolve(args)
    cov = CJ.coverage(convexity, optimization)
    if not convexity or not optimization:
        print(f"nothing to plot: {len(convexity)} convexity keys, "
              f"{len(optimization)} optimization keys")
        return 1

    losses = [l for l in args.losses.split(",") if l in set(cov["losses"])]
    stats = (args.stats.split(",") if args.stats
             else [s for s in CJ.STAT_NAMES if s in set(cov["stats"])])
    if not losses or not stats:
        print("no losses or descriptors in common between the two files")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.format
    foot = CJ.provenance(args)
    score = CJ.SCORE_LABEL.get(args.matrix, args.matrix)

    ranked, per_set = build_cells(convexity, optimization, losses, stats,
                                  ranked=True)
    raw, _ = build_cells(convexity, optimization, losses, stats, ranked=False)
    by_set, _ = build_cells(convexity, optimization, losses, stats, unit="set")

    n_sets = len(cov["sets"])
    if args.unit in ("seed", "both"):
        plot_heatmap(
            ranked, losses, stats, out_dir / f"spearman_grid.{ext}",
            title=f"Convexity vs {score} (within set, over seeds)",
            subtitle=f"{n_sets} parameter sets x 10 shared ideal seeds, both "
                     f"axes rank-normalized within set; expected sign in "
                     f"brackets, a red ring marks a cell against it",
            footnote=foot)
        plot_heatmap(
            raw, losses, stats, out_dir / f"spearman_grid_raw.{ext}",
            title=f"Convexity vs {score} (raw pooled over seeds)",
            subtitle=f"{n_sets} parameter sets x 10 shared ideal seeds, pooled "
                     f"WITHOUT rank-normalizing -- confounded by set difficulty",
            footnote=foot)
    if args.unit in ("set", "both"):
        plot_heatmap(
            by_set, losses, stats, out_dir / f"spearman_grid_by_set.{ext}",
            title=f"Convexity vs {score} (BETWEEN parameter sets)",
            subtitle=f"one point per parameter set (median over its 10 shared "
                     f"ideal seeds on both sides), n={n_sets} sets per cell",
            footnote=foot)

    if args.per_set:
        plot_per_set(per_set, losses, stats,
                     out_dir / f"spearman_per_set.{ext}")

    (out_dir / "tables.md").write_text(
        "# Convexity vs the region-of-success score\n\n"
        f"Score: **{score}**, optimizer **{args.optimizer}**, ladder "
        f"{list(CJ.SUCCESS_RADII)}. Higher is better, so the expected signs are "
        "the ones in `convexity.EXPECTED_SIGN` FLIPPED.\n\n"
        f"Parameter sets: **{len(cov['sets'])}** - losses: "
        f"**{len(cov['losses'])}** - descriptors: **{len(cov['stats'])}**. "
        f"Optimization cells {cov['n_optimization_cells']}, convexity cells "
        f"{cov['n_convexity_cells']}.\n\n"
        f"> {foot}\n\n"
        "`*` p<0.05, `**` Bonferroni over the cells drawn, `!` sign against "
        "expectation.\n\n"
        "## WITHIN parameter set, over ideal seeds (rank-normalized)\n\n"
        + markdown(ranked, losses, stats, cov)
        + "\n## BETWEEN parameter sets (one point per set)\n\n"
        + "The unit the first study reported. |rho| is far larger here because\n"
        + "most of the variance is between problems of different difficulty\n"
        + "rather than between target neurons of one problem.\n\n"
        + markdown(by_set, losses, stats, cov)
        + "\n## Raw pooled over seeds (confounded, for reference)\n\n"
        + markdown(raw, losses, stats, cov)
    )
    (out_dir / "scored.json").write_text(json.dumps({
        "provenance": foot,
        "matrix": args.matrix, "optimizer": args.optimizer,
        "radii": list(CJ.SUCCESS_RADII),
        "expected_sign": CJ.EXPECTED_SIGN,
        "coverage": cov,
        "ranked": {f"{l}|{s}": ranked[(l, s)] for l in losses for s in stats},
        "raw": {f"{l}|{s}": raw[(l, s)] for l in losses for s in stats},
        "by_set": {f"{l}|{s}": by_set[(l, s)] for l in losses for s in stats},
        "per_set": {f"{l}|{s}": per_set[(l, s)] for l in losses for s in stats},
    }, indent=1))
    print(f"  wrote {out_dir}/tables.md and scored.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
