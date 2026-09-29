"""Scatter: a convexity descriptor (x) against the region-of-success score (y).

The heatmaps in ``plot_convexity_vs_opt_grid.py`` report one rho per (loss,
descriptor) cell; this draws the cloud that rho was computed from, so the shape
behind the number is visible -- whether it is a clean monotone trend, a
saturating one, or two clusters of parameter sets with nothing in between.

Correlations use the same ``convexity_join``/``plot_style`` helpers as the
heatmaps and are computed on exactly the array that is plotted, so a panel's
annotation can never disagree with the corresponding heatmap cell.

The unit of observation decides what a point means and dominates the correlation:

``--unit set``   one point per parameter set (median over its 10 shared ideal
                 seeds, on BOTH sides) -- the between-problem view
``--unit seed``  one point per (parameter set, ideal seed), rank-normalized
                 within each set before pooling -- the strict within-problem
                 view. ``--raw`` plots the unnormalized values instead, which is
                 confounded by set difficulty.

    cd encoding
    # headline: the wide basin, whose 0.375 tolerance is the middle ladder rung
    ../.venv/bin/python scripts/plotting/plot_convexity_vs_opt_scatter.py \\
        --stat basin_wide --unit set
    # every descriptor x every loss, matching spearman_grid
    ../.venv/bin/python scripts/plotting/plot_convexity_vs_opt_scatter.py \\
        --stat all --unit both
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
from matplotlib.ticker import (LogLocator, MaxNLocator,  # noqa: E402
                               NullFormatter)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import convexity_join as CJ  # noqa: E402
import plot_style as PS  # noqa: E402
from helper import get_convexity_vs_opt_config  # noqa: E402


def points(convexity, optimization, stat, loss, *, unit="set", ranked=True):
    """The (x, y) arrays a panel plots, for one (loss, descriptor) cell.

    Mirrors ``plot_convexity_vs_opt_grid.build_cells`` so the scatter and the
    heatmap describe the same numbers.
    """
    pairs = {k: v for k, v in CJ.paired(convexity, optimization, stat).items()
             if k[1] == loss}
    if not pairs:
        return np.asarray([]), np.asarray([])
    if unit == "set":
        collapsed = CJ.aggregate_by_set(pairs)
        x = np.concatenate([np.asarray(v[0], float)
                            for v in collapsed.values()])
        y = np.concatenate([np.asarray(v[1], float)
                            for v in collapsed.values()])
    elif ranked:
        x, y = PS.rank_within_group(pairs)
    else:
        x = np.concatenate([np.asarray(v[0], float) for v in pairs.values()])
        y = np.concatenate([np.asarray(v[1], float) for v in pairs.values()])
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def draw(ax, x, y, stat, loss, *, unit, ranked, logx=False, n_cells=1,
         title=None, small=False, marker_label=None):
    """One scatter panel with an OLS trend line and the rho annotation."""
    colour = PS.COLORS.get(loss, PS.MUTED)
    # A between-set panel has 56 points and wants them legible; a per-seed panel
    # has hundreds and wants them transparent.
    size, alpha = (26.0, 0.85) if unit == "set" else (5.0, 0.22)
    ax.scatter(x, y, s=size, alpha=alpha, color=colour, linewidths=0.0,
               rasterized=(unit != "set"), label=marker_label, zorder=3)

    c = PS.corr(x, y)
    if not len(x):
        # `mse-zscore` has a flat fraction of exactly 0 on every landscape, so
        # rank_within_group drops every group and the panel is empty. An empty
        # panel reads as a bug; say what it actually is.
        ax.text(0.5, 0.5, "descriptor constant\n(no variation to correlate)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=6.0 if small else 8.5, color=PS.MUTED, zorder=5)
    # A log axis over less than ~1.5 decades gets labelled by its MINOR ticks
    # ("2x10^0 3x10^0 4x10^0"), which collide in a panel this size and buy
    # nothing -- roughness spans 1.0-5.3, so it stays linear.
    if logx and len(x) and np.all(x > 0) and (
            np.log10(x.max()) - np.log10(x.min()) >= 1.5):
        ax.set_xscale("log")
    if len(x) >= 3 and np.ptp(x) > 0:
        # Fit in the plotted coordinates so the line matches what is drawn.
        fx = np.log10(x) if ax.get_xscale() == "log" else x
        slope, intercept = np.polyfit(fx, y, 1)
        xx = np.linspace(fx.min(), fx.max(), 64)
        ax.plot(10.0 ** xx if ax.get_xscale() == "log" else xx,
                slope * xx + intercept, color=colour, linewidth=1.4,
                alpha=0.85, zorder=4)

    rho = c["spearman"]
    if np.isfinite(rho) and unit == "seed" and ranked:
        # Ranks fill (0,1] exactly, so the annotation would sit on the top row
        # of points; give it its own strip of axis.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.14 * (hi - lo))
    if np.isfinite(rho):
        agrees = np.sign(rho) == CJ.EXPECTED_SIGN[stat]
        txt = f"rho {rho:+.2f}{PS.stars(c['p_spearman'], n_cells)}  n={c['n']}"
        if not agrees:
            txt += "  !" if small else "  (sign against expectation)"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=6.5 if small else 9,
                color=PS.TEXT if agrees else "#b23a2f", zorder=5)

    # The region score is a bounded fraction; pinning the axis stops panels of
    # one figure being read on different scales.
    if not (unit == "seed" and ranked):
        ax.set_ylim(-0.04, 1.04)
    PS.style(ax, title=title or "")
    if small:
        ax.tick_params(labelsize=6)
        ax.title.set_fontsize(8)
    if unit == "seed" and ranked:
        ax.set_xticks([0.0, 0.5, 1.0])
        return c
    if ax.get_xscale() == "log":
        ax.xaxis.set_major_locator(LogLocator(base=10.0,
                                              numticks=4 if small else 6))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        # Descriptors span 1e-6 (flat) to 6e4 (locmin); the default formatter
        # runs the labels together at this panel size.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3 if small else 5))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3),
                            useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(5.5 if small else 8)
        ax.xaxis.get_offset_text().set_color(PS.MUTED)
    return c


def _axis_labels(stat, score, unit, ranked):
    xl, yl = CJ.STAT_LABELS[stat], score
    if unit == "seed" and ranked:
        return f"{xl}  (rank within set)", f"{yl}  (rank within set)"
    return xl, yl


def figure_per_loss(convexity, optimization, stat, losses, *, unit, ranked,
                    score, logx, foot, out_path):
    """One panel per loss for a single descriptor."""
    ncol = min(3, len(losses))
    nrow = int(np.ceil(len(losses) / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharex=True,
                             sharey=True, figsize=(3.5 * ncol, 3.2 * nrow))
    xlabel, ylabel = _axis_labels(stat, score, unit, ranked)
    results = {}
    for k, loss in enumerate(losses):
        ax = axes[k // ncol][k % ncol]
        x, y = points(convexity, optimization, stat, loss, unit=unit,
                      ranked=ranked)
        results[loss] = draw(ax, x, y, stat, loss, unit=unit, ranked=ranked,
                             logx=logx, n_cells=len(losses),
                             title=PS.LABELS.get(loss, loss))
    for k in range(len(losses), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    # supxlabel defaults to y~0.01, i.e. on top of the footnote; pin it above.
    fig.supxlabel(xlabel, color=PS.TEXT, fontsize=10, y=0.055)
    fig.supylabel(ylabel, color=PS.TEXT, fontsize=10)
    fig.text(0.005, 0.003, foot, color=PS.MUTED, fontsize=6, ha="left",
             va="bottom", wrap=True)
    # supxlabel sits below the rect, so this needs more bottom margin than the
    # matrix figures, whose column labels are ordinary axis labels.
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    PS.save(fig, out_path, {"stat": stat, "unit": unit, "ranked": ranked,
                            "results": results})
    return results


def figure_overlay(convexity, optimization, stat, losses, *, unit, ranked,
                   score, logx, foot, out_path):
    """All losses in one axes -- the single-figure version for the write-up."""
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    xlabel, ylabel = _axis_labels(stat, score, unit, ranked)
    allx = np.concatenate([points(convexity, optimization, stat, l, unit=unit,
                                  ranked=ranked)[0] for l in losses] or [[]])
    use_log = bool(logx and len(allx) and np.all(allx > 0)
                   and np.log10(allx.max()) - np.log10(allx.min()) >= 1.5)
    for loss in losses:
        x, y = points(convexity, optimization, stat, loss, unit=unit,
                      ranked=ranked)
        if not len(x):
            continue
        colour = PS.COLORS.get(loss, PS.MUTED)
        c = PS.corr(x, y)
        label = (f"{PS.LABELS.get(loss, loss)}  "
                 f"(rho {c['spearman']:+.2f}, n={c['n']})")
        ax.scatter(x, y, s=22 if unit == "set" else 4,
                   alpha=0.8 if unit == "set" else 0.18, color=colour,
                   linewidths=0.0, rasterized=(unit != "set"), label=label,
                   zorder=3)
        if len(x) >= 3 and np.ptp(x) > 0:
            fx = np.log10(x) if (use_log and np.all(x > 0)) else x
            slope, intercept = np.polyfit(fx, y, 1)
            xx = np.linspace(fx.min(), fx.max(), 64)
            ax.plot(10.0 ** xx if (use_log and np.all(x > 0)) else xx,
                    slope * xx + intercept, color=colour, linewidth=1.6,
                    alpha=0.9, zorder=4)
    if use_log:
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
    if not (unit == "seed" and ranked):
        ax.set_ylim(-0.04, 1.04)
    sign = "+" if CJ.EXPECTED_SIGN[stat] > 0 else "-"
    PS.style(ax, xlabel=xlabel, ylabel=ylabel,
             title=f"{CJ.STAT_LABELS[stat]} vs {score}  (expected {sign}, one "
                   f"point per "
                   f"{'parameter set' if unit == 'set' else 'ideal seed'})")
    ax.legend(frameon=False, fontsize=8, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    fig.text(0.005, 0.003, foot, color=PS.MUTED, fontsize=6, ha="left",
             va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    PS.save(fig, out_path)


def figure_matrix(convexity, optimization, stats, losses, *, unit, ranked,
                  score, logx, foot, out_path):
    """losses (rows) x descriptors (columns), the same layout as the heatmap."""
    nrow, ncol = len(losses), len(stats)
    is_rank = unit == "seed" and ranked
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(2.25 * ncol, 1.95 * nrow))
    results = {}
    for i, loss in enumerate(losses):
        for j, stat in enumerate(stats):
            ax = axes[i][j]
            x, y = points(convexity, optimization, stat, loss, unit=unit,
                          ranked=ranked)
            results[f"{loss}|{stat}"] = draw(
                ax, x, y, stat, loss, unit=unit, ranked=ranked, logx=logx,
                n_cells=nrow * ncol, small=True,
                title=CJ.STAT_LABELS[stat] if i == 0 else None)
            if j == 0:
                ax.set_ylabel(PS.SHORT_LABELS.get(loss, loss)
                              + ("\n(rank)" if is_rank else ""),
                              color=PS.TEXT, fontsize=7.5)
            if i == nrow - 1:
                ax.set_xlabel(CJ.STAT_LABELS[stat]
                              + (" (rank)" if is_rank else ""),
                              color=PS.MUTED, fontsize=7)
    if is_rank:
        head = (f"Convexity descriptor vs {score} -- BOTH AXES rank-normalized "
                f"within parameter set, on (0,1]; one point per ideal seed")
    else:
        head = (f"Convexity descriptor (x) vs {score} (y) -- one point per "
                f"{'parameter set' if unit == 'set' else 'ideal seed'}")
    fig.suptitle(head, color=PS.TEXT, fontsize=11, x=0.005, ha="left")
    fig.text(0.005, 0.003, foot, color=PS.MUTED, fontsize=6, ha="left",
             va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    PS.save(fig, out_path)
    return results


def main(argv=None) -> int:
    cfg = get_convexity_vs_opt_config()
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    CJ.add_common_args(ap, cfg)
    ap.add_argument("--stat", default="all",
                    help=f"one descriptor (e.g. {cfg['headline_stat']}), or "
                         f"'all' for the losses x descriptors matrix")
    ap.add_argument("--unit", default="set", choices=["seed", "set", "both"])
    ap.add_argument("--raw", action="store_true",
                    help="with --unit seed, plot the raw values instead of "
                         "rank-normalizing within each parameter set (pooling "
                         "raw is confounded by set difficulty)")
    ap.add_argument("--logx", action="store_true",
                    help="log x-axis where the descriptor is strictly positive "
                         "(useful for locmin and rough, which span decades)")
    args = ap.parse_args(argv)

    convexity, optimization = CJ.resolve(args)
    cov = CJ.coverage(convexity, optimization)
    if not convexity or not optimization:
        print(f"nothing to plot: {len(convexity)} convexity keys, "
              f"{len(optimization)} optimization keys")
        return 1

    losses = [l for l in args.losses.split(",") if l in set(cov["losses"])]
    stats = ([s for s in CJ.STAT_NAMES if s in set(cov["stats"])]
             if args.stat == "all" else [args.stat])
    for s in stats:
        if s not in CJ.STAT_NAMES:
            ap.error(f"unknown descriptor {s!r}; want one of "
                     f"{list(CJ.STAT_NAMES)} or 'all'")
    if not losses or not stats:
        print("no losses or descriptors in common between the two files")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.format
    foot = CJ.provenance(args)
    score = CJ.SCORE_LABEL.get(args.matrix, args.matrix)
    units = ["seed", "set"] if args.unit == "both" else [args.unit]
    ranked = not args.raw

    written, results = [], {}
    for unit in units:
        suffix = unit if (unit == "set" or ranked) else "seed_raw"
        if len(stats) == 1:
            stat = stats[0]
            p = out_dir / f"scatter_{stat}_{suffix}.{ext}"
            results[f"{unit}|{stat}"] = figure_per_loss(
                convexity, optimization, stat, losses, unit=unit,
                ranked=ranked, score=score, logx=args.logx, foot=foot,
                out_path=p)
            po = out_dir / f"scatter_{stat}_{suffix}_overlay.{ext}"
            figure_overlay(convexity, optimization, stat, losses, unit=unit,
                           ranked=ranked, score=score, logx=args.logx,
                           foot=foot, out_path=po)
            written += [p.name, po.name]
        else:
            p = out_dir / f"scatter_matrix_{suffix}.{ext}"
            results[unit] = figure_matrix(
                convexity, optimization, stats, losses, unit=unit,
                ranked=ranked, score=score, logx=args.logx, foot=foot,
                out_path=p)
            written.append(p.name)

    # Named after the figures it describes, so a matrix run and a
    # single-descriptor run can live in the same directory.
    tag = (f"{'matrix' if len(stats) > 1 else stats[0]}_{args.unit}"
           f"{'' if ranked else '_raw'}")
    (out_dir / f"scatter_{tag}.json").write_text(json.dumps({
        "provenance": foot, "matrix": args.matrix, "optimizer": args.optimizer,
        "unit": args.unit, "ranked": ranked, "coverage": cov,
        "figures": written, "results": results,
    }, indent=1))
    print(f"  wrote scatter_{tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
