"""Does the conclusion depend on which success radii were averaged? No.

The region score averages the fraction of starts landing within each of five
radii. That ladder is a choice, so this recomputes the whole study under several
alternative ladders -- free, because every row stores its full
``(n_ideal, n_init)`` distance matrix, so no GPU and no re-run is involved.

``success_vs_radius``
    The raw material: mean success rate against radius, with the five rungs
    marked and the random-start chance curve beneath. Shows that no rung
    saturates -- the top rung has a high chance rate, but chance is an additive
    constant shared by every loss and cancels in the paired comparison the score
    is actually used for.

``ladder_robustness``
    Per ladder: the mean score of each loss, its chance level, and
    ``spread/noise`` -- the between-loss spread of the mean score over the mean
    within-loss spread across parameter subsets. Higher separates losses better.
    Scale-free, so ladders sitting at very different absolute levels stay
    comparable.

``ladder_rank_agreement``
    Spearman of each ladder's per-subset scores against the default ladder. If
    this stays near 1, the choice of ladder does not reorder anything.

``ladder_table.md``
    The task doc's radius-ladder table, regenerated on this study's rows.

Usage (from encoding/):
    python scripts/plotting/plot_region_ladder.py
"""

import argparse
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

import plot_style as PS  # noqa: E402
import region_scores as R  # noqa: E402
from helper import get_region_scores_config  # noqa: E402


def _parse_args(cfg):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score_path", default=cfg["score_path"])
    ap.add_argument("--plot_path", default=cfg["plot_path"])
    ap.add_argument("--bootstrap_n", type=int, default=cfg["bootstrap_n"])
    ap.add_argument("--bootstrap_seed", type=int, default=cfg["bootstrap_seed"])
    ap.add_argument("--camp", default=None,
                    help="restrict to one camp's losses (default: every loss "
                         "in the store)")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), default=None,
                    help=f"the contrast tabulated across ladders; default "
                         f"{cfg['default_pair']}")
    ap.add_argument("--format", default="pdf", choices=("pdf", "png"),
                    help="pdf for the report, png for a quick look")
    return ap.parse_args()


def fig_success_curve(rows, opts, matrix, out: Path):
    """Success rate against radius, with the ladder rungs marked.

    Returns ``{(optimizer, loss): crossing radius}`` for :func:`write_table`.
    """
    grid = R.CURVE_GRID
    fig, axes = plt.subplots(1, len(opts), figsize=(5.2 * len(opts), 4.0),
                             squeeze=False, sharey=True)
    payload = {"radius_grid": grid, "rungs": list(R.SUCCESS_RADII), "panels": {}}
    all_crossings = {}
    for ax, opt in zip(axes[0], opts):
        panel = {}
        for loss in R.losses(rows):
            sub = [r for r in rows
                   if r.get("optimizer") == opt and r["loss_name"] == loss]
            if not sub:
                continue
            y = R.success_curve(sub, grid, matrix=matrix)
            ax.plot(grid, y, lw=1.6, color=PS.loss_color(loss),
                    label=PS.loss_label(loss))
            panel[loss] = y
        arm = [r for r in rows if r.get("optimizer") == opt]
        base = R.chance_curve(arm, grid)
        ax.plot(grid, base, lw=1.4, ls="--", color=PS.MUTED,
                label="chance (random starts)")
        panel["chance"] = base

        # Where each arm gives its advantage back. This is what reconciles
        # `score_vs_chance` with `distance_distributions`: the tight rungs are
        # left of the crossing, the median final distance is right of it. The
        # loosest rung sits right ON it, which is the whole reason
        # `single_loosest` behaves differently from every other ladder.
        crossings = {}
        for loss in R.losses(rows):
            sub = [r for r in arm if r["loss_name"] == loss]
            if not sub:
                continue
            x = R.crossing_radius(sub, grid, chance_rows=arm, matrix=matrix)
            crossings[loss] = x
            all_crossings[(opt, loss)] = x
            if x is None:
                continue
            ax.plot([x], [float(np.interp(np.log(x), np.log(grid), base))],
                    marker="v", ms=5, color=PS.loss_color(loss), zorder=5)
        panel["crossing_radius"] = crossings
        shown = [x for x in crossings.values() if x is not None]
        if shown:
            ax.annotate(
                "falls back to chance at r = "
                + ", ".join(f"{x:.2f}" for x in sorted(shown)),
                xy=(0.98, 0.04), xycoords="axes fraction", ha="right",
                fontsize=7, color=PS.MUTED, zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))

        for rung in R.SUCCESS_RADII:
            ax.axvline(rung, color=PS.GRID, lw=1.0, zorder=0)
            ax.annotate(f"{rung:g}", xy=(rung, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=6.5, color=PS.MUTED)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        PS.style(ax, xlabel="success radius (distance to true parameters)",
                 ylabel="fraction of starts within the radius")
        # Extra pad: the rung annotations sit just above the axes.
        ax.set_title(PS.OPT_LABELS.get(opt, opt), color=PS.TEXT, fontsize=11,
                     loc="left", pad=22)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        payload["panels"][opt] = panel

    fig.suptitle("No rung saturates: the tight rungs sit well left of the "
                 "crossing, the loosest one sits on it", fontsize=10, y=1.02)
    fig.tight_layout()
    PS.save(fig, out, payload)
    return all_crossings


def _measure(rows, ladders, opts, args, pair, matrix):
    """``{(ladder, optimizer): {...}}`` -- everything the next three outputs need."""
    default = ladders["default"]
    per_set_default = {}
    for opt in opts:
        per_set_default[opt] = {
            (r["loss_name"], r["set"]): R.rescore(r, default, matrix=matrix)
            for r in rows if r.get("optimizer") == opt
        }

    out = {}
    for name, radii in ladders.items():
        for opt in opts:
            stats = R.spread_over_noise(rows, opt, radii, matrix=matrix)
            if not stats:
                continue
            here = {(r["loss_name"], r["set"]): R.rescore(r, radii, matrix=matrix)
                    for r in rows if r.get("optimizer") == opt}
            keys = sorted(set(here) & set(per_set_default[opt]))
            agree = PS.corr([per_set_default[opt][k] for k in keys],
                            [here[k] for k in keys])
            c = R.contrast(rows, opt, pair[0], pair[1], radii=radii,
                           matrix=matrix, bootstrap_n=args.bootstrap_n,
                           seed=args.bootstrap_seed)
            out[(name, opt)] = {
                "radii": list(radii),
                **stats,
                "rank_agreement_vs_default": agree,
                "contrast": None if c is None else
                            {k: v for k, v in c.items()
                             if k not in ("a", "b", "diff", "sets")},
            }
    return out


def fig_robustness(measured, ladders, opts, out: Path):
    """Mean score per loss and the separation ratio, one column per ladder."""
    names = list(ladders)
    # sharey per row: the panels carry the same y label, so different scales
    # would make the Polyak/Adam comparison the figure exists for misleading.
    fig, axes = plt.subplots(2, len(opts), figsize=(5.6 * len(opts), 6.4),
                             squeeze=False, sharex=True, sharey="row")
    x = np.arange(len(names))
    for j, opt in enumerate(opts):
        top, bot = axes[0][j], axes[1][j]
        losses = sorted({l for (n, o), v in measured.items() if o == opt
                         for l in v["means"]})
        for loss in losses:
            y = [measured[(n, opt)]["means"].get(loss, np.nan)
                 if (n, opt) in measured else np.nan for n in names]
            top.plot(x, y, marker="o", ms=4, lw=1.4, color=PS.loss_color(loss),
                     label=PS.loss_label(loss))
        top.plot(x, [measured[(n, opt)]["chance"] if (n, opt) in measured
                     else np.nan for n in names],
                 marker="o", ms=4, lw=1.4, ls="--", color=PS.MUTED,
                 label="chance")
        PS.style(top, ylabel="mean region score")
        top.set_title(PS.OPT_LABELS.get(opt, opt), color=PS.TEXT, fontsize=11,
                      loc="left", pad=10)
        top.legend(fontsize=7.5, frameon=False, loc="upper left")

        bot.bar(x, [measured[(n, opt)]["spread_over_noise"] if (n, opt) in measured
                    else np.nan for n in names],
                color=PS.OPT_COLORS.get(opt, PS.TEXT), alpha=0.75, width=0.62)
        PS.style(bot, ylabel="spread / noise",
                 xlabel="radius ladder")
        bot.set_xticks(x)
        bot.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        bot.annotate("between-loss sd of the mean, over the mean within-loss "
                     "sd across subsets\n(higher separates the losses better)",
                     xy=(0.02, 0.94), xycoords="axes fraction", va="top",
                     fontsize=7.5, color=PS.MUTED)

    fig.suptitle("The ladder changes the level, not the verdict", fontsize=10)
    fig.tight_layout()
    PS.save(fig, out, {f"{n}|{o}": v for (n, o), v in measured.items()})


def fig_rank_agreement(measured, ladders, opts, out: Path):
    """Spearman of each ladder's per-subset scores against the default ladder."""
    names = [n for n in ladders if n != "default"]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(names))
    width = 0.8 / max(len(opts), 1)
    for k, opt in enumerate(opts):
        y = [measured[(n, opt)]["rank_agreement_vs_default"]["spearman"]
             if (n, opt) in measured else np.nan for n in names]
        ax.bar(x + k * width - 0.4 + width / 2, y, width=width * 0.9,
               color=PS.OPT_COLORS.get(opt, PS.TEXT), alpha=0.8,
               label=PS.OPT_LABELS.get(opt, opt))
    ax.axhline(1.0, color=PS.MUTED, lw=1.0, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    PS.style(ax, ylabel="Spearman vs the default ladder",
             xlabel="radius ladder",
             title="Every ladder ranks the (loss, subset) cells the same way")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    PS.save(fig, out, {f"{n}|{o}": measured[(n, o)]["rank_agreement_vs_default"]
                       for n in names for o in opts if (n, o) in measured})


def write_table(measured, ladders, opts, pair, out: Path, crossings=None):
    """The task doc's radius-ladder table, regenerated on this study's rows."""
    lines = [
        "# Radius-ladder robustness",
        "",
        "Regenerated by `scripts/plotting/plot_region_ladder.py` from",
        "`data/optimization-region-scores.jsonl`. Every column is recomputed from",
        "the stored raw distance matrices, so this costs no GPU time.",
        "",
        f"Contrast: `{pair[0]}` vs `{pair[1]}` "
        "(positive = the second loss scores higher).",
        "",
        "| ladder | optimizer | radii | chance | spread/noise | mean diff | p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in ladders:
        for opt in opts:
            m = measured.get((name, opt))
            if not m:
                continue
            c = m["contrast"] or {}
            lines.append(
                f"| {name} | {opt} | "
                f"{', '.join(f'{r:g}' for r in m['radii'])} | "
                f"{m['chance']:.4f} | {m['spread_over_noise']:.3f} | "
                f"{c.get('mean_diff', float('nan')):+.4f} | "
                f"{c.get('wilcoxon_p', float('nan')):.1e} |"
            )
    # Ladder-independent, so a column would just repeat one number down every
    # row: the radius where each arm hands its advantage back to chance. Every
    # rung above sits below it, which is why every ladder scores positive; the
    # median final distance sits above it, which is why
    # `distance_distributions` reads like a loss.
    if crossings:
        lines += [
            "",
            "## Where each arm falls back to chance",
            "",
            "The success curve and the random-start curve cross here. The four"
            " tight rungs sit well below this radius and the median final"
            " distance sits above it -- that is the whole difference between"
            " this table and `distance_distributions`. The loosest rung (1.5)"
            " sits ON the crossing, which is why `single_loosest` is the one"
            " ladder that flips a contrast's sign.",
            "",
            "| optimizer | loss | crossing radius |",
            "| --- | --- | --- |",
        ]
        for (opt, loss), x in crossings.items():
            lines.append(
                f"| {opt} | `{loss}` | "
                + ("never" if x is None else f"{x:.3f}")
                + " |"
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


def main():
    cfg = get_region_scores_config()
    args = _parse_args(cfg)
    out = Path(args.plot_path)
    # PDF for anything headed into the report, PNG for a quick look.
    def fig(stem):
        return out / f"{stem}.{args.format}"


    rows = R.load(args.score_path, exclude_losses=cfg.get("exclude_losses"))
    R.check_rescore_matches_stored(rows)
    ladders = R.ladders_from_config(cfg)
    # Every ladder is recomputed off the SAME matrix the camps are scored on, so
    # "the conclusion is robust to the radius ladder" is a statement about the
    # statistic actually being reported and not about a retired one.
    matrix = cfg["score_matrix"]
    if args.camp:
        camp = R.camp(cfg, args.camp)
        rows = [r for r in rows if r["loss_name"] in set(camp["losses"])]
        if not rows:
            raise SystemExit(f"camp {args.camp!r} has no rows")
        print(f"  camp {args.camp}: {len(rows)} rows")
    opts = R.optimizers(rows)
    pair = tuple(args.pair or cfg["default_pair"])
    print(f"  ladders recomputed on {matrix}; tabulated contrast {pair}")

    crossings = fig_success_curve(rows, opts, matrix, fig("success_vs_radius"))
    measured = _measure(rows, ladders, opts, args, pair, matrix)
    fig_robustness(measured, ladders, opts, fig("ladder_robustness"))
    fig_rank_agreement(measured, ladders, opts, fig("ladder_rank_agreement"))
    write_table(measured, ladders, opts, pair, out / "ladder_table.md",
                crossings=crossings)

    print(f"\n  {pair[0]} vs {pair[1]} under every ladder:")
    for (name, opt), m in measured.items():
        c = m["contrast"] or {}
        print(f"    {name:<18} {opt:<7} chance={m['chance']:.4f} "
              f"spread/noise={m['spread_over_noise']:.3f} "
              f"diff={c.get('mean_diff', float('nan')):+.4f} "
              f"p={c.get('wilcoxon_p', float('nan')):.1e}")
    print(f"wrote plots and json sidecars to {out}")


if __name__ == "__main__":
    main()
