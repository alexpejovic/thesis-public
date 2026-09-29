"""What the optimizer actually did: the inherited Polyak setup diverges.

These figures describe the *methodology*, not the losses. Every published
optimization score in this project was produced by the same 100-step Polyak
trajectory, and the diagnostics stored with the region scores say that
trajectory routinely ends farther from the truth than it started:

* ``distance_distributions`` -- how much closer than a random start the BEST
  iterate gets, per loss and per arm. Only the start and the best distance are
  drawn: the score moved to the best iterate on 2026-08-30 and the final
  iterate is no longer plotted anywhere. The reason it moved is the console
  summary this script prints, where the two are still reported side by side --
  under Polyak the median final distance sits ABOVE the median start while the
  median best distance sits below it, so a final-iterate figure was largely
  measuring how far a diverging optimizer wandered after passing the truth.
  The ladder rungs are drawn across the distance axis with the chance mass
  below each: the score is the mass below those rungs, where a uniform start
  essentially never lands, while the quantiles are the location. Both readings
  are of the same 1000 runs; see ``success_vs_radius`` for the radius at which
  the curves cross.
* ``divergence_diagnostics`` -- how permanent it is. ``escape_frac`` (ended
  beyond the region any start could have come from), ``moving_frac`` (still
  moving when the step budget ran out), ``worse_than_start_frac``, and the size
  of the last step.
* ``cap_frac`` -- how often Polyak's trust region binds, per loss. It is the
  quantity the final-iterate masking artifact was coupled to (Spearman -0.88,
  ``claude-tasks/2026-08-28-region-success-score.md``), and it does not exist
  for Adam. Together with ``divergence_diagnostics`` it is the evidence behind
  the eligibility gate in ``plot_region_optimizers.py``.

None of this invalidates the scores as a *ranking* -- every loss faces the same
optimizer and the same starts -- but it is the caveat the numbers need, and the
argument for the Adam arm.

Usage (from encoding/):
    python scripts/plotting/plot_region_diagnostics.py
    python scripts/plotting/plot_region_diagnostics.py --camp activation
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import plot_style as PS  # noqa: E402
import region_scores as R  # noqa: E402
from helper import get_region_scores_config  # noqa: E402

# The stored `quantiles()` blocks, in the order they are drawn as a whisker.
BANDS = ("p10", "p25", "p50", "p75", "p90")

DIAGS = (
    ("diag.escape_frac", "fraction of runs that escaped",
     "ended beyond $\\|x\\|_\\infty$ = init bound + log 3"),
    ("diag.moving_frac", "fraction still moving at step 100",
     "mean displacement over the last 10 steps > 0.01"),
    ("diag.worse_than_start_frac", "fraction worse than their start",
     "$d_\\mathrm{final} > d_0$"),
    ("diag.last_disp_p90", "size of the final step (p90)",
     "log scale: Adam has converged, Polyak has not"),
)


def _parse_args(cfg):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score_path", default=cfg["score_path"])
    ap.add_argument("--plot_path", default=cfg["plot_path"])
    ap.add_argument("--camp", default=None,
                    help="restrict to one camp's losses (default: every loss "
                         "in the store). Diagnostics are properties of the "
                         "OPTIMIZER, so the default is deliberately the whole "
                         "store -- a camp narrows it for a closer look.")
    ap.add_argument("--format", default="pdf", choices=("pdf", "png"),
                    help="pdf for the report, png for a quick look")
    return ap.parse_args()


def fig_distances(rows, opts, cfg, out: Path):
    """Start and BEST distance quantiles, per optimizer and loss.

    `d_final` is deliberately absent: the headline statistic is the best
    iterate, and plotting the final one beside it is what this figure set
    stopped doing on 2026-08-30. It is still printed in the console summary,
    where it is the argument for the switch rather than a result.
    """
    fields = [("d0", "start"), ("d_best", "best")]
    fig, axes = plt.subplots(1, len(opts), figsize=(5.6 * len(opts), 4.4),
                             squeeze=False, sharey=True)
    payload = {}
    for ax, opt in zip(axes[0], opts):
        pos, ticks, labels, groups = 0.0, [], [], []
        panel = {}
        for loss in R.losses(rows):
            sub = [r for r in rows
                   if r.get("optimizer") == opt and r["loss_name"] == loss]
            if not sub:
                continue
            first = pos
            for field, fname in fields:
                # Each parameter set contributes its own quantile block; the
                # figure shows the distribution ACROSS SETS of each quantile, so
                # a wide whisker means the subsets disagree, not that one run
                # was strange.
                q = {b: np.array([r[field][b] for r in sub
                                  if r[field][b] is not None]) for b in BANDS}
                c = PS.MUTED if fname == "start" else PS.loss_color(loss)
                ax.plot([pos, pos], [q["p10"].mean(), q["p90"].mean()],
                        lw=1.4, color=c, alpha=0.55)
                ax.plot([pos, pos], [q["p25"].mean(), q["p75"].mean()],
                        lw=5.0, color=c, alpha=0.8, solid_capstyle="butt")
                ax.plot([pos], [q["p50"].mean()], marker="_", ms=13, mew=2.2,
                        color=PS.TEXT, zorder=5)
                ticks.append(pos)
                labels.append(fname)
                panel[f"{loss}|{field}"] = {b: float(q[b].mean()) for b in BANDS}
                pos += 1.0
            groups.append(((first + pos - 1.0) / 2.0, loss))
            pos += 0.8

        arm = [r for r in rows if r.get("optimizer") == opt]

        # The ladder rungs, on the axis the score is actually read off. Without
        # them this figure and `score_vs_chance` look like they contradict each
        # other: the quantiles below say the optimizer ended up worse than its
        # start, the score says it beat chance. Both are true, and these five
        # lines are where the second one lives -- the score is the mean over
        # them of P(d_final < r), and at the tight rungs a uniform start
        # essentially never lands, so a small minority of runs carries it.
        chance_below = {}
        for rung in R.SUCCESS_RADII:
            base = float(np.mean([(R.raw(r, "raw_d0") < rung).mean()
                                  for r in arm]))
            chance_below[f"{rung:g}"] = base
            ax.axhline(rung, color=PS.GRID, lw=1.0, zorder=0)
            ax.annotate(f"r={rung:g}  (chance {base:.4f})",
                        xy=(1.0, rung), xycoords=("axes fraction", "data"),
                        xytext=(-2, 2), textcoords="offset points",
                        ha="right", va="bottom", fontsize=6.5, color=PS.MUTED,
                        # The rules run behind the whiskers, so the labels need
                        # to sit on something, in front of them.
                        zorder=6,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))

        # Log distance axis. The rungs are dyadic by construction
        # (gen_region_success_scores.py:33-36), so on a linear axis the four
        # tight ones pile up in the bottom eighth and their labels collide --
        # and it is exactly that pile-up the reader needs to be able to read.
        # Ordering and the above/below-the-start-line reading are unchanged.
        ax.set_yscale("log")
        # Log minor ticks are unlabelled by default, which would leave two
        # labelled decades on an axis whose whole range is one and a half.
        ax.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        # The reference the whole finding turns on.
        d0_med = float(np.mean([r["d0"]["p50"] for r in arm]))
        ax.axhline(d0_med, color=PS.MUTED, lw=1.1, ls="--")
        ax.annotate(f"median start distance {d0_med:.2f}",
                    xy=(0.02, d0_med), xycoords=("axes fraction", "data"),
                    xytext=(0, -3), textcoords="offset points",
                    va="top", fontsize=8, color=PS.MUTED, zorder=6,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8, rotation=45)
        ax.set_xlim(-0.8, pos - 0.6)
        # The other reading of the same runs, in words, so the two halves of
        # the finding cannot be separated from each other.
        worse = float(np.mean([r["diag"]["worse_than_start_frac"]
                               for r in arm]))
        # Rescored on cfg["score_matrix"], NOT the stored `region_score`: that
        # field was computed on `raw_d_final` and would put a final-iterate
        # number on a best-iterate figure.
        score = float(np.mean([R.score(r, cfg) for r in arm]))
        base = float(np.mean([R.chance(r, R.SUCCESS_RADII) for r in arm]))
        PS.style(ax, ylabel="distance to true parameters",
                 xlabel=(f"{worse:.0%} of runs end farther out than they "
                         f"started, yet region score {score:.3f} vs chance "
                         f"{base:.3f}"))
        ax.xaxis.label.set_color(PS.MUTED)
        ax.xaxis.label.set_fontsize(8)
        # Extra pad: the per-loss group labels sit just above the axes, and the
        # default title pad puts the title straight through them.
        ax.set_title(PS.opt_label(opt), color=PS.TEXT, fontsize=11,
                     loc="left", pad=30 if len(groups) > 4 else 24)
        # One label per loss, centred over its own triple of whiskers, in data
        # coordinates -- an axes-fraction guess drifts as soon as a loss is
        # missing from an arm.
        # Beyond four losses per panel the horizontal group labels collide, so
        # they tilt. The store now holds seven and grows with every camp.
        crowded = len(groups) > 4
        for centre, loss in groups:
            ax.annotate(PS.loss_short(loss), xy=(centre, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, 4), textcoords="offset points",
                        ha="left" if crowded else "center",
                        rotation=22 if crowded else 0,
                        rotation_mode="anchor",
                        fontsize=6.5 if crowded else 7.5,
                        color=PS.loss_color(loss))
        panel["median_start_distance"] = d0_med
        panel["success_radii"] = list(R.SUCCESS_RADII)
        panel["chance_below_rung"] = chance_below
        panel["worse_than_start_frac"] = worse
        panel["region_score"] = score
        panel["region_score_baseline"] = base
        payload[opt] = panel

    fig.suptitle(
        "How close the best iterate gets, against the start it came from\n"
        "the near tail below the tight rungs beats a random start by two to "
        "three orders of magnitude -- that mass is what the region score reads",
        fontsize=10, y=1.06)
    fig.tight_layout()
    PS.save(fig, out, payload)


def fig_diagnostics(rows, opts, out: Path):
    """Four divergence diagnostics, Polyak against Adam, over all 56 subsets."""
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.8))
    payload = {}
    for ax, (field, title, note) in zip(axes.ravel(), DIAGS):
        pos, ticks, labels = 0, [], []
        for opt in opts:
            for loss in R.losses(rows):
                vals = np.array([
                    R._dig(r, field) for r in rows
                    if r.get("optimizer") == opt and r["loss_name"] == loss
                    and R._dig(r, field) is not None
                ], dtype=float)
                if not vals.size:
                    continue
                c = PS.loss_color(loss)
                ax.scatter(np.full(len(vals), pos)
                           + np.linspace(-0.17, 0.17, len(vals)),
                           vals, s=7, alpha=0.5, color=c, linewidths=0)
                ax.plot([pos - 0.3, pos + 0.3], [np.median(vals)] * 2, lw=2.4,
                        color=PS.OPT_COLORS.get(opt, PS.TEXT), zorder=4)
                ticks.append(pos)
                labels.append(PS.loss_short(loss))
                payload.setdefault(field, {})[f"{opt}|{loss}"] = {
                    "median": float(np.median(vals)),
                    "p10": float(np.percentile(vals, 10)),
                    "p90": float(np.percentile(vals, 90)),
                    "n": int(vals.size),
                }
                pos += 1
            pos += 0.5
        if field == "diag.last_disp_p90":
            ax.set_yscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        PS.style(ax, title=title)
        ax.annotate(note, xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=7.5, color=PS.MUTED)
        for opt in opts:
            ax.plot([], [], lw=2.4, color=PS.OPT_COLORS.get(opt, PS.TEXT),
                    label=PS.OPT_LABELS.get(opt, opt))
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    fig.suptitle("Divergence diagnostics -- one dot per parameter subset, "
                 "thick bar = arm median", fontsize=10)
    fig.tight_layout()
    PS.save(fig, out, payload)


def fig_cap_frac(rows, order, out: Path):
    """Polyak's trust-region cap fraction per loss. Null on every Adam row."""
    idx = R.index(rows)
    have = [l for l in R.losses(rows)
            if any(("polyak", l, s) in idx
                   and idx[("polyak", l, s)]["diag"]["polyak_cap_frac"] is not None
                   for s in order)]
    if not have:
        print("  no polyak rows with a cap fraction; skipping cap_frac figure")
        return

    fig, ax = plt.subplots(figsize=(13, 3.8))
    x = np.arange(len(order))
    payload = {"sets": order}
    spreads = []
    for loss in have:
        def pick(key):
            return np.array([
                (idx[("polyak", loss, s)]["diag"][key]
                 if ("polyak", loss, s) in idx else np.nan) or np.nan
                for s in order
            ], dtype=float)
        mid, lo, hi = pick("polyak_cap_frac"), pick("polyak_cap_frac_p10"), \
            pick("polyak_cap_frac_p90")
        c = PS.loss_color(loss)
        ax.plot(x, mid, marker="o", ms=3.0, lw=1.3, color=c,
                label=PS.loss_label(loss))
        # The p10-p90 spread over starts is drawn as a NUMBER, not a band: it
        # runs from roughly 0 to roughly 1 in every set, so a shaded band covers
        # the whole panel and hides the between-loss differences that are the
        # point. It stays in the sidecar for anyone who wants it.
        spreads.append((np.nanmedian(lo), np.nanmedian(hi)))
        payload[loss] = {"cap_frac": mid, "p10": lo, "p90": hi}

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    lo_med = np.nanmedian([s[0] for s in spreads])
    hi_med = np.nanmedian([s[1] for s in spreads])
    PS.style(ax, xlabel="parameter set (ordered by difficulty)",
             ylabel="mean fraction of steps at the cap")
    ax.set_title("Polyak trust-region cap fraction", color=PS.TEXT,
                 fontsize=11, loc="left", pad=22)
    # Anchored at zero so the level is readable, autoscaled above so the
    # between-loss differences are not squashed into the bottom fifth.
    ax.set_ylim(0, None)
    ax.annotate(
        f"line = mean over the 1000 runs of a subset; across runs the cap "
        f"fraction spans p10 {lo_med:.2f} to p90 {hi_med:.2f} (median over "
        "subsets), so almost the whole range",
        xy=(0.0, 1.02), xycoords="axes fraction", fontsize=7.5,
        color=PS.MUTED, va="bottom")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    PS.save(fig, out, payload)


def main():
    cfg = get_region_scores_config()
    args = _parse_args(cfg)
    out = Path(args.plot_path)
    # PDF for anything headed into the report, PNG for a quick look.
    def fig(stem):
        return out / f"{stem}.{args.format}"


    rows = R.load(args.score_path, exclude_losses=cfg.get("exclude_losses"))
    # These figures read the raw matrices directly, same as the other scripts,
    # so they run under the same guard that the rescoring path reproduces the
    # published score.
    R.check_rescore_matches_stored(rows)
    # The shared subset order is computed BEFORE any camp filter, so a camp view
    # and the full view put the subsets in the same places.
    order = R.set_order(rows, optimizer=cfg["set_order_optimizer"],
                        matrix=cfg["score_matrix"])
    if args.camp:
        camp = R.camp(cfg, args.camp)
        rows = R.camp_rows(rows, camp) if camp["axis"] == "optimizer" else [
            r for r in rows if r["loss_name"] in set(camp["losses"])
        ]
        if not rows:
            raise SystemExit(f"camp {args.camp!r} has no rows")
        print(f"  camp {args.camp}: {len(rows)} rows")
    opts = R.optimizers(rows)

    fig_distances(rows, opts, cfg, fig("distance_distributions"))
    fig_diagnostics(rows, opts, fig("divergence_diagnostics"))
    fig_cap_frac(rows, order, fig("cap_frac"))

    for opt in opts:
        sub = [r for r in rows if r.get("optimizer") == opt]
        # d_final is reported here and nowhere on a figure: it is the
        # justification for scoring the best iterate, not a result.
        print(f"  {opt}: d0 p50 {np.mean([r['d0']['p50'] for r in sub]):.3f} | "
              f"d_final p50 {np.mean([r['d_final']['p50'] for r in sub]):.3f} | "
              f"d_best p50 {np.mean([r['d_best']['p50'] for r in sub]):.3f} | "
              f"escape {np.mean([r['diag']['escape_frac'] for r in sub]):.3f} | "
              f"moving {np.mean([r['diag']['moving_frac'] for r in sub]):.3f}")
    print(f"wrote plots and json sidecars to {out}")


if __name__ == "__main__":
    main()
