"""Region-of-success scores, one camp at a time.

A *camp* is one question -- does masking help, which activation, which KL
weight -- declared in ``configs/region_scores.yaml`` rather than hardcoded here.
This script draws the same three figures for each of them, so adding a camp is a
config entry and not a new script.

Three things are true of every camp figure, and all three are deliberate:

* **The best iterate, never the final one.** ``cfg["score_matrix"]`` is
  ``raw_d_best``: the question is whether a loss's landscape lets the optimizer
  FIND the neuron, not whether that particular optimizer stayed there. The
  inherited polyak setup diverges on most starts (median final distance 2.94
  against a median start of 1.77, median best 1.56), so a final-iterate score
  was largely measuring the step size. See
  ``claude-tasks/2026-08-28-region-success-score.md``.
* **One optimizer, chosen once.** Every loss-axis camp is drawn under
  ``region_scores.best_optimizer``; the chosen arm is named in every sidecar and
  the full ranking behind the choice lives in
  ``plots/opt_region_scores/best_optimizer.json``, written by
  ``plot_region_optimizers.py``. Panels drawn under different arms cannot be
  read against each other, and the mistake is silent.
* **A noise floor on the same axis.** Three config-identical encoders
  (``noise_floor_group``) are drawn as a band. An effect inside it is not a
  finding however small its p-value: the paired test runs over 56 parameter
  subsets and resolves differences far below what retraining one configuration
  moves the score.

The ``optimizer`` camp is not drawn here -- it varies the optimizer rather than
the loss, and ``plot_region_optimizers.py`` owns it.

Usage (from encoding/):
    python scripts/plotting/plot_region_camps.py
    python scripts/plotting/plot_region_camps.py --camp beta --format png
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
    ap.add_argument("--camp", default=None,
                    help="draw one camp (default: every loss-axis camp)")
    ap.add_argument("--optimizer", default=None,
                    help="override the arm the camps are drawn under; by "
                         "default region_scores.best_optimizer decides")
    ap.add_argument("--bootstrap_n", type=int, default=cfg["bootstrap_n"])
    ap.add_argument("--bootstrap_seed", type=int, default=cfg["bootstrap_seed"])
    ap.add_argument("--format", default="pdf", choices=("pdf", "png"),
                    help="pdf for the report, png for a quick look")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# camp presentation
# ---------------------------------------------------------------------------


def _label(camp, loss) -> str:
    """The camp's own name for an arm, else the global short label.

    A camp may rename an arm: the beta ladder's top rung is `encoder-zscore-2`,
    which is the masking camp's unmasked encoder under another hat, and calling
    it "ZScore (repl.)" in a figure about KL weights would be actively wrong.
    """
    return R.member_label(camp, loss) or PS.loss_short(loss)


def _colors(camp, members) -> list[str]:
    """Per-arm colours: a sequential ramp for an ordered camp, else categorical.

    beta is an ordered axis, so its rungs get a light-to-dark ramp and the
    ordering can be read off the figure without the legend.
    """
    if camp.get("color_ramp") == "beta":
        return PS.beta_ramp(len(members))
    return [PS.loss_color(m) for m in members]


def _mean_ci(values, n_boot, seed):
    """Percentile bootstrap CI for a mean over parameter sets.

    The same resampler the paired contrasts use (``region_scores._boot_ci``),
    on the raw per-set scores rather than on differences -- resampling the SETS
    is the right unit either way, since the 56 subsets are the sample that
    generalizes.
    """
    return R._boot_ci(np.asarray(values, float), n_boot, seed)


def _chance_by_set(rows, order):
    """The random-start floor per parameter set.

    A property of the starts alone, so it is identical across the losses of a
    set (the C1 invariant, ``d0_hash``); the median over whichever rows are
    present is just a robust way of reading that one number back.
    """
    per: dict[str, list[float]] = {}
    for r in rows:
        per.setdefault(r["set"], []).append(R.chance(r, R.SUCCESS_RADII))
    return np.array([np.median(per[s]) if s in per else np.nan for s in order])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_scores(camp, rows, members, floor, baseline, args, out: Path):
    """Mean score per arm: the camp's headline, with everything it must clear."""
    scores = {m: [] for m in members}
    for r in rows:
        if r["loss_name"] in scores:
            scores[r["loss_name"]].append(R.score(r, args.cfg))

    means = [float(np.mean(scores[m])) for m in members]
    cis = [_mean_ci(scores[m], args.bootstrap_n, args.bootstrap_seed)
           for m in members]
    chance = float(np.mean([R.chance(r, R.SUCCESS_RADII) for r in rows]))

    fig, ax = plt.subplots(figsize=(1.5 + 1.15 * len(members), 4.6))
    x = np.arange(len(members))
    colors = _colors(camp, members)
    ax.bar(x, means, width=0.62, color=colors, edgecolor="none", zorder=2)
    ax.errorbar(x, means,
                yerr=[[m - lo for m, (lo, _) in zip(means, cis)],
                      [hi - m for m, (_, hi) in zip(means, cis)]],
                fmt="none", ecolor=PS.TEXT, elinewidth=1.1, capsize=3, zorder=3)

    # Reference lines are labelled in AXES-fraction x so the text cannot land
    # outside the data limits and be silently clipped, which is what happens
    # when a bar chart's x range is left to matplotlib's 5% margin.
    # A reference label sits over whichever bar happens to be under it, so it
    # carries the panel colour behind it -- otherwise "chance 0.083" in muted
    # grey lands on an orange bar and is unreadable.
    box = dict(facecolor=PS.PANEL, edgecolor="none", alpha=0.85, pad=1.4)

    def _rule(y, text, color, ls, left=False):
        ax.axhline(y, ls=ls, lw=1.25, color=color, zorder=1)
        ax.annotate(text, xy=(0.015 if left else 0.985, y),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 2.5), textcoords="offset points",
                    ha="left" if left else "right", va="bottom",
                    fontsize=7.5, color=color, bbox=box, zorder=4)

    _rule(chance, f"chance {chance:.3f}", PS.MUTED, "--")
    if baseline is not None:
        _rule(baseline["mean"],
              f"{PS.loss_short(baseline['loss'])} {baseline['mean']:.3f}",
              PS.COLORS["mse-zscore"], "-.", left=True)

    # The floor is drawn on the SCORE axis, not as an error bar: it is three
    # config-identical encoders' own mean scores, so the band is directly
    # comparable with the bars beside it.
    fm = floor.get("means") or {}
    if len(fm) > 1:
        lo, hi = min(fm.values()), max(fm.values())
        ax.axhspan(lo, hi, color=PS.MUTED, alpha=0.16, lw=0, zorder=0)
        ax.annotate(f"retraining noise ({hi - lo:.4f})",
                    xy=(0.985, lo), xycoords=("axes fraction", "data"),
                    xytext=(0, -4), textcoords="offset points",
                    ha="right", va="top", fontsize=7.5, color=PS.MUTED,
                    bbox=box, zorder=4)

    # An arm that leaves the box far more than its neighbours is not being
    # measured on the same footing as them; its bar is hatched so the figure
    # cannot be read as a clean comparison.
    escapes = {}
    for m in members:
        v = [R._dig(r, "diag.escape_frac") for r in rows
             if r["loss_name"] == m]
        v = [float(x) for x in v if x is not None and np.isfinite(x)]
        if v:
            escapes[m] = float(np.mean(v))
    max_gap = (args.cfg.get("cross_camp") or {}).get("max_escape_gap")
    flagged = []
    if max_gap is not None and len(escapes) > 1:
        median_esc = float(np.median(list(escapes.values())))
        for bar, m in zip(ax.patches, members):
            if abs(escapes.get(m, median_esc) - median_esc) > float(max_gap):
                bar.set_hatch("///")
                bar.set_edgecolor("#c0392b")
                bar.set_linewidth(1.0)
                flagged.append(m)
    if flagged:
        ax.annotate(
            "hatched: escape fraction more than "
            f"{float(max_gap):g} from the camp median "
            f"({', '.join(f'{m}={escapes[m]:.2f}' for m in flagged)})",
            xy=(0.5, -0.14), xycoords="axes fraction", ha="center",
            fontsize=7, color="#c0392b")

    ax.set_xticks(x)
    ax.set_xticklabels([_label(camp, m) for m in members], fontsize=8.5)
    PS.style(ax, ylabel="region score (best iterate)",
             title=f"{camp['title']} -- {PS.opt_label(args.optimizer)}")
    ax.set_xlim(-0.62, len(members) - 0.38)
    ax.set_ylim(0, max(max(hi for _, hi in cis), chance,
                       baseline["mean"] if baseline else 0) * 1.18)

    PS.save(fig, out, {
        "camp": camp["name"], "optimizer": args.optimizer,
        "score_matrix": args.cfg["score_matrix"],
        "members": members, "labels": [_label(camp, m) for m in members],
        "mean": dict(zip(members, means)),
        "ci": {m: list(c) for m, c in zip(members, cis)},
        "n_sets": {m: len(scores[m]) for m in members},
        "chance": chance, "baseline": baseline, "noise_floor": floor,
        "escape_frac": escapes, "escape_flagged": flagged,
    })


def fig_contrasts(camp, rows, members, floor, args, out: Path):
    """The camp's paired within-camp differences, and what they have to clear."""
    # The same colour the arm has on `scores` -- for a ramped camp that is its
    # ramp step, not its categorical colour, or the two figures disagree about
    # which rung is which.
    colour = dict(zip(members, _colors(camp, members)))
    # The comparability limit the cross-camp figure gates on. Within a camp the
    # contrast is the camp's own question, so it is drawn either way -- but a
    # gap this size means the two arms are not facing the same effective
    # optimizer, and that has to be visible ON the figure. The beta camp is why:
    # its beta = 0 arm escapes on 0.94 of starts against 0.53 for the top rung,
    # so every one of its contrasts carries a gap of 0.24-0.41.
    max_gap = (args.cfg.get("cross_camp") or {}).get("max_escape_gap")
    specs = [tuple(c) for c in camp["contrasts"]]
    got = []
    for a, b, label in specs:
        c = R.contrast(rows, args.optimizer, a, b,
                       matrix=args.cfg["score_matrix"],
                       bootstrap_n=args.bootstrap_n, seed=args.bootstrap_seed)
        if c is not None:
            got.append((label, c))
    if not got:
        print(f"  {camp['name']}: no contrast has both arms; skipping the forest")
        return

    fig, ax = plt.subplots(figsize=(7.6, 1.0 + 0.62 * len(got)))
    y = np.arange(len(got))[::-1]
    spread = floor.get("spread")
    if spread is not None and np.isfinite(spread):
        ax.axvspan(-spread, spread, color=PS.MUTED, alpha=0.16, lw=0, zorder=0,
                   label=f"retraining noise (+-{spread:.4f})")
    ax.axvline(0.0, color=PS.TEXT, lw=1.0, zorder=1)

    payload = {}
    for yi, (label, c) in zip(y, got):
        ax.plot([c["ci_lo"], c["ci_hi"]], [yi, yi], lw=2.2,
                color=PS.TEXT, alpha=0.75, solid_capstyle="round", zorder=2)
        ax.plot([c["mean_diff"]], [yi], "o", ms=6.5,
                color=colour.get(c["loss_b"], PS.TEXT),
                mec=PS.TEXT, mew=0.8, zorder=3)
        # Bonferroni over the camp's own contrasts: they are one family, asked
        # together, and a camp of four rungs would otherwise get a free
        # significant result roughly one time in five.
        star = PS.stars(c["wilcoxon_p"], n_cells=len(got))
        over = (np.isfinite(spread) and abs(c["mean_diff"]) > spread
                if spread is not None else True)
        gap = R.escape_gap(rows, args.optimizer, c["loss_a"], c["loss_b"])
        confounded = (max_gap is not None and np.isfinite(gap)
                      and gap > float(max_gap))
        note = ""
        if not over:
            note = "  [inside noise]"
        if confounded:
            note += f"  [escape gap {gap:.2f}: NOT comparability-clean]"
        ax.annotate(
            f"{c['mean_diff']:+.4f}{star}  p={c['wilcoxon_p']:.1e}  "
            f"{c['n_pos']}/{c['n']}{note}",
            (c["ci_hi"], yi), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=7.5,
            color="#c0392b" if confounded else (PS.TEXT if over else PS.MUTED))
        payload[label] = {k: v for k, v in c.items()
                          if k not in ("a", "b", "diff", "sets")}
        payload[label]["over_noise_floor"] = bool(over)
        # C5's statistic per pair. Recorded rather than gated on: within a camp
        # the arms differ in one thing and the comparison IS the camp's
        # question, so suppressing it would hide the point. Across camps it is
        # a gate -- see plot_region_cross.py.
        payload[label]["escape_gap"] = gap
        payload[label]["comparability_clean"] = not confounded

    ax.set_yticks(y)
    ax.set_yticklabels([lb for lb, _ in got], fontsize=8.5)
    ax.set_ylim(-0.7, len(got) - 0.3)
    PS.style(ax, xlabel="paired mean difference (positive = the second arm scores higher)",
             title=f"{camp['title']} -- {PS.opt_label(args.optimizer)}")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    # Leave room for the annotations, which sit outside the CI.
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.62 * (hi - lo))

    PS.save(fig, out, {"camp": camp["name"], "optimizer": args.optimizer,
                       "score_matrix": args.cfg["score_matrix"],
                       "noise_floor": floor, "contrasts": payload})


def fig_by_set(camp, rows, members, order, args, out: Path):
    """Score against parameter subset: where a camp's mean difference comes from.

    The score varies far more across parameter subsets than across arms, which
    is exactly why every comparison in this study is PAIRED on the subset. This
    figure is what makes that visible instead of asserted.
    """
    table = R.arm_scores(rows, args.cfg, optimizer=args.optimizer)
    fig, ax = plt.subplots(figsize=(13, 4.0))
    x = np.arange(len(order))
    colors = _colors(camp, members)
    payload = {"sets": order, "arms": {}}

    for m, c in zip(members, colors):
        y = np.array([table.get(m, {}).get(s, np.nan) for s in order])
        ax.plot(x, y, marker="o", ms=3.0, lw=1.3, color=c, label=_label(camp, m))
        payload["arms"][m] = y

    chance = _chance_by_set(rows, order)
    ax.plot(x, chance, ls="--", lw=1.2, color=PS.MUTED, label="chance")
    payload["chance"] = chance

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90, fontsize=6)
    PS.style(ax, xlabel="parameter subset (ordered by difficulty)",
             ylabel="region score (best iterate)",
             title=f"{camp['title']} -- {PS.opt_label(args.optimizer)}")
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    PS.save(fig, out, payload)


def write_summary(camp, rows, members, floor, baseline, args, out: Path):
    """The camp's numbers as markdown, so a figure's claims are quotable."""
    lines = [f"# {camp['title']}", "",
             f"Optimizer: **{PS.opt_label(args.optimizer)}**. "
             f"Statistic: `{args.cfg['score_matrix']}` "
             f"(best iterate) over radii {list(R.SUCCESS_RADII)}.", ""]

    chance = float(np.mean([R.chance(r, R.SUCCESS_RADII) for r in rows]))
    lines += ["| arm | mean score | 95% CI | sets |", "| --- | --- | --- | --- |"]
    for m in members:
        v = [R.score(r, args.cfg) for r in rows if r["loss_name"] == m]
        lo, hi = _mean_ci(v, args.bootstrap_n, args.bootstrap_seed)
        lines.append(f"| {_label(camp, m)} | {np.mean(v):.4f} | "
                     f"[{lo:.4f}, {hi:.4f}] | {len(v)} |")
    lines.append(f"| _chance_ | {chance:.4f} | | |")
    if baseline is not None:
        lines.append(f"| _{PS.loss_short(baseline['loss'])} (baseline)_ | "
                     f"{baseline['mean']:.4f} | | {baseline['n']} |")
    lines += ["", f"Retraining noise floor ({', '.join(sorted(floor['means']))}): "
                  f"**{floor['spread']:.4f}**. `esc gap` is check C5's "
                  "between-arm escape-fraction difference; above ~0.10 the two "
                  "arms are not facing the same effective optimizer and the "
                  "contrast carries that caveat.", ""]

    if camp["contrasts"]:
        lines += ["| contrast | diff | 95% CI | p | n+ / n | clears noise | "
                  "esc gap |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for a, b, label in (tuple(c) for c in camp["contrasts"]):
            c = R.contrast(rows, args.optimizer, a, b,
                           matrix=args.cfg["score_matrix"],
                           bootstrap_n=args.bootstrap_n,
                           seed=args.bootstrap_seed)
            if c is None:
                lines.append(f"| {label} | _missing an arm_ | | | | | |")
                continue
            over = abs(c["mean_diff"]) > floor["spread"]
            gap = R.escape_gap(rows, args.optimizer, a, b)
            lines.append(
                f"| {label} | {c['mean_diff']:+.4f} | "
                f"[{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | "
                f"{c['wilcoxon_p']:.1e} | {c['n_pos']}/{c['n']} | "
                f"{'yes' if over else 'NO'} | {gap:.3f} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------


def main():
    cfg = get_region_scores_config()
    args = _parse_args(cfg)
    # The figure functions already take six arguments each; the config rides on
    # `args` beside the bootstrap knobs it belongs with rather than becoming a
    # seventh. `args.cfg` is read for `score_matrix` and nothing else.
    args.cfg = cfg
    plot_root = Path(args.plot_path)
    ext = args.format

    rows = R.load(args.score_path, exclude_losses=cfg["exclude_losses"])
    R.check_rescore_matches_stored(rows)

    # An explicit --optimizer wins everywhere; otherwise each camp may name its
    # own arm and the rest fall back to the global choice.
    forced = args.optimizer
    default_arm = forced
    if default_arm is None:
        # The ranking behind the choice is written out by
        # plot_region_optimizers.py, which owns the decision; here only the
        # chosen arm is needed.
        default_arm, _ = R.best_optimizer(rows, cfg)
        print(f"  default arm '{default_arm}' "
              f"({'pinned' if cfg.get('best_optimizer') != 'auto' else 'chosen'}"
              f" by best_optimizer_rule)")

    # One order for every camp, so the by-set panels can be read against each
    # other and against the optimizer camp.
    order = R.set_order(rows, optimizer=cfg["set_order_optimizer"],
                        matrix=cfg["score_matrix"])

    wanted = [args.camp] if args.camp else R.camp_names(cfg)
    for name in wanted:
        camp = R.camp(cfg, name)
        if camp["axis"] == "optimizer":
            if args.camp:
                raise SystemExit(
                    f"camp {name!r} varies the optimizer; "
                    "plot_region_optimizers.py draws it")
            continue

        # The floor and the baseline are properties of the ARM, so they are
        # recomputed per camp now that a camp can choose its own.
        args.optimizer = forced or camp["optimizer"] or default_arm
        floor = R.noise_floor(rows, cfg, args.optimizer)
        base_rows = R.baseline_rows(rows, cfg, args.optimizer)
        baseline = None
        if base_rows:
            baseline = {
                "loss": cfg["baseline_loss"], "n": len(base_rows),
                "mean": float(np.mean([R.score(r, cfg) for r in base_rows])),
            }

        sub = R.camp_rows(rows, camp, optimizer=args.optimizer)
        members = R.camp_members(rows, camp, optimizer=args.optimizer)
        missing = [m for m in camp["losses"] if m not in members]
        own = " (camp's own arm)" if camp["optimizer"] and not forced else ""
        print(f"\n[{name}] under '{args.optimizer}'{own} | {len(sub)} rows, "
              f"arms {members}" + (f", MISSING {missing}" if missing else ""))
        print(f"  noise floor {floor['spread']:.4f}"
              + (f", {cfg['baseline_loss']} {baseline['mean']:.4f}"
                 if baseline else
                 f", NO {cfg['baseline_loss']} rows under this arm"))
        if len(members) < 2:
            print(f"  skipping {name}: needs at least two arms with rows under "
                  f"'{args.optimizer}'. Score them first.")
            continue

        out = plot_root / name
        fig_scores(camp, sub, members, floor, baseline, args,
                   out / f"scores.{ext}")
        fig_contrasts(camp, sub, members, floor, args,
                      out / f"contrast_forest.{ext}")
        fig_by_set(camp, sub, members, order, args, out / f"per_set.{ext}")
        write_summary(camp, sub, members, floor, baseline, args,
                      out / "summary.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
