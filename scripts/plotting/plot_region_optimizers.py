"""The optimizer camp: which optimizer to believe, and why.

Every loss-axis camp is drawn under ONE optimizer, so which one is a decision
the whole figure set rests on. This script makes that decision visible instead
of implicit. It draws every camp declared ``axis: optimizer`` in
``configs/region_scores.yaml`` and writes ``best_optimizer.json`` beside the
figures. There are two such camps:

``optimizer``   the three-loss spine under all four arms -- which arm to trust.
``beta_arms``   the KL ladder under all four arms. The ladder's polyak-only
                result was monotone in beta AND monotone in escape fraction, so
                its ordering had to be re-asked of an arm that cannot produce
                that artifact. Whether the ordering survives under adam is the
                whole question, and this is the figure that answers it.

The rule (``best_optimizer_rule``) is not "highest score". An arm is eligible
only if its between-loss ``escape_frac`` spread is within
``max_escape_spread``: an optimizer that escapes far more on one loss than
another is not facing them equally, so its RANKING of the losses says as much
about the optimizer as about the landscapes. That is check C5 in
``gen_region_success_scores.py --check``, and the 2026-08-29 activation study
failed it on polyak (spread 0.135, relu 0.625 against mse-zscore 0.489). An arm
can top the score table and still be disqualified, which is exactly what
``optimizer_decision`` draws.

Scale invariance is the other axis the arms differ on, and it is not something
a figure can measure -- it is a property of the update rule:

* **adam** ``m/sqrt(v)`` is exactly invariant to multiplying the loss by a
  constant, up to its own ``eps``. It is C8's reference for that reason.
* **polyak** is invariant while its trust-region cap does not bind, which on
  these landscapes is about half the time.
* **jaxley-polyak** ``x <- x - (f/3) g/||g||^0.8`` is not invariant at all: its
  step length scales with the loss. It is here because it is the loop the
  Jaxley fitting examples actually run.
* **rmsprop** normalizes by ``sqrt(E[g^2])``, so its step is ~lr regardless of
  the loss scale.

Usage (from encoding/):
    python scripts/plotting/plot_region_optimizers.py
    python scripts/plotting/plot_region_optimizers.py --camp beta_arms --format png
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

import plot_style as PS  # noqa: E402
import region_scores as R  # noqa: E402
from helper import get_region_scores_config  # noqa: E402

# The diagnostics that say whether an arm is optimizing or wandering. Each is
# stored per row as a mean over that row's 1000 runs.
DIAGNOSTICS = (
    ("diag.escape_frac", "ended outside the box"),
    ("diag.worse_than_start_frac", "ended worse than it started"),
    ("diag.moving_frac", "still moving at the last step"),
    ("diag.polyak_cap_frac", "polyak cap bound (polyak only)"),
)


def _parse_args(cfg):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score_path", default=cfg["score_path"])
    ap.add_argument("--plot_path", default=cfg["plot_path"])
    ap.add_argument("--camp", default=None,
                    help="draw one axis:optimizer camp (default: all of them)")
    ap.add_argument("--bootstrap_n", type=int, default=cfg["bootstrap_n"])
    ap.add_argument("--bootstrap_seed", type=int, default=cfg["bootstrap_seed"])
    ap.add_argument("--format", default="pdf", choices=("pdf", "png"),
                    help="pdf for the report, png for a quick look")
    return ap.parse_args()


def _optimizer_camps(cfg, name=None):
    """Every camp that varies the optimizer, or just the one named.

    There is more than one now: `optimizer` compares the arms on the three-loss
    spine, and `beta_arms` holds the KL ladder fixed and varies the arm. Both
    are the same figure shape, so both are drawn here rather than in a second
    script.
    """
    if name:
        c = R.camp(cfg, name)
        if c["axis"] != "optimizer":
            raise SystemExit(
                f"camp {name!r} is not an axis:optimizer camp; "
                "plot_region_camps.py draws the loss-axis camps"
            )
        return [c]
    found = [R.camp(cfg, n) for n in R.camp_names(cfg)]
    found = [c for c in found if c["axis"] == "optimizer"]
    if not found:
        raise SystemExit("no axis:optimizer camp in configs/region_scores.yaml")
    return found


def _label(camp, loss) -> str:
    """The camp's own name for a loss, else the global short label.

    `beta_arms` renames `encoder-zscore-2` to "beta 0.20": it IS the ladder's
    top rung, and calling it "ZScore (repl.)" in a figure about KL weights would
    be actively wrong. Same rule as plot_region_camps._label.
    """
    return R.member_label(camp, loss) or PS.loss_short(loss)


def _present(rows, camp):
    """The camp's optimizers and losses that actually have rows, in order."""
    opts = {r.get("optimizer") for r in rows}
    losses = {r["loss_name"] for r in rows}
    return ([o for o in camp["optimizers"] if o in opts],
            [x for x in camp["losses"] if x in losses])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_scores(camp, rows, opts, losses, cfg, args, out: Path):
    """Mean score per (loss, optimizer): the comparison the decision rests on."""
    fig, ax = plt.subplots(figsize=(2.6 + 1.5 * len(losses), 4.6))
    width = 0.8 / max(len(opts), 1)
    payload = {"losses": losses, "optimizers": opts, "mean": {}, "ci": {}}

    for k, opt in enumerate(opts):
        means, los, his = [], [], []
        for loss in losses:
            v = [R.score(r, cfg) for r in rows
                 if r.get("optimizer") == opt and r["loss_name"] == loss]
            m = float(np.mean(v)) if v else np.nan
            lo, hi = (R._boot_ci(np.asarray(v, float), args.bootstrap_n,
                                 args.bootstrap_seed) if v else (np.nan, np.nan))
            means.append(m)
            los.append(m - lo)
            his.append(hi - m)
        x = np.arange(len(losses)) + (k - (len(opts) - 1) / 2) * width
        ax.bar(x, means, width=width * 0.92, color=PS.opt_color(opt),
               edgecolor="none", label=PS.opt_label(opt), zorder=2)
        ax.errorbar(x, means, yerr=[los, his], fmt="none", ecolor=PS.TEXT,
                    elinewidth=1.0, capsize=2.5, zorder=3)
        payload["mean"][opt] = dict(zip(losses, means))
        payload["ci"][opt] = {l: [m - a, m + b]
                              for l, m, a, b in zip(losses, means, los, his)}

    chance = float(np.mean([R.chance(r, R.SUCCESS_RADII) for r in rows]))
    ax.axhline(chance, ls="--", lw=1.2, color=PS.MUTED, zorder=1)
    ax.annotate(f"chance {chance:.3f}", xy=(0.985, chance),
                xycoords=("axes fraction", "data"), xytext=(0, 2.5),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=7.5, color=PS.MUTED,
                bbox=dict(facecolor=PS.PANEL, edgecolor="none", alpha=0.85,
                          pad=1.4), zorder=4)
    payload["chance"] = chance

    ax.set_xticks(np.arange(len(losses)))
    ax.set_xticklabels([_label(camp, x) for x in losses], fontsize=8.5)
    PS.style(ax, ylabel="region score (best iterate)",
             title=f"{camp['title']} -- score per loss, per arm")
    # Headroom for the legend, which would otherwise sit on the tallest bar's
    # error bar.
    tops = [hi for v in payload["ci"].values() for _, hi in v.values()]
    ax.set_ylim(0, max(tops + [chance]) * 1.22)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    PS.save(fig, out, payload)


def fig_diagnostics(camp, rows, opts, losses, out: Path):  # noqa: D401
    """Is the arm optimizing, or wandering? One panel per diagnostic.

    These are what separate "scores low because the landscape is hard" from
    "scores low because the step size is wrong", and they are the evidence
    behind the eligibility gate in the decision figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.0))
    payload = {}
    for ax, (field, title) in zip(axes.ravel(), DIAGNOSTICS):
        width = 0.8 / max(len(opts), 1)
        panel = {}
        for k, opt in enumerate(opts):
            vals = []
            for loss in losses:
                v = [R._dig(r, field) for r in rows
                     if r.get("optimizer") == opt and r["loss_name"] == loss]
                v = [float(x) for x in v if x is not None and np.isfinite(x)]
                vals.append(float(np.mean(v)) if v else np.nan)
            x = np.arange(len(losses)) + (k - (len(opts) - 1) / 2) * width
            ax.bar(x, vals, width=width * 0.92, color=PS.opt_color(opt),
                   edgecolor="none", label=PS.opt_label(opt), zorder=2)
            panel[opt] = dict(zip(losses, vals))
        ax.set_xticks(np.arange(len(losses)))
        ax.set_xticklabels([_label(camp, x) for x in losses], fontsize=7.5,
                           rotation=20, ha="right")
        PS.style(ax, ylabel="fraction of runs", title=title)
        payload[field] = panel
    fig.suptitle("Optimizer behaviour, averaged over runs and parameter subsets",
                 fontsize=11, y=0.995)
    # One figure-level legend rather than one inside a panel: an in-panel legend
    # lands on the tallest bars of whichever diagnostic happens to be first.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=len(opts),
               loc="upper center", bbox_to_anchor=(0.5, 0.962))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    PS.save(fig, out, payload)


def fig_agreement(camp, rows, opts, cfg, out: Path):
    """Every arm's score against the reference arm's, one point per (loss, set).

    Two arms that rank the parameter subsets alike are measuring the same thing
    about the landscape and differ only in level; two that do not are measuring
    the optimizer. This is the pooled version of C8's spearman.
    """
    from gen_region_success_scores import REFERENCE_OPTIMIZER as REF

    if REF not in opts:
        print(f"  no {REF} rows; skipping the agreement figure")
        return
    others = [o for o in opts if o != REF]
    if not others:
        print("  only the reference arm is present; skipping the agreement figure")
        return

    scores = {}
    for r in rows:
        scores[(r.get("optimizer"), r["loss_name"], r["set"])] = R.score(r, cfg)
    cells = sorted({(l, s) for (_, l, s) in scores})

    fig, axes = plt.subplots(1, len(others), figsize=(4.4 * len(others), 4.4),
                             squeeze=False)
    payload = {}
    for ax, opt in zip(axes[0], others):
        px, py, cols = [], [], []
        for loss, s in cells:
            if (REF, loss, s) in scores and (opt, loss, s) in scores:
                px.append(scores[(REF, loss, s)])
                py.append(scores[(opt, loss, s)])
                cols.append(PS.loss_color(loss))
        if not px:
            continue
        ax.scatter(px, py, s=13, c=cols, alpha=0.75, linewidths=0)
        lim = [0, max(max(px), max(py)) * 1.06]
        ax.plot(lim, lim, ls="--", lw=1.0, color=PS.MUTED)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        st = PS.corr(px, py)
        PS.style(ax, xlabel=f"{PS.opt_short(REF)} score",
                 ylabel=f"{PS.opt_short(opt)} score",
                 title=f"{PS.opt_short(opt)} vs {PS.opt_short(REF)}: "
                       f"rho = {st['spearman']:+.3f}")
        payload[opt] = {"spearman": st, "n": len(px)}
    fig.tight_layout()
    PS.save(fig, out, payload)


def fig_decision(camp, ranking, chosen, cfg, out: Path):
    """The rule, drawn: score against the comparability gate that can veto it."""
    limit = float((cfg.get("best_optimizer_rule") or {}).get(
        "max_escape_spread", 1.0))
    opts = [o for o in ranking if ranking[o]["mean_score"] is not None]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.set_xlim(0, max([ranking[o]["escape_spread"] for o in opts]
                       + [limit]) * 1.55)
    # Everything right of the limit is vetoed, whatever it scores.
    ax.axvspan(limit, ax.get_xlim()[1], color="#c0392b", alpha=0.08, lw=0,
               zorder=0)
    ax.axvline(limit, ls="--", lw=1.2, color="#c0392b", zorder=1)
    ax.annotate(f"comparability limit {limit:g}\n(C5: between-loss escape spread)",
                xy=(limit, 0.02), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", va="bottom",
                fontsize=7.5, color="#c0392b")

    for o in opts:
        v = ranking[o]
        ax.plot([v["escape_spread"]], [v["mean_score"]], "o", ms=11,
                color=PS.opt_color(o),
                mec=PS.TEXT if o == chosen else "none",
                mew=2.0 if o == chosen else 0, zorder=3)
        ax.annotate(PS.opt_short(o) + ("  <- chosen" if o == chosen else
                                       ("" if v["eligible"] else "  (vetoed)")),
                    (v["escape_spread"], v["mean_score"]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=PS.TEXT if v["eligible"] else PS.MUTED)

    PS.style(ax, xlabel="between-loss escape-fraction spread",
             ylabel="mean region score (best iterate)",
             title="Choosing the arm every other camp is drawn under")
    PS.save(fig, out, {"chosen": chosen, "limit": limit, "ranking": ranking})


def write_summary(camp, ranking, chosen, opts, losses, cfg, out: Path):
    lines = [f"# {camp['title']}", "",
             f"Chosen arm: **{PS.opt_label(chosen)}**. Statistic: "
             f"`{cfg['score_matrix']}` (best iterate).", "",
             "| optimizer | mean score | escape spread | eligible | note |",
             "| --- | --- | --- | --- | --- |"]
    for o in opts:
        v = ranking.get(o, {})
        note = []
        if v.get("missing_losses"):
            note.append("missing " + ", ".join(v["missing_losses"]))
        if not v.get("eligible") and not v.get("missing_losses"):
            note.append("escape spread over the C5 limit")
        if o == chosen:
            note.append("chosen")
        ms = v.get("mean_score")
        lines.append(
            f"| {PS.opt_label(o)} | {'--' if ms is None else f'{ms:.4f}'} | "
            f"{v.get('escape_spread', float('nan')):.3f} | "
            f"{'yes' if v.get('eligible') else 'no'} | {'; '.join(note)} |")
    lines += ["", "Losses in this camp: "
              + ", ".join(f"{_label(camp, x)} (`{x}`)" for x in losses) + ".", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------


def main():
    cfg = get_region_scores_config()
    args = _parse_args(cfg)
    root = Path(args.plot_path)
    ext = args.format

    allrows = R.load(args.score_path, exclude_losses=cfg["exclude_losses"])
    R.check_rescore_matches_stored(allrows)

    # The ranking is computed over the WHOLE store, not just a camp's rows,
    # because best_optimizer_rule names its own `over_losses` spine and the two
    # need not be the same set. It is therefore one number for every camp here.
    chosen, ranking = R.best_optimizer(allrows, cfg)
    print(f"  chosen arm: {chosen}")

    for camp in _optimizer_camps(cfg, args.camp):
        rows = R.camp_rows(allrows, camp)
        if not rows:
            raise SystemExit(
                f"camp {camp['name']!r} has no rows: it wants losses "
                f"{camp['losses']} under optimizers {camp['optimizers']}"
            )
        opts, losses = _present(rows, camp)
        missing_opts = [o for o in camp["optimizers"] if o not in opts]
        missing_losses = [x for x in camp["losses"] if x not in losses]
        print(f"\n[{camp['name']}] {len(rows)} rows | arms {opts} | "
              f"losses {losses}"
              + (f" | ARMS NOT YET RUN: {missing_opts}" if missing_opts else "")
              + (f" | LOSSES MISSING: {missing_losses}" if missing_losses else ""))
        if len(opts) < 2:
            print(f"  skipping {camp['name']}: needs at least two arms with "
                  "rows. Score them first.")
            continue

        out = root / camp["name"]
        fig_scores(camp, rows, opts, losses, cfg, args, out / f"scores.{ext}")
        fig_diagnostics(camp, rows, opts, losses, out / f"diagnostics.{ext}")
        fig_agreement(camp, rows, opts, cfg, out / f"agreement.{ext}")
        fig_decision(camp, ranking, chosen, cfg, out / f"decision.{ext}")
        write_summary(camp, ranking, chosen, opts, losses, cfg,
                      out / "summary.md")

    root.mkdir(parents=True, exist_ok=True)
    (root / "best_optimizer.json").write_text(json.dumps(
        {"chosen": chosen, "rule": cfg.get("best_optimizer_rule"),
         "pinned": cfg.get("best_optimizer"), "ranking": ranking},
        indent=1, sort_keys=True, default=PS._jsonable))
    print(f"  wrote {root / 'best_optimizer.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
