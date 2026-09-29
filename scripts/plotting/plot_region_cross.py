"""Comparisons ACROSS camps, drawn only when they survive both gates.

Within a camp, two arms differ in exactly one thing -- masking on or off, one
activation or another, one KL weight or another -- so any difference between
them is attributable. Across camps they differ in several things at once, and
the number of possible comparisons explodes: the four camps in
``configs/region_scores.yaml`` generate dozens of pairs, of which a handful
would clear p < 0.05 by construction even if every arm were identical.

So every cross-camp pair is computed and only some are drawn. Two gates, both
configured under ``cross_camp``:

1. **Holm-Bonferroni** over the whole cross-camp family. Step-down, so it is
   uniformly more powerful than plain Bonferroni at the same familywise error.
2. **Larger than the retraining noise floor.** The paired test runs over 56
   parameter subsets and resolves differences far below what retraining one
   configuration moves the score, so significance alone is not evidence of a
   real difference between two encoders. The floor is the spread of three
   config-identical encoders' mean scores.
3. **Comparability.** The two arms must escape the box at similar rates
   (check C5's statistic, per pair). Under polyak `mse-zscore` escapes on 0.489
   of starts against `encoder-act-relu`'s 0.625, so an encoder-vs-MSE gap there
   is partly a statement about the optimizer. The activation contrasts among
   themselves are well inside the limit; this gate is what tells the two cases
   apart instead of drawing them the same way.

Nothing is hidden, only un-drawn: the complete table, suppressed rows included
with the reason each was suppressed, goes into the figure's ``.json`` sidecar
and into ``cross_camp.md``.

Usage (from encoding/):
    python scripts/plotting/plot_region_cross.py
    python scripts/plotting/plot_region_cross.py --format png --show_all
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

# The pseudo-camp the baseline loss belongs to. It is not a camp in the config
# -- it is one arm, drawn as a reference line inside every camp -- but an
# encoder-vs-MSE gap is the most consequential cross comparison in the study, so
# it has to be in the family that gets corrected rather than quoted loose.
BASELINE_CAMP = "baseline"


def _parse_args(cfg):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score_path", default=cfg["score_path"])
    ap.add_argument("--plot_path", default=cfg["plot_path"])
    ap.add_argument("--optimizer", default=None,
                    help="override the arm; by default best_optimizer decides")
    ap.add_argument("--alpha", type=float,
                    default=float((cfg.get("cross_camp") or {}).get("alpha", 0.05)))
    ap.add_argument("--show_all", action="store_true",
                    help="draw every pair, gates annotated but not applied. For "
                         "inspection only -- not for the report.")
    ap.add_argument("--bootstrap_n", type=int, default=cfg["bootstrap_n"])
    ap.add_argument("--bootstrap_seed", type=int, default=cfg["bootstrap_seed"])
    ap.add_argument("--format", default="pdf", choices=("pdf", "png"),
                    help="pdf for the report, png for a quick look")
    return ap.parse_args()


def _membership(cfg, rows, optimizer):
    """``{loss: {camp names}}`` over the loss-axis camps, plus the baseline.

    A loss can belong to more than one camp -- the beta ladder's top rung is the
    masking camp's unmasked encoder -- and that is exactly why membership is a
    SET. A pair counts as cross-camp only when the two share no camp at all;
    otherwise the comparison is already drawn inside the camp they share, and
    drawing it again here would double-count it in the correction.
    """
    present = {r["loss_name"] for r in rows if r.get("optimizer") == optimizer}
    member: dict[str, set] = {}
    labels: dict[str, str] = {}
    for name in R.camp_names(cfg):
        camp = R.camp(cfg, name)
        if camp["axis"] != "loss":
            continue
        for loss in camp["losses"]:
            if loss in present:
                member.setdefault(loss, set()).add(name)
                labels.setdefault(loss, R.member_label(camp, loss)
                                  or PS.loss_short(loss))
    base = cfg.get("baseline_loss")
    if base and base in present:
        member.setdefault(base, set()).add(BASELINE_CAMP)
        labels.setdefault(base, PS.loss_short(base))
    return member, labels


def _pairs(member):
    """Every pair of arms sharing no camp, in a stable order."""
    names = sorted(member)
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
            if not (member[a] & member[b])]


def _measure(rows, member, labels, optimizer, cfg, args) -> list[dict]:
    """Every cross-camp contrast, Holm-adjusted, with the gate verdicts."""
    out = []
    for a, b in _pairs(member):
        c = R.contrast(rows, optimizer, a, b, matrix=cfg["score_matrix"],
                       bootstrap_n=args.bootstrap_n, seed=args.bootstrap_seed)
        if c is None:
            continue
        out.append({
            "loss_a": a, "loss_b": b,
            "camp_a": "+".join(sorted(member[a])),
            "camp_b": "+".join(sorted(member[b])),
            "label_a": labels[a], "label_b": labels[b],
            "label": f"{labels[a]} vs {labels[b]}",
            "mean_diff": c["mean_diff"], "ci_lo": c["ci_lo"],
            "ci_hi": c["ci_hi"], "p": c["wilcoxon_p"],
            "n": c["n"], "n_pos": c["n_pos"],
            "mean_a": c["mean_a"], "mean_b": c["mean_b"],
            "escape_gap": R.escape_gap(rows, optimizer, a, b),
        })
    for row, adj in zip(out, R.holm([r["p"] for r in out])):
        row["p_holm"] = float(adj)
    return out


def _gate(measured, floor, cfg, alpha):
    """Apply the two gates, recording WHY each suppressed row was suppressed."""
    cc = cfg.get("cross_camp") or {}
    require_floor = bool(cc.get("require_over_noise_floor", True))
    max_gap = cc.get("max_escape_gap")
    spread = floor.get("spread")
    for row in measured:
        reasons = []
        if not (row["p_holm"] < alpha):
            reasons.append(f"Holm p = {row['p_holm']:.3f} >= {alpha}")
        if require_floor and spread is not None and np.isfinite(spread):
            if abs(row["mean_diff"]) <= spread:
                reasons.append(
                    f"|diff| {abs(row['mean_diff']):.4f} <= noise floor "
                    f"{spread:.4f}")
        gap = row.get("escape_gap")
        if max_gap is not None and gap is not None and np.isfinite(gap):
            if gap > float(max_gap):
                reasons.append(
                    f"escape gap {gap:.3f} > {float(max_gap):g} (C5: the two "
                    "arms are not facing the same effective optimizer)")
        row["suppressed_because"] = reasons
        row["drawn"] = not reasons
    return measured


def fig_cross(measured, floor, optimizer, cfg, args, out: Path):
    shown = [m for m in measured if m["drawn"] or args.show_all]
    shown = sorted(shown, key=lambda m: m["mean_diff"])
    n_drawn = sum(1 for m in measured if m["drawn"])

    fig, ax = plt.subplots(figsize=(8.6, 1.4 + 0.42 * max(len(shown), 3)))
    spread = floor.get("spread")
    if spread is not None and np.isfinite(spread):
        ax.axvspan(-spread, spread, color=PS.MUTED, alpha=0.16, lw=0, zorder=0,
                   label=f"retraining noise (+-{spread:.4f})")
    ax.axvline(0.0, color=PS.TEXT, lw=1.0, zorder=1)

    if not shown:
        ax.text(0.5, 0.5, "no cross-camp comparison clears both gates",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=PS.MUTED)
        ax.set_yticks([])
    else:
        y = np.arange(len(shown))[::-1]
        for yi, m in zip(y, shown):
            drawn = m["drawn"]
            col = PS.loss_color(m["loss_b"]) if drawn else PS.MUTED
            ax.plot([m["ci_lo"], m["ci_hi"]], [yi, yi], lw=2.0, color=PS.TEXT,
                    alpha=0.75 if drawn else 0.25, solid_capstyle="round",
                    zorder=2)
            ax.plot([m["mean_diff"]], [yi], "o", ms=6.0, color=col,
                    mec=PS.TEXT if drawn else PS.MUTED, mew=0.8, zorder=3,
                    alpha=1.0 if drawn else 0.45)
            ax.annotate(
                f"{m['mean_diff']:+.4f}  Holm p={m['p_holm']:.1e}  "
                f"{m['n_pos']}/{m['n']}",
                (max(m["ci_hi"], m["mean_diff"]), yi), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=7,
                color=PS.TEXT if drawn else PS.MUTED)
        ax.set_yticks(y)
        ax.set_yticklabels([m["label"] for m in shown], fontsize=7.5)
        ax.set_ylim(-0.7, len(shown) - 0.3)

    PS.style(
        ax,
        xlabel="paired mean difference (positive = the second arm scores higher)",
        title=(f"Cross-camp comparisons -- {PS.opt_label(optimizer)}\n"
               f"{n_drawn} of {len(measured)} pairs clear Holm alpha="
               f"{args.alpha} and the noise floor"
               + ("  [--show_all: greyed rows do NOT]" if args.show_all else "")))
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.5 * (hi - lo))
    PS.save(fig, out, {
        "optimizer": optimizer, "score_matrix": cfg["score_matrix"],
        "alpha": args.alpha, "correction": (cfg.get("cross_camp") or {}).get(
            "correction", "holm"),
        "noise_floor": floor, "n_pairs": len(measured), "n_drawn": n_drawn,
        "pairs": measured,
    })


def write_table(measured, floor, optimizer, cfg, args, out: Path):
    lines = ["# Cross-camp comparisons", "",
             f"Optimizer: **{PS.opt_label(optimizer)}**. Statistic: "
             f"`{cfg['score_matrix']}`. Family size {len(measured)}, "
             f"Holm-corrected at alpha = {args.alpha}. Noise floor "
             f"{floor['spread']:.4f}. Max escape gap "
             f"{(cfg.get('cross_camp') or {}).get('max_escape_gap')}.", "",
             "Every pair is listed. `drawn` is what appears in the figure.", "",
             "| a | b | diff | 95% CI | p | Holm p | n+ / n | esc gap | "
             "drawn | why not |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in sorted(measured, key=lambda r: r["p_holm"]):
        lines.append(
            f"| {m['label_a']} ({m['camp_a']}) "
            f"| {m['label_b']} ({m['camp_b']}) "
            f"| {m['mean_diff']:+.4f} | [{m['ci_lo']:+.4f}, {m['ci_hi']:+.4f}] "
            f"| {m['p']:.1e} | {m['p_holm']:.1e} | {m['n_pos']}/{m['n']} "
            f"| {m['escape_gap']:.3f} "
            f"| {'yes' if m['drawn'] else 'no'} "
            f"| {'; '.join(m['suppressed_because'])} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")


def main():
    cfg = get_region_scores_config()
    args = _parse_args(cfg)
    root = Path(args.plot_path)
    ext = args.format

    rows = R.load(args.score_path, exclude_losses=cfg["exclude_losses"])
    R.check_rescore_matches_stored(rows)

    optimizer = args.optimizer
    if optimizer is None:
        optimizer, _ = R.best_optimizer(rows, cfg)
    floor = R.noise_floor(rows, cfg, optimizer)

    member, labels = _membership(cfg, rows, optimizer)
    measured = _gate(_measure(rows, member, labels, optimizer, cfg, args),
                     floor, cfg, args.alpha)
    if not measured:
        raise SystemExit(
            "no cross-camp pair could be measured: every arm present under "
            f"'{optimizer}' shares a camp with every other. Score another "
            "camp's encoders first."
        )
    n_drawn = sum(1 for m in measured if m["drawn"])
    print(f"  {len(measured)} cross-camp pairs under '{optimizer}', "
          f"{n_drawn} clear both gates")

    out = root / "cross_camp"
    fig_cross(measured, floor, optimizer, cfg, args, out / f"cross_camp.{ext}")
    write_table(measured, floor, optimizer, cfg, args, out / "cross_camp.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
