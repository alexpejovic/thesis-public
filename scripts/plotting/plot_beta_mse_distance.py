"""Rank agreement with MSE against encoder beta, averaged over VAE training seeds.

Reads the json records written by ``scripts/datagen/gen_beta_mse_distance.py``. Each
record is one 3-parameter subset and holds, per (beta, training seed) encoder, the
Spearman rank correlation between its loss landscape and the MSE landscape. Two
levels of averaging, kept separate on purpose:

* over **training seeds** within a beta -- removes VAE training variance, which is
  what the extra seeds were trained for;
* over **parameter subsets** -- removes landscape-to-landscape variance.

Rank agreement is the headline metric because it is invariant to how a landscape is
mapped onto [0, 1]: it asks whether the encoder orders parameter settings the way
MSE does, which is the property the optimizer actually consumes. The squared
distances (min-max and percentile-anchored) are kept as scale-sensitive diagnostics
in one comparison figure, since where they disagree with each other is where a few
extreme grid points were setting a landscape's scale.

The headline correlation is computed on subset-level values that have already been
averaged over training seeds, so each (subset, beta) contributes one point and
training noise does not inflate the sample size.

Thresholds, paths and the choice of headline metric come from
``configs/beta_mse.yaml`` -- the same file the datagen script reads. Every figure
gets a sibling ``.json`` holding the numbers it draws, so a figure in the report
is re-derivable without re-running the sweep.

Usage (from encoding/):
    python scripts/plotting/plot_beta_mse_distance.py
    python scripts/plotting/plot_beta_mse_distance.py --scatter_stem <stem>
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from helper import get_beta_mse_config, unit  # noqa: E402


def _dump(path: Path, obj) -> None:
    """Write a figure's numbers beside the figure itself.

    A PDF in the report is otherwise only re-derivable by re-running the whole
    sweep, and the statistics annotated onto these figures exist nowhere else.
    """
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_jsonable))


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def _parse_args(cfg: dict):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channel_type", default=cfg["channel_type"])
    ap.add_argument("--opt_samples", type=int, default=None,
                    help="only use records at this grid resolution "
                         "(default: the finest present)")
    ap.add_argument("--scatter_stem", default=None,
                    help="file stem of a --save_losses run, for the "
                         "feat-vs-mse scatter")
    ap.add_argument("--overlay_stem", default=None,
                    help="file stem of a --save_losses run, for the 2-D landscape "
                         "overlay (MSE beside each beta). Defaults to "
                         "--scatter_stem when that is given.")
    ap.add_argument("--scatter_train_seed", type=int, default=None,
                    help="which training seed to draw in the scatter "
                         "(default: the lowest present)")
    return ap.parse_args()


def load_records(cfg: dict, channel_type: str, opt_samples: int | None) -> list[dict]:
    data_base = Path(cfg["save_path"])
    min_ideal_v_range = float(cfg["min_ideal_v_range"])
    min_p98_span = float(cfg["min_p98_span"])

    paths = sorted((data_base / channel_type).glob("*.json"))
    if not paths:
        raise SystemExit(f"no records under {data_base / channel_type}")
    records = [json.loads(p.read_text()) for p in paths]

    if opt_samples is None:
        opt_samples = max(r["opt_samples"] for r in records)
    kept = [r for r in records if r["opt_samples"] == opt_samples]
    print(f"{len(kept)}/{len(records)} records at opt_samples={opt_samples}")

    flat = [r for r in kept if r["ideal_v_range"] < min_ideal_v_range]
    for r in flat:
        print(
            f"  dropping non-spiking {'+'.join(r['params'])} seed={r['seed']} "
            f"(ideal_v_range={r['ideal_v_range']:.1f} mV)"
        )
    kept = [r for r in kept if r["ideal_v_range"] >= min_ideal_v_range]

    # Min-max normalization lets one diverging grid point set a landscape's scale.
    # Flag rather than drop: which subsets are outlier-dominated is itself a result,
    # and dropping them silently would hide it. Rank agreement is immune to this,
    # so the note now only qualifies the distance diagnostics.
    for r in kept:
        d = r.get("diagnostics")
        if not d:
            continue
        worst = min([d["mse_p98_span"]] + list(d["feat_p98_span"]))
        if worst < min_p98_span:
            print(
                f"  NOTE outlier-dominated {'+'.join(r['params'])} "
                f"seed={r['seed']} (p98_span={worst:.3f}); its distances are set "
                "by a few extreme grid points"
            )

    # Every record must score the same encoder set, or a per-beta mean would rest
    # on different models between subsets. The manifest exists to guarantee this;
    # this check is what makes the guarantee visible if it ever breaks.
    sigs = {
        tuple(sorted((e["beta"], e["train_seed"]) for e in r["encoders"]))
        for r in kept
    }
    if len(sigs) > 1:
        raise SystemExit(
            f"records disagree on the encoder set ({len(sigs)} distinct sets). "
            "Rebuild the manifest and re-run the affected subsets."
        )
    return kept


def _seed_averaged(records, betas, values) -> np.ndarray:
    """(n_subsets, n_betas), each cell averaged over training seeds at that beta.

    ``values(record)`` returns a per-encoder list, aligned with
    ``record["encoders"]`` by construction in the datagen script.
    """
    out = np.empty((len(records), len(betas)))
    for i, r in enumerate(records):
        cell = defaultdict(list)
        for e, v in zip(r["encoders"], values(r)):
            cell[e["beta"]].append(v)
        out[i] = [np.mean(cell[b]) for b in betas]
    return out


def _subset_averaged(records, betas, seeds, values) -> np.ndarray:
    """(n_betas, n_seeds), each cell averaged over parameter subsets."""
    cell = defaultdict(list)
    for r in records:
        for e, v in zip(r["encoders"], values(r)):
            cell[(e["beta"], e["train_seed"])].append(v)
    return np.array([[np.mean(cell[(b, s)]) for s in seeds] for b in betas])


def main():
    cfg = get_beta_mse_config()
    args = _parse_args(cfg)
    data_base = Path(cfg["save_path"])
    plot_base = Path(cfg["plot_path"])
    rank_key = cfg["rank_key"]

    records = load_records(cfg, args.channel_type, args.opt_samples)
    if not records:
        raise SystemExit("no usable records")

    missing = [r for r in records if rank_key not in (r.get("diagnostics") or {})]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(records)} records predate the '{rank_key}' "
            "diagnostic; re-run those subsets to get rank agreement."
        )

    encs = records[0]["encoders"]
    betas = sorted({e["beta"] for e in encs})
    seeds = sorted({e["train_seed"] for e in encs})
    print(
        f"{len(records)} subsets x {len(betas)} betas x {len(seeds)} training "
        f"seeds (seeds={seeds})"
    )
    plot_base.mkdir(parents=True, exist_ok=True)

    names = ["+".join(r["params"]) + f" s{r['seed']}" for r in records]

    # (n_subsets, n_betas), already averaged over training seeds.
    A = _seed_averaged(records, betas, lambda r: r["diagnostics"][rank_key])
    # The scale-sensitive diagnostics, same shape.
    D = _seed_averaged(records, betas, lambda r: r["distances"])
    DR = None
    if all("distances_robust" in r["diagnostics"] for r in records):
        DR = _seed_averaged(
            records, betas, lambda r: r["diagnostics"]["distances_robust"]
        )
    # (n_betas, n_seeds) averaged over subsets, for the training-variance panel
    by_seed = _subset_averaged(
        records, betas, seeds, lambda r: r["diagnostics"][rank_key]
    )

    betas_flat = np.tile(betas, len(records))
    rho, pval = spearmanr(betas_flat, A.ravel())
    print(
        f"spearman(beta, seed-averaged rank agreement) = {rho:.3f} "
        f"(p={pval:.2g}, n={A.size}) [expect NEGATIVE: agreement should fall "
        "as beta rises]"
    )
    wins = int(np.sum(np.argmax(A, axis=1) == 0))
    print(f"beta=0 agrees most with MSE in {wins}/{len(A)} subsets")
    per_beta = []
    for j, b in enumerate(betas):
        stats = {
            "beta": float(b),
            "mean": float(A[:, j].mean()),
            "median": float(np.median(A[:, j])),
            "sd_over_subsets": float(A[:, j].std()),
            "sd_over_train_seeds": float(by_seed[j].std()),
        }
        per_beta.append(stats)
        print(
            f"  beta={b:<5g} mean={stats['mean']:.5g} "
            f"median={stats['median']:.5g} "
            f"sd_over_subsets={stats['sd_over_subsets']:.3g} "
            f"sd_over_train_seeds={stats['sd_over_train_seeds']:.3g}"
        )

    rho_d, p_d = spearmanr(betas_flat, D.ravel())
    print(
        f"[diagnostic] spearman(beta, min-max distance) = {rho_d:.3f} "
        f"(p={p_d:.2g}) [expect POSITIVE]"
    )
    # Where the metrics disagree on beta=0 is where the [0,1] mapping, not the
    # ordering, is doing the work, so name those subsets instead of averaging over.
    disagree = [
        names[i]
        for i in range(len(A))
        if (np.argmax(A[i]) == 0) != (np.argmin(D[i]) == 0)
    ]
    print(
        f"rank and min-max distance disagree on the beta=0 verdict in "
        f"{len(disagree)}/{len(A)} subsets"
        + (f"; e.g. {disagree[:3]}" if disagree else "")
    )
    rho_dr = p_dr = None
    flips = []
    if DR is not None:
        rho_dr, p_dr = spearmanr(betas_flat, DR.ravel())
        print(
            f"[diagnostic] spearman(beta, percentile-anchored distance) = "
            f"{rho_dr:.3f} (p={p_dr:.2g}) [expect POSITIVE]"
        )
        flips = [
            names[i]
            for i in range(len(D))
            if (np.argmin(D[i]) == 0) != (np.argmin(DR[i]) == 0)
        ]
        print(
            f"anchor changes the beta=0 verdict in {len(flips)}/{len(D)} subsets"
            + (f"; e.g. {flips[:3]}" if flips else "")
        )

    # --- 0. the metrics side by side ----------------------------------------
    # One figure, because the metrics disagree exactly when a landscape keeps
    # MSE's shape but is mapped onto [0,1] differently; showing them apart would
    # hide the one case a reader needs to see.
    panels = [
        (
            A,
            "Spearman rank agreement with MSE",
            "darkgreen",
            rho,
            "primary metric (higher = closer)",
        ),
        (
            D,
            "min-max distance from MSE",
            "crimson",
            rho_d,
            "scale-sensitive (lower = closer)",
        ),
    ]
    if DR is not None:
        panels.append(
            (
                DR,
                "percentile-anchored distance",
                "darkorange",
                rho_dr,
                "same formula, robust anchor",
            )
        )
    fig, axs = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.4))
    for ax, (M, ylabel, color, rho_m, subtitle) in zip(np.atleast_1d(axs), panels):
        for row in M:
            ax.plot(betas, row, color="0.8", lw=0.7)
        ax.plot(betas, np.median(M, axis=0), color=color, lw=2.5, marker="o")
        ax.set_xlabel(r"encoder $\beta$")
        ax.set_ylabel(ylabel)
        ax.set_title(rf"{subtitle} ($\rho$={rho_m:.2f})", fontsize=10)
    fig.suptitle(
        f"{len(records)} subsets, averaged over {len(seeds)} training seeds",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(plot_base / "rank_and_distance_vs_beta.pdf")
    plt.close(fig)
    _dump(
        plot_base / "rank_and_distance_vs_beta.json",
        {
            "betas": betas,
            "subsets": names,
            "metrics": {
                "rank_agreement": {"values": A, "median": np.median(A, axis=0),
                                   "spearman_vs_beta": rho, "p": pval},
                "minmax_distance": {"values": D, "median": np.median(D, axis=0),
                                    "spearman_vs_beta": rho_d, "p": p_d},
                **(
                    {
                        "percentile_anchored_distance": {
                            "values": DR,
                            "median": np.median(DR, axis=0),
                            "spearman_vs_beta": rho_dr,
                            "p": p_dr,
                        }
                    }
                    if DR is not None
                    else {}
                ),
            },
        },
    )

    # --- 1. per-subset lines + median ---------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in A:
        ax.plot(betas, row, color="0.75", lw=0.7, zorder=1)
    ax.plot(
        betas,
        np.median(A, axis=0),
        color="darkgreen",
        lw=2.5,
        marker="o",
        zorder=3,
        label="median over subsets",
    )
    ax.set_xlabel(r"encoder $\beta$")
    ax.set_ylabel(
        "rank agreement with MSE\n"
        r"Spearman$(\hat{L}_{enc}, \hat{L}_{MSE})$ over grid points"
    )
    ax.set_title(
        f"Encoder loss landscape vs MSE\n{len(records)} subsets, "
        f"averaged over {len(seeds)} training seeds"
    )
    ax.annotate(
        f"Spearman$(\\beta$, agreement$)$ = {rho:.3f}\n"
        f"$p$ = {pval:.2g}, $n$ = {A.size}",
        xy=(0.03, 0.06),
        xycoords="axes fraction",
        va="bottom",
        fontsize=9,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_base / "rank_vs_beta.pdf")
    plt.close(fig)
    _dump(
        plot_base / "rank_vs_beta.json",
        {
            "rank_key": rank_key,
            "opt_samples": records[0]["opt_samples"],
            "n_subsets": len(records),
            "train_seeds": seeds,
            "betas": betas,
            "subsets": names,
            "rank_agreement_seed_averaged": A,
            "median_over_subsets": np.median(A, axis=0),
            "spearman_beta_vs_agreement": {"rho": rho, "p": pval, "n": int(A.size)},
            "beta0_wins": wins,
            "per_beta": per_beta,
            "disagree_with_minmax": disagree,
            "anchor_flips": flips,
        },
    )

    # --- 2. spread over subsets ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(A, tick_labels=[f"{b:g}" for b in betas], showfliers=False)
    for j in range(len(betas)):
        ax.scatter(
            np.full(len(A), j + 1) + np.linspace(-0.12, 0.12, len(A)),
            A[:, j],
            s=6,
            alpha=0.5,
            color="steelblue",
            zorder=3,
        )
    ax.set_xlabel(r"encoder $\beta$")
    ax.set_ylabel("rank agreement with MSE (seed-averaged)")
    ax.set_title("Spread across parameter subsets")
    fig.tight_layout()
    fig.savefig(plot_base / "rank_vs_beta_box.pdf")
    plt.close(fig)
    _dump(
        plot_base / "rank_vs_beta_box.json",
        {
            "betas": betas,
            "subsets": names,
            "rank_agreement_seed_averaged": A,
            "quartiles": {
                f"{b:g}": np.percentile(A[:, j], [25, 50, 75]).tolist()
                for j, b in enumerate(betas)
            },
        },
    )

    # --- 3. how much did averaging over training seeds buy? -----------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, s in enumerate(seeds):
        ax.plot(
            betas,
            by_seed[:, k],
            lw=1.0,
            marker="o",
            ms=4,
            alpha=0.8,
            label=f"train seed {s}",
        )
    sem = (
        by_seed.std(axis=1, ddof=1) / np.sqrt(len(seeds)) if len(seeds) > 1 else None
    )
    ax.errorbar(
        betas,
        by_seed.mean(axis=1),
        yerr=sem,
        color="black",
        lw=2.5,
        marker="s",
        capsize=4,
        label="mean $\\pm$ s.e.m.",
        zorder=5,
    )
    ax.set_xlabel(r"encoder $\beta$")
    ax.set_ylabel("rank agreement with MSE (subset-averaged)")
    ax.set_title("One line per VAE training seed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_base / "rank_per_train_seed.pdf")
    plt.close(fig)
    _dump(
        plot_base / "rank_per_train_seed.json",
        {
            "betas": betas,
            "train_seeds": seeds,
            "rank_agreement_subset_averaged": by_seed,
            "mean_over_seeds": by_seed.mean(axis=1),
            "sem_over_seeds": sem,
        },
    )

    # --- 4. feat-vs-mse scatter, one panel per beta -------------------------
    if args.scatter_stem:
        npz_path = data_base / args.channel_type / f"{args.scatter_stem}_losses.npz"
        if not npz_path.exists():
            raise SystemExit(
                f"{npz_path} not found; re-run that sweep with --save_losses"
            )
        z = np.load(npz_path)
        want = args.scatter_train_seed
        if want is None:
            want = int(np.min(z["train_seeds"]))
        idx = [i for i, s in enumerate(z["train_seeds"]) if int(s) == want]
        if not idx:
            raise SystemExit(
                f"train seed {want} not in {sorted(set(z['train_seeds']))}"
            )
        mse_u = unit(z["mse"])
        fig, axes = plt.subplots(
            1,
            len(idx),
            figsize=(3 * len(idx), 3.2),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        scatter_stats = []
        for k, i in enumerate(idx):
            ax = axes[0][k]
            feat_u = unit(z["feat"][i])
            ax.scatter(mse_u, feat_u, s=3, alpha=0.3, color="steelblue")
            # The fit and identity lines stay as visual guides; the number quoted
            # is the rank agreement, so the panel matches the headline metric.
            slope, intercept = np.polyfit(mse_u, feat_u, 1)
            xs = np.array([0.0, 1.0])
            ax.plot(xs, slope * xs + intercept, color="crimson", lw=1.5)
            ax.plot(xs, xs, color="0.4", lw=0.8, ls="--")
            rho_panel = spearmanr(mse_u, feat_u).statistic
            ax.set_title(
                rf"$\beta$={z['betas'][i]:g} ($\rho$={rho_panel:.2f})", fontsize=9
            )
            ax.set_xlabel("normalized MSE")
            if k == 0:
                ax.set_ylabel("normalized encoder loss")
            scatter_stats.append(
                {
                    "beta": float(z["betas"][i]),
                    "spearman_vs_mse": float(rho_panel),
                    "fit_slope": float(slope),
                    "fit_intercept": float(intercept),
                }
            )
        fig.suptitle(
            f"Encoder loss vs MSE per grid point — train seed {want} — "
            f"{args.scatter_stem}"
        )
        fig.tight_layout()
        fig.savefig(plot_base / "scatter_feat_vs_mse.pdf")
        plt.close(fig)
        # The grids themselves are the npz; only the per-panel statistics are
        # stored here, so the sidecar stays a few hundred bytes.
        _dump(
            plot_base / "scatter_feat_vs_mse.json",
            {
                "stem": args.scatter_stem,
                "train_seed": want,
                "n_grid_points": int(mse_u.size),
                "panels": scatter_stats,
            },
        )

    # --- 5. landscape overlay: MSE beside each beta, at the ideal slice ------
    # The qualitative "it looks like MSE every single time" claim. The legacy
    # compare_encoder_to_mse2d.py draws this from data/comp_losses runs, a format
    # these sweeps do not produce, so it is rebuilt here from the saved 3-D grid:
    # hold the third parameter at its true value and show the remaining 2-D plane.
    overlay_stem = args.overlay_stem or args.scatter_stem
    if overlay_stem:
        npz_path = data_base / args.channel_type / f"{overlay_stem}_losses.npz"
        rec_path = data_base / args.channel_type / f"{overlay_stem}.json"
        if not npz_path.exists():
            raise SystemExit(f"{npz_path} not found; re-run with --save_losses")
        z = np.load(npz_path)
        rec = json.loads(rec_path.read_text())
        axis = z["axis"]
        n, dim = len(axis), len(rec["params"])
        if dim != 3:
            raise SystemExit(f"overlay expects a 3-parameter sweep, got {dim}")

        # Slice the LAST axis at the index nearest its true value, so the plane
        # shown actually contains the optimum rather than an arbitrary cut.
        held = rec["params"][-1]
        k = int(np.argmin(np.abs(axis - rec["ideal_params"][held])))

        want = args.scatter_train_seed
        if want is None:
            want = int(np.min(z["train_seeds"]))
        idx = [i for i, sd in enumerate(z["train_seeds"]) if int(sd) == want]

        panels = [("MSE", unit(z["mse"]).reshape((n,) * dim)[:, :, k])]
        for i in idx:
            panels.append(
                (
                    rf"$\beta$={z['betas'][i]:g}",
                    unit(z["feat"][i]).reshape((n,) * dim)[:, :, k],
                )
            )

        fig, axes = plt.subplots(
            1, len(panels), figsize=(2.6 * len(panels), 3.0), squeeze=False
        )
        for (title, plane), ax in zip(panels, axes[0]):
            im = ax.pcolormesh(axis, axis, plane.T, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(rec["params"][0], fontsize=7)
            ax.set_aspect("equal")
        axes[0][0].set_ylabel(rec["params"][1], fontsize=7)
        fig.colorbar(im, ax=axes[0], fraction=0.02, label="normalized loss")
        fig.suptitle(
            f"{overlay_stem} — {held} held at its true value (train seed {want})",
            fontsize=9,
        )
        fig.savefig(plot_base / "landscape_overlay.pdf", bbox_inches="tight")
        plt.close(fig)
        # The planes are recoverable from the npz given these slice details; the
        # sidecar records the slice, not 5 x n^2 numbers.
        _dump(
            plot_base / "landscape_overlay.json",
            {
                "stem": overlay_stem,
                "train_seed": want,
                "params": rec["params"],
                "held_param": held,
                "held_at_true_value": rec["ideal_params"][held],
                "held_axis_index": k,
                "held_axis_value": float(axis[k]),
                "axis": axis,
                "panels": [title for title, _ in panels],
            },
        )

    print(f"wrote plots and json sidecars to {plot_base}")


if __name__ == "__main__":
    main()
