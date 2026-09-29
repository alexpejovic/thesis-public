"""Shared plotting style, colours and correlation helpers.

Ported from the ``big-loss-scoring`` branch, which lifted it in turn from
``analyze_landscape_vs_optimization.py`` on ``large-landscape-analysis``, so the
score figures cannot end up reporting two different p-values for the same
number. Trimmed on the way over: the ``convexity`` import and everything hanging
off it (``EXPECTED_SIGN``, ``SCORE_NAMES``, ``set_metric_orientation``) is gone
because ``convexity.py`` does not exist on ``main`` and that machinery only ever
described convexity descriptors. ``encoder-zscore-2`` is new here -- the
same-config replicate postdates the port.

Uses ``scipy.stats``, which IS installed in the working venv but is NOT declared
in ``pyproject.toml`` -- as is already latently true of every plotting script in
this directory.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from scipy.stats import pearsonr, rankdata, spearmanr  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from losses import LOSS_NAMES  # noqa: E402

# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------

TEXT, MUTED, GRID = "#0b0b0b", "#8a8984", "#e3e2dd"
PANEL = "#fcfcfb"

LOSS_ORDER = list(LOSS_NAMES)

LABELS = {
    "encoder-minmax": "Encoder (MinMax)",
    "encoder-zscore": "Encoder (ZScoreGlobal)",
    "encoder-zscore-2": "Encoder (ZScoreGlobal, replicate)",
    "encoder-zscore-mask": "Encoder (ZScoreGlobal + mask)",
    "encoder-act-elu": "Encoder (activation ELU)",
    "encoder-act-gelu": "Encoder (activation GELU)",
    "encoder-act-relu": "Encoder (activation ReLU)",
    "encoder-beta-0.00": "Encoder (beta 0.00)",
    "encoder-beta-0.05": "Encoder (beta 0.05)",
    "encoder-beta-0.10": "Encoder (beta 0.10)",
    "encoder-beta-0.15": "Encoder (beta 0.15)",
    "mse-minmax": "MSE floor (MinMax)",
    "mse-zscore": "MSE floor (ZScoreGlobal)",
    "jaxley-paper": "Jaxley-paper summary stats (MAE)",
    "spike-w1": "spike W1",
    "spike-w1-multi": "spike W1 (multi-channel)",
}

# The beta ladder is an ORDERED axis, so its five rungs get a sequential ramp
# (light -> dark) rather than five categorical hues: the eye should read the
# ordering off the figure without consulting the legend. The beta = 0.20 rung is
# `encoder-zscore-2`, which already has a categorical colour because it is also
# the unmasked arm of the masking contrast -- `beta_ramp()` below hands the camp
# driver the ramp colour to override it with, so one encoder can be dark orange
# in the masking camp and the darkest ramp step in the beta camp without either
# figure lying about which arm it is drawing.
BETA_RAMP = ("#cfe3f5", "#93c0e8", "#4f92d0", "#2a6fb0", "#17436d")

COLORS = {
    "encoder-minmax": "#2a78d6",
    "encoder-zscore": "#eb6834",
    # A near-neighbour of `encoder-zscore`, on purpose: the two are a
    # same-config replicate pair and the eye should read them as one family.
    "encoder-zscore-2": "#c2410c",
    "encoder-zscore-mask": "#1baf7a",
    # The activation sweep: three hues of one family, since it is one camp.
    "encoder-act-elu": "#7b4bd6",
    "encoder-act-gelu": "#a97ce8",
    "encoder-act-relu": "#5a2ea6",
    "encoder-beta-0.00": BETA_RAMP[0],
    "encoder-beta-0.05": BETA_RAMP[1],
    "encoder-beta-0.10": BETA_RAMP[2],
    "encoder-beta-0.15": BETA_RAMP[3],
    "mse-minmax": "#8a8984",
    "mse-zscore": "#52514e",
    "jaxley-paper": "#9b59b6",
    "spike-w1": "#d4af37",
    "spike-w1-multi": "#8b6f1f",
}

# The optimizer is an axis of the region-score study, not of the older convexity
# one, so these are new. `polyak` is the inherited setup and `adam` the
# scale-invariant control, which is why adam gets the colder, more "reference"
# colour.
OPT_LABELS = {
    "polyak": "Polyak SGD (max lr 1.0)",
    "adam": "Adam (lr 0.05, cosine)",
    "rmsprop": "RMSProp",
    "jaxley-polyak": "Jaxley Polyak (f/3, beta 0.8)",
}

OPT_COLORS = {
    "polyak": "#c0392b",
    "adam": "#2c6fbb",
    "rmsprop": "#7f8c8d",
    # A warm neighbour of `polyak`: the two are the same idea (a loss-sized
    # step) differing only in how the gradient norm enters, and neither is
    # loss-scale invariant. Reading them as one family is the point.
    "jaxley-polyak": "#e08214",
}

OPT_SHORT = {
    "polyak": "Polyak",
    "adam": "Adam",
    "rmsprop": "RMSProp",
    "jaxley-polyak": "Jaxley Polyak",
}


def style(ax, *, xlabel="", ylabel="", title=""):
    ax.set_facecolor(PANEL)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, loc="left", pad=10)


# Cramped axes need something shorter than LABELS. Only the part that actually
# distinguishes the losses survives -- a row of "Encoder (ZScoreGlobal ...)"
# labels overlaps and tells the reader nothing.
SHORT_LABELS = {
    "encoder-minmax": "MinMax",
    "encoder-zscore": "ZScore",
    "encoder-zscore-2": "ZScore (repl.)",
    "encoder-zscore-mask": "ZScore + mask",
    "encoder-act-elu": "act ELU",
    "encoder-act-gelu": "act GELU",
    "encoder-act-relu": "act ReLU",
    "encoder-beta-0.00": "b 0.00",
    "encoder-beta-0.05": "b 0.05",
    "encoder-beta-0.10": "b 0.10",
    "encoder-beta-0.15": "b 0.15",
    "mse-minmax": "MSE MinMax",
    "mse-zscore": "MSE ZScore",
    "jaxley-paper": "Jaxley stats",
    "spike-w1": "spike W1",
    "spike-w1-multi": "spike W1 multi",
}


def loss_label(name: str) -> str:
    return LABELS.get(name, name)


def loss_short(name: str) -> str:
    return SHORT_LABELS.get(name, name)


def loss_color(name: str) -> str:
    return COLORS.get(name, MUTED)


def opt_label(name: str) -> str:
    return OPT_LABELS.get(name, name)


def opt_short(name: str) -> str:
    return OPT_SHORT.get(name, name)


def opt_color(name: str) -> str:
    return OPT_COLORS.get(name, MUTED)


def beta_ramp(n: int) -> list[str]:
    """`n` colours off :data:`BETA_RAMP`, light to dark.

    Used by the beta camp so its top rung -- which is `encoder-zscore-2` under
    another name -- takes the ramp's darkest step instead of its categorical
    masking-camp colour.
    """
    if n <= len(BETA_RAMP):
        # Keep the darkest end pinned: the ordering is what the ramp encodes.
        return list(BETA_RAMP[len(BETA_RAMP) - n:])
    import matplotlib.colors as mcolors

    cmap = mcolors.LinearSegmentedColormap.from_list("beta", BETA_RAMP)
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def save(fig, path, obj=None) -> None:
    """Write the figure and, beside it, the numbers it draws.

    A PDF in the report is otherwise only re-derivable by re-running the study,
    and the statistics annotated onto these figures exist nowhere else. Same
    habit as ``plot_beta_mse_distance.py``, factored out because four scripts
    now need it.
    """
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    if obj is not None:
        path.with_suffix(".json").write_text(
            json.dumps(obj, indent=1, sort_keys=True, default=_jsonable)
        )
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------


def corr(x, y) -> dict:
    """Spearman and Pearson with the degenerate cases handled.

    Fewer than 3 usable pairs, or either side constant, gives NaN rather than a
    misleading number or an exception.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"spearman": float("nan"), "pearson": float("nan"),
                "p_spearman": float("nan"), "p_pearson": float("nan"),
                "n": int(len(x))}
    sr, pr = spearmanr(x, y), pearsonr(x, y)
    return {"spearman": float(sr.statistic), "p_spearman": float(sr.pvalue),
            "pearson": float(pr.statistic), "p_pearson": float(pr.pvalue),
            "n": int(len(x))}


def rank_within_group(pairs) -> tuple:
    """Rank-normalize (x, y) inside each group, then pool.

    ``pairs`` maps a group key to ``(xs, ys)``.

    Pooling raw across parameter sets is confounded: anything that moves both the
    descriptor and the optimization difficulty shows up as correlation that is
    really "harder problems are harder". Ranking inside each set first removes
    the between-set component and leaves the within-set effect.

    A group where either side is constant carries no within-group information and
    is dropped rather than contributing a block of tied ranks.
    """
    xs, ys = [], []
    for x, y in pairs.values():
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        n = int(ok.sum())
        if n < 3 or np.all(x[ok] == x[ok][0]) or np.all(y[ok] == y[ok][0]):
            continue
        xs.append(rankdata(x[ok]) / n)
        ys.append(rankdata(y[ok]) / n)
    if not xs:
        return np.asarray([]), np.asarray([])
    return np.concatenate(xs), np.concatenate(ys)


def fisher_ci(r, n, level=1.96):
    """Fisher-z confidence band for a Pearson r. Returns (lo, hi)."""
    r, n = float(r), int(n)
    if not np.isfinite(r) or n < 4 or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - level * se)), float(np.tanh(z + level * se))


def stars(p, n_cells=1) -> str:
    """`*` for p < 0.05, `**` for a Bonferroni-corrected hit."""
    if not np.isfinite(p):
        return ""
    if p < 0.05 / max(n_cells, 1):
        return "**"
    if p < 0.05:
        return "*"
    return ""
