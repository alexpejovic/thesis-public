"""The eight trace-to-trace losses being compared, in one place.

Every loss has the same signature::

    loss(traces, ideal_traces) -> scalar float64

where both arguments are ``(n_amps, n_t)`` -- already ``compress_traces``'d and
``nan_to_num``'d by the caller (see ``sim.Sim.traces_of``). Everything else the
loss needs -- encoder weights, normalizer, sample spacing, window indices -- is
captured in the closure by a factory.

Two properties every loss here guarantees:

* it is **exactly 0** when the two arguments are equal. This is what makes
  ``f_min=0.0`` exact rather than a guess for ``optax.polyak_sgd``.
* it is jit-able, vmap-able, and differentiable in its first argument.

Why a factory, and why the datestr is static under jit
------------------------------------------------------
An encoder is selected by a datestr like ``2026_08_12_12_30_21_338929``. That
string is consumed *only* here, at Python level (a yaml read plus
``eqx.tree_deserialise_leaves``). The returned closure takes two array arguments
and has no string-typed parameter at all, so nothing string-typed *can* be
traced -- the "static under jit" requirement is satisfied structurally rather
than by remembering to pass ``static_argnames``.

That is also the point. The three call sites in this repo
(``train_comp.py``, ``scripts/datagen/generate_losses.py``,
``scripts/datagen/generate_encoder_optimization_quality.py``) each carry their
own ``if feat_loss_type == ...`` ladder behind
``@jax.jit(static_argnames=["feat_loss_type"])``, and the copies have already
drifted apart. Callers should jit the closure instead and delete the ladder.

Drift resolved here, deliberately
---------------------------------
* **Encoder loss is latent L2**, ``mean((ideal_mean - v_mean) ** 2)``, following
  ``generate_losses.py``. ``train_comp.py`` uses latent *L1*
  (``mean(sum(abs(...), axis=1))``); every landscape and optimization number
  published so far came from the L2 form, so that is the one kept. Switching
  ``train_comp.py`` over will change its gradient magnitudes by roughly
  ``latent_dim``, so its step size may need re-tuning.
* **The bounds terms ``b1``/``b2`` are not included.** They were normalizer
  repair for ``MinMax``/``ZScore``, and are identically zero for
  ``ZScoreGlobal``/``MinMaxHard`` anyway. So these losses reproduce the legacy
  ``feat_losses.npz`` and ``mses.npz`` arrays, never ``losses.npz`` (which was
  ``feat + b1 + b2``).
* **Masking is not applied**, even for the encoder trained with
  ``masking: true``. Masking was a training-time augmentation; the encoder is
  fed whole traces at loss time. This is intentional -- do not "fix" it.
* **Inputs must be finite.** The legacy code ``nan_to_num``s the candidate
  traces but not the ideal ones, and ``MinMax``'s internal nan->0 masks that
  under exactly one normalizer. Sanitize once, in the simulator.
"""

from functools import lru_cache
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from helper import (
    ChannelType,
    get_datagen_config,
    get_trace_normal_and_denormal,
)
from sim import dt_eff, trace_len
from spike_metrics import spike_w1, spike_w1_multi
from train import get_stim_window
from vae.conv_vae import TimeVAEBase
from vae.vae_helper import get_vae_config, load_encoder

LossFn = Callable[[Array, Array], Array]

# Bump on ANY change to a loss definition below. Cached sweeps record this, so a
# changed definition invalidates them instead of silently mixing old and new
# numbers in one correlation.
#
# Adding a NEW name is not such a change: it leaves every existing definition
# byte-identical, so the rows already on disk stay valid and a bump would only
# invalidate them for nothing. Bump when an existing entry's meaning moves.
LOSS_SET_VERSION = 2

# The trained encoders. Single source of truth -- param_groups.py and the
# plotting scripts import these rather than restating the datestrs.
#
# `zscore` and `zscore-2` have byte-identical config.yaml files -- same seed
# 5678, same ZScoreGlobal, masking off -- but different weights (val_loss 0.1328
# vs 0.1317), because training is not deterministic given the config seed. That
# makes them a same-config REPLICATE pair, and the gap between them is the
# encoder-training noise floor that the `zscore-mask` vs `zscore-2` masking
# contrast has to clear to mean anything. Do not "deduplicate" them.
ENCODERS = {
    "minmax": "2026_08_12_12_30_21_338929",       # normalizer MinMax
    "zscore": "2026_08_12_15_20_26_225838",       # normalizer ZScoreGlobal
    "zscore-mask": "2026_08_12_15_20_52_343885",  # ZScoreGlobal, trained with masking
    "zscore-2": "2026_08_17_14_02_29_129424",     # ZScoreGlobal, replicate of `zscore`
    # Activation sweep, 2026-08-28. Byte-identical config.yaml files except
    # `activation`; the activation is read back out of each config by
    # vae_helper.load_encoder, so nothing here has to know about it.
    #
    # `act-gelu` shares its whole config with `zscore` and `zscore-2` (gelu is
    # the configs/vae.yaml default), so it is a THIRD draw from the same
    # replicate family -- val_loss 0.1396 against their 0.1328 / 0.1317. That is
    # the noise floor an elu-vs-relu gap has to clear, which is why it is scored
    # alongside them rather than dropped as a duplicate.
    "act-elu": "2026_08_28_12_15_36_586159",      # ZScoreGlobal, activation elu
    "act-gelu": "2026_08_28_12_16_06_530703",     # ZScoreGlobal, activation gelu
    "act-relu": "2026_08_28_12_16_37_567835",     # ZScoreGlobal, activation relu
    # The KL-weight (beta) ladder, at training seed 5678 -- the rungs of
    # configs/beta_mse.yaml, pinned in data/beta_mse_distance/encoder_manifest.json.
    # One training seed only: the region score costs ~300 GPU-s per (loss, set)
    # cell over 56 sets, so all four ladder seeds would be four studies, not one.
    #
    # THE beta = 0.2 RUNG IS NOT LISTED HERE. Its seed-5678 encoder is
    # 2026_08_17_14_02_29_129424 -- literally `zscore-2` above, the unmasked arm
    # of the masking contrast. A second name for the same datestr would draw the
    # same encoder twice in one figure and inflate every pooled statistic, so the
    # beta camp lists `encoder-zscore-2` as its top rung and relabels it through
    # configs/region_scores.yaml. This identity is invisible from the datestrs,
    # which is why it is written down here.
    "beta-0.00": "2026_08_17_14_10_26_874636",    # ZScoreGlobal, beta 0.00
    "beta-0.05": "2026_08_17_14_10_34_201126",    # ZScoreGlobal, beta 0.05
    "beta-0.10": "2026_08_17_14_10_34_201133",    # ZScoreGlobal, beta 0.10
    "beta-0.15": "2026_08_17_14_09_03_870431",    # ZScoreGlobal, beta 0.15
}

# Standardization from the Jaxley paper: "we divided the mean voltages by 8.0
# and standard deviations by 4.0". Traces are raw mV, so these apply directly.
JAXLEY_MEAN_SCALE = 8.0
JAXLEY_STD_SCALE = 4.0

# spec = (kind, argument). The argument is an encoder datestr for the kinds that
# need one, a variant name for `jaxley_paper`, and "" otherwise.
LossSpec = tuple[str, str]

LOSS_SPECS: dict[str, LossSpec] = {
    "encoder-minmax": ("encoder_loss", ENCODERS["minmax"]),
    "encoder-zscore": ("encoder_loss", ENCODERS["zscore"]),
    "encoder-zscore-mask": ("encoder_loss", ENCODERS["zscore-mask"]),
    "encoder-zscore-2": ("encoder_loss", ENCODERS["zscore-2"]),
    "encoder-act-elu": ("encoder_loss", ENCODERS["act-elu"]),
    "encoder-act-gelu": ("encoder_loss", ENCODERS["act-gelu"]),
    "encoder-act-relu": ("encoder_loss", ENCODERS["act-relu"]),
    # beta = 0.20 is `encoder-zscore-2`; see the note in ENCODERS.
    "encoder-beta-0.00": ("encoder_loss", ENCODERS["beta-0.00"]),
    "encoder-beta-0.05": ("encoder_loss", ENCODERS["beta-0.05"]),
    "encoder-beta-0.10": ("encoder_loss", ENCODERS["beta-0.10"]),
    "encoder-beta-0.15": ("encoder_loss", ENCODERS["beta-0.15"]),
    "mse-minmax": ("mse", ENCODERS["minmax"]),
    "mse-zscore": ("mse", ENCODERS["zscore"]),
    "jaxley-paper": ("jaxley_paper", "stim"),
    "spike-w1": ("spike_w1", ""),
    "spike-w1-multi": ("spike_w1_multi", ""),
}

LOSS_NAMES = tuple(LOSS_SPECS)


# ---------------------------------------------------------------------------
# cached loaders -- one encoder / normalizer per datestr per process
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _normalizer_for_datestr(datestr: str, channel_type: str):
    """The normalizer that encoder was trained with, from its own config.yaml.

    The normalizer belongs to the checkpoint, not the caller -- which is why the
    two MSE losses are keyed by an encoder datestr even though they never touch
    the weights. `mse-minmax` and `mse-zscore` are genuinely different losses:
    MinMax rescales each trace by its own min/max (so it is invariant to a
    per-amplitude affine change), while ZScoreGlobal applies two dataset-wide
    constants (so it is a fixed affine rescaling of raw-mV MSE).
    """
    ntype = get_vae_config(channel_type, TimeVAEBase, datestr)["normalizer_type"]
    normalizer, _ = get_trace_normal_and_denormal(
        ntype, channel_type, get_datagen_config()
    )
    return jax.vmap(normalizer)


def _check_checkpoint(channel_type: str, datestr: str, epoch: int | None) -> None:
    """Fail with a readable message when the requested checkpoint is missing.

    `load_encoder` would otherwise raise a bare FileNotFoundError from inside
    `eqx.tree_deserialise_leaves`, which in an array job means N identical
    opaque tracebacks and no indication of which encoder or epoch was wanted.
    A run still in progress has no `final-model.eqx` at all, so this is the
    likely failure, not a hypothetical one.
    """
    root = Path(f"./data/models/vae/{channel_type}/{TimeVAEBase.__name__}/{datestr}")
    want = root / "final-model.eqx" if epoch is None else (
        root / f"models/model{epoch:07d}.eqx"
    )
    if want.is_file():
        return
    models = root / "models"
    have = sorted(models.glob("model*.eqx")) if models.is_dir() else []
    latest = have[-1].name if have else "none"
    raise FileNotFoundError(
        f"encoder {datestr!r} has no checkpoint for epoch={epoch!r}: {want} is "
        f"missing ({len(have)} checkpoints present, highest {latest}). "
        "A training run still in progress has no final-model.eqx; pass an "
        "explicit --encoder_epoch to score it at a checkpoint."
    )


@lru_cache(maxsize=None)
def _encoder_for_datestr(
    datestr: str, channel_type: str, n_t: int, epoch: int | None = None
):
    # `epoch` is part of the cache key on purpose: without it, a process that
    # scores the same encoder at two epochs would silently reuse the first
    # weights for both.
    _check_checkpoint(channel_type, datestr, epoch)
    encoder = load_encoder(TimeVAEBase, channel_type, n_t, datestr, epoch=epoch)
    return jax.vmap(encoder)


# ---------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------


def _make_encoder_loss(
    datestr: str, channel_type: str, config: dict, epoch: int | None = None
) -> LossFn:
    """Squared distance between the two traces' latent means."""
    normalize = _normalizer_for_datestr(datestr, channel_type)
    encode = _encoder_for_datestr(datestr, channel_type, trace_len(config), epoch)

    def loss(traces: Array, ideal_traces: Array) -> Array:
        # The encoder's weights are float32 while the simulation runs in x64, so
        # the cast is required rather than cosmetic.
        mean, _ = encode(normalize(traces).astype(jnp.float32))
        ideal_mean, _ = encode(normalize(ideal_traces).astype(jnp.float32))
        return jnp.mean((ideal_mean - mean) ** 2).astype(jnp.float64)

    return loss


def _make_mse_loss(
    datestr: str, channel_type: str, config: dict, _epoch: int | None = None
) -> LossFn:
    """Mean squared error between normalized traces. No encoder weights used."""
    normalize = _normalizer_for_datestr(datestr, channel_type)

    def loss(traces: Array, ideal_traces: Array) -> Array:
        return jnp.mean((normalize(ideal_traces) - normalize(traces)) ** 2)

    return loss


def _make_jaxley_paper_loss(
    variant: str, channel_type: str, config: dict
) -> LossFn:
    """The Jaxley-paper summary-statistic loss.

    From the paper: "We split the voltage trace into two windows and used the
    mean and standard deviation of these two windows as summary statistics. This
    led to a total of four summary statistics. To standardize the data, we
    divided the mean voltages by 8.0 and standard deviations by 4.0. We used a
    mean absolute error loss on the standardized summary statistics."

    ``variant`` selects what gets split in half:

    ``"stim"`` (default)
        The stimulus window only -- compressed samples [300, 650) and
        [650, 1000) for the standard config, i.e. 35 ms each. This is the
        early-vs-late contrast the two windows exist to capture (spike-rate
        adaptation), and it is what the paper's split amounts to when the
        recording is roughly coextensive with the stimulus.

    ``"full"``
        The whole trace in half. Our traces carry ~30 ms of pre-stimulus rest
        and ~30 ms of post-stimulus rest (t_filter=10, delay=40, duration=70,
        t_max=150), so half #1 is rest + early stimulus and half #2 is late
        stimulus + rest. The two windows stop being comparable and each mean and
        std is diluted by a flat DC segment. Kept as an ablation.
    """
    n_t = trace_len(config)
    if variant in ("", "stim"):
        lo, hi = get_stim_window(config, n_t)
    elif variant == "full":
        lo, hi = 0, n_t
    else:
        raise ValueError(
            f"unknown jaxley_paper variant {variant!r}; want 'stim' or 'full'"
        )
    mid = lo + (hi - lo) // 2
    windows = ((lo, mid), (mid, hi))

    def summary(x: Array) -> Array:
        """(n_amps, n_t) -> (n_amps, 4) standardized summary statistics."""
        parts = []
        for start, end in windows:
            seg = x[..., start:end]
            parts.append(jnp.mean(seg, axis=-1) / JAXLEY_MEAN_SCALE)
            # Population std (ddof=0), matching jnp.std's default.
            parts.append(jnp.std(seg, axis=-1) / JAXLEY_STD_SCALE)
        return jnp.stack(parts, axis=-1)

    def loss(traces: Array, ideal_traces: Array) -> Array:
        return jnp.mean(jnp.abs(summary(traces) - summary(ideal_traces)))

    return loss


def _make_spike_w1_loss(
    _arg: str, channel_type: str, config: dict, _epoch: int | None = None
) -> LossFn:
    step = dt_eff(config)

    def loss(traces: Array, ideal_traces: Array) -> Array:
        # Argument order matters: spike_w1's second argument is the reference.
        return spike_w1(traces, ideal_traces, step)

    return loss


def _make_spike_w1_multi_loss(
    _arg: str, channel_type: str, config: dict, _epoch: int | None = None
) -> LossFn:
    step = dt_eff(config)

    def loss(traces: Array, ideal_traces: Array) -> Array:
        # Order is load-bearing here: each channel is normalized by the scale of
        # the *reference*, so the distance is not symmetric in its arguments.
        #
        # The `with_fi` branch inside tests `voltage.ndim >= 2 and
        # voltage.shape[-2] > 1` in Python. Under an outer vmap the logical
        # shape is still (n_amps, n_t), so the branch stays static -- it looks
        # data-dependent but is resolved at trace time.
        return spike_w1_multi(traces, ideal_traces, step)

    return loss


_FACTORIES: dict[str, Callable[[str, str, dict, int | None], LossFn]] = {
    "encoder_loss": _make_encoder_loss,
    "mse": _make_mse_loss,
    "jaxley_paper": _make_jaxley_paper_loss,
    "spike_w1": _make_spike_w1_loss,
    "spike_w1_multi": _make_spike_w1_multi_loss,
}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def get_loss_fn(
    spec: LossSpec,
    channel_type: ChannelType = "Pospischil",
    config: dict | None = None,
    *,
    epoch: int | None = None,
) -> LossFn:
    """Build one loss from its ``(kind, argument)`` spec.

    ``epoch`` selects an encoder checkpoint (``None`` = ``final-model.eqx``). It
    is accepted and ignored by the kinds that use no encoder weights, so a
    caller can pass it unconditionally and record one value per row.
    """
    kind, arg = spec
    if kind not in _FACTORIES:
        raise ValueError(f"unknown loss kind {kind!r}; known: {sorted(_FACTORIES)}")
    config = config if config is not None else get_datagen_config()
    return _FACTORIES[kind](arg, channel_type, config, epoch)


def spec_for_name(name: str) -> LossSpec:
    if name not in LOSS_SPECS:
        raise ValueError(f"unknown loss {name!r}; known: {list(LOSS_SPECS)}")
    return LOSS_SPECS[name]


def get_loss_fns(
    names=LOSS_NAMES,
    channel_type: ChannelType = "Pospischil",
    config: dict | None = None,
) -> dict[str, LossFn]:
    """Build several losses at once, sharing cached encoders and normalizers.

    This is what a grid sweep uses: it evaluates every loss on one simulated
    grid, so the three encoders and the ZScoreGlobal dataset statistics are
    loaded once between them.
    """
    config = config if config is not None else get_datagen_config()
    return {
        name: get_loss_fn(LOSS_SPECS[name], channel_type, config) for name in names
    }
