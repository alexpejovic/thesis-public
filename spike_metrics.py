"""Differentiable spike-train distances between voltage traces.

The Allen ephys features (``first_isi``, ``median_isi``, ...) carry the right
information about a trace but encode it badly for gradient descent: spike times
come out as integer sample indices, so the features are quantised onto the
sample grid, and the spike count is an integer, so they jump whenever a spike
appears or disappears.  Both make the loss a staircase.

The functions here encode the same information as a *cumulative spike count*
``C(t)``, which fixes both problems:

* a spike appearing changes ``C`` by 1 only after its own time, so the distance
  moves continuously as the spike emerges instead of jumping;
* shifting a spike by ``dt`` changes the L1 distance between two ``C`` curves by
  exactly ``dt``, however far apart they are, so the loss never saturates the
  way MSE does once spikes stop overlapping.

The L1 distance between cumulative count curves is the 1-Wasserstein distance
between the two spike trains.
"""

from typing import Literal, Sequence

import jax
import jax.numpy as jnp
from jax import Array

Rectifier = Literal["relu", "abs"]

RECTIFIERS = {"relu": jax.nn.relu, "abs": jnp.abs}

# Half the spike height above rest: well inside every spike, well above every
# subthreshold fluctuation.  Verified stable for V_TH in [-40, 0] and BETA in
# [1, 10] -- see scripts/research/verify.py.
DEFAULT_V_TH = -20.0
DEFAULT_BETA = 4.0

# Thresholds for the multi-channel variant.  Crossings at different heights
# respond to different parts of the spike waveform, which adds the rank that a
# single cumulative-count channel lacks.  Kept below ~+15 mV: above that the
# crossing becomes intermittent (not every spike reaches it) and the channel
# goes back to being a staircase -- see scripts/research/ablate_channels.py.
DEFAULT_THRESHOLDS = (-40.0, -20.0, 0.0, 15.0)


def soft_spike_train(
    voltage: Array,
    dt: float,
    v_th: float = DEFAULT_V_TH,
    beta: float = DEFAULT_BETA,
    rectifier: Rectifier = "relu",
) -> Array:
    """Differentiable spike-event density along the last axis.

    ``sigmoid((v - v_th) / beta)`` is a soft indicator of "currently spiking";
    its rectified time derivative is a bump at each edge that integrates to one,
    so the whole thing integrates to the spike count.  ``beta`` sets how sharply
    a crossing is resolved, in mV.

    The rectifier has to do two things, and both matter (measured in
    scripts/research/ablate_rectifier.py):

    * be non-negative, so the cumulative sum accumulates.  Without it the sum
      telescopes back to ``u`` itself, which is just "currently above
      threshold" and carries no count at all -- 16% basin of attraction against
      100%.
    * be exactly zero away from the edges, so nothing leaks.  ``softplus`` looks
      like the obvious smooth substitute and is a trap: ``softplus(0) ~ 0.69``
      means every sample contributes, and the count is buried under a linear
      ramp (``C(end)`` of 93 where the true count is 4; 3% basin).

    ``relu`` (default) keeps only rising edges, so ``C(end)`` is exactly the
    spike count.

    ``abs`` keeps both edges and scored identically on every landscape tested
    (1.00 local minima, 100% basin over 8 sweeps).  It is worth having because
    the falling edge carries its own timing, so the channel also responds to
    spike *width* and to downstroke kinetics -- extra information for free,
    which is worth something given how close the joint parameter fit is to being
    rank limited.  Note that it double counts, so ``C(end)`` is twice the spike
    count and no longer directly interpretable.

    Crucially ``abs`` stays smooth because it is still an *event-timing*
    quantity -- zero except at the edges.  That is what separates it from
    :func:`cumulative_time_above`, which integrates the whole waveform and is
    correspondingly rough (3-38% basin).
    """
    u = jax.nn.sigmoid((voltage - v_th) / beta)
    du = jnp.diff(u, axis=-1, prepend=u[..., :1]) / dt
    return RECTIFIERS[rectifier](du)


def cumulative_spike_count(
    voltage: Array,
    dt: float,
    v_th: float = DEFAULT_V_TH,
    beta: float = DEFAULT_BETA,
    rectifier: Rectifier = "relu",
) -> Array:
    """``C(t)``: the (soft) number of spikes that have happened by time ``t``.

    With ``rectifier="abs"`` this counts both edges, so it is twice the spike
    count -- a monotone rescaling, which leaves the Wasserstein distance below
    well defined.
    """
    return jnp.cumsum(
        soft_spike_train(voltage, dt, v_th, beta, rectifier), axis=-1
    ) * dt


def cumulative_time_above(
    voltage: Array,
    dt: float,
    v_th: float = DEFAULT_V_TH,
    beta: float = DEFAULT_BETA,
) -> Array:
    """Cumulative time spent above ``v_th``.

    Sensitive to spike *width* rather than spike count, so it separates traces
    that :func:`cumulative_spike_count` alone cannot tell apart.
    """
    u = jax.nn.sigmoid((voltage - v_th) / beta)
    return jnp.cumsum(u, axis=-1) * dt


def spike_w1(
    voltage: Array,
    ref_voltage: Array,
    dt: float,
    v_th: float = DEFAULT_V_TH,
    beta: float = DEFAULT_BETA,
    rectifier: Rectifier = "relu",
) -> Array:
    """1-Wasserstein distance between the spike trains of two trace sets.

    ``voltage`` and ``ref_voltage`` broadcast against each other; the last axis
    is time.  Returns a scalar (the mean over every leading axis).

    ``rectifier="abs"`` additionally makes the distance sensitive to spike width
    and downstroke timing; see :func:`soft_spike_train`.
    """
    c = cumulative_spike_count(voltage, dt, v_th, beta, rectifier)
    c_ref = cumulative_spike_count(ref_voltage, dt, v_th, beta, rectifier)
    return jnp.mean(jnp.abs(c - c_ref))


def spike_w1_multi(
    voltage: Array,
    ref_voltage: Array,
    dt: float,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    beta: float = DEFAULT_BETA,
    rectifier: Rectifier = "relu",
    *,
    with_fi: bool = True,
) -> Array:
    """Multi-channel spike-Wasserstein distance.

    A single cumulative-count channel is close to one dimensional, which is not
    enough to identify eight parameters at once.  This stacks channels that are
    each smooth for the same reason but respond to different things:

    * up-crossing counts at several thresholds, which pick out different points
      on the rising edge and so respond differently to spike shape;
    * the cumulative f-I curve across stimulus amplitudes (rate vs drive), which
      needs a leading amplitude axis and is skipped for a single trace.

    Only *event-timing* channels are included.  Amplitude channels such as
    :func:`cumulative_time_above` were measured to be far rougher than the count
    channels (3-38% basin of attraction versus 94-99%) because they track the
    area under the spike rather than when it happened, which brings back exactly
    the saturation MSE suffers from.

    Each channel is normalised by the scale of the reference so no single one
    dominates.
    """
    total = 0.0
    n = 0

    for v_th in thresholds:
        c = cumulative_spike_count(voltage, dt, v_th, beta, rectifier)
        c_ref = cumulative_spike_count(ref_voltage, dt, v_th, beta, rectifier)
        scale = jnp.mean(jnp.abs(c_ref)) + 1e-8
        total = total + jnp.mean(jnp.abs(c - c_ref)) / scale
        n += 1

    if with_fi and voltage.ndim >= 2 and voltage.shape[-2] > 1:
        # counts per amplitude, then cumulated over amplitude: a smooth encoding
        # of the f-I curve, by the same argument that makes C(t) smooth in time.
        counts = jnp.sum(
            soft_spike_train(voltage, dt, DEFAULT_V_TH, beta, rectifier), -1
        ) * dt
        counts_ref = jnp.sum(
            soft_spike_train(ref_voltage, dt, DEFAULT_V_TH, beta, rectifier), -1
        ) * dt
        fi = jnp.cumsum(counts, axis=-1)
        fi_ref = jnp.cumsum(counts_ref, axis=-1)
        scale = jnp.mean(jnp.abs(fi_ref)) + 1e-8
        total = total + jnp.mean(jnp.abs(fi - fi_ref)) / scale
        n += 1

    return total / n
