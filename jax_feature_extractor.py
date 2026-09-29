"""JAX re-implementation of the Allen ``EphysSweepFeatureExtractor`` sweep features.

ALL the code and comments from this file are purely generated using Claude Code,
using model Opus 5.

Motivation
----------
``featuregrad/jaxley_ephys/allen_ephys/ephys_extractor.py`` implements the Allen
Institute spike-feature pipeline with numpy/scipy/pandas.  Every stage of that
pipeline uses *data-dependent shapes* (``np.flatnonzero``, boolean indexing,
``np.delete``, python ``for`` loops over the detected spikes, a ``DataFrame``),
which makes it impossible to put under ``jax.vmap``/``jax.jit``.

This module reproduces the seven sweep-level features that
``jax-allen/make_features_numpy.py`` consumes::

    adapt, avg_rate, first_isi, isi_cv, latency, mean_isi, median_isi

with pure ``jax.numpy`` internals, so the whole extraction can be
``jit``-compiled and ``vmap``-ed over a batch of voltage traces.

How the shape problem is solved
-------------------------------
1.  Everything that only depends on the *time base* (``t``) and on the static
    detection parameters is resolved once, eagerly, in ``__init__``: the window
    indices, ``has_fixed_dt``, the Bessel filter coefficients, the ``dt`` array.
    ``t`` is therefore a concrete array (numpy or a concrete ``jnp`` array), not
    a traced one -- which is exactly how the numpy script uses it (one shared
    time grid for all traces).
2.  The per-spike arrays get a *fixed* capacity ``max_spikes`` plus a boolean
    ``valid`` mask.  Valid entries are always kept compacted at the front and in
    ascending order, so the numpy semantics of "neighbouring spike" (``x[1:]``
    vs ``x[:-1]``) carry over unchanged.  Dropping spikes becomes masking +
    re-compacting instead of ``np.delete``.
3.  The three kinds of data-dependent scans in the numpy code are replaced by
    O(N) primitives:

    * ``np.any(dvdt[a:b] < 0)``               -> prefix sum of ``dvdt < 0``
    * "last index <= i where ``dvdt <= target``" -> ``lax.cummax``
    * ``np.argmax(v[a:b])``                   -> masked ``argmax`` over an
      ``(max_spikes, n_samples)`` grid (works for overlapping ranges, which the
      ``check_thresholds_and_peaks`` re-detection branch can produce).

Numerical parity
----------------
Verified **bit-exact** (``==``, not ``allclose``) against
``EphysSweepFeatureExtractor`` for all seven features on all 400 traces in
``jax-allen/data`` -- see ``test_jax_vs_numpy.py``.  Getting there needed four
things beyond a straight transcription:

* dtypes are reproduced rather than promoted.  ``1e-3 * np.diff(v) / np.diff(t)``
  does not upcast under NEP 50, so ``dvdt`` stays in the dtype of ``v``/``t``
  (float32 for these traces).
* ``ft.norm_diff`` (behind ``adapt``) does ``a.astype(float)`` == float64, so
  x64 has to be on::

      import jax
      jax.config.update("jax_enable_x64", True)

  Note this also changes what ``jnp.arange`` produces, so a time base built with
  ``jnp.arange`` must pin ``dtype=jnp.float32`` to stay identical to the numpy
  script's input.  Without x64, ``adapt`` is computed in float32 and differs in
  the last 1-2 significant digits.
* numpy sums with *pairwise summation*, not left-to-right; see
  :func:`_numpy_sum`.  This matters beyond cosmetics: a 1-ULP difference in
  ``mean(dvdt[upstrokes])`` can flip a ``<=`` in threshold refinement and change
  the detected spike set.
* two XLA rewrites have to be blocked with an optimization barrier; see
  :func:`_round_trip`.

The Bessel ``filtfilt`` branch (see below) agrees with scipy to ~1e-12 relative
rather than bit-exactly, because scipy's ``lfilter`` accumulates the recursion
slightly differently than the equivalent ``lax.scan``.  That branch is *not*
taken for the shipped traces (their float32 time base makes
``ft.has_fixed_dt`` False), but on a uniform time base a threshold comparison
could in principle land differently.

Scope / assumptions
-------------------
* Only the seven sweep-level features above are implemented.  The per-spike
  ``_spikes_df`` columns (troughs, widths, ADP, ``isi_type``, ...) are *not*:
  they are irrelevant to those features (the sweep-level block only reads
  ``threshold_index``) and ``isi_type`` is a string column that has no sensible
  vmap-able representation.  ``process_spikes`` does return the threshold, peak
  and upstroke indexes plus the ``clipped`` flags if you want to build more.
* The stimulus current ``i`` is not needed: the numpy extractor only uses it to
  fill ``*_i`` columns of the spike DataFrame.
* ``t`` must be strictly increasing with no zero ``dt`` (checked in
  ``__init__``), and ``v`` must not contain NaNs -- the numpy code drops NaN
  ``dvdt`` entries, which would be a data-dependent shape.
* If a trace contains more than ``max_spikes`` dV/dt crossings the extra ones
  are dropped; ``process_spikes`` reports this via the ``overflow`` flag.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal as signal

__all__ = [
    "JaxEphysSweepFeatureExtractor",
    "SpikeIndexes",
    "make_feature_extractor",
]


# --------------------------------------------------------------------------- #
# small shape-stable primitives
# --------------------------------------------------------------------------- #
def _compact(mask: jax.Array, arrays: tuple[jax.Array, ...]):
    """Move the ``True`` entries of ``mask`` to the front, order preserved.

    Replaces numpy's ``x = x[boolean_mask]`` / ``np.delete(x, idx)``: the arrays
    keep their static length, the surviving entries are packed at the front and
    the returned mask says how many there are.
    """
    order = jnp.argsort(jnp.where(mask, 0, 1))  # stable -> preserves order
    return mask[order], tuple(a[order] for a in arrays)


def _prefix_count(flags: jax.Array) -> jax.Array:
    """``out[k] = flags[:k].sum()``, length ``len(flags) + 1``."""
    return jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(flags.astype(jnp.int32))]
    )


def _any_in_range(prefix: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    """``np.any(flags[lo:hi])`` from a prefix count (empty range -> False)."""
    n = prefix.shape[0] - 1
    lo_c = jnp.clip(lo, 0, n)
    hi_c = jnp.clip(hi, 0, n)
    return (hi_c > lo_c) & ((prefix[hi_c] - prefix[lo_c]) > 0)


def _last_true_upto(flags: jax.Array) -> jax.Array:
    """``out[i] = max{j <= i : flags[j]}`` or ``-1``."""
    pos = jnp.arange(flags.shape[0], dtype=jnp.int32)
    return jax.lax.cummax(jnp.where(flags, pos, jnp.int32(-1)))


def _range_argextremum(
    values: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    valid: jax.Array,
    maximize: bool,
) -> jax.Array:
    """``argmax``/``argmin`` of ``values[lo_j:hi_j]`` (absolute index) per ``j``.

    Mirrors ``np.argmax(values[lo:hi]) + lo``: ties go to the lowest index.
    Empty/invalid ranges return ``lo`` (numpy would raise on an empty slice;
    the numpy pipeline never hits that case for well-formed traces).
    Ranges may overlap.
    """
    n = values.shape[0]
    pos = jnp.arange(n, dtype=jnp.int32)
    in_range = (
        (pos[None, :] >= lo[:, None]) & (pos[None, :] < hi[:, None]) & valid[:, None]
    )
    fill = -jnp.inf if maximize else jnp.inf
    masked = jnp.where(in_range, values[None, :], jnp.asarray(fill, values.dtype))
    arg = jnp.argmax(masked, axis=1) if maximize else jnp.argmin(masked, axis=1)
    return jnp.where(jnp.any(in_range, axis=1), arg.astype(jnp.int32), lo)


def _shift_left(arr: jax.Array, fill) -> jax.Array:
    """``out[j] = arr[j + 1]``, last entry filled with ``fill``."""
    return jnp.concatenate([arr[1:], jnp.asarray([fill], arr.dtype)])


def _shift_right(arr: jax.Array, fill) -> jax.Array:
    """``out[j] = arr[j - 1]``, first entry filled with ``fill``."""
    return jnp.concatenate([jnp.asarray([fill], arr.dtype), arr[:-1]])


def _take(arr: jax.Array, idx: jax.Array) -> jax.Array:
    """Gather with indices clamped into range (invalid slots are masked later)."""
    return arr[jnp.clip(idx, 0, arr.shape[0] - 1)]


def _round_trip(x: jax.Array) -> jax.Array:
    """Force ``x`` to be materialised at its own dtype before it is used again.

    Two XLA rewrites cost exactly 1 ULP against numpy without this:

    * a compile-time-constant divisor makes the algebraic simplifier turn
      ``x / c`` into ``x * (1 / c)`` (used for ``dvdt``'s ``dt`` and for
      ``avg_rate``'s ``end - start``);
    * fusing two float32 reductions straight into a quotient
      (``std / mean`` for ``isi_cv``) keeps more precision than numpy, which
      rounds both operands to float32 first.

    An optimization barrier is the narrowest way to pin both down.
    """
    return jax.lax.optimization_barrier(x)


def _numpy_sum(values: jax.Array, n: jax.Array) -> jax.Array:
    """``values[:n].sum()`` with numpy's *pairwise summation* order.

    ``values`` must be exactly zero outside ``[0, n)`` (adding 0.0 never changes
    a finite accumulator, so the padding is inert).

    numpy does not sum left-to-right: for ``n >= 8`` it keeps eight interleaved
    accumulators, combines them with a balanced tree and then folds in the
    ``n % 8`` leftovers.  Reproducing that matters because a 1-ULP difference in
    ``mean(dvdt[upstrokes])`` can flip a ``<=`` comparison during threshold
    refinement and change which samples are detected as spikes.

    Exact for ``n <= 128``; above that numpy recurses on halves and this keeps
    using the eight-accumulator form (a <=1-ULP difference).
    """
    dtype = values.dtype
    zero = jnp.zeros((), dtype)
    length = values.shape[0]

    def fold(init, xs):
        return jax.lax.scan(lambda acc, e: (acc + e, None), init, xs)[0]

    # r[j] = a[j] + a[8 + j] + a[16 + j] + ...  over the full 8-blocks only
    pad = (-length) % 8
    padded = jnp.concatenate([values, jnp.zeros((pad,), dtype)]) if pad else values
    grid = padded.reshape(-1, 8)
    n_blocks = n // 8
    rows_used = (jnp.arange(grid.shape[0]) < n_blocks)[:, None]
    r = fold(jnp.zeros((8,), dtype), jnp.where(rows_used, grid, zero))

    total = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))

    # ... then the n % 8 trailing elements, left to right
    idx = jnp.arange(length)
    tail = jnp.where((idx >= 8 * n_blocks) & (idx < n), values, zero)
    return fold(total, tail)


def _masked_sum(values: jax.Array, mask: jax.Array, count: jax.Array) -> jax.Array:
    """``values[mask].sum()`` -- numpy summation order, numpy accumulator dtype."""
    return _numpy_sum(jnp.where(mask, values, jnp.zeros((), values.dtype)), count)


def _masked_mean(values: jax.Array, mask: jax.Array, count: jax.Array) -> jax.Array:
    """``values[mask].mean()``."""
    return _masked_sum(values, mask, count) / count.astype(values.dtype)


# --------------------------------------------------------------------------- #
# ft.calculate_dvdt, planned once per (time base, cutoff)
# --------------------------------------------------------------------------- #
def _has_fixed_dt(t: np.ndarray) -> bool:
    """``ft.has_fixed_dt`` on a concrete time base."""
    dt = np.diff(t)
    return bool(np.allclose(dt, np.ones_like(dt) * dt[0]))


def _odd_ext(x: jax.Array, n: int) -> jax.Array:
    """``scipy.signal._arraytools.odd_ext`` for 1-D input."""
    if n < 1:
        return x
    left = 2.0 * x[0] - x[n:0:-1]
    right = 2.0 * x[-1] - x[-2 : -(n + 2) : -1]
    return jnp.concatenate([left, x, right])


def _lfilter(b: np.ndarray, a: np.ndarray, zi: jax.Array, x: jax.Array):
    """``scipy.signal.lfilter`` (transposed direct form II) as a ``lax.scan``.

    ``b``/``a`` are concrete, equal-length and already normalised by ``a[0]``.
    """
    b_j = jnp.asarray(b)
    a_j = jnp.asarray(a)

    def step(z, x_n):
        y_n = b_j[0] * x_n + z[0]
        shifted = jnp.concatenate([z[1:], jnp.zeros((1,), z.dtype)])
        return shifted + b_j[1:] * x_n - a_j[1:] * y_n, y_n

    z_f, y = jax.lax.scan(step, zi, x)
    return y, z_f


class _DvdtPlan:
    """Static plan reproducing ``ft.calculate_dvdt(v, t, filter)``.

    All the branching in ``calculate_dvdt`` depends on ``t`` and ``filter``
    only, so it is decided here (eagerly) and the ``__call__`` is branch-free.
    """

    def __init__(self, t: np.ndarray, filter_khz: float | None):
        t = np.asarray(t)
        if t.ndim != 1 or t.shape[0] < 2:
            raise ValueError("t must be 1-D with at least two samples")

        dt = np.diff(t)
        self.use_filter = bool(filter_khz) and _has_fixed_dt(t)

        self._b = self._a = self._zi = None
        self._padlen = 0
        if self.use_filter:
            delta_t = t[1] - t[0]
            sample_freq = 1.0 / delta_t
            # filter kHz -> Hz, then fraction of the Nyquist frequency
            filt_coeff = (filter_khz * 1e3) / (sample_freq / 2.0)
            if filt_coeff < 0 or filt_coeff >= 1:
                raise ValueError(
                    "bessel coeff ({:f}) is outside of valid range [0,1); cannot "
                    "filter sampling frequency {:.1f} kHz with cutoff frequency "
                    "{:.1f} kHz.".format(filt_coeff, sample_freq / 1e3, filter_khz)
                )
            b, a = signal.bessel(4, filt_coeff, "low")
            # scipy's filtfilt/lfilter normalise by a[0] internally.
            b, a = b / a[0], a / a[0]
            n_taps = max(len(a), len(b))
            b = np.r_[b, np.zeros(n_taps - len(b))]
            a = np.r_[a, np.zeros(n_taps - len(a))]
            self._b, self._a = b, a
            self._zi = signal.lfilter_zi(b, a)
            self._padlen = 3 * n_taps
            if t.shape[0] <= self._padlen:
                raise ValueError(
                    f"trace of length {t.shape[0]} is too short for filtfilt "
                    f"padding (padlen={self._padlen})"
                )

        # ft.calculate_dvdt drops entries with dt == 0; that only depends on t.
        valid = np.flatnonzero(dt != 0)
        self._valid_idx = jnp.asarray(valid)
        self._dt = jnp.asarray(dt[valid])
        self._full = valid.shape[0] == dt.shape[0]
        self.size = int(valid.shape[0])

    def _filtfilt(self, v: jax.Array) -> jax.Array:
        """``scipy.signal.filtfilt(b, a, v, axis=0)`` with the scipy defaults.

        padtype='odd', padlen=3*max(len(a), len(b)), method='pad'.
        """
        n = self._padlen
        # scipy builds the odd extension in the dtype of v, then lfilter
        # upcasts to result_type(b, a, ext).
        ext = _odd_ext(v, n).astype(jnp.result_type(self._b, v.dtype))
        zi = jnp.asarray(self._zi, ext.dtype)
        y, _ = _lfilter(self._b, self._a, zi * ext[0], ext)
        y_rev = y[::-1]
        y, _ = _lfilter(self._b, self._a, zi * y_rev[0], y_rev)
        y = y[::-1]
        return y[n:-n] if n > 0 else y

    def __call__(self, v: jax.Array) -> jax.Array:
        dv = jnp.diff(self._filtfilt(v) if self.use_filter else v)
        if not self._full:
            dv = dv[self._valid_idx]
        return 1e-3 * dv / _round_trip(self._dt)  # in V/s = mV/ms


# --------------------------------------------------------------------------- #
# public results container
# --------------------------------------------------------------------------- #
class SpikeIndexes(NamedTuple):
    """Fixed-capacity result of :meth:`JaxEphysSweepFeatureExtractor.process_spikes`.

    Valid entries are packed at the front of every array; ``valid[j]`` says
    whether slot ``j`` holds a spike.  These are exactly the
    ``threshold_index`` / ``peak_index`` / ``upstroke_index`` / ``clipped``
    columns the numpy extractor would put into ``_spikes_df``.
    """

    threshold: jax.Array  # int32 (max_spikes,)
    peak: jax.Array  # int32 (max_spikes,)
    upstroke: jax.Array  # int32 (max_spikes,)
    clipped: jax.Array  # bool  (max_spikes,)
    valid: jax.Array  # bool  (max_spikes,)
    n_spikes: jax.Array  # int32 scalar
    overflow: jax.Array  # bool scalar -- max_spikes was too small


# --------------------------------------------------------------------------- #
# the extractor
# --------------------------------------------------------------------------- #
class JaxEphysSweepFeatureExtractor:
    """``vmap``-able equivalent of ``EphysSweepFeatureExtractor`` sweep features.

    The time base and all detection parameters are bound at construction time;
    the resulting object is a pure function of the voltage trace::

        ext = JaxEphysSweepFeatureExtractor(time_array)
        feats = jax.jit(jax.vmap(ext.features))(voltages)   # dict of (B,) arrays

    Parameters mirror ``EphysSweepFeatureExtractor.__init__``.  ``check_thresh_frac``
    is the ``thresh_frac`` used inside ``ft.check_thresholds_and_peaks``; the
    numpy extractor calls that function without forwarding ``self.thresh_frac``,
    so it always sees the function default of 0.05.  It is exposed separately
    here to keep that (surprising) behaviour reproducible.
    """

    def __init__(
        self,
        t,
        start=None,
        end=None,
        filter=10.0,
        dv_cutoff=20.0,
        max_interval=0.005,
        min_height=2.0,
        min_peak=-30.0,
        thresh_frac=0.05,
        check_thresh_frac=0.05,
        clip_tol=1.0,
        max_spikes=32,
    ):
        t_np = np.asarray(t)
        if t_np.ndim != 1:
            raise ValueError("t must be 1-D")
        if t_np.shape[0] < 3:
            raise ValueError("t must have at least three samples")
        dt = np.diff(t_np)
        if np.any(dt <= 0):
            raise ValueError(
                "t must be strictly increasing (zero/negative dt would make the "
                "numpy implementation drop samples, which is not shape-stable)"
            )
        if max_spikes < 1:
            raise ValueError("max_spikes must be >= 1")

        self.t = jnp.asarray(t_np)
        self.n_samples = int(t_np.shape[0])
        self.max_spikes = int(max_spikes)

        self.start = start
        self.end = end
        self.filter = filter
        self.dv_cutoff = dv_cutoff
        self.max_interval = max_interval
        self.min_height = min_height
        self.min_peak = min_peak
        self.thresh_frac = thresh_frac
        self.check_thresh_frac = check_thresh_frac
        self.clip_tol = clip_tol

        def find_time_index(t_0):
            hits = np.flatnonzero(t_np >= t_0)
            if not hits.size:
                raise ValueError("Could not find given time in time vector")
            return int(hits[0])

        # ft.detect_putative_spikes: `if start is None: start = t[0]` etc.
        dps_start = t_np[0] if start is None else start
        dps_end = t_np[-1] if end is None else end
        self._win_lo = find_time_index(dps_start)
        self._win_hi = find_time_index(dps_end)  # window is [lo, hi] inclusive
        if self._win_hi <= self._win_lo:
            raise ValueError("empty detection window")

        # ft.find_peak_indexes / ft.check_thresholds_and_peaks: `if not end`.
        peak_end = t_np[-1] if not end else end
        self._peak_end_index = find_time_index(peak_end)
        self._clip_end_index = self._peak_end_index

        # ft.latency / ft.average_rate: `if start is None: start = t[0]`.
        self._lat_start = t_np[0] if start is None else start
        self._rate_start = t_np[0] if start is None else start
        self._rate_end = t_np[-1] if end is None else end
        self._rate_denom = np.asarray(self._rate_end) - np.asarray(self._rate_start)

        # _process_individual_spikes computes dvdt on the full trace, while
        # detect_putative_spikes computes its own dvdt on the window.
        self._dvdt_full = _DvdtPlan(t_np, filter)
        if self._win_lo == 0 and self._win_hi == self.n_samples - 1:
            self._dvdt_win = self._dvdt_full
        else:
            self._dvdt_win = _DvdtPlan(t_np[self._win_lo : self._win_hi + 1], filter)

    # ------------------------------------------------------------------ #
    def _detect_putative_spikes(self, v: jax.Array):
        """``ft.detect_putative_spikes`` -> (indexes, valid mask, overflow)."""
        m = self.max_spikes
        v_win = v[self._win_lo : self._win_hi + 1]
        dvdt = self._dvdt_win(v_win)

        # positive-going crossings of the dV/dt cutoff level
        above = (dvdt >= self.dv_cutoff).astype(jnp.int32)
        crossings = jnp.diff(above) == 1
        cand = jnp.flatnonzero(crossings, size=m, fill_value=-1).astype(jnp.int32)
        overflow = jnp.sum(crossings) > m
        mask = cand >= 0

        # keep spike j (j > 0) only if dV/dt dropped below zero since spike j-1
        neg_prefix = _prefix_count(dvdt < 0)
        prev = _shift_right(cand, 0)
        has_neg = _any_in_range(neg_prefix, prev, cand)
        keep = mask & ((jnp.arange(m) == 0) | has_neg)

        spikes = cand + self._win_lo
        mask, (spikes,) = _compact(keep, (spikes,))
        return jnp.where(mask, spikes, self.n_samples), mask, overflow

    def _find_peak_indexes(self, v, spikes, mask):
        """``ft.find_peak_indexes``: argmax of v between consecutive spikes."""
        next_mask = _shift_left(mask, False)
        next_spike = _shift_left(spikes, self.n_samples)
        hi = jnp.where(next_mask, next_spike, jnp.int32(self._peak_end_index))
        return _range_argextremum(v, spikes, hi, mask, maximize=True)

    def _filter_putative_spikes(self, v, dvdt, spikes, peaks, mask):
        """``ft.filter_putative_spikes``.

        Note the asymmetry in the numpy code: a failing ``diff_mask[j]`` drops
        ``peaks[j]`` but ``spikes[j + 1]``.  Both masks always keep the same
        number of entries, so compacting them independently and re-pairing by
        position reproduces it exactly.
        """
        m = self.max_spikes
        neg_prefix = _prefix_count(dvdt < 0)
        next_mask = _shift_left(mask, False)
        next_spike = _shift_left(spikes, self.n_samples)

        # diff_mask[j] = np.any(dvdt[peaks[j]:spikes[j+1]] < 0), j = 0..K-2
        pair = mask & next_mask
        diff_mask = pair & _any_in_range(neg_prefix, peaks, next_spike)

        is_last = mask & ~next_mask
        keep_peak = mask & (diff_mask | is_last)
        keep_spike = mask & ((jnp.arange(m) == 0) | _shift_right(diff_mask, False))

        mask, (spikes,) = _compact(keep_spike, (spikes,))
        _, (peaks,) = _compact(keep_peak, (peaks,))

        # absolute peak level and threshold-to-peak height
        peak_v = _take(v, peaks)
        keep = mask & (peak_v >= self.min_peak)
        keep = keep & ((peak_v - _take(v, spikes)) >= self.min_height)

        mask, (spikes, peaks) = _compact(keep, (spikes, peaks))
        spikes = jnp.where(mask, spikes, self.n_samples)
        peaks = jnp.where(mask, peaks, self.n_samples)
        return spikes, peaks, mask

    def _find_upstroke_indexes(self, dvdt, spikes, peaks, mask):
        """``ft.find_upstroke_indexes``: argmax of dV/dt between spike and peak."""
        return _range_argextremum(dvdt, spikes, peaks, mask, maximize=True)

    def _refine_threshold_indexes(self, dvdt, upstrokes, mask, count):
        """``ft.refine_threshold_indexes``.

        For each spike, walk *backwards* from its upstroke to the previous
        upstroke (exclusive) and take the last sample with
        ``dvdt <= thresh_frac * mean(dvdt[upstrokes])``; fall back to the
        previous upstroke (0 for the first spike) when there is none.  The
        backwards search is a ``cummax`` lookup.
        """
        avg_upstroke = _masked_mean(_take(dvdt, upstrokes), mask, count)
        target = avg_upstroke * self.thresh_frac
        last_below = _last_true_upto(dvdt <= target)

        lo = _shift_right(upstrokes, 0)  # np.append([0], upstroke_indexes)
        found = last_below[jnp.clip(upstrokes, 0, dvdt.shape[0] - 1)]
        return jnp.where(found > lo, found, lo)

    def _check_thresholds_and_peaks(self, v, dvdt, spikes, peaks, upstrokes, mask):
        """``ft.check_thresholds_and_peaks``."""
        m = self.max_spikes
        t = self.t
        n_dvdt = dvdt.shape[0]

        # -- merge spikes whose threshold overlaps the previous peak --------
        next_mask = _shift_left(mask, False)
        overlap = mask & next_mask & (_shift_left(spikes, self.n_samples) <= peaks + 1)
        keep_spike = mask & ~_shift_right(overlap, False)
        keep_other = mask & ~overlap

        mask, (spikes,) = _compact(keep_spike, (spikes,))
        _, (peaks, upstrokes) = _compact(keep_other, (peaks, upstrokes))
        count = jnp.sum(mask.astype(jnp.int32))

        # -- peaks that occur too long after their threshold ---------------
        t_spike = _take(t, spikes)
        t_peak = _take(t, peaks)
        too_long = mask & ((t_peak - t_spike) >= self.max_interval)

        avg_upstroke = _masked_mean(_take(dvdt, upstrokes), mask, count)
        target = avg_upstroke * self.check_thresh_frac
        last_below = _last_true_upto(dvdt <= target)

        # (a) assume the peak is right and re-find the threshold
        t_0 = jnp.searchsorted(t, t_peak - self.max_interval, side="left")
        below = last_below[jnp.clip(upstrokes, 0, n_dvdt - 1)]
        found = too_long & (below > t_0.astype(jnp.int32))
        spikes = jnp.where(found, below, spikes)

        # (b) otherwise assume the threshold is right and re-find the peak
        retry_peak = too_long & ~found
        t_0b = jnp.searchsorted(t, t_spike + 2.0 * self.max_interval, side="left")
        # ft.find_time_index raises when the time is past the end of the trace;
        # check_thresholds_and_peaks catches that and uses len(t) - 1.
        t_0b = jnp.where(t_0b >= self.n_samples, self.n_samples - 1, t_0b)
        new_peak = _range_argextremum(
            v, spikes, t_0b.astype(jnp.int32), retry_peak, maximize=True
        )
        t_new_peak = _take(t, new_peak)
        is_last = mask & ~_shift_left(mask, False)
        accept = (t_new_peak - t_spike < self.max_interval) & (
            is_last | (t_new_peak < _take(t, _shift_left(spikes, self.n_samples)))
        )
        peaks = jnp.where(retry_peak & accept, new_peak, peaks)

        keep = mask & ~(retry_peak & ~accept)
        mask, (spikes, peaks, upstrokes) = _compact(keep, (spikes, peaks, upstrokes))
        spikes = jnp.where(mask, spikes, self.n_samples)
        peaks = jnp.where(mask, peaks, self.n_samples)
        count = jnp.sum(mask.astype(jnp.int32))

        # -- was the last spike cut off by the end of the window? ----------
        last_slot = count - 1
        last_peak = _take(peaks, last_slot)
        last_thresh_v = _take(v, _take(spikes, last_slot)) + self.clip_tol
        pos = jnp.arange(self.n_samples, dtype=jnp.int32)
        in_tail = (pos >= last_peak) & (pos <= self._clip_end_index)
        returned = jnp.any(in_tail & (v <= last_thresh_v))
        clipped = (jnp.arange(m) == last_slot) & (count > 0) & ~returned

        return spikes, peaks, upstrokes, clipped, mask, count

    # ------------------------------------------------------------------ #
    def process_spikes(self, v) -> SpikeIndexes:
        """``EphysSweepFeatureExtractor._process_individual_spikes``.

        Returns the spike index arrays instead of a ``DataFrame``.
        """
        v = jnp.asarray(v)
        if v.shape != (self.n_samples,):
            raise ValueError(
                f"v must have shape ({self.n_samples},), got {v.shape}. "
                "Map over the batch axis with jax.vmap instead of passing it in."
            )
        dvdt = self._dvdt_full(v)

        spikes, mask, overflow = self._detect_putative_spikes(v)
        peaks = self._find_peak_indexes(v, spikes, mask)
        spikes, peaks, mask = self._filter_putative_spikes(v, dvdt, spikes, peaks, mask)
        count = jnp.sum(mask.astype(jnp.int32))

        upstrokes = self._find_upstroke_indexes(dvdt, spikes, peaks, mask)
        thresholds = self._refine_threshold_indexes(dvdt, upstrokes, mask, count)
        thresholds = jnp.where(mask, thresholds, self.n_samples)

        thresholds, peaks, upstrokes, clipped, mask, count = (
            self._check_thresholds_and_peaks(
                v, dvdt, thresholds, peaks, upstrokes, mask
            )
        )
        # _process_individual_spikes re-derives the upstrokes from the refined
        # thresholds before storing them in the DataFrame.
        upstrokes = self._find_upstroke_indexes(dvdt, thresholds, peaks, mask)

        return SpikeIndexes(
            threshold=thresholds,
            peak=peaks,
            upstroke=jnp.where(mask, upstrokes, self.n_samples),
            clipped=clipped & mask,
            valid=mask,
            n_spikes=count,
            overflow=overflow,
        )

    # ------------------------------------------------------------------ #
    def sweep_features(self, v) -> dict[str, jax.Array]:
        """``EphysSweepFeatureExtractor._process_spike_related_features``.

        Returns all of :data:`FEATS` as 0-d arrays.  For a sweep without spikes
        the numpy code only sets ``avg_rate = 0`` and leaves the rest missing;
        here they come back as NaN (and ``avg_rate`` is 0.0, same value).
        """
        spikes = self.process_spikes(v)
        return self._sweep_features_from(spikes)

    def _sweep_features_from(self, spikes: SpikeIndexes) -> dict[str, jax.Array]:
        t = self.t
        nan = jnp.asarray(np.nan, t.dtype)
        mask = spikes.valid
        count = spikes.n_spikes

        t_thresh = _take(t, spikes.threshold)

        # ft.get_isis: t[spikes[1:]] - t[spikes[:-1]]
        isis = t_thresh[1:] - t_thresh[:-1]
        isi_valid = mask[:-1] & mask[1:]
        n_isi = jnp.maximum(count - 1, 0)
        n_isi_safe = jnp.maximum(n_isi, 1)
        isi_mean = _masked_mean(isis, isi_valid, n_isi_safe)
        # np.std: mean, then sum of squared deviations, then sqrt -- all float32
        centred = jnp.where(isi_valid, isis - isi_mean, jnp.zeros((), t.dtype))
        isi_var = _numpy_sum(centred * centred, n_isi_safe) / n_isi_safe.astype(t.dtype)
        isi_std = jnp.sqrt(isi_var)

        has_isi = n_isi >= 1
        first_isi = jnp.where(has_isi, isis[0], nan)
        # numpy rounds std and mean to float32 before dividing them
        isi_cv = jnp.where(has_isi, _round_trip(isi_std) / _round_trip(isi_mean), nan)
        mean_isi = jnp.where(has_isi, 1e3 * isi_mean, nan)

        # np.median: sort, then average the one/two central order statistics.
        ordered = jnp.sort(jnp.where(isi_valid, isis, jnp.inf))
        lo = jnp.clip((n_isi - 1) // 2, 0, ordered.shape[0] - 1)
        hi = jnp.clip(n_isi // 2, 0, ordered.shape[0] - 1)
        median_isi = jnp.where(has_isi, (ordered[lo] + ordered[hi]) / 2.0, nan)

        adapt = self._adaptation_index(isis, isi_valid, n_isi)

        latency = jnp.where(count > 0, t_thresh[0] - self._lat_start, nan)

        # len(spikes_in_interval) / (end - start): a python int over a float32,
        # so the division happens in the dtype of t.
        in_interval = (
            mask & (t_thresh >= self._rate_start) & (t_thresh <= self._rate_end)
        )
        n_in_interval = jnp.sum(in_interval.astype(jnp.int32))
        avg_rate = n_in_interval.astype(self._rate_denom.dtype) / _round_trip(
            jnp.asarray(self._rate_denom)
        )

        return {
            "adapt": adapt,
            "avg_rate": avg_rate,
            "first_isi": first_isi,
            "isi_cv": isi_cv,
            "latency": latency,
            "mean_isi": mean_isi,
            "median_isi": median_isi,
        }

    def _adaptation_index(self, isis, isi_valid, n_isi):
        """``ft.adaptation_index`` -> ``ft.norm_diff`` (computed in float64)."""
        double = jnp.result_type(float)  # float64 iff jax_enable_x64
        a = isis.astype(double)
        pair = isi_valid[:-1] & isi_valid[1:]

        total = a[1:] + a[:-1]
        delta = a[1:] - a[:-1]

        # np.allclose(a[1:] + a[:-1], 0.0) -> |x| <= atol (1e-8)
        all_zero = jnp.all(jnp.where(pair, jnp.abs(total) <= 1e-8, True))

        denom = jnp.where(pair & (total != 0), total, jnp.ones((), double))
        norm_diffs = delta / denom
        both_zero = pair & (a[1:] == 0) & (a[:-1] == 0)
        norm_diffs = jnp.where(both_zero, jnp.zeros((), double), norm_diffs)

        # np.nanmean: NaNs -> 0, sum over *all* n_pairs entries, divide by the
        # number of non-NaN ones.
        n_pairs = jnp.maximum(n_isi - 1, 0)
        finite = pair & ~jnp.isnan(norm_diffs)
        n_use = jnp.sum(finite.astype(jnp.int32))
        total = _numpy_sum(jnp.where(finite, norm_diffs, 0.0), n_pairs)
        avg = total / jnp.maximum(n_use, 1).astype(double)
        avg = jnp.where(n_use > 0, avg, jnp.asarray(np.nan, double))

        value = jnp.where(all_zero, jnp.zeros((), double), avg)
        # adaptation_index -> nan for no ISIs, norm_diff -> nan for a single one
        return jnp.where(n_isi <= 1, jnp.asarray(np.nan, double), value)

    # ------------------------------------------------------------------ #
    def features(self, v) -> dict[str, jax.Array]:
        """Drop-in equivalent of ``make_features_numpy._get_features``.

        The numpy helper returns ``{}`` when ``adapt`` is missing (i.e. when no
        spike was detected) and ``resave_allen`` later turns those into
        ``NAN_FEATS``; so a spike-less sweep yields NaN for *all* seven
        features, ``avg_rate`` included.
        """
        spikes = self.process_spikes(v)
        feats = self._sweep_features_from(spikes)
        has_spikes = spikes.n_spikes > 0
        return {
            k: jnp.where(has_spikes, val, jnp.asarray(np.nan, val.dtype))
            for k, val in feats.items()
        }

    __call__ = features


def make_feature_extractor(t, **kwargs):
    """Convenience wrapper returning a jitted, vmap-ed ``features`` function.

    ``fn(voltages)`` maps over the leading axis of ``voltages`` and returns a
    dict of ``(batch,)`` arrays.
    """
    ext = JaxEphysSweepFeatureExtractor(t, **kwargs)
    return jax.jit(jax.vmap(ext.features))
