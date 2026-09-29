"""Region-of-success optimization scores: how often gradient descent finds the neuron.

The score
---------
For one (loss, parameter set): draw ``n_init`` random starts in transformed
space, run the optimizer on each, and record the fraction that finish inside a
"region of success" -- a radius ``r`` of the true parameters. Repeat over
``n_ideal`` draws of the true parameters. The score is

    S = mean over radii r of [ mean over ideal seeds i of
                               ( fraction of starts landing within r of x*_i ) ]

The nesting order is load-bearing. The fraction over starts comes first because
that is the estimator whose variance ``n_init`` controls; the ideal-seed mean is
next because the ideal seed is the outer random effect (measured on the
100-seed rows of experiment-optimization-scores-narrow.jsonl, 73-81% of the
per-seed variance is between ideal seeds, not binomial within them); the radius
average is outermost. Averaging in any other order weights the radii by
something other than 1/n.

Why this replaces the escalating-distance score
-----------------------------------------------
``generate_encoder_optimization_quality.py`` reports the first offset at which
the optimizer stops converging. It uses exactly two starts per distance level,
``x* + d`` and ``x* - d``, with the same scalar added to every free coordinate --
two corners on the all-ones diagonal, not a sample of the space. The score
therefore collapses onto whichever single parameter is hardest to optimize. It
is also quantized to ``dist_stepsize`` (0.2): 105 of the 380 runs on disk
returned exactly 0.2 and carry no information, which ``todo.md`` already flags.

Why a geometric radius ladder
-----------------------------
``SUCCESS_RADII`` is 0.375 x 2^{-2..2}: dyadic around 0.375, the ``basin_wide``
radius (0.25 x the 1.5 half-span), so the middle rung asks exactly the question
the convexity ``basin_wide`` descriptor asks.

Geometric rather than uniform, deliberately. Averaging success over a UNIFORM
grid on (0, R] converges to ``1 - E[min(d, R)]/R`` -- a censored MEAN ERROR
(checked numerically: a 2400-rung uniform ladder gives 0.098789 against the
closed form 0.098713). That would throw away exactly the boundedness that made
success rates beat mean error by +23% rank correlation in the first study,
where one start flying out to distance 6 could swamp ten that converged.
Geometric spacing is uniform in log r, so each halving of the error gets equal
weight. ``ausc@R`` is stored anyway, since it costs nothing.

No rung saturates. On the real ``(start_for, ideal_params_for_seed)`` draws the
random-start success rate is 0.00012 / 0.00098 / 0.00817 / 0.0627 / 0.376 at the
five rungs, and the ladder average is S0 = 0.090. That baseline is recomputed
per row from the actual starts and stored, so "better than guessing" is
checkable rather than assumed.

Comparability, and why there is an Adam arm
-------------------------------------------
Polyak's displacement is exactly scale-invariant while its cap does not bind,
which is what makes two losses of different magnitude comparable at all. The cap
binds ~52% of steps here. Worse, cap fraction is not an innocent bystander:
across the 56 sets, the paired score difference and the paired cap-fraction
difference correlate at Spearman -0.81, so matching *average* cap fractions is
necessary and not sufficient. Both are downstream of local curvature (for
``f = a|d|^2`` the raw Polyak step is ``1/(4a)``), and cap fraction alone cannot
separate that from a genuine gradient-magnitude advantage.

Adam's ``m/sqrt(v)`` update IS exactly invariant to multiplying the loss by a
constant. Running every cell under both and requiring the sign, the ranking and
the significance to agree turns comparability from an assumption into a
measurement. That is check C8 in ``--check``.

The four arms
-------------
``polyak``         optax ``polyak_sgd``, the inherited setup; invariant while
                   its trust-region cap does not bind, which it does ~52% of
                   the time here.
``adam``           cosine-decayed, lr 0.05 (calibrated); the ONLY exactly
                   scale-invariant arm, and therefore C8's reference.
``rmsprop``        a second adaptive arm.
``jaxley-polyak``  ``x <- x - (f/3) g/||g||^0.8``, the loop used in the Jaxley
                   fitting examples. Its step length scales with the loss, so
                   it is not scale-invariant; it is here because it is what
                   downstream users of this model class actually run.

Usage (from encoding/):
    python scripts/datagen/gen_region_success_scores.py \\
        --set_key gLeak+eLeak+gNa --loss encoder-zscore-mask \\
        --ideal_seeds 0 10 --init_seeds 0 100
    python scripts/datagen/gen_region_success_scores.py --check
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # encoding/
sys.path.insert(0, _HERE)

import jsonl_store as JS  # noqa: E402
import losses as L  # noqa: E402
import param_groups as pg  # noqa: E402
import sim as S  # noqa: E402
from helper import get_datagen_config  # noqa: E402
from logger import get_std_logger  # noqa: E402

OUT_PATH = Path("./data/optimization-region-scores.jsonl")

# `steps`, `optimizer`, `max_learning_rate` and `encoder_epoch` are in the key so
# the two optimizer arms, a learning-rate calibration sweep, and any re-run at a
# different checkpoint all coexist in one file instead of resume-skipping each
# other. Anything in here must also be a TOP-LEVEL row field, not just a `meta`
# entry -- `jsonl_store.row_key` reads the row, not the meta block.
KEY_FIELDS = (
    "params", "ideal_seeds", "init_seeds", "init_bounds",
    "loss", "encoder_epoch", "optimizer", "steps", "max_learning_rate",
    "grad_norm_beta",
)

# See the module docstring.
SUCCESS_RADII = (0.09375, 0.1875, 0.375, 0.75, 1.5)

# gen_optimization_scores.SUCCESS_TOLS, verbatim, so a new row can be compared
# directly against data/optimization-scores-{v2,betweenset}.jsonl.
LEGACY_TOLS = (0.1, 0.173, 0.375, 0.433, 0.866)

AUSC_RMAX = (1.0, 1.5)

# Guards the 0/0 polyak step at the exact optimum without moving the convergence
# floor. Measured on the branch: 1e-30 leaves the floor at 7.2e-9, 1e-12
# degrades it to 5.7e-8.
POLYAK_EPS = 1e-30

OPTIMIZERS = ("polyak", "adam", "rmsprop", "jaxley-polyak")

# Optimizers whose update needs the loss VALUE at the current iterate, not
# only its gradient. `build_runner` passes `value=` for exactly these.
NEEDS_VALUE = frozenset({"polyak", "jaxley-polyak"})

# Per-optimizer step scale. Polyak sets its own step and uses this only as a
# trust-region cap, so 1.0 is right there. Adam's per-coordinate step is ~lr, so
# reusing 1.0 would move ~50 units over 100 cosine-decayed steps inside a +-1.5
# box.
#
# Adam's 0.05 is measured, not guessed (job 2799826, gLeak+eLeak+gNa,
# encoder-zscore-mask, 3 ideal x 64 init):
#
#     lr     region score    escape_frac
#     0.02   0.100           0.000
#     0.05   0.132           0.000
#     0.10   0.138           0.010
#     0.20   0.094           0.255
#
# 0.10 scores marginally higher than 0.05 but the gap (0.006) is well inside the
# standard error (~0.016) and it is already leaking runs out of the box, with
# escape rising steeply just beyond it. 0.05 is the last lr with zero escapes.
# RMSProp's 0.03 is measured, not guessed (job 2822181, same cell as adam's
# sweep, picked by --lr_report's rule -- smallest median best-iterate distance
# among the rates that keep their runs in the box):
#
#     lr     d_best p50   escape   region score
#     0.003  1.7926       0.000    0.0917        barely leaves the start
#     0.01   1.7311       0.000    0.1062        <- the previous GUESS
#     0.03   1.6666       0.000    0.1552        <- picked
#     0.10   1.5968       0.130    0.1219        leaks out of the box
#
# The old 0.01 was never measured and scored a third less. 0.03 still has
# moving_frac 0.95, so its best iterate may improve on a longer budget -- worth
# knowing, not a reason to prefer a rate that barely moves.
#
# jaxley-polyak's entry is the `loss_val / 3.0` of the reference loop, i.e. the
# coefficient multiplying the loss to get the step size, so 1/3 rather than a
# learning rate in the usual sense. It is PINNED TO THE REFERENCE VALUE, against
# what the same sweep says, because the brief for this arm is that it behave
# exactly like that loop:
#
#     lr      d_best p50   escape   region score
#     1/6     1.6836       0.036    0.1396        <- what the rule would pick
#     1/3     1.5285       0.219    0.1719        <- the reference value, used
#     2/3     1.7779       0.536    0.0469
#
# So at this loss scale the reference coefficient leaks 22% of runs out of the
# box. That is a real property of the loop as written, not a defect here: the
# step length is `lr * f * ||g||^0.2` and the encoder loss is ~9 at a random
# start, where Jaxley's own examples fit an MSE on standardised voltages that is
# orders of magnitude smaller. It is also why the arm may fail C5's
# escape-spread gate and be disqualified as the reference optimizer. To run a
# TUNED jaxley arm instead, set this to 1/6 -- but do not mix the two in one
# store: region_scores.index() keys on (optimizer, loss, set) and two learning
# rates under one optimizer name collide.
DEFAULT_LR = {
    "polyak": 1.0, "adam": 0.05, "rmsprop": 0.03, "jaxley-polyak": 1.0 / 3.0,
}

# The gradient-normalisation exponent of the jaxley-polyak arm: the step follows
# `g / ||g||^GRAD_NORM_BETA`. 0.8 is the value the reference loop uses.
GRAD_NORM_BETA = 0.8

# The arm C8 measures every other arm against. adam is the only optimizer here
# whose update is exactly invariant to multiplying the loss by a constant, which
# is the whole property C8 tests -- see the module docstring.
REFERENCE_OPTIMIZER = "adam"

# Steps at the end of the run whose mean displacement decides "still moving".
TAIL_STEPS = 10
MOVING_EPS = 0.01

# x* is drawn from the middle half of each parameter's bounds, and
# SigmoidTransform.inverse maps that to +-logit(0.75) = +-log(3) in transformed
# space -- identically for all eight parameters, because the bounds cancel. So a
# coordinate beyond (init_hi + LOGIT_Q3) is outside any region that could hold
# both the start box and the truth: the run has escaped, not converged.
LOGIT_Q3 = float(np.log(3.0))


def _jaxley_polyak(lr_scale, beta):
    """Jaxley's Polyak-style SGD: a loss-sized step along a norm-damped gradient.

    A transcription of the reference loop (``todo.md``)::

        optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=3.0)
        ...
        grad_val = tree_map(lambda x: x / grad_norm ** beta, grad_val)
        opt_state.hyperparams["learning_rate"] = loss_val / 3.0
        updates, opt_state = optimizer.update(grad_val, opt_state)
        opt_params = optax.apply_updates(opt_params, updates)

    ``optax.sgd(lr)`` with no momentum emits ``-lr * g``, so that loop is exactly

        x <- x - (lr_scale * f(x)) * g / ||g||^beta

    with ``lr_scale`` carrying the ``1/3``. The transformation is stateless, so
    ``inject_hyperparams`` has nothing to inject here -- it exists in the
    reference only because the learning rate is rewritten each iteration.

    What this is NOT: with ``beta = 0.8`` the step length is
    ``lr_scale * f * ||g||^{1-beta}``, so multiplying the loss by ``c`` scales
    the step by ``c``. Textbook polyak is the ``beta = 2`` case, where the two
    factors of ``c`` cancel and the step is exactly invariant. This arm is
    therefore NOT loss-scale invariant and inherits the between-loss
    comparability caveat in the module docstring; adam remains the only exactly
    invariant arm, which is what C8 rests on.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None, *, value, **extra):
        del params, extra
        gnorm = optax.global_norm(updates)
        # `where` rather than `gnorm + eps`: exact wherever the gradient is
        # non-zero, and finite AT the optimum, where the reference loop's 0/0
        # would hand a NaN to the trailing zero_nans instead of standing still.
        denom = jnp.where(gnorm > 0, gnorm ** beta, 1.0)
        step = lr_scale * value
        return jax.tree.map(lambda g: -step * g / denom, updates), state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _make_optimizer(name, max_lr, eps, steps, grad_norm_beta=GRAD_NORM_BETA):
    """The gradient transformation, with NaN guards on both sides.

    ``zero_nans`` comes first to sanitize the incoming gradient (``nan_to_num``
    inside the simulator has a NaN cotangent), and again last because polyak can
    itself produce a NaN from a 0/0 step -- a guard placed only before it cannot
    catch that.
    """
    if name == "polyak":
        core = optax.polyak_sgd(max_learning_rate=max_lr, f_min=0.0, eps=eps)
    elif name == "adam":
        core = optax.adam(optax.cosine_decay_schedule(max_lr, steps))
    elif name == "rmsprop":
        core = optax.rmsprop(max_lr)
    elif name == "jaxley-polyak":
        core = _jaxley_polyak(max_lr, grad_norm_beta)
    else:
        raise ValueError(f"unknown optimizer {name!r}; want one of {OPTIMIZERS}")
    return optax.chain(optax.zero_nans(), core, optax.zero_nans())


def build_runner(
    traces_of,
    loss_fn,
    idx,
    *,
    steps=100,
    max_lr=1.0,
    eps=POLYAK_EPS,
    optimizer="polyak",
    grad_norm_beta=GRAD_NORM_BETA,
):
    """A jitted, vmapped optimizer run returning a dict of per-run diagnostics.

    ``traces_of(x8, states6) -> (n_amps, n_t)`` is injected rather than imported
    so the whole scan/vmap/optax/scatter path can be exercised against an
    analytic simulator in a test, with no simulation at all.

    Vmapped over ``(target8, states6, ideal_traces, x0)``, so one call handles a
    batch of (ideal seed, init seed) pairs. Every diagnostic below is an O(1)
    addition to the scan carry: the scan is latency-bound on the 6001-step
    jaxley solve, so they are free in wall clock.

    Returned keys, each ``(batch,)``:
        d_final     ||x_final - x*_sub||
        d_best      ||x_at_min_loss - x*_sub||   (a capped polyak is plain GD at
                    lr=max_lr inside a small box, where oscillation is plausible
                    and d_final alone would not see it)
        f0, f_final, f_best
        cap_frac    fraction of steps where the polyak cap bound (0.0 otherwise)
        last_disp   ||x_final - x_{final-1}||
        tail_disp   mean step displacement over the last TAIL_STEPS steps
        max_absx    max over steps of ||x||_inf   (escape detector)
        nan_frac    fraction of steps with a non-finite value or gradient
    """
    opt = _make_optimizer(optimizer, max_lr, eps, steps, grad_norm_beta)
    # Only optax's polyak has a trust-region cap to recompute; `jaxley-polyak`
    # has none, so cap_frac stays 0 there and is reported as null in the row.
    is_polyak = optimizer == "polyak"
    needs_value = optimizer in NEEDS_VALUE
    tail_from = steps - TAIL_STEPS

    def run_one(target8, states6, ideal_traces, x0):
        x_star = target8[idx]

        def objective(x_sub):
            # Scatter the optimized subset into the full 8-vector. Gradients
            # reach only x_sub, so the frozen parameters stay pinned at truth for
            # free and the D-dim error norm equals the 8-dim one exactly.
            full = target8.at[idx].set(x_sub)
            return loss_fn(traces_of(full, states6), ideal_traces)

        def body(carry, k):
            x, state, n_cap, f_best, x_best, n_nan, max_absx, disp, tail, = carry
            value, grad = jax.value_and_grad(objective)(x)

            finite = jnp.isfinite(value) & jnp.all(jnp.isfinite(grad))
            n_nan = n_nan + jnp.where(finite, 0.0, 1.0)

            if is_polyak:
                # Recompute the uncapped step to see whether the cap bound.
                raw = value / (jnp.sum(grad ** 2) + eps)
                n_cap = n_cap + jnp.where(raw > max_lr, 1.0, 0.0)
            if needs_value:
                updates, state = opt.update(grad, state, x, value=value)
            else:
                updates, state = opt.update(grad, state, x)

            # `value` is the loss AT x, so the iterate to remember is x itself,
            # not the one the update produces.
            better = value < f_best
            f_best = jnp.where(better, value, f_best)
            x_best = jnp.where(better, x, x_best)

            x_new = optax.apply_updates(x, updates)
            disp = jnp.linalg.norm(x_new - x)
            tail = tail + jnp.where(k >= tail_from, disp, 0.0)
            max_absx = jnp.maximum(max_absx, jnp.max(jnp.abs(x_new)))
            return (x_new, state, n_cap, f_best, x_best, n_nan, max_absx,
                    disp, tail), None

        zero = jnp.zeros((), dtype=x0.dtype)
        init = (x0, opt.init(x0), zero, jnp.array(jnp.inf, dtype=x0.dtype), x0,
                zero, jnp.max(jnp.abs(x0)), zero, zero)
        (x_final, _, n_cap, f_best, x_best, n_nan, max_absx, disp, tail), _ = (
            jax.lax.scan(body, init, jnp.arange(steps))
        )

        # The scan never scores its own last iterate, so f_final is one extra
        # forward pass (<1% on top of 100 fwd+bwd) rather than an approximation.
        f_final = objective(x_final)
        take_final = f_final < f_best
        f_best = jnp.where(take_final, f_final, f_best)
        x_best = jnp.where(take_final, x_final, x_best)

        return {
            "d_final": jnp.linalg.norm(x_final - x_star),
            "d_best": jnp.linalg.norm(x_best - x_star),
            "f0": objective(x0),
            "f_final": f_final,
            "f_best": f_best,
            "cap_frac": n_cap / steps,
            "last_disp": disp,
            "tail_disp": tail / TAIL_STEPS,
            "max_absx": max_absx,
            "nan_frac": n_nan / steps,
        }

    return jax.jit(jax.vmap(run_one, in_axes=(0, 0, 0, 0)))


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def success_table(raw, radii):
    """``{radius: (n_ideal,) fraction of starts within it}`` from an (I, J) matrix."""
    return {float(r): (raw < r).mean(axis=1) for r in radii}


def ladder_score(raw, radii=SUCCESS_RADII):
    """Per-ideal-seed ladder average, and its mean -- the headline score.

    Returns ``(per_seed, scalar)``. ``per_seed`` is what the standard error is
    computed from: each element already contains its own binomial noise, so the
    spread across seeds captures both variance components without a
    decomposition that could get the weights wrong.
    """
    per_seed = np.mean([(raw < r).mean(axis=1) for r in radii], axis=0)
    return per_seed, float(per_seed.mean())


def ausc(raw, r_max):
    """1 - E[min(d, R)]/R: the continuum limit of a uniform success ladder.

    A non-finite distance is censored to R, the same thing the success ladder
    does with it implicitly (`nan < r` is False, i.e. a failure). Without this
    a single diverged run in a thousand turns the whole statistic into nan while
    the headline score still looks healthy -- easy to miss in a 336-row file.
    """
    d = np.where(np.isfinite(raw), raw, np.inf)
    return float(1.0 - np.minimum(d, r_max).mean() / r_max)


def quantiles(a):
    """Summary stats over the FINITE entries, plus a count of the rest.

    np.percentile propagates nan, so one non-finite run would nan out every
    field here. Reporting the finite distribution and `n_nonfinite` alongside it
    keeps the row readable and makes the bad runs countable instead of hiding
    them inside a nan.
    """
    a = np.asarray(a, dtype=float).ravel()
    good = a[np.isfinite(a)]
    n_bad = int(a.size - good.size)
    if good.size == 0:
        return {k: None for k in
                ("mean", "std", "p10", "p25", "p50", "p75", "p90", "p99",
                 "max")} | {"n_nonfinite": n_bad}
    q = np.percentile(good, [10, 25, 50, 75, 90, 99])
    return {
        "mean": float(good.mean()), "std": float(good.std()),
        "p10": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
        "p75": float(q[3]), "p90": float(q[4]), "p99": float(q[5]),
        "max": float(good.max()), "n_nonfinite": n_bad,
    }


def start_distances(targets, starts, idx):
    """``(I, J)`` distances from every start to every truth. No simulation.

    ``start_for`` depends only on (init seed, dim) and the targets only on the
    ideal seed, so this is pure host arithmetic -- and it is identical across
    losses and optimizers by construction. ``d0_hash`` turns that into a
    checkable invariant (C1).
    """
    x_star = np.asarray(targets)[:, np.asarray(idx)]        # (I, D)
    d0 = np.linalg.norm(x_star[:, None, :] - np.asarray(starts)[None, :, :], axis=-1)
    return d0


def _hash_matrix(a) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.round(np.asarray(a), 8), dtype=np.float64).tobytes()
    ).hexdigest()[:16]


def score_block(
    simulator,
    loss_fn,
    params,
    ideal_seeds,
    init_seeds,
    *,
    steps=100,
    max_lr=1.0,
    eps=POLYAK_EPS,
    optimizer="polyak",
    grad_norm_beta=GRAD_NORM_BETA,
    chunk=2048,
    init_bounds=(-1.5, 1.5),
    logger=None,
):
    """Run every (ideal seed, init seed) pair and return the raw diagnostics.

    Returns ``(raw, d0, diag_arrays, f_at_truth)`` where ``raw`` and ``d0`` are
    ``(I, J)`` final and initial distances and ``diag_arrays`` maps each
    per-run diagnostic name to its own ``(I, J)`` matrix.
    """
    idx = S.param_indices(params)
    dim = len(params)

    ideal_list = list(range(*ideal_seeds))
    init_list = list(range(*init_seeds))
    n_i, n_j = len(ideal_list), len(init_list)

    targets = simulator.targets_for_seeds(ideal_list)           # (I, 8)
    states = simulator.init_states_for_seeds(ideal_list)        # (I, 6)
    ideal_traces = jax.jit(jax.vmap(simulator.traces_of, in_axes=(0, 0)))(
        targets, states
    )                                                           # (I, n_amps, n_t)

    # The loss must be exactly 0 at the truth or polyak's f_min=0.0 is a lie.
    f_at_truth = float(
        jnp.max(jax.jit(jax.vmap(loss_fn))(ideal_traces, ideal_traces))
    )

    lo_b, hi_b = init_bounds
    starts = jnp.stack(
        [S.start_for(j, dim, lo=lo_b, hi=hi_b) for j in init_list]
    )                                                           # (J, D)

    d0 = start_distances(targets, starts, idx)

    runner = build_runner(
        simulator.traces_of, loss_fn, idx,
        steps=steps, max_lr=max_lr, eps=eps, optimizer=optimizer,
        grad_norm_beta=grad_norm_beta,
    )

    # Ideal-major flattening, so a chunk usually covers whole ideal seeds.
    ii, jj = np.meshgrid(np.arange(n_i), np.arange(n_j), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    total = n_i * n_j
    out = None

    t0 = time.time()
    for lo in range(0, total, chunk):
        hi = min(lo + chunk, total)
        sl = slice(lo, hi)
        res = runner(
            targets[ii[sl]], states[ii[sl]], ideal_traces[ii[sl]], starts[jj[sl]]
        )
        # The only host sync, once per chunk rather than once per step.
        if out is None:
            out = {k: np.empty(total) for k in res}
        for k, v in res.items():
            out[k][sl] = np.asarray(v)
        if logger:
            logger.info(f"{hi}/{total} cells ({time.time() - t0:.0f}s)")

    diag_arrays = {k: v.reshape(n_i, n_j) for k, v in out.items()}
    return diag_arrays.pop("d_final"), d0, diag_arrays, f_at_truth


def build_row(
    *,
    probe,
    raw,
    d0,
    diag_arrays,
    f_at_truth,
    loss_name,
    set_key,
    params,
    optimizer,
    meta_common,
    save_raw,
):
    """Assemble the JSONL row. Everything here is derived from `raw` and `d0`."""
    per_seed, score = ladder_score(raw)
    base_per_seed, base_score = ladder_score(d0)
    n_i = raw.shape[0]

    succ = success_table(raw, SUCCESS_RADII)
    legacy = success_table(raw, LEGACY_TOLS)
    base = success_table(d0, SUCCESS_RADII)

    escape_cut = meta_common["init_bounds"][1] + LOGIT_Q3
    cap = diag_arrays["cap_frac"]

    row = {
        **probe,
        "set": set_key,
        "loss_name": loss_name,
        "dim": len(params),

        "region_score": score,
        "region_score_baseline": base_score,
        # (S - S0) / (1 - S0): the share of the available improvement captured.
        # Secondary -- S0 varies only ~0.054-0.063 across the 56 sets, so this is
        # near-affine and changes no ranking. It is here so "better than
        # guessing" is a number on the page, not an assumption.
        "region_score_norm": float((score - base_score) / (1.0 - base_score)),
        "region_score_per_seed": [float(v) for v in per_seed],
        # Across-seed spread of a value that already contains its own binomial
        # noise, so this is the total standard error, not just the outer part.
        "region_score_se": float(per_seed.std(ddof=1) / np.sqrt(n_i))
        if n_i > 1 else None,

        "success_radii": list(SUCCESS_RADII),
        **{f"success@{r:g}": [float(v) for v in succ[float(r)]]
           for r in SUCCESS_RADII},
        "success_baseline": {f"{r:g}": float(base[float(r)].mean())
                             for r in SUCCESS_RADII},

        # Back-compat with optimization-scores-{v2,betweenset}.jsonl.
        **{f"optimization-success@{t:g}": [float(v) for v in legacy[float(t)]]
           for t in LEGACY_TOLS},
        "optimization-scores": [float(v) for v in raw.mean(axis=1)],
        "optimization-scores-std": [float(v) for v in raw.std(axis=1)],
        "optimization-scores-median": [float(v) for v in np.median(raw, axis=1)],

        **{f"ausc@{r:g}": ausc(raw, r) for r in AUSC_RMAX},
        **{f"ausc_baseline@{r:g}": ausc(d0, r) for r in AUSC_RMAX},

        "d_final": quantiles(raw),
        "d_best": quantiles(diag_arrays["d_best"]),
        "d0": quantiles(d0),
        "d0_hash": _hash_matrix(d0),
        # Diagnostic only, never a pass/fail: on these landscapes the mean final
        # distance routinely EXCEEDS the mean start distance (the published
        # betweenset rows do it too) because a few runs fly out to distance 5-6
        # and drag the average past every run that converged. That is precisely
        # why the headline is a bounded success rate. Finite entries only.
        "improvement": float(
            d0[np.isfinite(d0)].mean() - raw[np.isfinite(raw)].mean()
        ) if np.isfinite(raw).any() else None,

        "diag": {
            "polyak_cap_frac": float(cap.mean()) if optimizer == "polyak" else None,
            "polyak_cap_frac_p10": float(np.percentile(cap, 10))
            if optimizer == "polyak" else None,
            "polyak_cap_frac_p90": float(np.percentile(cap, 90))
            if optimizer == "polyak" else None,
            "moving_frac": float((diag_arrays["tail_disp"] > MOVING_EPS).mean()),
            "last_disp_p50": float(np.median(diag_arrays["last_disp"])),
            "last_disp_p90": float(np.percentile(diag_arrays["last_disp"], 90)),
            "escape_frac": float((diag_arrays["max_absx"] > escape_cut).mean()),
            "escape_cut": float(escape_cut),
            "nonfinite_frac": float((~np.isfinite(raw)).mean()),
            "nan_step_frac": float(diag_arrays["nan_frac"].mean()),
            "f_at_truth": f_at_truth,
            "f0_p50": float(np.median(diag_arrays["f0"])),
            "f_final_p50": float(np.median(diag_arrays["f_final"])),
            # nanmedian, not median: f0 is 0 for a start that happens to sit at
            # the optimum, and one such run would nan the whole field.
            "f_ratio_p50": float(np.nanmedian(
                diag_arrays["f_final"] / np.where(diag_arrays["f0"] == 0, np.nan,
                                                  diag_arrays["f0"])
            )),
            "worse_than_start_frac": float((raw > d0).mean()),
        },
        "meta": meta_common,
    }
    if save_raw:
        row["raw_d_final"] = np.round(raw, 5).tolist()
        row["raw_d0"] = np.round(d0, 5).tolist()
        # d_best too, because on these landscapes it is a materially different
        # question. Measured in the pilot: d_final p50 3.15 against d_best p50
        # 1.77 -- the optimizer passes near the truth and then wanders off. The
        # user's specification is the FINAL position, so that stays the
        # headline, but "could this loss find the neuron at all" is answerable
        # from the same run and would otherwise need a full re-run to recover.
        row["raw_d_best"] = np.round(diag_arrays["d_best"], 5).tolist()
    return row


def _commit() -> str:
    try:
        from git import Repo

        return Repo("../").commit().hexsha
    except Exception:  # noqa: BLE001
        return "unknown"


def _device() -> str:
    try:
        return jax.devices()[0].device_kind
    except Exception:  # noqa: BLE001
        return "unknown"


# Recorded in every row because `commit` alone is not enough to say what
# produced a number. The five score files already on disk were all written
# before the venv was reinstalled on 2026-08-26, no lockfile is tracked, and a
# 100-step trajectory on a rough landscape turns a 1e-15 kernel change into an
# O(1) change in the final distance -- so the same code and the same commit can
# give different scores across an upgrade. Without this field there is no way,
# after the fact, to tell which rows are comparable with which.
_VERSIONED = ("jax", "jaxlib", "jaxley", "optax", "equinox", "numpy")


def _versions() -> dict:
    import importlib.metadata as md

    out = {}
    for pkg in _VERSIONED:
        try:
            out[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            out[pkg] = "unknown"
    return out


def _epoch_arg(v):
    """`--encoder_epoch` as a string, because a study spec's YAML `null` arrives
    as the literal token "None" and `type=int` would then fail in every task."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("", "none", "null", "final"):
        return None
    return int(s)


# ---------------------------------------------------------------------------
# --check: the comparability report (pure pandas/scipy, no GPU)
# ---------------------------------------------------------------------------


def _pivot(rows, optimizer, field):
    """``{loss_name: {set: value}}`` for one optimizer arm."""
    out = {}
    for r in rows:
        if r.get("optimizer") != optimizer:
            continue
        out.setdefault(r["loss_name"], {})[r["set"]] = _dig(r, field)
    return out


def _dig(row, field):
    if "." in field:
        a, b = field.split(".", 1)
        return (row.get(a) or {}).get(b)
    return row.get(field)


def _paired(a: dict, b: dict):
    """Values of two {set: value} maps over the sets they share, in a fixed order."""
    keys = sorted(set(a) & set(b))
    return keys, np.array([a[k] for k in keys]), np.array([b[k] for k in keys])


def check_scores(out_path, pair, logger=print) -> int:
    """Print the C1-C8 comparability table. Returns 0 if every check passed."""
    from scipy.stats import spearmanr, wilcoxon

    rows = JS.load_rows(out_path)
    if not rows:
        logger(f"no rows in {out_path}")
        return 1
    logger(f"{len(rows)} rows from {out_path}")
    losses = sorted({r["loss_name"] for r in rows})
    opts = sorted({r.get("optimizer") for r in rows})
    logger(f"losses: {losses}")
    logger(f"optimizers: {opts}")
    logger("")

    failures = []

    def report(tag, ok, msg):
        # `ok` must be coerced before the identity test. Several checks compute
        # their condition from numpy (np.median, .mean()), which yields np.bool_
        # -- and `np.False_ is False` is False, so an un-coerced numpy verdict
        # would print FAIL and still exit 0. That made C6 unenforceable.
        if ok is not None:
            ok = bool(ok)
        state = "INFO" if ok is None else ("PASS" if ok else "FAIL")
        logger(f"  [{state}] {tag}: {msg}")
        if ok is False:
            failures.append(tag)

    # C1 -- the paired design held.
    groups = {}
    for r in rows:
        k = (r["set"], tuple(r["ideal_seeds"]), tuple(r["init_seeds"]),
             tuple(r["init_bounds"]))
        groups.setdefault(k, set()).add(r["d0_hash"])
    bad = {k: v for k, v in groups.items() if len(v) > 1}
    report("C1 pairing", not bad,
           f"{len(groups)} (set, seed-range) groups, "
           f"{len(bad)} with differing d0_hash"
           + (f" -- e.g. {sorted(bad)[0]}" if bad else ""))

    # C2 -- the loss really is minimized at the truth.
    worst = max(rows, key=lambda r: abs(_dig(r, "diag.f_at_truth") or 0.0))
    wv = abs(_dig(worst, "diag.f_at_truth") or 0.0)
    report("C2 optimum at truth", wv < 1e-12,
           f"max |f(x*)| = {wv:.3e} ({worst['loss_name']} {worst['set']})")

    # C3 -- better than guessing.
    #
    # The failure mode this exists to catch is the v1 pathology: a loss whose
    # optimization systematically ends farther from the truth than it started,
    # which happened because the starts were drawn from a wider window than the
    # landscapes were swept over. That is an ARM-level defect, so it is tested
    # at arm level. A handful of individual rows sitting at chance is a
    # different thing entirely -- a real result about a few hard parameter
    # triplets -- and failing the whole study for it would train the reader to
    # ignore C3.
    worse = [r for r in rows if r["region_score"] <= r["region_score_baseline"]]
    arms = {}
    for r in rows:
        arms.setdefault((r["loss_name"], r.get("optimizer")), []).append(
            (r["region_score"], r["region_score_baseline"])
        )
    bad_arms = [k for k, v in arms.items()
                if np.mean([a for a, _ in v]) <= np.mean([b for _, b in v])]
    frac = len(worse) / max(len(rows), 1)
    report("C3 beats chance", not bad_arms and frac <= 0.05,
           f"{len(worse)}/{len(rows)} rows ({frac:.1%}) at or below chance, "
           f"{len(bad_arms)} whole arms below chance"
           + (f" -- {bad_arms}" if bad_arms else ""))
    if worse:
        report("C3 rows at chance", None,
               "; ".join(f"{r['set']}/{r['loss_name']}/{r.get('optimizer')}"
                         for r in worse[:6])
               + " -- hard triplets, not a defect")

    # C4/C5 -- these ask whether the two losses are treated the SAME, not
    # whether the optimizer is well behaved in absolute terms.
    #
    # The distinction is load-bearing. The inherited polyak setup diverges on
    # most starts (measured: escape_frac ~0.5, moving_frac ~0.9, 62% of runs
    # ending farther out than they began), so an absolute threshold would fail
    # every polyak row and say nothing about comparability. A diverging
    # optimizer that diverges EQUALLY for both losses still gives a fair paired
    # comparison -- just a noisier one. What would actually invalidate the
    # comparison is one loss being budget-limited or escaping more than the
    # other, so that is what is tested. The absolute level is reported as INFO,
    # per optimizer, because it is a real caveat on the result.
    for opt in opts:
        sub = [r for r in rows if r.get("optimizer") == opt]
        if not sub:
            continue
        for tag, field, lim in (("C4 budget", "diag.moving_frac", 0.10),
                                ("C5 divergence", "diag.escape_frac", 0.10)):
            per = {}
            for r in sub:
                per.setdefault(r["loss_name"], []).append(_dig(r, field) or 0.0)
            avg = {k: float(np.mean(v)) for k, v in per.items()}
            spread = (max(avg.values()) - min(avg.values())) if len(avg) > 1 else 0.0
            report(f"{tag} [{opt}]", spread < lim,
                   "between-loss spread {:.3f} (limit {:g}); levels ".format(
                       spread, lim)
                   + ", ".join(f"{k}={v:.3f}" for k, v in sorted(avg.items())))
    nf = max(_dig(r, "diag.nonfinite_frac") or 0.0 for r in rows)
    report("C5b non-finite", nf < 0.005, f"max nonfinite_frac {nf:.4f}")
    worst = max(rows, key=lambda r: _dig(r, "diag.escape_frac") or 0.0)
    report("absolute optimizer health", None,
           f"worst escape_frac {_dig(worst, 'diag.escape_frac'):.3f} "
           f"({worst['loss_name']} {worst['optimizer']} {worst['set']}); "
           "a high level is a caveat on the result, not a comparability defect")

    a_name, b_name = pair
    for opt in opts:
        s = _pivot(rows, opt, "region_score")
        if a_name not in s or b_name not in s:
            continue
        keys, sa, sb = _paired(s[a_name], s[b_name])
        if len(keys) < 3:
            continue
        diff = sb - sa
        try:
            w = wilcoxon(diff).pvalue
        except ValueError:
            w = float("nan")
        logger("")
        logger(f"  --- {opt}: {b_name} minus {a_name} over {len(keys)} sets ---")
        logger(f"      mean {sa.mean():.4f} vs {sb.mean():.4f}; "
               f"paired diff {diff.mean():+.4f}; "
               f"{int((diff > 0).sum())}/{len(keys)} sets favour {b_name}; "
               f"wilcoxon p={w:.2e}")

        if opt == "polyak":
            c = _pivot(rows, opt, "diag.polyak_cap_frac")
            if a_name in c and b_name in c:
                _, ca, cb = _paired(c[a_name], c[b_name])
                report("C6 cap parity",
                       abs(np.median(ca) - np.median(cb)) < 0.05,
                       f"median cap_frac {np.median(ca):.3f} vs "
                       f"{np.median(cb):.3f}")
                rho = spearmanr(diff, cb - ca).statistic
                report("C7 cap coupling", None,
                       f"spearman(dScore, dCapFrac) = {rho:+.3f} "
                       "(diagnostic only; C8 is what settles comparability)")

    # C8 -- does the conclusion survive an exactly scale-invariant optimizer?
    #
    # `adam` is the reference arm, not one half of a hardcoded pair: its
    # m/sqrt(v) update is the only one here exactly invariant to multiplying the
    # loss by a constant. polyak's trust-region cap and jaxley-polyak's
    # ||g||^beta damping each break that in their own way, so every other
    # present arm is checked against adam separately.
    sad = _pivot(rows, REFERENCE_OPTIMIZER, "region_score")
    others = [o for o in opts if o != REFERENCE_OPTIMIZER]
    if a_name not in sad or b_name not in sad or not others:
        report("C8 scale-invariance", None,
               f"the {REFERENCE_OPTIMIZER} reference arm and at least one other "
               "arm must both be present for this pair")
    for opt in others:
        sp = _pivot(rows, opt, "region_score")
        if a_name not in sp or b_name not in sp:
            report(f"C8 scale-invariance ({opt})", None,
                   f"{opt} has no rows for this pair")
            continue
        kp, _, _ = _paired(sp[a_name], sp[b_name])
        ka, _, _ = _paired(sad[a_name], sad[b_name])
        shared = sorted(set(kp) & set(ka))
        if len(shared) < 3:
            report(f"C8 scale-invariance ({opt})", None,
                   f"only {len(shared)} sets shared with {REFERENCE_OPTIMIZER}")
            continue
        dp = np.array([sp[b_name][k] - sp[a_name][k] for k in shared])
        da = np.array([sad[b_name][k] - sad[a_name][k] for k in shared])
        pp, pad = wilcoxon(dp).pvalue, wilcoxon(da).pvalue
        rho = spearmanr(
            [sp[a_name][k] for k in shared] + [sp[b_name][k] for k in shared],
            [sad[a_name][k] for k in shared] + [sad[b_name][k] for k in shared],
        ).statistic
        agree = (np.sign(dp.mean()) == np.sign(da.mean())
                 and (pp < 0.05) == (pad < 0.05) and rho > 0.7)
        report(f"C8 scale-invariance ({opt})", bool(agree),
               f"{opt} diff {dp.mean():+.4f} (p={pp:.2e}), "
               f"{REFERENCE_OPTIMIZER} diff {da.mean():+.4f} (p={pad:.2e}), "
               f"spearman(S_{opt}, S_{REFERENCE_OPTIMIZER}) = {rho:+.3f}")

    logger("")
    logger("ALL CHECKS PASSED" if not failures else f"FAILED: {', '.join(failures)}")
    return 1 if failures else 0


# The one hard gate: runs must stay in the box. A run that leaves the region any
# start could have come from is no longer measuring the landscape, and an lr
# that leaks runs out scores the escape rate instead. 0.05 is the threshold
# Adam's 0.05 was picked under (job 2799826), reused so the arms are comparably
# tuned. Among admissible lrs, the smallest median BEST-iterate distance wins --
# the same statistic the study reports (configs/region_scores.yaml:score_matrix).
LR_MAX_ESCAPE = 0.05

# "Still moving at the last step" is REPORTED, not gated on. It was a gate while
# the score was the final iterate, where an unsettled run means the budget cut
# it off mid-descent. Scoring the best iterate makes that much weaker: a run
# that wanders on after passing close to the truth still contributes its best
# point. Gating on it here picked rmsprop's 0.003 -- an lr so small the runs
# barely leave their start and score 0.0917 against a chance of 0.0865 -- over
# lrs that score half again as much. Above this fraction the arm is flagged, so
# "its best iterate might improve with a longer budget" stays on the record.
LR_MOVING_NOTE = 0.90


def lr_report(path, logger=print) -> int:
    """Tabulate an lr-calibration sweep and name the pick.

    The rule lives here rather than in the sbatch script so that "how the
    learning rate was chosen" is one function every arm went through, instead of
    a shell loop plus a reading of the log. Returns non-zero if no lr in the
    sweep is admissible -- that is a real result (the whole range is too hot)
    and must not look like a success.
    """
    path = Path(path)
    rows = JS.load_rows(path)
    if not rows:
        logger(f"no rows in {path}")
        return 1

    by_arm: dict[tuple, list] = {}
    for r in rows:
        by_arm.setdefault(
            (r.get("optimizer"), r.get("grad_norm_beta"), r["loss_name"],
             r["set"]), []).append(r)

    failed = []
    for (opt, beta, loss, sset), group in sorted(by_arm.items(), key=str):
        head = f"{opt}" + (f" (beta {beta})" if beta is not None else "")
        logger(f"\n{head}  {loss}  {sset}   "
               f"[admissible: escape < {LR_MAX_ESCAPE}; "
               f"pick = min d_best p50]")
        logger(f"  {'lr':>8}  {'d_best p50':>10}  {'d_final p50':>11}  "
               f"{'moving':>7}  {'escape':>7}  {'score':>7}")
        best, best_lr = None, None
        for r in sorted(group, key=lambda x: x["max_learning_rate"]):
            d = r["diag"]
            ok = d["escape_frac"] < LR_MAX_ESCAPE
            note = "" if ok else "   (escapes: not admissible)"
            if ok and d["moving_frac"] >= LR_MOVING_NOTE:
                note = "   (still moving: best iterate may improve on a longer budget)"
            logger(f"  {r['max_learning_rate']:8.4f}  {r['d_best']['p50']:10.4f}  "
                   f"{r['d_final']['p50']:11.4f}  {d['moving_frac']:7.3f}  "
                   f"{d['escape_frac']:7.3f}  {r['region_score']:7.4f}{note}")
            if ok and (best is None or r["d_best"]["p50"] < best):
                best, best_lr = r["d_best"]["p50"], r["max_learning_rate"]
        if best_lr is None:
            logger("  PICK: none -- no admissible lr in this sweep")
            failed.append(head)
        else:
            logger(f"  PICK: lr = {best_lr:g}  (d_best p50 {best:.4f})")

    if failed:
        logger(f"\nNO ADMISSIBLE LR for: {', '.join(failed)}. "
               "Widen the sweep downwards.")
        return 1
    logger("")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _range_flag(ap, name, default, typ, help_):
    """A range as `--name LO HI`, plus scalar `--name_lo/_hi` overrides.

    Both forms exist because they serve different callers. A human writes
    `--ideal_seeds 0 10`. A study spec cannot: `study.argv_for` emits one flag
    and one token per axis/flag entry, so a two-valued flag would arrive
    truncated. The scalar overrides are what the spec sets.
    """
    ap.add_argument(f"--{name}", nargs=2, type=typ, default=list(default),
                    metavar=("LO", "HI"), help=help_)
    ap.add_argument(f"--{name}_lo", type=typ, default=None,
                    help=f"scalar override for {name} LO (study specs use this)")
    ap.add_argument(f"--{name}_hi", type=typ, default=None,
                    help=f"scalar override for {name} HI (study specs use this)")


def _resolve_range(args, name):
    lo, hi = getattr(args, name)
    lo = getattr(args, f"{name}_lo") if getattr(args, f"{name}_lo") is not None else lo
    hi = getattr(args, f"{name}_hi") if getattr(args, f"{name}_hi") is not None else hi
    return lo, hi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--set_key", help="parameter set as a param_groups key, e.g. "
                                      "gLeak+eLeak+gNa (one token, so a study "
                                      "spec can drive it)")
    ap.add_argument("--params", nargs="+", help="parameters, instead of --set_key")
    ap.add_argument("--loss", choices=list(L.LOSS_NAMES),
                    help="required unless --merge or --check")
    ap.add_argument("--encoder_epoch", type=_epoch_arg, default=None,
                    help="encoder checkpoint; omit or 'None' for final-model.eqx")
    ap.add_argument("--channel_type", default="Pospischil")

    _range_flag(ap, "ideal_seeds", (0, 10), int,
                "half-open range of true-parameter draws")
    _range_flag(ap, "init_seeds", (0, 100), int,
                "half-open range of optimization starts")
    _range_flag(ap, "init_bounds", pg.GRID_PLAN[pg.SET_DIM]["bounds"], float,
                "range the starts are drawn from, in transformed space. Defaults "
                "to GRID_PLAN bounds and MUST match the landscape window the "
                "convexity descriptors were computed on, or the two score "
                "families describe different regions of parameter space. This "
                "is what made optimization-scores.jsonl v1 invalid.")

    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--optimizer", default="polyak", choices=OPTIMIZERS)
    ap.add_argument("--max_learning_rate", type=float, default=None,
                    help=f"default per optimizer: {DEFAULT_LR}")
    ap.add_argument("--polyak_eps", type=float, default=POLYAK_EPS)
    ap.add_argument("--grad_norm_beta", type=float, default=GRAD_NORM_BETA,
                    help="jaxley-polyak only: the exponent in g/||g||^beta. "
                         f"Default {GRAD_NORM_BETA}, the reference loop's value. "
                         "Recorded as null for every other optimizer, so adding "
                         "it left the resume keys of existing rows unchanged.")
    ap.add_argument("--chunk", type=int, default=2048,
                    help="(ideal, init) pairs per jitted call. Batched steps/sec "
                         "is constant in chunk (the 6001-step scan is "
                         "latency-bound), so throughput scales linearly with it "
                         "and the only limit is memory. A100-40GB: 512 without "
                         "checkpointing, 2048 with.")
    ap.add_argument("--checkpoint", nargs="*", type=int, default=[78, 78],
                    metavar="LEN",
                    help="jx.integrate recursive checkpointing, e.g. 78 78 "
                         "(78*78 >= 6001). ~1.5x compute for 4x the chunk, a net "
                         "2.7x speedup. Pass with no values to disable.")
    ap.add_argument("--save_raw", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="store the full (ideal, init) distance matrices in the "
                         "row, so any radius or ladder can be recomputed later "
                         "without re-running")
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--force", action="store_true", help="ignore the resume key")
    ap.add_argument("--merge", action="store_true",
                    help="de-duplicate and rewrite the output file, then exit")
    ap.add_argument("--lr_report", metavar="PATH", default=None,
                    help="tabulate an lr-calibration sweep written by "
                         "slurm_scripts/region_lr_sweep.sh and name the pick")
    ap.add_argument("--check", action="store_true",
                    help="print the C1-C8 comparability report, then exit")
    ap.add_argument("--check_pair", nargs=2,
                    default=["encoder-zscore-mask", "encoder-zscore-2"],
                    metavar=("A", "B"), help="the contrast --check reports on")
    args = ap.parse_args(argv)

    out_path = Path(args.out)

    if args.lr_report:
        return lr_report(args.lr_report)
    if args.check:
        return check_scores(out_path, tuple(args.check_pair))
    if args.merge:
        kept = JS.collect(out_path, KEY_FIELDS)
        print(f"merged {kept} rows into {out_path}")
        return 0

    if args.loss is None:
        ap.error("--loss is required unless --merge or --check is given")
    if (args.set_key is None) == (args.params is None):
        ap.error("give exactly one of --set_key or --params")

    params = S.canonical(args.params if args.params else pg.params_for(args.set_key))
    set_key = pg.set_key(params)

    i_lo, i_hi = _resolve_range(args, "ideal_seeds")
    j_lo, j_hi = _resolve_range(args, "init_seeds")
    b_lo, b_hi = _resolve_range(args, "init_bounds")
    if i_hi <= i_lo or j_hi <= j_lo:
        ap.error("seed ranges must be non-empty half-open [LO, HI)")
    if b_hi <= b_lo:
        ap.error("init_bounds must satisfy LO < HI")

    max_lr = (args.max_learning_rate if args.max_learning_rate is not None
              else DEFAULT_LR[args.optimizer])

    spec = L.LOSS_SPECS[args.loss]
    # An epoch means nothing for a loss that loads no encoder; recording it
    # anyway would split the resume key across rows that are actually identical.
    epoch = args.encoder_epoch if spec[0] == "encoder_loss" else None

    probe = {
        "params": params,
        "ideal_seeds": [i_lo, i_hi],
        "init_seeds": [j_lo, j_hi],
        # Top-level, not only in meta, because it is part of the resume key.
        "init_bounds": [float(b_lo), float(b_hi)],
        "loss": list(spec),
        "encoder_epoch": epoch,
        "optimizer": args.optimizer,
        "steps": args.steps,
        # Top-level, not only in meta, because it is part of the resume key: an
        # lr sweep writes several rows that are otherwise identical.
        "max_learning_rate": float(max_lr),
        # None for every optimizer that does not read it, which is what keeps
        # the resume key of a polyak/adam/rmsprop row identical to the rows
        # written before this field existed (`row_key` reads a missing field as
        # None). Storing 0.8 unconditionally would orphan all of them.
        "grad_norm_beta": (float(args.grad_norm_beta)
                           if args.optimizer == "jaxley-polyak" else None),
    }

    if not args.force:
        if JS.row_key(probe, KEY_FIELDS) in JS.existing_keys(out_path, KEY_FIELDS):
            print(f"skip {set_key} {args.loss} {args.optimizer}  (already done)")
            return 0

    logger = get_std_logger()
    config = get_datagen_config()
    checkpoint = args.checkpoint if args.checkpoint else None
    simulator = S.Sim(args.channel_type, config, checkpoint_lengths=checkpoint)
    loss_fn = L.get_loss_fn(spec, args.channel_type, config, epoch=epoch)

    t0 = time.time()
    beta_note = (f" beta={args.grad_norm_beta}"
                 if args.optimizer == "jaxley-polyak" else "")
    logger.info(
        f"{set_key} {args.loss} {args.optimizer} lr={max_lr}{beta_note} "
        f"ideal[{i_lo},{i_hi}) x init[{j_lo},{j_hi}) bounds=({b_lo},{b_hi})"
    )
    raw, d0, diag_arrays, f_at_truth = score_block(
        simulator, loss_fn, params, (i_lo, i_hi), (j_lo, j_hi),
        steps=args.steps, max_lr=max_lr, eps=args.polyak_eps,
        optimizer=args.optimizer, grad_norm_beta=args.grad_norm_beta,
        chunk=args.chunk,
        init_bounds=(b_lo, b_hi), logger=logger,
    )

    meta_common = {
        "polyak_eps": args.polyak_eps,
        "f_min": 0.0,
        "init_bounds": [float(b_lo), float(b_hi)],
        "n_amps": int(simulator.n_amps),
        "chunk": args.chunk,
        "checkpoint": checkpoint,
        "x64": True,
        "loss_set_version": L.LOSS_SET_VERSION,
        "success_radii": list(SUCCESS_RADII),
        "commit": _commit(),
        "device": _device(),
        "versions": _versions(),
        "wall_s": round(time.time() - t0, 1),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    row = build_row(
        probe=probe, raw=raw, d0=d0, diag_arrays=diag_arrays,
        f_at_truth=f_at_truth, loss_name=args.loss, set_key=set_key,
        params=params, optimizer=args.optimizer, meta_common=meta_common,
        save_raw=args.save_raw,
    )
    JS.append_row(out_path, row)

    d = row["diag"]
    # `region_score_se` is None with a single ideal seed -- an across-seed
    # spread needs at least two. The row is already written by this point, so
    # formatting None here would crash AFTER the work was done.
    se = row["region_score_se"]
    print(
        f"wrote {set_key} {args.loss} {args.optimizer}: "
        f"score {row['region_score']:.4f} "
        f"(baseline {row['region_score_baseline']:.4f}, "
        f"se {'n/a (1 ideal seed)' if se is None else f'{se:.4f}'}), "
        f"moving {d['moving_frac']:.3f}, escape {d['escape_frac']:.3f} "
        f"({time.time() - t0:.0f}s)"
    )
    if args.optimizer == "polyak" and (d["polyak_cap_frac"] or 0) > 0.5:
        print("  NOTE: the polyak cap bound on most steps, so this loss's score "
              "is NOT scale-invariant -- the adam arm is what settles the "
              "comparison. See the module docstring.")
    if args.optimizer == "jaxley-polyak":
        print(f"  NOTE: jaxley-polyak (beta={args.grad_norm_beta}) is not "
              "loss-scale invariant either -- its step length scales with the "
              "loss. Between-loss claims belong to the adam arm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
