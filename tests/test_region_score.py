"""Pins the region-of-success score's optimizer path, with no simulation.

``build_runner`` takes ``traces_of`` as an injected argument precisely so this
file can drive the whole scan / vmap / optax / scatter / diagnostics path
against an analytic problem whose answer is known in closed form. That makes
every assertion here exact rather than a tolerance, and the file runs in
seconds on a CPU.

The analytic problem
--------------------
``traces_of(x8, states6)`` returns ``x8`` tiled into a (2, 8) "trace", and the
loss is ``a/2 * ||traces[0] - ideal[0]||^2``. Since the frozen coordinates of
``x8`` are pinned at the truth by the scatter in ``build_runner``, that is

    f(x_sub) = a/2 * d^2,   ||g|| = a*d,   raw polyak step = f/||g||^2 = 1/(2a)

so the cap binds exactly when ``1/(2a) > max_lr`` -- a prediction, not a fit.
Uncapped, the displacement is ``step*||g|| = d/2``: the distance halves every
step regardless of ``a``, which is the scale-invariance the whole comparison
between differently-scaled losses depends on.

What the scale-invariance test establishes
------------------------------------------
``test_scale_invariance`` multiplies the loss by 1000 and requires:

* adam            -- same trajectory to ~1e-6 relative
* polyak UNCAPPED -- same trajectory to ~1e-11 relative
* polyak CAPPED   -- a DIFFERENT trajectory, by O(1)

The first two tolerances are measured, not nominal, and neither optimizer is
invariant *exactly*:

* adam's update is ``m/(sqrt(v) + eps)``. Scaling the loss by c scales m and
  sqrt(v) alike but NOT eps, so optax's default ``eps=1e-8`` leaves a residual.
  Measured here: 3e-8 relative at 5 steps growing to 7e-7 at 100.
* polyak's step is ``f/(||g||^2 + eps)``, exact to ~1e-11 while the run is above
  its convergence floor. Drive the same quadratic to machine precision and the
  two arms bottom out at different depths (7.5e-10 vs 1.5e-12 at 40 steps) and
  the comparison stops meaning anything. So the polyak check runs 20 steps, in
  the regime the real study lives in -- d_final there is O(0.1-2), nowhere near
  the floor.

Both residuals are ~1e-6 or smaller. The polyak CAP changes results by O(1).
That six-order-of-magnitude gap is the whole argument for the adam arm.

The third is not a bug being tolerated, it is the reason the study runs an adam
arm at all. Scaling a loss by c divides the raw polyak step by c, so a run that
was capped can become uncapped and the invariance is lost. On the real
landscapes the cap binds ~52% of steps, which is why "the two encoders have the
same mean cap fraction" is not sufficient evidence that their scores are
comparable. See gen_region_success_scores.__doc__.

    ../.venv/bin/python tests/test_region_score.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                       # encoding/
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts", "datagen"))

from gen_region_success_scores import (  # noqa: E402
    DEFAULT_LR,
    GRAD_NORM_BETA,
    LEGACY_TOLS,
    SUCCESS_RADII,
    _make_optimizer,
    ausc,
    build_runner,
    ladder_score,
    start_distances,
    success_table,
)

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print("FAIL:", msg)


# ---------------------------------------------------------------------------
# the analytic problem
# ---------------------------------------------------------------------------

DIM = 3
IDX = jnp.array([0, 3, 7])          # a non-contiguous subset, like a real triplet


def _traces_of(x8, states6):
    """(2, 8): the parameter vector itself, so the loss can be exactly quadratic."""
    del states6
    return jnp.stack([x8, x8])


def _loss(a):
    def loss(traces, ideal_traces):
        return 0.5 * a * jnp.sum((traces[0] - ideal_traces[0]) ** 2)

    return loss


def _batch(n=8, seed=0, spread=1.0):
    """(target8, states6, ideal_traces, x0) for `n` runs."""
    rng = np.random.default_rng(seed)
    target8 = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, 8)))
    states6 = jnp.zeros((n, 6))
    ideal = jax.vmap(_traces_of)(target8, states6)
    x0 = target8[:, IDX] + jnp.asarray(rng.uniform(-spread, spread, size=(n, DIM)))
    return target8, states6, ideal, x0


def _run(a, *, optimizer="polyak", max_lr=1.0, steps=100, n=8, seed=0, spread=1.0,
         grad_norm_beta=GRAD_NORM_BETA):
    runner = build_runner(_traces_of, _loss(a), IDX,
                          steps=steps, max_lr=max_lr, optimizer=optimizer,
                          grad_norm_beta=grad_norm_beta)
    return {k: np.asarray(v) for k, v in runner(*_batch(n, seed, spread)).items()}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_converges_and_diagnostics_are_clean():
    """Uncapped polyak halves the distance each step, so 100 steps is exact."""
    r = _run(a=1.0, max_lr=1.0)                     # raw step 1/(2a) = 0.5 < 1
    if not np.all(r["d_final"] < 1e-6):
        fail(f"did not converge: max d_final {r['d_final'].max():.3e}")
    if not np.allclose(r["d_best"], r["d_final"], atol=1e-12):
        fail("d_best != d_final on a monotone descent")
    if not np.all(r["cap_frac"] == 0.0):
        fail(f"cap bound when 1/(2a)=0.5 < max_lr=1: {r['cap_frac']}")
    for k, lim in (("nan_frac", 0.0), ("tail_disp", 1e-6), ("last_disp", 1e-6)):
        if not np.all(r[k] <= lim):
            fail(f"{k} = {r[k].max():.3e} > {lim}")
    if not np.all(r["f_final"] <= r["f0"]):
        fail("loss increased over the run")
    if not np.all(np.isfinite(r["max_absx"])):
        fail("max_absx non-finite")


def test_cap_fraction_matches_the_closed_form():
    """cap_frac is 1 or 0 exactly as 1/(2a) crosses max_lr."""
    for a, max_lr in ((0.1, 1.0), (1.0, 1.0), (0.4, 1.0), (1.0, 0.1)):
        want = 1.0 if 1.0 / (2 * a) > max_lr else 0.0
        got = _run(a=a, max_lr=max_lr)["cap_frac"]
        if not np.allclose(got, want):
            fail(f"a={a} max_lr={max_lr}: cap_frac {got.mean():.3f}, want {want} "
                 f"(raw step 1/(2a) = {1/(2*a):.3f})")


def test_f0_is_the_loss_at_the_start():
    r = _run(a=1.0)
    _, _, _, x0 = _batch()
    tgt, _, _, _ = _batch()
    want = 0.5 * 1.0 * np.sum((np.asarray(x0) - np.asarray(tgt[:, IDX])) ** 2, axis=1)
    if not np.allclose(r["f0"], want):
        fail(f"f0 wrong: {r['f0'][:3]} vs {want[:3]}")


def test_scale_invariance():
    """The claim the adam arm rests on. See the module docstring."""
    # adam: invariant up to its own eps (see the module docstring).
    for a in (0.1, 1.0):
        base = _run(a=a, optimizer="adam", max_lr=0.05)
        scaled = _run(a=1000 * a, optimizer="adam", max_lr=0.05)
        rel = np.max(np.abs(base["d_final"] - scaled["d_final"])
                     / np.abs(base["d_final"]))
        if rel > 1e-5:
            fail(f"adam scale residual {rel:.2e} at a={a}, want <1e-5")

    # polyak, uncapped both times (1/(2a) = 0.05 and 5e-5, both < max_lr=1), and
    # stopped at 20 steps -- above the convergence floor where the two arms would
    # otherwise bottom out at different depths and the comparison lose meaning.
    base, scaled = _run(a=10.0, steps=20), _run(a=10000.0, steps=20)
    if not np.all(base["cap_frac"] == 0) or not np.all(scaled["cap_frac"] == 0):
        fail("expected both polyak runs uncapped")
    rel = np.max(np.abs(base["d_final"] - scaled["d_final"]) / np.abs(base["d_final"]))
    if rel > 1e-9:
        fail(f"uncapped polyak scale residual {rel:.2e}, want <1e-9")
    if base["d_final"].max() > 1e-5:
        fail("the 20-step polyak reference did not get close enough to be a "
             "meaningful invariance check")

    # polyak, capped at a=0.1 (raw 5.0 > 1) but uncapped at a=100 (raw 5e-3).
    capped, uncapped = _run(a=0.1), _run(a=100.0)
    if not np.all(capped["cap_frac"] == 1.0):
        fail("expected the a=0.1 run to be capped")
    if np.allclose(capped["d_final"], uncapped["d_final"], rtol=1e-6):
        fail("capped polyak looks scale-invariant -- it must not be; if this "
             "ever passes, the C8 adam arm has lost its purpose")


def _reference_jaxley(a, x0, x_star, steps, lr_scale, beta):
    """The todo.md loop, in plain NumPy, on the analytic quadratic.

    Deliberately transcribed from the reference rather than refactored: the
    point of the test is that the optax arm agrees with THIS, line for line.

        loss_val, grad_val = grad_fn(opt_params)
        grad_norm = l2_norm(grad_val)
        grad_val = tree_map(lambda x: x / grad_norm ** beta, grad_val)
        if loss_val < best_loss: best_loss, best_params = loss_val, opt_params
        opt_state.hyperparams["learning_rate"] = loss_val / 3.0
        updates, opt_state = optimizer.update(grad_val, opt_state)
        opt_params = optax.apply_updates(opt_params, updates)

    `optax.sgd(lr)` emits `-lr * g`, so the update is `x - lr_scale*f * ghat`.
    Returns (x_final, x_best), with `x_best` including the final iterate the way
    `build_runner` does -- the scan never scores its own last position, so the
    runner adds one extra forward pass at the end and so does this.
    """
    x = np.array(x0, dtype=float)
    f_best, x_best = np.inf, x.copy()
    for _ in range(steps):
        d = x - x_star
        value = 0.5 * a * float(d @ d)
        grad = a * d
        gnorm = float(np.linalg.norm(grad))
        ghat = grad / gnorm ** beta if gnorm > 0 else grad
        if value < f_best:
            f_best, x_best = value, x.copy()
        x = x - (lr_scale * value) * ghat
    d = x - x_star
    f_final = 0.5 * a * float(d @ d)
    if f_final < f_best:
        f_best, x_best = f_final, x.copy()
    return x, x_best


def test_jaxley_polyak_matches_the_reference_loop():
    """The decisive test: the optax arm IS the loop in todo.md, step for step.

    The whole brief for this optimizer is "behaves exactly like the one below",
    so agreement with a literal transcription is the specification, and anything
    looser (does it converge, is it monotone) would pass for a wrong optimizer.
    """
    a, steps = 1.0, 40
    lr_scale, beta = DEFAULT_LR["jaxley-polyak"], GRAD_NORM_BETA
    got = _run(a=a, optimizer="jaxley-polyak", max_lr=lr_scale, steps=steps,
               grad_norm_beta=beta)
    target8, _, _, x0 = _batch()
    idx = np.asarray(IDX)
    worst_final = worst_best = 0.0
    for k in range(len(np.asarray(x0))):
        x_star = np.asarray(target8)[k][idx]
        xf, xb = _reference_jaxley(a, np.asarray(x0)[k], x_star, steps,
                                   lr_scale, beta)
        worst_final = max(worst_final,
                          abs(np.linalg.norm(xf - x_star) - got["d_final"][k]))
        worst_best = max(worst_best,
                         abs(np.linalg.norm(xb - x_star) - got["d_best"][k]))
    if worst_final > 1e-12 or worst_best > 1e-12:
        fail(f"jaxley-polyak does not reproduce the reference loop: "
             f"max |d_final diff| {worst_final:.3e}, "
             f"max |d_best diff| {worst_best:.3e}")

    # beta is a real knob, not a decoration: a different exponent must move the
    # trajectory, or the test above would pass for any beta.
    other = _run(a=a, optimizer="jaxley-polyak", max_lr=lr_scale, steps=steps,
                 grad_norm_beta=0.2)
    if np.allclose(other["d_final"], got["d_final"], rtol=1e-6):
        fail("grad_norm_beta does not reach the optimizer")


def test_jaxley_polyak_has_no_cap_and_stands_still_at_the_optimum():
    """cap_frac belongs to optax's polyak alone; and 0/0 must not become NaN.

    The reference loop divides by ``grad_norm ** beta`` with no guard, so at the
    exact optimum it produces a NaN that only the trailing ``zero_nans`` would
    catch. `_jaxley_polyak` uses `where(gnorm > 0, ...)` instead, which is exact
    everywhere the gradient is non-zero and stationary where it is zero.
    """
    r = _run(a=1.0, optimizer="jaxley-polyak", max_lr=DEFAULT_LR["jaxley-polyak"],
             steps=50)
    if not np.all(r["cap_frac"] == 0.0):
        fail(f"jaxley-polyak reported a cap fraction: {r['cap_frac']}")
    if not np.all(r["nan_frac"] == 0.0):
        fail(f"jaxley-polyak produced non-finite steps: {r['nan_frac']}")

    opt = _make_optimizer("jaxley-polyak", DEFAULT_LR["jaxley-polyak"], 1e-30, 10,
                          GRAD_NORM_BETA)
    x = jnp.zeros(DIM)
    upd, _ = opt.update(jnp.zeros(DIM), opt.init(x), x, value=jnp.array(0.0))
    if not (np.all(np.isfinite(np.asarray(upd))) and np.allclose(upd, 0.0)):
        fail(f"the update at the exact optimum is not a finite zero: {upd}")

    try:
        _make_optimizer("nope", 1.0, 1e-30, 10, GRAD_NORM_BETA)
        fail("an unknown optimizer name was accepted")
    except ValueError:
        pass


def test_jaxley_polyak_is_not_scale_invariant():
    """Pins the caveat, so it cannot be quietly forgotten in an analysis.

    The step length is ``lr_scale * f * ||g||^{1-beta}``, so multiplying the
    loss by c scales the step by c^{2-beta} = c^{1.2} at beta = 0.8. Textbook
    polyak is beta = 2, where the two factors of c cancel exactly. This arm
    therefore carries the same between-loss comparability caveat polyak does,
    and adam remains C8's reference.
    """
    lr = DEFAULT_LR["jaxley-polyak"]
    base = _run(a=1.0, optimizer="jaxley-polyak", max_lr=lr, steps=30)
    scaled = _run(a=1000.0, optimizer="jaxley-polyak", max_lr=lr, steps=30)
    if np.allclose(base["d_final"], scaled["d_final"], rtol=1e-3):
        fail("jaxley-polyak looks scale-invariant -- it must not be; if this "
             "ever passes, the note in its docstring and in the task doc is "
             "wrong and every between-loss claim under it needs revisiting")

    # beta = 2 IS invariant, which is the closed-form statement of why.
    b2 = _run(a=1.0, optimizer="jaxley-polyak", max_lr=1.0, steps=20,
              grad_norm_beta=2.0)
    b2s = _run(a=1000.0, optimizer="jaxley-polyak", max_lr=1.0, steps=20,
               grad_norm_beta=2.0)
    rel = np.max(np.abs(b2["d_final"] - b2s["d_final"]) / np.abs(b2["d_final"]))
    if rel > 1e-9:
        fail(f"beta=2 should be exactly scale-invariant, residual {rel:.2e}")


def test_escape_and_nonfinite_are_detected():
    """A wildly over-long step must leave a trace in max_absx."""
    # rmsprop normalizes by sqrt(E[g^2]), so its step is ~lr regardless of the
    # loss scale -- a tiny `a` converges rather than diverging. An oversized lr
    # is what escapes.
    r = _run(a=1.0, optimizer="rmsprop", max_lr=50.0, steps=20, spread=1.0)
    if not np.all(r["max_absx"] > 2.6):
        fail(f"escape not visible: max_absx {r['max_absx']}")
    clean = _run(a=1.0)
    if np.any(clean["max_absx"] > 2.6):
        fail("a converging run was flagged as escaping")


def test_frozen_coordinates_never_move():
    """The scatter must leave the 5 frozen parameters pinned at truth.

    Checked through the objective rather than by inspection: the loss at the
    end is the subset error alone, so if a frozen coordinate had drifted the
    converged loss would not be ~0.
    """
    r = _run(a=1.0)
    if not np.all(r["f_final"] < 1e-12):
        fail(f"converged loss not ~0: {r['f_final'].max():.3e} -- a frozen "
             "coordinate moved, or the scatter index is wrong")


def test_aggregation_helpers():
    """ladder_score / success_table / ausc on a matrix with a known answer."""
    raw = np.array([[0.05, 0.2, 0.5, 2.0],       # ideal seed 0
                    [0.05, 0.05, 0.05, 0.05]])   # ideal seed 1
    s = success_table(raw, SUCCESS_RADII)
    want = {0.09375: [0.25, 1.0], 0.1875: [0.25, 1.0], 0.375: [0.5, 1.0],
            0.75: [0.75, 1.0], 1.5: [0.75, 1.0]}
    for r, w in want.items():
        if not np.allclose(s[r], w):
            fail(f"success@{r}: {s[r]} vs {w}")
    per_seed, score = ladder_score(raw)
    if not np.allclose(per_seed, [np.mean([0.25, 0.25, 0.5, 0.75, 0.75]), 1.0]):
        fail(f"ladder per-seed wrong: {per_seed}")
    if not np.isclose(score, per_seed.mean()):
        fail("ladder score is not the mean of the per-seed values")
    # ausc = 1 - E[min(d,R)]/R
    if not np.isclose(ausc(raw, 1.5),
                      1 - np.minimum(raw, 1.5).mean() / 1.5):
        fail("ausc closed form wrong")
    # A uniform ladder must converge to it -- the identity that rules a uniform
    # ladder OUT as the headline, since it is a censored mean error.
    rungs = (np.arange(4000) + 0.5) * 1.5 / 4000
    if not np.isclose(np.mean([(raw < r).mean() for r in rungs]),
                      ausc(raw, 1.5), atol=1e-3):
        fail("uniform ladder does not converge to ausc")


def test_legacy_radii_unchanged():
    """LEGACY_TOLS must stay bit-equal to gen_optimization_scores.SUCCESS_TOLS,
    or new rows stop being comparable with the score files already on disk."""
    from gen_optimization_scores import SUCCESS_TOLS

    if tuple(LEGACY_TOLS) != tuple(SUCCESS_TOLS):
        fail(f"LEGACY_TOLS {LEGACY_TOLS} != SUCCESS_TOLS {SUCCESS_TOLS}")


def test_start_distances_are_pair_invariant():
    """d0 depends only on (ideal seed, init seed, dim) -- never on the loss."""
    targets = jnp.asarray(np.random.default_rng(1).uniform(-1, 1, size=(5, 8)))
    starts = jnp.asarray(np.random.default_rng(2).uniform(-1.5, 1.5, size=(7, DIM)))
    a = start_distances(targets, starts, np.asarray(IDX))
    b = start_distances(targets, starts, np.asarray(IDX))
    if not np.array_equal(a, b):
        fail("start_distances is not deterministic")
    if a.shape != (5, 7):
        fail(f"start_distances shape {a.shape}, want (5, 7)")
    manual = np.linalg.norm(np.asarray(targets)[0][np.asarray(IDX)]
                            - np.asarray(starts)[3])
    if not np.isclose(a[0, 3], manual):
        fail("start_distances value wrong")


def _synthetic_file(tmpdir, *, cap_gap, break_pairing=False, n_sets=16):
    """A minimal two-loss, two-optimizer score file built through build_row.

    Deliberately constructed to PASS every check when nothing is wrong, so that
    a single injected defect is what makes it fail:

    * `tail_disp` well under MOVING_EPS, or C4 flags the whole file as
      budget-limited;
    * per-set difficulty drawn over a wide range and 10x100 cells per row, so
      the between-set signal dominates binomial noise and C8's
      spearman(S_polyak, S_adam) is high;
    * both optimizer arms share the set's difficulty, as they would in reality;
    * a real mask penalty, so both arms reach significance and C8's
      "the two arms agree" test has something to agree about;
    * a little jitter on cap_frac, or C7's spearman gets a constant input;
    * non-converged runs left at a distance BELOW the random-start mean (1.72).
      Scattering them at 2.0 instead makes the fixture worse than guessing and
      C3 fails on a file with no defect in it -- real failed optimizations stop
      early, they do not systematically flee the truth.
    """
    import jsonl_store as JS
    from gen_region_success_scores import MOVING_EPS, build_row

    keys = ["d_best", "f0", "f_final", "f_best", "cap_frac", "last_disp",
            "tail_disp", "max_absx", "nan_frac"]
    path = os.path.join(tmpdir, "synthetic.jsonl")
    rng = np.random.default_rng(3)
    shape = (10, 100)
    for si in range(n_sets):
        d0 = np.abs(rng.normal(1.72, 0.63, size=shape))
        base = rng.uniform(0.15, 0.70)                    # wide: the C8 signal
        for li, ln in enumerate(("encoder-zscore-mask", "encoder-zscore-2")):
            p_conv = np.clip(base + (0.0 if li else -0.06), 0.02, 0.95)
            for op in ("polyak", "adam"):
                raw = np.where(rng.random(shape) < p_conv,
                               np.abs(rng.normal(0.15, 0.10, shape)),
                               np.abs(rng.normal(1.4, 0.50, shape)))
                diag = {k: np.abs(rng.normal(0.02, 0.005, shape)) for k in keys}
                diag["tail_disp"] = np.abs(rng.normal(MOVING_EPS / 20, 1e-4, shape))
                diag["last_disp"] = diag["tail_disp"]
                diag["cap_frac"] = np.clip(
                    rng.normal(0.5 + (cap_gap if li else 0.0), 0.01, shape), 0, 1)
                diag["max_absx"] = np.full(shape, 1.0)
                diag["nan_frac"] = np.zeros(shape)
                dd = d0 * (1.001 if (break_pairing and li) else 1.0)
                probe = {"params": ["Leak_gLeak", "Leak_eLeak", "Na_gNa"],
                         "ideal_seeds": [0, shape[0]], "init_seeds": [0, shape[1]],
                         "init_bounds": [-1.5, 1.5],
                         "loss": ["encoder_loss", ln], "encoder_epoch": None,
                         "optimizer": op, "steps": 100, "max_learning_rate": 1.0}
                JS.append_row(path, build_row(
                    probe=probe, raw=raw, d0=dd, diag_arrays=diag, f_at_truth=0.0,
                    loss_name=ln, set_key=f"synthetic-set-{si}",
                    params=probe["params"], optimizer=op,
                    meta_common={"init_bounds": [-1.5, 1.5]}, save_raw=False))
    return path


def test_check_reports_numpy_valued_failures():
    """A FAIL computed with numpy must set the exit code, not just print.

    `np.False_ is False` is False, so a check whose verdict comes from
    np.median (C6 does) printed FAIL and still exited 0 until `report` started
    coercing. That is invisible in a log full of PASS lines, so it is pinned.
    """
    import tempfile

    from gen_region_success_scores import check_scores

    if np.False_ is False:                       # pragma: no cover
        fail("numpy bool is now Python bool; this regression test is moot")

    quiet = lambda *a, **k: None                 # noqa: E731
    pair = ("encoder-zscore-mask", "encoder-zscore-2")
    with tempfile.TemporaryDirectory() as tmp:
        healthy = _synthetic_file(tmp, cap_gap=0.0)
        if check_scores(healthy, pair, logger=quiet) != 0:
            fail("a healthy synthetic file did not pass --check")
    with tempfile.TemporaryDirectory() as tmp:
        capped = _synthetic_file(tmp, cap_gap=0.30)
        if check_scores(capped, pair, logger=quiet) != 1:
            fail("C6 cap-parity failure did not set the exit code "
                 "(numpy bool not coerced in report())")
    with tempfile.TemporaryDirectory() as tmp:
        unpaired = _synthetic_file(tmp, cap_gap=0.0, break_pairing=True)
        if check_scores(unpaired, pair, logger=quiet) != 1:
            fail("C1 did not catch a broken paired design")


def test_resume_skips_a_completed_unit():
    """A re-submitted unit must skip, and must skip BEFORE building the Sim.

    CLAUDE.md's "resume, don't regenerate" is what makes re-running
    `submit_study.sh` cheap, and the ledger only records successes -- so a unit
    whose row already exists gets handed to a task again on every resubmit. If
    the skip were checked after `S.Sim(...)` and the encoder load, each skip
    would still cost ~90 s of startup across 336 units.

    Also pins that `--force` overrides the skip, and that a run differing only
    in a KEY_FIELD (here max_learning_rate) is NOT treated as done -- the bug
    that made an lr sweep silently write one row instead of four.
    """
    import tempfile

    import jsonl_store as JS
    from gen_region_success_scores import KEY_FIELDS, main

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "resume.jsonl")
        probe = {
            "params": ["Leak_gLeak", "Leak_eLeak", "Na_gNa"],
            "ideal_seeds": [0, 10], "init_seeds": [0, 100],
            "init_bounds": [-1.5, 1.5],
            "loss": ["encoder_loss", "2026_08_12_15_20_52_343885"],
            "encoder_epoch": None, "optimizer": "polyak", "steps": 100,
            "max_learning_rate": 1.0,
        }
        JS.append_row(out, {**probe, "set": "gLeak+eLeak+gNa",
                            "loss_name": "encoder-zscore-mask"})

        argv = ["--set_key", "gLeak+eLeak+gNa", "--loss", "encoder-zscore-mask",
                "--ideal_seeds", "0", "10", "--init_seeds", "0", "100",
                "--init_bounds", "-1.5", "1.5", "--steps", "100",
                "--optimizer", "polyak", "--out", out]

        # Poison Sim so that reaching it is a hard failure rather than a slow test.
        import sim as S
        real_sim = S.Sim
        S.Sim = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("resume check ran AFTER Sim construction"))
        try:
            if main(argv) != 0:
                fail("a completed unit did not skip cleanly")
            # A different learning rate is a different configuration.
            try:
                main(argv + ["--max_learning_rate", "0.05"])
                fail("a differing max_learning_rate was treated as already done "
                     "-- it must be part of KEY_FIELDS")
            except AssertionError as exc:
                if "resume check ran AFTER" not in str(exc):
                    raise
            # --force must bypass the skip too.
            try:
                main(argv + ["--force"])
                fail("--force did not bypass the resume skip")
            except AssertionError as exc:
                if "resume check ran AFTER" not in str(exc):
                    raise
        finally:
            S.Sim = real_sim

        if len(JS.load_rows(out)) != 1:
            fail("the skip path wrote a row")
        if JS.row_key(probe, KEY_FIELDS) not in JS.existing_keys(out, KEY_FIELDS):
            fail("row_key does not match the row it was built from")

        # Adding `grad_norm_beta` to KEY_FIELDS must not orphan the rows written
        # before it existed. `row_key` reads a missing field as None, so a row
        # with no such field and a row carrying an explicit None must key alike;
        # that is what lets a resume still skip the 784 rows already on disk.
        # It also means non-jaxley runs MUST write None rather than 0.8.
        if (JS.row_key(probe, KEY_FIELDS)
                != JS.row_key({**probe, "grad_norm_beta": None}, KEY_FIELDS)):
            fail("an explicit grad_norm_beta=None does not key like an absent "
                 "field -- every pre-existing row would be recomputed")
        if (JS.row_key(probe, KEY_FIELDS)
                == JS.row_key({**probe, "grad_norm_beta": 0.8}, KEY_FIELDS)):
            fail("grad_norm_beta is not actually part of the resume key")


def test_nonfinite_runs_are_counted_not_propagated():
    """One diverged run must not nan out a whole row's diagnostics.

    `np.percentile` and `np.minimum` both propagate nan, so a single
    non-finite d_final in a thousand would turn every quantile and `ausc` into
    nan -- while `region_score` stayed healthy-looking, because `nan < r` is
    already False. In a 336-row file that is easy to miss, so the failure is
    made countable (`n_nonfinite`) instead of contagious.
    """
    from gen_region_success_scores import ausc, ladder_score, quantiles

    raw = np.array([[0.1, 0.2, np.nan, 3.0], [0.05, np.inf, 0.5, 1.0]])

    # the ladder already treats non-finite as a failure; pin that too
    per_seed, score = ladder_score(raw, (0.375,))
    if not np.allclose(per_seed, [0.5, 0.25]):
        fail(f"non-finite runs not counted as failures: {per_seed}")

    a = ausc(raw, 1.5)
    if not np.isfinite(a):
        fail("ausc propagated a non-finite distance instead of censoring it")
    # censored to R, so it must equal the value with nan/inf replaced by 1.5
    want = 1.0 - np.minimum(np.nan_to_num(raw, nan=1.5, posinf=1.5), 1.5).mean() / 1.5
    if not np.isclose(a, want):
        fail(f"ausc censoring wrong: {a} vs {want}")

    q = quantiles(raw)
    if q.get("n_nonfinite") != 2:
        fail(f"n_nonfinite = {q.get('n_nonfinite')}, want 2")
    if not all(np.isfinite(q[k]) for k in ("mean", "std", "p50", "max")):
        fail(f"quantiles propagated nan: {q}")

    empty = quantiles(np.array([np.nan, np.inf]))
    if empty["n_nonfinite"] != 2 or empty["mean"] is not None:
        fail(f"all-non-finite case not handled: {empty}")


if __name__ == "__main__":
    test_converges_and_diagnostics_are_clean()
    test_cap_fraction_matches_the_closed_form()
    test_f0_is_the_loss_at_the_start()
    test_scale_invariance()
    test_jaxley_polyak_matches_the_reference_loop()
    test_jaxley_polyak_has_no_cap_and_stands_still_at_the_optimum()
    test_jaxley_polyak_is_not_scale_invariant()
    test_escape_and_nonfinite_are_detected()
    test_frozen_coordinates_never_move()
    test_aggregation_helpers()
    test_legacy_radii_unchanged()
    test_start_distances_are_pair_invariant()
    test_check_reports_numpy_valued_failures()
    test_resume_skips_a_completed_unit()
    test_nonfinite_runs_are_counted_not_propagated()

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES[:25]:
            print("  ", f)
        sys.exit(1)
    print("all region-score tests passed")
