"""Pins the two facts the whole study rests on.

1. ``sim.ideal_params_for_seed`` reproduces ``generate_losses.py``'s PRNG stream
   exactly. If it does not, seed `s` means a different neuron here than in the
   landscapes already on disk, and the convexity<->optimization join -- which is
   by seed index -- silently compares unrelated things.

2. ``Sim.traces_of`` (all 8 parameters plus the 6 gating states as ``data_set``
   values) is numerically identical to the legacy ``comp.set`` + ``init_states``
   path. If it is not, the two generators descend different landscapes.

Fact 2 needs real simulation, so this file takes ~30 s. Everything else is
instant.

    ../.venv/bin/python tests/test_sim.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # encoding/

import sim as S  # noqa: E402
from helper import (  # noqa: E402
    compress_traces,
    get_datagen_config,
    get_param_bounds,
    insert_channels,
    set_const_params,
)
from jaxley import Compartment  # noqa: E402
from train_comp import (  # noqa: E402
    _backward_transform_params,
    _get_transform_list,
    set_frozen_params,
)

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


CONFIG = get_datagen_config()
PB = get_param_bounds(CONFIG["param_bounds"], "Pospischil")


# ---------------------------------------------------------------------------
# 1. PRNG compatibility with the legacy recipe
# ---------------------------------------------------------------------------


def _legacy_ideal_params(seed, param_bounds):
    """Verbatim transcription of generate_losses.py:159-177.

    Deliberately duplicated rather than imported: this is the spec, and it must
    keep working even if that script is later refactored.
    """
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)  # spent on get_traces_to_match_rand
    ideal_params_dict = {}
    for param_name, bounds in param_bounds.items():
        key, subkey = jax.random.split(key)
        bound_center = (bounds[0] + bounds[1]) / 2
        q1 = (bounds[0] + bound_center) / 2
        q2 = (bound_center + bounds[1]) / 2
        val = jax.random.uniform(subkey, minval=q1, maxval=q2)
        ideal_params_dict[param_name] = val
    return ideal_params_dict


def test_prng_bit_compat():
    for seed in range(20):
        want = _legacy_ideal_params(seed, PB)
        got = S.ideal_params_for_seed(seed, PB)
        if list(got) != list(want):
            fail(f"seed {seed}: key order {list(got)} != {list(want)}")
        for name in want:
            a, b = float(got[name]), float(want[name])
            if a != b:  # must be BIT identical, not close
                fail(f"seed {seed} {name}: {a!r} != {b!r}")

    # Different seeds must give different neurons (guards a dropped fold_in).
    p0 = S.ideal_params_for_seed(0, PB)
    p1 = S.ideal_params_for_seed(1, PB)
    if all(float(p0[k]) == float(p1[k]) for k in p0):
        fail("seeds 0 and 1 gave identical parameters")

    # And every draw must sit inside the middle half of its bounds.
    for seed in (0, 5, 99):
        p = S.ideal_params_for_seed(seed, PB)
        for name, bounds in PB.items():
            centre = (bounds[0] + bounds[1]) / 2
            q1, q2 = (bounds[0] + centre) / 2, (centre + bounds[1]) / 2
            v = float(p[name])
            if not (q1 <= v <= q2):
                fail(f"seed {seed} {name}={v} outside middle half [{q1}, {q2}]")


def test_param_order():
    if list(PB) != S.ALL_PARAMS:
        fail(f"config order {list(PB)} != ALL_PARAMS {S.ALL_PARAMS}")
    S.check_param_order(PB)  # must not raise

    # canonical() sorts and rejects junk, so the cache key is order-independent.
    if S.canonical(["K_gK", "Na_gNa"]) != ["Na_gNa", "K_gK"]:
        fail(f"canonical did not reorder: {S.canonical(['K_gK', 'Na_gNa'])}")
    if S.canonical(["vt", "Leak_gLeak"]) != ["Leak_gLeak", "vt"]:
        fail("canonical did not reorder vt/gLeak")
    for bad in (["nope"], ["Na_gNa", "Na_gNa"]):
        try:
            S.canonical(bad)
            fail(f"canonical accepted {bad}")
        except ValueError:
            pass

    idx = S.param_indices(["Km_gKm", "Na_gNa"])  # canonical: Na_gNa, Km_gKm
    if list(np.asarray(idx)) != [2, 4]:
        fail(f"param_indices gave {np.asarray(idx)}, want [2, 4]")


def test_scatter_roundtrip():
    """target8.at[IDX].set(target8[IDX]) is a no-op for every parameter set."""
    rng = np.random.default_rng(0)
    x8 = jnp.asarray(rng.normal(size=8))
    from itertools import combinations

    for combo in combinations(S.ALL_PARAMS, 3):
        idx = S.param_indices(list(combo))
        back = x8.at[idx].set(x8[idx])
        if not bool(jnp.all(back == x8)):
            fail(f"scatter roundtrip failed for {combo}")
        # The 3-dim norm must equal the 8-dim norm when only those 3 differ.
        pert = x8.at[idx].add(0.5)
        n3 = float(jnp.linalg.norm(pert[idx] - x8[idx]))
        n8 = float(jnp.linalg.norm(pert - x8))
        if abs(n3 - n8) > 1e-12:
            fail(f"norm mismatch for {combo}: {n3} vs {n8}")


def test_start_for_stability():
    """Widening the init-seed range must not move existing starts."""
    for dim in (1, 3, 8):
        first = jnp.stack([S.start_for(j, dim) for j in range(32)])
        again = jnp.stack([S.start_for(j, dim) for j in range(64)])[:32]
        if not bool(jnp.all(first == again)):
            fail(f"start_for unstable when range extended (dim={dim})")
        if float(jnp.min(first)) < -3.0 or float(jnp.max(first)) > 3.0:
            fail(f"start_for out of [-3, 3] (dim={dim})")
        # Distinct starts, and independent of the ideal seed by construction.
        if len({tuple(np.asarray(r)) for r in first}) != 32:
            fail(f"start_for produced duplicates (dim={dim})")


def test_shapes():
    if S.trace_len(CONFIG) != 1300:
        fail(f"trace_len {S.trace_len(CONFIG)} != 1300")
    if abs(S.dt_eff(CONFIG) - 0.1) > 1e-12:
        fail(f"dt_eff {S.dt_eff(CONFIG)} != 0.1")


# ---------------------------------------------------------------------------
# 2. data_set path == legacy comp.set path
# ---------------------------------------------------------------------------


def _legacy_traces(seed, amp_subset=None):
    """The legacy comp.set + init_states path, transcribed from generate_losses.py."""
    cfg = CONFIG
    dt, t_max = cfg["dt"], cfg["t_max"]
    cp = cfg["current"]

    comp = insert_channels(Compartment(), "Pospischil")
    comp = set_const_params(cfg["param_consts"], "Pospischil", comp)
    comp.record("v")

    ideal = _legacy_ideal_params(seed, PB)
    transforms = _get_transform_list(PB)

    # The legacy code freezes the non-trainable subset via comp.set. Here every
    # parameter is "frozen" at truth, which is exactly the configuration the
    # ideal trace is generated in.
    comp = set_frozen_params(PB, comp, ideal)
    comp.init_states(delta_t=dt)

    x8 = _backward_transform_params([ideal[p] for p in S.ALL_PARAMS], transforms)
    phys = [ideal[p] for p in S.ALL_PARAMS]

    amps = amp_subset if amp_subset is not None else jnp.asarray(
        np.load(f"{cfg['save_path']}/Pospischil_amps.npz")["arr_0"]
    )

    def one(amp):
        current = jx.step_current(
            i_delay=cp["delay"], i_dur=cp["duration"], i_amp=amp,
            delta_t=dt, t_max=t_max,
        )
        ps = None
        for i, name in enumerate(S.ALL_PARAMS):
            ps = comp.data_set(name, phys[i], ps)
        return jnp.array(
            jx.integrate(
                comp, data_stimuli=comp.data_stimulate(current, None),
                param_state=ps, delta_t=dt, t_max=t_max,
            ).flatten()
        )

    volts = jax.vmap(one)(amps)
    return jnp.nan_to_num(compress_traces(volts, cfg["t_filter"], dt, cfg["avg_scale"])), x8


def test_data_set_matches_legacy():
    """The headline equivalence, on real simulations."""
    sim = S.Sim("Pospischil", CONFIG)

    for seed in (0, 1, 7):
        legacy, x8 = _legacy_traces(seed)
        states6 = sim.init_states_for_seeds([seed])[0]
        got = sim.traces_of(x8, states6)

        if got.shape != legacy.shape:
            fail(f"seed {seed}: shape {got.shape} != legacy {legacy.shape}")
            continue
        diff = float(jnp.max(jnp.abs(got - legacy)))
        scale = float(jnp.max(jnp.abs(legacy)))
        # Roundoff over a 6001-step integration, not a semantic difference.
        if diff > 1e-6:
            fail(f"seed {seed}: max|diff| {diff:.3e} vs legacy (scale {scale:.1f})")

        # Sanity: the trace is a real spiking trace, not zeros.
        if scale < 10.0:
            fail(f"seed {seed}: legacy trace suspiciously flat, max|v|={scale}")

    # states_for_params agrees with init_states_for_seeds on the same neuron.
    targets = sim.targets_for_seeds([3])
    a = sim.init_states_for_seeds([3])[0]
    b = sim.states_for_params(targets[0])
    if float(jnp.max(jnp.abs(a - b))) > 1e-9:
        fail(f"states_for_params != init_states_for_seeds: {a} vs {b}")


def test_omitting_states_is_wrong():
    """Guard the reason the gating states are passed at all.

    If a future refactor drops them, this test fails loudly rather than the
    study quietly simulating the wrong neuron.
    """
    sim = S.Sim("Pospischil", CONFIG)
    legacy, x8 = _legacy_traces(0)
    good = sim.traces_of(x8, sim.init_states_for_seeds([0])[0])
    # jaxley's own defaults, i.e. what you get by not passing states at all.
    default_states = jnp.asarray([0.2] * len(S.GATING_STATES))
    bad = sim.traces_of(x8, default_states)
    err = float(jnp.max(jnp.abs(bad - legacy)))
    if err < 1.0:
        fail(
            "omitting the gating states changed the trace by only "
            f"{err:.3e} mV -- the equivalence test above may be vacuous"
        )
    if float(jnp.max(jnp.abs(good - legacy))) > 1e-6:
        fail("with states, traces still differ from legacy")


def test_batching():
    """traces_of vmaps over parameters, over seeds, and over both."""
    sim = S.Sim("Pospischil", CONFIG)
    seeds = [0, 1]
    targets = sim.targets_for_seeds(seeds)          # (2, 8)
    states = sim.init_states_for_seeds(seeds)       # (2, 6)

    # one ideal seed, several parameter vectors (the grid-sweep pattern)
    grid = jnp.stack([targets[0], targets[0] + 0.05])
    swept = jax.vmap(sim.traces_of, in_axes=(0, None))(grid, states[0])
    if swept.shape != (2, sim.n_amps, sim.n_t):
        fail(f"sweep vmap shape {swept.shape}")
    single = sim.traces_of(targets[0], states[0])
    # Tolerance is 1e-6 absolute (~1e-8 relative), not exact equality: batched
    # and unbatched execution use different XLA kernels, so a 6001-step implicit
    # integration diverges at roundoff -- measured 3e-9 to 5e-9 absolute on
    # traces of magnitude ~80 mV. Anything larger would mean a real batching bug.
    if float(jnp.max(jnp.abs(swept[0] - single))) > 1e-6:
        fail("sweep vmap row 0 != unbatched call")

    # paired over both axes (the optimization pattern)
    paired = jax.vmap(sim.traces_of, in_axes=(0, 0))(targets, states)
    if paired.shape != (2, sim.n_amps, sim.n_t):
        fail(f"paired vmap shape {paired.shape}")
    for i in range(2):
        one = sim.traces_of(targets[i], states[i])
        if float(jnp.max(jnp.abs(paired[i] - one))) > 1e-6:
            fail(f"paired vmap row {i} != unbatched call")

    # differentiable, and the gradient reaches the parameters
    def obj(x8):
        return jnp.mean(sim.traces_of(x8, states[0]) ** 2)

    g = jax.grad(obj)(targets[0])
    if not bool(jnp.all(jnp.isfinite(g))):
        fail(f"gradient not finite: {g}")
    if float(jnp.max(jnp.abs(g))) == 0.0:
        fail("gradient identically zero")


def test_amps():
    sim = S.Sim("Pospischil", CONFIG)
    if sim.n_amps < 2:
        fail(f"n_amps {sim.n_amps} < 2 -- spike_w1_multi's f-I channel needs >1")
    amps = np.asarray(sim.amps)
    if not np.all(np.diff(amps) > 0):
        fail("amps not increasing")
    if not (0.0 < amps.min() < 0.01):
        fail(f"amps out of expected microamp range: {amps.min()}..{amps.max()}")


if __name__ == "__main__":
    test_prng_bit_compat()
    test_param_order()
    test_scatter_roundtrip()
    test_start_for_stability()
    test_shapes()
    test_amps()
    test_data_set_matches_legacy()
    test_omitting_states_is_wrong()
    test_batching()

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES[:25]:
            print("  ", f)
        sys.exit(1)
    print("all sim tests passed")
