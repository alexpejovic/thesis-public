"""The one simulator both score generators share.

Why this module exists
----------------------
The convexity sweep and the optimization run must descend *the same* loss
landscape. If they differ in how the non-optimized ("frozen") parameters or the
initial gating states are set, they describe different problems and every
correlation between their two scores is noise. So the simulator lives here,
once, and both generators import it.

Why everything goes through ``data_set``
----------------------------------------
The legacy path (``generate_losses.py``, ``train_comp.py``) bakes the frozen
parameters in with ``comp.set(...)`` followed by ``comp.init_states(delta_t)``.
Both mutate the compartment's ``nodes`` DataFrame on the host, so the ideal seed
becomes part of the *compiled program* -- one XLA executable per seed. That is
fatal for the optimization score, which needs to vmap thousands of
(ideal seed, init seed) pairs through one compiled graph.

Every one of the 8 parameters is a ``nodes`` column, and so are the 6 channel
gating states, and ``jx.integrate`` splices ``param_state`` into the pstate
before ``get_all_parameters`` / ``get_all_states``. So both can be passed as
*traced values* instead, which makes the ideal seed an ordinary batched array.

The gating states are not optional. Measured against the legacy path on seed 0:

    data_set the 8 parameters only ................ max|diff| = 1.4e+01 mV
    data_set the 8 parameters AND the 6 states .... max|diff| = 5.3e-10 mV

``init_states`` computes the steady-state gating variables from whatever is in
``nodes`` at the time, and ``get_all_states`` does *not* re-run
``channel.init_state``. So leaving the states out silently simulates a neuron
whose gates were initialised for jaxley's default ``vt``/``taumax`` rather than
this seed's -- a 14 mV error, not a rounding difference. ``init_states_for_seeds``
therefore computes them on the host (pure pandas, no simulation, microseconds
per seed) and they ride along as 6 more ``data_set`` values.

``tests/test_sim.py`` pins both facts: the PRNG stream against the legacy
recipe, and the traces against the legacy ``comp.set`` path.
"""

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
from jax import Array
from jaxley import Compartment

from helper import (
    ChannelType,
    compress_traces,
    get_datagen_config,
    get_param_bounds,
    insert_channels,
    set_const_params,
)
from train_comp import (
    _backward_transform_params,
    _forward_transform_params,
    _get_transform_list,
)

# The canonical parameter order. This list IS the dict<->array contract for the
# whole study: a flat parameter vector's element `i` is ALL_PARAMS[i], on disk
# and in memory, everywhere. It is asserted against the config at import time by
# `check_param_order` so that reordering `param_bounds` in datagen.yaml fails
# loudly instead of silently transposing landscapes.
ALL_PARAMS = [
    "Leak_gLeak",
    "Leak_eLeak",
    "Na_gNa",
    "K_gK",
    "Km_gKm",
    "CaL_gCaL",
    "Km_taumax",
    "vt",
]

# Channel gating states, in the order they are passed as data_set values. Any
# fixed order works -- unlike ALL_PARAMS this order is never used to index a
# stored array by position alone -- but it is kept explicit for the same reason.
GATING_STATES = ["Na_m", "Na_h", "K_n", "Km_p", "CaL_q", "CaL_r"]


def check_param_order(param_bounds: dict) -> None:
    """Fail loudly if the config's parameter order drifts from ALL_PARAMS."""
    got = list(param_bounds)
    if got != ALL_PARAMS:
        raise RuntimeError(
            "parameter order changed: datagen.yaml gives\n"
            f"  {got}\nbut sim.ALL_PARAMS is\n  {ALL_PARAMS}\n"
            "Every stored landscape and score vector is positional, so this "
            "would silently mis-pair grid axes with true values."
        )


def canonical(params) -> list[str]:
    """Sort a requested parameter subset into canonical order.

    Called at every CLI boundary so `--params K_gK Na_gNa` and
    `--params Na_gNa K_gK` produce the same axis order and the same cache key.
    """
    want = list(params)
    unknown = [p for p in want if p not in ALL_PARAMS]
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}; known: {ALL_PARAMS}")
    if len(set(want)) != len(want):
        raise ValueError(f"duplicate parameter(s) in {want}")
    return [p for p in ALL_PARAMS if p in set(want)]


def param_indices(params) -> Array:
    """Indices into an 8-vector for a canonical parameter subset."""
    return jnp.array([ALL_PARAMS.index(p) for p in canonical(params)])


def trace_len(config: dict) -> int:
    """Samples in a compressed trace: 1300 for the standard config."""
    return int(
        (config["t_max"] - 2 * config["t_filter"])
        / (config["dt"] * config["avg_scale"])
    )


def dt_eff(config: dict) -> float:
    """Sample spacing of a compressed trace in ms: 0.1 for the standard config."""
    return config["dt"] * config["avg_scale"]


def ideal_params_for_seed(seed: int, param_bounds: dict) -> dict[str, Array]:
    """The "true" neuron for one ideal-trace seed, in physical units.

    Draws every parameter -- optimized and frozen alike -- uniformly from the
    middle half of its bounds, so the target is never pinned against a sigmoid
    tail where the transform has no gradient.

    This reproduces ``generate_losses.py`` exactly, INCLUDING the leading
    ``split`` whose subkey that script spends on picking a random amplitude set
    and then discards. Dropping it would shift the entire PRNG stream, so seed
    `s` would mean a different neuron here than in every landscape already on
    disk -- and the join between convexity and optimization scores is by seed
    index. Pinned by ``tests/test_sim.py``.
    """
    key = jax.random.PRNGKey(seed)
    key, _ = jax.random.split(key)  # spent on `amps` upstream; keep the stream aligned
    out = {}
    for name, bounds in param_bounds.items():
        key, subkey = jax.random.split(key)
        centre = (bounds[0] + bounds[1]) / 2
        q1 = (bounds[0] + centre) / 2
        q2 = (centre + bounds[1]) / 2
        out[name] = jax.random.uniform(subkey, minval=q1, maxval=q2)
    return out


def start_for(init_seed: int, dim: int, lo: float = -3.0, hi: float = 3.0) -> Array:
    """One optimization starting point, in transformed space.

    Depends only on ``(init_seed, dim)``, deliberately: start `j` is then
    identical across every ideal seed, every loss and every parameter set of the
    same dimensionality. That paired design cancels start difficulty out of
    loss-vs-loss comparisons, and makes widening the init-seed range purely
    additive -- the first N starts never move, so existing rows stay valid.

    Bounds default to [-3, 3] because past roughly +-3 the sigmoid transform has
    already pinned the parameter to its bound; a wider range would only sample
    dead gradient.
    """
    key = jax.random.fold_in(jax.random.PRNGKey(0), init_seed)
    return jax.random.uniform(key, (dim,), minval=lo, maxval=hi)


class Sim:
    """A compiled-once Pospischil compartment, driven entirely by traced values.

    ``traces_of(x8, states6)`` is the single entry point. Both arguments are
    arrays, so it vmaps over either or both:

        vmap(traces_of, in_axes=(0, None))   # a grid sweep, one ideal seed
        vmap(traces_of, in_axes=(0, 0))      # many (ideal, init) pairs at once
    """

    def __init__(
        self,
        channel_type: ChannelType = "Pospischil",
        config: dict | None = None,
        checkpoint_lengths: list[int] | None = None,
    ):
        # `checkpoint_lengths` is forwarded to jx.integrate. Reverse-mode through
        # the 6001-step solve is what dominates the optimization cost -- measured
        # ~220x the forward pass, not the ~3x a naive model predicts -- and its
        # residual tape is what caps the batch size. Recursive checkpointing
        # trades recompute for memory: e.g. [78, 78] (78*78 = 6084 >= 6001) cuts
        # stored residuals by ~78x. Since batched steps/sec is constant in batch
        # size (the scan is latency-bound), a bigger batch is nearly free, so
        # trading memory for batch width is the main lever on total cost.
        self.checkpoint_lengths = checkpoint_lengths
        self.channel_type = channel_type
        self.config = config if config is not None else get_datagen_config()
        cfg = self.config

        self.dt = cfg["dt"]
        self.t_max = cfg["t_max"]
        self.t_filter = cfg["t_filter"]
        self.avg_scale = cfg["avg_scale"]
        self.current = cfg["current"]
        self.dt_eff = dt_eff(cfg)
        self.n_t = trace_len(cfg)

        self.param_bounds = get_param_bounds(cfg["param_bounds"], channel_type)
        check_param_order(self.param_bounds)
        self.transforms = _get_transform_list(self.param_bounds)

        self.amps = self._load_amps()
        self.n_amps = len(self.amps)

        self.comp = self._build_comp()
        # init_states once, at jaxley's defaults. Every per-seed state value is
        # supplied through data_set instead, so this call only fixes the shapes.
        self.comp.init_states(delta_t=self.dt)

        # A scratch compartment used only to compute steady-state gating values
        # on the host. Kept separate so the simulation compartment is never
        # mutated after construction.
        self._scratch = self._build_comp()

    # -- construction helpers ------------------------------------------------

    def _build_comp(self) -> Compartment:
        comp = insert_channels(Compartment(), self.channel_type)
        comp = set_const_params(self.config["param_consts"], self.channel_type, comp)
        comp.record("v")
        return comp

    def _load_amps(self) -> Array:
        """Stimulus amplitudes, read straight from the dataset's amps file.

        The legacy path calls ``get_traces_to_match_rand``, which indexes a 20 GB
        memmap of voltage traces purely to return this constant 10-element array
        and then throws the traces away.
        """
        path = f"{self.config['save_path']}/{self.channel_type}_amps.npz"
        amps = jnp.asarray(np.load(path)["arr_0"])
        if amps.ndim != 1 or amps.size < 2:
            raise RuntimeError(f"{path}: expected a 1-D amps array, got {amps.shape}")
        return amps

    # -- per-seed quantities (host side) ------------------------------------

    def targets_for_seeds(self, seeds) -> Array:
        """``(I, 8)`` true parameter vectors in TRANSFORMED space."""
        rows = []
        for s in seeds:
            ideal = ideal_params_for_seed(int(s), self.param_bounds)
            rows.append(
                _backward_transform_params(
                    [ideal[p] for p in ALL_PARAMS], self.transforms
                )
            )
        return jnp.stack(rows)

    def init_states_for_seeds(self, seeds) -> Array:
        """``(I, 6)`` steady-state gating values, one row per ideal seed.

        Pure host-side pandas: set the 8 physical parameters, run
        ``init_states``, read the 6 columns back. No simulation, so this is
        microseconds per seed -- and it is what makes the ``data_set`` route
        numerically identical to the legacy ``comp.set`` path.
        """
        rows = []
        for s in seeds:
            ideal = ideal_params_for_seed(int(s), self.param_bounds)
            for name in ALL_PARAMS:
                self._scratch.set(name, float(ideal[name]))
            self._scratch.init_states(delta_t=self.dt)
            rows.append(
                [float(self._scratch.nodes[g].iloc[0]) for g in GATING_STATES]
            )
        return jnp.asarray(rows)

    def states_for_params(self, x8) -> Array:
        """``(6,)`` steady-state gating values for one TRANSFORMED parameter vector.

        Same as ``init_states_for_seeds`` but for an arbitrary parameter vector
        rather than a seed. Host-side, so ``x8`` must be concrete.
        """
        phys = _forward_transform_params(jnp.asarray(x8), self.transforms)
        for i, name in enumerate(ALL_PARAMS):
            self._scratch.set(name, float(phys[i]))
        self._scratch.init_states(delta_t=self.dt)
        return jnp.asarray(
            [float(self._scratch.nodes[g].iloc[0]) for g in GATING_STATES]
        )

    # -- the simulator ------------------------------------------------------

    def _simulate(self, x8: Array, states6: Array, amp) -> Array:
        """One raw voltage trace for one amplitude. All inputs may be traced."""
        phys = _forward_transform_params(x8, self.transforms)

        current = jx.step_current(
            i_delay=self.current["delay"],
            i_dur=self.current["duration"],
            i_amp=amp,
            delta_t=self.dt,
            t_max=self.t_max,
        )

        param_state = None
        for i, name in enumerate(ALL_PARAMS):
            param_state = self.comp.data_set(name, phys[i], param_state)
        # The gating states: see the module docstring -- omitting these is a
        # 14 mV error, not a rounding difference.
        for i, name in enumerate(GATING_STATES):
            param_state = self.comp.data_set(name, states6[i], param_state)

        kwargs = {}
        if self.checkpoint_lengths is not None:
            kwargs["checkpoint_lengths"] = self.checkpoint_lengths
        return jnp.array(
            jx.integrate(
                self.comp,
                data_stimuli=self.comp.data_stimulate(current, None),
                param_state=param_state,
                delta_t=self.dt,
                t_max=self.t_max,
                **kwargs,
            ).flatten()
        )

    def traces_of(self, x8: Array, states6: Array) -> Array:
        """``(n_amps, n_t)`` compressed, NaN-free traces for one parameter vector.

        ``x8`` is a full 8-vector in transformed space; ``states6`` the matching
        gating states from ``init_states_for_seeds`` / ``states_for_params``.
        """
        volts = jax.vmap(self._simulate, in_axes=(None, None, 0))(x8, states6, self.amps)
        volts = compress_traces(volts, self.t_filter, self.dt, self.avg_scale)
        return jnp.nan_to_num(volts)
