"""Registry of parameter sets and landscape grid settings.

Defined once here and imported by the score generators and the plotting
scripts, so a set cannot be defined two ways. Set keys are used verbatim in
filenames and in the score files, so renaming one orphans its existing rows.

This study sweeps **all 56 three-parameter subsets** of the eight neuronal
parameters (C(8,3) = 56), each giving a 3-D loss landscape.

Independent of `losses`
-----------------------
This module knows about parameters, not about losses. It used to re-export
`ENCODERS`/`LOSS_NAMES`/`LOSS_SPECS` purely so the deleted `score_jobs()` could
enumerate (loss, set) pairs; `study.py` owns that product now, so the import is
gone with it. Do not reintroduce it -- a loss registry reachable from two
modules is how the datestrs drift.

(This is a layering property, not an import-cost one. `sim` imports
`train_comp` for the sigmoid-transform helpers and `train_comp` imports the VAE
stack at module level, so `import param_groups` still costs ~4 s and pulls in
jaxley and equinox regardless. Restating `ALL_PARAMS` here to avoid that would
recreate the "a set defined two ways" failure this module exists to prevent.)

No job-enumeration CLI, either
------------------------------
This module used to expose `sweep_jobs`/`score_jobs`/`--dump` so the old bash
array scripts could `eval` `SET=... PARAMS=...` lines. `study.py` owns job
expansion now (see slurm_scripts/README_studies.md); keeping the old path would
mean maintaining a second enumeration that can drift from the ledger.
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from sim import ALL_PARAMS, canonical  # noqa: E402

# Short display name per parameter, used to build set keys. Chosen to match the
# names already used in the first study's group keys (`gNa`, `vt`, ...).
SHORT = {
    "Leak_gLeak": "gLeak",
    "Leak_eLeak": "eLeak",
    "Na_gNa": "gNa",
    "K_gK": "gK",
    "Km_gKm": "gKm",
    "CaL_gCaL": "gCaL",
    "Km_taumax": "taumax",
    "vt": "vt",
}

SET_DIM = 3


def set_key(params) -> str:
    """Stable key for a parameter set: short names joined in canonical order."""
    return "+".join(SHORT[p] for p in canonical(params))


# key -> the parameters left TRAINABLE (everything else is frozen at truth).
PARAM_SETS = {
    set_key(combo): list(combo)
    for combo in itertools.combinations(ALL_PARAMS, SET_DIM)
}

STUDY_SETS = list(PARAM_SETS)

assert len(PARAM_SETS) == 56, f"expected 56 sets, got {len(PARAM_SETS)}"


def params_for(key: str) -> list[str]:
    if key not in PARAM_SETS:
        raise KeyError(f"unknown parameter set {key!r}")
    return PARAM_SETS[key]


def frozen_for(key: str) -> list[str]:
    """Parameters NOT optimized in this set (the legacy --freeze_params list)."""
    trainable = set(params_for(key))
    return [p for p in ALL_PARAMS if p not in trainable]


def set_dim(key: str) -> int:
    return len(params_for(key))


# ---------------------------------------------------------------------------
# Landscape grid, per dimensionality: (bounds, samples per axis).
#
# Bounds follow the first study's reasoning: tight beats wide, because past
# roughly +-3 the sigmoid transform has already pinned the parameter to its
# bound, so extra span buys saturated tail while a finer step buys resolution.
#
# Samples per axis is set by COST PARITY with the optimization runs, which is
# what the study is correlating against. Measured per-unit costs on a 20-core
# worker (sim of 10 amps = 3.4 ms; one encoder forward = 3.0 ms; the five cheap
# losses = 1.79 ms total) give:
#
#   one grid point      = sim + all 8 losses      = 14.19 ms
#   one optimizer step  = 3 x (sim + one loss)    = 14.25 ms   (fwd+bwd ~= 3x fwd)
#
# Within 0.4% of each other, so parity is nearly "equal counts". The
# optimization side is 56 sets x 8 losses x 100 ideal x 1000 init x 100 steps =
# 4.48e9 cell-steps, so the sweeps want ~4.5e9 grid points over 56 x 100 = 5600
# sweeps: N = (4.5e9 / 5600) ** (1/3) ~= 93.
#
#   N     pts/sweep   total pts   cost ratio   step
#   31       29,791     1.7e8        0.04      0.1000     (too coarse to match)
#   81      531,441     3.0e9        0.66      0.0375
#   91      753,571     4.2e9        0.94      0.0333  <- chosen
#   101   1,030,301     5.8e9        1.28      0.0300
#
# 91 gives ratio 0.94 and a step of exactly 1/30.
#
# Two consequences to keep in mind when reading results:
#
#  * `locmin` and `rough` are NOT comparable across dimensionalities -- a finer
#    grid resolves more local minima. At step 0.0333 these 3-D numbers are not
#    comparable with the first study's 4-D numbers at step 0.10.
#  * the cost model carries roughly 2x uncertainty, and the two sides scale
#    differently on GPU (the sweep is encoder-throughput bound, the optimizer is
#    backward-pass bound). Re-solve `samples` from the pilot's measurements
#    before committing to the full run; it is one constant.
GRID_PLAN = {
    3: {"bounds": (-1.5, 1.5), "samples": 91},
}

# Cost model fitted from the measurements above, for reporting estimates.
COST_STARTUP_S = 21.0
COST_PER_POINT_S = 0.01419


def grid_for(key: str) -> dict:
    dim = set_dim(key)
    if dim not in GRID_PLAN:
        raise KeyError(
            f"no grid defined for {dim}-D set {key!r}; known: {sorted(GRID_PLAN)}"
        )
    return GRID_PLAN[dim]


def grid_step(key: str) -> float:
    g = grid_for(key)
    lo, hi = g["bounds"]
    return (hi - lo) / (g["samples"] - 1)


def grid_points(key: str) -> int:
    return grid_for(key)["samples"] ** set_dim(key)


def est_runtime_s(key: str) -> float:
    """Estimated single-sweep wall clock on a 20-core CPU worker."""
    return COST_STARTUP_S + COST_PER_POINT_S * grid_points(key)


# Ideal-trace seeds. Half-open [lo, hi), so the default is 100 landscapes.
IDEAL_SEEDS = (0, 100)
# Optimization-init seeds, likewise half-open: 1000 starts per ideal seed.
INIT_SEEDS = (0, 1000)

