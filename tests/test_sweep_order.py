"""Guards the axis-order contract between the grid, the parameters and x_star.

This is the regression test for the bug the first study hit: ``yaml.safe_dump``
defaults to ``sort_keys=True``, which re-sorted the stored parameter mapping
alphabetically, so each grid axis got paired with the WRONG true value for every
parameter set whose canonical order is not alphabetical. It inflated
``argmin_err`` identically across every loss -- the give-away, since the loss
cannot move where the true parameters are.

Four checks, cheapest first:

1. the reshape convention (``itertools.product`` + ``order="C"``) -- instant
2. cache-key invariance under a reordered ``--params`` -- instant
3. a real sweep's ``argmin_err`` is within a couple of grid steps -- ~40 s
4. that check has teeth: transposing the landscape blows ``argmin_err`` up

Check 3 uses ``Na_gNa, K_gK``, whose canonical order is deliberately NOT
alphabetical, so a re-sort would corrupt it.

    ../.venv/bin/python tests/test_sweep_order.py
"""

import itertools
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # encoding/
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts", "datagen"))

import convexity as cx  # noqa: E402
import gen_loss_sweep as G  # noqa: E402
import sim as S  # noqa: E402
from helper import get_datagen_config  # noqa: E402

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


CONFIG = get_datagen_config()

# Non-alphabetical canonical order: K_gK sorts before Na_gNa alphabetically, but
# Na_gNa comes first in ALL_PARAMS. Exactly the case the old bug corrupted.
PARAMS = ["Na_gNa", "K_gK"]
BOUNDS = (-1.5, 1.5)
SAMPLES = 41
LOSSES = ["mse-zscore", "spike-w1"]


def test_reshape_convention():
    """product() varies the LAST axis fastest, so reshape is C order.

    Getting this backwards would silently transpose every landscape, and on a
    symmetric grid it would be invisible -- hence an asymmetric probe function.
    """
    for dim in (2, 3, 4):
        n = 5
        axis = np.linspace(-1.0, 1.0, n)

        def probe(*xs):
            # Weighted so no two axes are interchangeable.
            return sum((i + 1) * 10 ** i * x for i, x in enumerate(xs))

        flat = np.array([probe(*combo) for combo in itertools.product(*[axis] * dim)])
        y = flat.reshape((n,) * dim, order="C")

        for _ in range(20):
            idx = tuple(np.random.default_rng(0).integers(0, n, size=dim))
            want = probe(*[axis[i] for i in idx])
            if abs(y[idx] - want) > 1e-12:
                fail(f"{dim}-D reshape mismatch at {idx}: {y[idx]} != {want}")
                break

        # And an explicit statement of the convention: the LAST index moves
        # fastest through the flat array.
        if abs(flat[1] - probe(*([axis[0]] * (dim - 1) + [axis[1]]))) > 1e-12:
            fail(f"{dim}-D: flat[1] is not the second value of the LAST axis")


def test_canonical_key_invariance():
    """Reordering --params must not create a second cache entry."""
    a = G.sweep_key("Pospischil", S.canonical(["K_gK", "Na_gNa"]), 0,
                    BOUNDS, SAMPLES, CONFIG)
    b = G.sweep_key("Pospischil", S.canonical(["Na_gNa", "K_gK"]), 0,
                    BOUNDS, SAMPLES, CONFIG)
    if G.key_hash(a) != G.key_hash(b):
        fail("reordered --params produced different cache keys")
    if a["params"] != PARAMS:
        fail(f"canonical params {a['params']} != {PARAMS}")

    # The key must be a LIST, so JSON cannot reorder it.
    if not isinstance(a["params"], list):
        fail("sweep_key stored params as a non-list")

    # Changing anything that alters the traces must change the key.
    for mutate in (
        lambda k: G.sweep_key("Pospischil", PARAMS, 1, BOUNDS, SAMPLES, CONFIG),
        lambda k: G.sweep_key("Pospischil", PARAMS, 0, (-2.0, 2.0), SAMPLES, CONFIG),
        lambda k: G.sweep_key("Pospischil", PARAMS, 0, BOUNDS, 21, CONFIG),
        lambda k: G.sweep_key("Pospischil", ["Na_gNa", "vt"], 0, BOUNDS, SAMPLES, CONFIG),
    ):
        if G.key_hash(mutate(a)) == G.key_hash(a):
            fail("a trace-affecting change did not change the cache key")


def test_real_sweep_argmin():
    """A real landscape's minimum must sit at the true parameters."""
    out = G.generate_sweep(
        "Pospischil", PARAMS, 0, BOUNDS, SAMPLES, LOSSES,
        config=CONFIG, batch_size=512, keep_grid=True,
    )
    descriptors = json.loads((out / "descriptors.json").read_text())
    step = (BOUNDS[1] - BOUNDS[0]) / (SAMPLES - 1)

    for name in LOSSES:
        err = descriptors["scores"][name]["argmin_err"]
        # Measured 0.7-1.3 grid steps; 4 steps is generous but still an order of
        # magnitude below what a transposition produces (see the next test).
        if err > 4 * step:
            fail(f"{name}: argmin_err {err:.4f} > {4 * step:.4f} "
                 f"({err / step:.1f} grid steps) -- axes may be mis-paired")

    # The stored x_star must match the target's transformed values, in order.
    y, grids, x_star = G.load_grid(out, LOSSES[0])
    simulator = S.Sim("Pospischil", CONFIG)
    x8 = simulator.targets_for_seeds([0])[0]
    want = np.asarray(x8[S.param_indices(PARAMS)])
    if np.max(np.abs(x_star - want)) > 1e-12:
        fail(f"stored x_star {x_star} != target {want}")
    if len(x_star) != len(PARAMS):
        fail(f"x_star has {len(x_star)} entries for {len(PARAMS)} params")

    # load_grid re-asserts shape; make sure it actually would catch a bad file.
    if y.shape != (SAMPLES,) * len(PARAMS):
        fail(f"grid shape {y.shape} != {(SAMPLES,) * len(PARAMS)}")
    return out


def test_transposition_is_detected(out):
    """Prove the argmin check has teeth by transposing on purpose."""
    y, grids, x_star = G.load_grid(out, "mse-zscore")
    step = float(grids[0][1] - grids[0][0])

    good = cx.SCORES["argmin_err"](y, grids, x_star)
    bad = cx.SCORES["argmin_err"](y.T, grids, x_star)

    if good > 4 * step:
        fail(f"baseline argmin_err {good} already large")
    # x_star for seed 0 is roughly [-0.71, +1.00], so swapping the axes moves
    # the reported minimum by well over a full unit.
    if bad < 10 * step:
        fail(f"transposed argmin_err {bad:.4f} is not clearly worse than "
             f"{good:.4f} -- this test cannot detect a transposition")
    if abs(x_star[0] - x_star[1]) < 0.5:
        fail("x_star components too close for the transposition probe to bite")


if __name__ == "__main__":
    test_reshape_convention()
    test_canonical_key_invariance()
    out = test_real_sweep_argmin()
    if out is not None:
        test_transposition_is_detected(out)

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES[:25]:
            print("  ", f)
        sys.exit(1)
    print("all sweep-order tests passed")
