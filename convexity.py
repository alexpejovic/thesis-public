"""Loss-landscape descriptors, generalized from 1-D to N-D.

Each function reduces **exactly** to the 1-D reference implementation it
generalizes, so 1-D
numbers stay comparable with the first study while 2-D and 4-D landscapes get
the same descriptors.

How each one generalizes:

``n_local_min``  The 1-D version collapses runs of equal values, then counts
    strict interior minima. In N-D the equivalent is: mark every cell with no
    strictly lower axis-neighbour, group those cells into connected plateaus,
    and count the plateaus that do not touch the grid boundary.

``basin``       Unchanged in spirit: point every cell at its steepest strictly
    lower axis-neighbour (ties, or no lower neighbour, mean stay put), follow
    those pointers to a fixed point, and report the fraction of cells that end
    up within ``tol`` of the true parameters. Distance is the max-norm, which
    is |.| in 1-D.

``roughness``   Total variation divided by the descent from the two ends to the
    minimum, evaluated on axis-aligned lines. 1-D has exactly one line, so the
    formula is unchanged; in N-D it is averaged over every line, skipping lines
    too flat for the ratio to mean anything.

``flat_frac``   Fraction of adjacent cell pairs with zero difference, counted
    over every axis.

``argmin_err``  Max-norm distance from the grid's argmin to the true value.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

# Lines flatter than this fraction of the landscape's full range are dropped
# from the roughness average: their denominator is ~0 and the ratio explodes.
_ROUGHNESS_MIN_DESCENT = 1e-3


def _as_grids(grids, y):
    """Accept a single 1-D grid or a per-axis sequence of them."""
    if y.ndim == 1 and np.ndim(grids[0]) == 0:
        return [np.asarray(grids, dtype=float)]
    return [np.asarray(g, dtype=float) for g in grids]


def _neighbour_indices(shape: tuple) -> np.ndarray:
    """Flat index of each axis neighbour; off-grid entries point at themselves.

    Returns shape (2 * ndim, size).
    """
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    out = []
    for ax in range(len(shape)):
        for shift in (+1, -1):
            nb = np.roll(idx, shift, axis=ax)
            # np.roll wraps around; the wrapped face has no neighbour, so make
            # it point at itself and mark it invalid later.
            face = [slice(None)] * len(shape)
            face[ax] = 0 if shift == +1 else -1
            nb = nb.copy()
            nb[tuple(face)] = idx[tuple(face)]
            out.append(nb.ravel())
    return np.stack(out)


def _equal_value_plateaus(y: np.ndarray) -> tuple[int, np.ndarray]:
    """Label connected regions of exactly equal value. Returns (n, flat labels).

    This is the N-D counterpart of the reference's ``collapse``: it is what
    turns a run of identical samples into a single feature.
    """
    flat = y.ravel()
    idx = np.arange(flat.size).reshape(y.shape)
    rows, cols = [], []
    for ax in range(y.ndim):
        if y.shape[ax] < 2:
            continue
        a = np.take(idx, np.arange(y.shape[ax] - 1), axis=ax).ravel()
        b = np.take(idx, np.arange(1, y.shape[ax]), axis=ax).ravel()
        same = flat[a] == flat[b]
        rows.append(a[same])
        cols.append(b[same])

    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
    else:
        rows = cols = np.empty(0, dtype=int)

    graph = sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)),
        shape=(flat.size, flat.size),
    )
    return connected_components(graph, directed=False)


def n_local_min(y) -> int:
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return 0

    flat = y.ravel()
    nb_idx = _neighbour_indices(y.shape)
    self_idx = np.arange(flat.size)
    nb_val = np.where(nb_idx == self_idx, np.inf, flat[nb_idx])

    # No strictly lower cell next to this one.
    cand = flat <= nb_val.min(axis=0)

    # Decide at the plateau level, not the cell level. A cell in the middle of
    # a flat shoulder has no lower neighbour of its own, but the plateau it
    # belongs to drains off one end -- so the whole plateau must be candidate
    # for it to count as a minimum.
    n_plateaus, labels = _equal_value_plateaus(y)
    is_min = np.ones(n_plateaus, dtype=bool)
    np.logical_and.at(is_min, labels, cand)

    # Plateaus touching the boundary are excluded: the reference counts
    # interior minima only, and an edge cell has an unobserved side.
    edge = np.zeros(y.shape, dtype=bool)
    for ax in range(y.ndim):
        face = [slice(None)] * y.ndim
        face[ax] = 0
        edge[tuple(face)] = True
        face[ax] = -1
        edge[tuple(face)] = True
    interior = np.ones(n_plateaus, dtype=bool)
    np.logical_and.at(interior, labels, ~edge.ravel())

    return int(np.count_nonzero(is_min & interior))


def basin(y, grids, x_star, tol=0.1) -> float:
    y = np.asarray(y, dtype=float)
    grids = _as_grids(grids, y)
    x_star = np.atleast_1d(np.asarray(x_star, dtype=float))

    flat = y.ravel()
    size = flat.size
    self_idx = np.arange(size)

    nb_idx = _neighbour_indices(y.shape)
    nb_val = np.where(nb_idx == self_idx, np.inf, flat[nb_idx])

    lowest = nb_val.min(axis=0)
    move = lowest < flat
    if y.ndim == 1:
        # The reference halts on a two-way tie -- in 1-D that means a symmetric
        # ridge, and which way it falls is genuinely arbitrary. Kept for exact
        # agreement.
        move &= (nb_val == lowest).sum(axis=0) == 1
    # In N-D, ties are instead the common case: any cell equidistant along two
    # axes of a symmetric bowl ties. Halting there would strand whole diagonals
    # and understate every basin (a perfect 4-D bowl scores 0.18 instead of 1),
    # so ties break deterministically toward the lowest axis. argmin already
    # returns the first occurrence.
    best = nb_idx[np.argmin(nb_val, axis=0), self_idx]
    nxt = np.where(move, best, self_idx)

    # Path doubling: log2(size) squarings reach the fixed point.
    f = nxt
    for _ in range(int(np.ceil(np.log2(max(size, 2)))) + 2):
        f = f[f]

    mesh = np.meshgrid(*grids, indexing="ij")
    coords = np.stack([m.ravel() for m in mesh])          # (ndim, size)
    dist = np.max(np.abs(coords[:, f] - x_star[:, None]), axis=0)
    return float(np.mean(dist <= tol))


def roughness(y) -> float:
    y = np.asarray(y, dtype=float)
    span = y.max() - y.min()
    yn = (y - y.min()) / (span + 1e-30)

    ratios = []
    for ax in range(y.ndim):
        lines = np.moveaxis(yn, ax, -1).reshape(-1, yn.shape[ax])
        if lines.shape[1] < 2:
            continue
        tv = np.abs(np.diff(lines, axis=1)).sum(axis=1)
        lo = lines.min(axis=1)
        descent = (lines[:, 0] - lo) + (lines[:, -1] - lo)
        if y.ndim == 1:
            # Single line: reproduce the reference exactly, guard and all.
            ratios.append(tv / (descent + 1e-30))
        else:
            keep = descent > _ROUGHNESS_MIN_DESCENT
            if keep.any():
                ratios.append(tv[keep] / descent[keep])

    if not ratios:
        return float("nan")
    return float(np.mean(np.concatenate(ratios)))


def flat_frac(y) -> float:
    y = np.asarray(y, dtype=float)
    flat_pairs = 0
    total = 0
    for ax in range(y.ndim):
        d = np.diff(y, axis=ax)
        flat_pairs += int((d == 0).sum())
        total += d.size
    if total == 0:
        return float("nan")
    return flat_pairs / total


def argmin_err(y, grids, x_star) -> float:
    y = np.asarray(y, dtype=float)
    grids = _as_grids(grids, y)
    x_star = np.atleast_1d(np.asarray(x_star, dtype=float))
    where = np.unravel_index(int(np.argmin(y)), y.shape)
    pos = np.array([grids[a][where[a]] for a in range(y.ndim)])
    return float(np.max(np.abs(pos - x_star)))


def score(y, grids, x_star, tol=0.1) -> dict:
    """All five descriptors for one landscape.

    ``tol`` is the basin radius. Callers sweeping coarse high-dimensional grids
    should raise it to at least one grid step, or nothing can land inside it.
    """
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(y)):
        finite = y[np.isfinite(y)]
        fill = np.nanmax(finite) if finite.size else 0.0
        y = np.nan_to_num(y, nan=fill, posinf=fill, neginf=fill)
    return dict(
        locmin=n_local_min(y),
        basin=basin(y, grids, x_star, tol=tol),
        rough=roughness(y),
        flat=flat_frac(y),
        argmin_err=argmin_err(y, grids, x_star),
    )


# ===========================================================================
# The six convexity scores, as a registry.
#
# Everything above this line is the N-D descriptor library, copied verbatim
# from the `large-landscape-analysis` branch. It reduces EXACTLY to the frozen
# 1-D reference implementation of that branch, which is what keeps the 1-D
# numbers comparable with the first study. Do not edit it -- any change here
# silently breaks that comparability. Everything below is the selection
# layer the score generators need: a uniform `f(y, grids, x_star) -> float`
# signature so a score can be picked by name, plus `basin_wide`, which used to
# live inline in the caller.
# ===========================================================================

# Radius for `basin_wide`, as a fraction of the swept half-span.
#
# The strict basin (tolerance = one grid step) saturates at exactly 0 on rugged
# high-dimensional landscapes -- measured on a real 2-D sweep, both the encoder
# loss and its MSE floor scored 0.000 at tol 0.1, while at a wider radius they
# separated 0.023 vs 0.113. The wide radius asks "does descent reach the right
# neighbourhood" rather than "does it land on the exact node".
#
# It is a fraction rather than a fixed distance because the bounds differ by
# dimensionality: a fixed 0.5 would be a sixth of a +-3 window but half of a
# +-1 one. A quarter of the half-span is always well clear of the grid step and
# always the same share of the region actually swept.
#
# NOTE: this constant and `_basin_strict`'s rule below are baked into every
# cached `descriptors.json`. Changing either invalidates every cached sweep, so
# both are recorded alongside the scores rather than assumed.
BASIN_WIDE_FRACTION = 0.25

# Floor for the strict basin radius. On a coarse grid a resting point can never
# land within 0.1 of the true value, so the tolerance is raised to at least one
# grid step or the descriptor reads 0 everywhere and carries no signal.
BASIN_STRICT_FLOOR = 0.1


def _sanitize(y) -> np.ndarray:
    """Replace non-finite cells with the largest finite value present.

    A NaN loss means the simulation blew up, which is the worst possible
    outcome, so treating it as the observed maximum is the honest reading. Same
    repair as ``score()`` does inline; lifted out so the registry can apply it
    once per call while the bare descriptors stay byte-identical to the
    reference-checked originals.
    """
    y = np.asarray(y, dtype=np.float64)
    if np.all(np.isfinite(y)):
        return y
    finite = y[np.isfinite(y)]
    fill = np.nanmax(finite) if finite.size else 0.0
    return np.nan_to_num(y, nan=fill, posinf=fill, neginf=fill)


def _grid_step(grids) -> float:
    g = grids[0]
    return float(g[1] - g[0]) if len(g) > 1 else 0.0


def basin_strict_tol(grids) -> float:
    """The strict basin radius actually used, so callers can record it."""
    return max(BASIN_STRICT_FLOOR, _grid_step(grids))


def basin_wide_tol(grids, fraction: float = BASIN_WIDE_FRACTION) -> float:
    """The wide basin radius actually used, so callers can record it."""
    g = grids[0]
    return fraction * (float(g[-1]) - float(g[0])) / 2.0


def basin_wide(y, grids, x_star, fraction: float = BASIN_WIDE_FRACTION) -> float:
    """``basin`` with a radius of ``fraction`` x the swept half-span."""
    grids = _as_grids(grids, np.asarray(y, dtype=float))
    return basin(y, grids, x_star, tol=basin_wide_tol(grids, fraction))


def _basin_strict(y, grids, x_star) -> float:
    """``basin`` with tol = max(0.1, one grid step)."""
    grids = _as_grids(grids, np.asarray(y, dtype=float))
    return basin(y, grids, x_star, tol=basin_strict_tol(grids))


# name -> f(y, grids, x_star) -> float. The three shape-only descriptors ignore
# `grids` and `x_star`; the uniform signature is what lets a score be selected
# by string from the CLI.
SCORES = {
    "locmin": lambda y, g, x: float(n_local_min(_sanitize(y))),
    "basin": lambda y, g, x: _basin_strict(_sanitize(y), g, x),
    "basin_wide": lambda y, g, x: basin_wide(_sanitize(y), g, x),
    "rough": lambda y, g, x: roughness(_sanitize(y)),
    "flat": lambda y, g, x: flat_frac(_sanitize(y)),
    "argmin_err": lambda y, g, x: argmin_err(_sanitize(y), g, x),
}

SCORE_NAMES = tuple(SCORES)

# Sign each descriptor is expected to take against optimization distance, for
# reading the correlation tables. Higher `basin` should mean the optimizer more
# often lands near truth (so a NEGATIVE correlation with distance); more local
# minima, more roughness, more flat cells and a more displaced global minimum
# should all mean worse optimization (POSITIVE correlation with distance).
EXPECTED_SIGN = {
    "locmin": +1,
    "basin": -1,
    "basin_wide": -1,
    "rough": +1,
    "flat": +1,
    "argmin_err": +1,
}


def score_all(y, grids, x_star) -> dict:
    """All six descriptors for one landscape, sanitized once."""
    y = _sanitize(y)
    grids = _as_grids(grids, y)
    return {name: fn(y, grids, x_star) for name, fn in SCORES.items()}
