"""Load and aggregate ``data/optimization-region-scores.jsonl`` for plotting.

The ``score_join.py`` analogue for the region-of-success study. One row is one
(optimizer, loss, parameter set) cell holding a full ``(n_ideal, n_init)``
matrix of final distances, so every derived quantity a figure needs -- any
radius ladder, the best-iterate score, the chance baseline -- is host arithmetic
over ``raw_d_final`` / ``raw_d_best`` / ``raw_d0``. No GPU, no re-run.

Two things make the figures trustworthy:

* **The score is imported, never restated.** ``ladder_score``, ``ausc``,
  ``success_table`` and the pivot helpers come from
  ``scripts/datagen/gen_region_success_scores.py``, the module that wrote the
  file. A second implementation here is exactly how a figure and the ``--check``
  report end up quoting two different numbers for one result.
* **A camp is a filter, not a file.** :func:`camp_rows` narrows the rows to one
  camp's arms before anything is computed, which is what makes it safe for
  several studies to append to one store. The alternative -- one flat figure set
  over every row -- made `set_order` (a median over whichever losses are
  present) and `spread_over_noise` (a property of the loss SET) move whenever an
  unrelated study landed.
* **Budgets are never merged.** The score is a bounded-budget statistic, so rows
  computed with 100 initializations are not comparable with rows computed with
  1000, and a calibration run writes rows for the same (loss, set) cells the
  production run will. Rows are grouped by
  ``(steps, ideal_seeds, init_seeds, init_bounds, loss_set_version)`` and only
  one budget is returned. This is the same guard ``score_join`` carries, and it
  is what the task doc's "do not pool new rows with the old score files" finding
  asks for.
"""

import os
import sys
from pathlib import Path

# Plotting never needs an accelerator, and importing the generator below pulls
# in jax/jaxley/equinox, which would otherwise claim a GPU on a shared node.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENCODING = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ENCODING)
sys.path.insert(0, os.path.join(_ENCODING, "scripts", "datagen"))

import jsonl_store as JS  # noqa: E402
import param_groups as pg  # noqa: E402

# The score, its ladder and the pivot helpers, straight from the generator.
from gen_region_success_scores import (  # noqa: E402,F401
    LEGACY_TOLS,
    SUCCESS_RADII,
    _dig,
    _paired,
    _pivot,
    ausc,
    ladder_score,
    quantiles,
    success_table,
)

PATH = Path("./data/optimization-region-scores.jsonl")

# Fields that together define a comparable measurement. Two rows differing on
# any of these answer different questions and must not share an axis.
BUDGET_FIELDS = ("steps", "ideal_seeds", "init_seeds", "init_bounds")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _budget(row) -> tuple:
    meta = row.get("meta") or {}
    return (
        row.get("steps"),
        tuple(row.get("ideal_seeds") or ()),
        tuple(row.get("init_seeds") or ()),
        tuple(float(v) for v in (row.get("init_bounds") or ())),
        meta.get("loss_set_version"),
    )


def load(path=PATH, *, budget=None, quiet=False,
         exclude_losses=None) -> list[dict]:
    """Rows of ONE budget, the largest unless ``budget`` names another.

    ``exclude_losses`` drops named losses before anything else, so the budget
    choice, the reported counts and `set_order` all see the same set of arms
    the figures will draw. It is config-driven (`exclude_losses` in
    `configs/region_scores.yaml`) rather than per-script, because a loss present
    in one figure and absent from another cannot be read across them, and
    because unrelated studies write into the same store.

    Also warns when the returned rows were not all produced by the same library
    versions. A 100-step trajectory on a rough landscape amplifies a 1e-15
    kernel difference into an O(1) change in the final distance, so a mixed-
    version set of rows is not a valid paired comparison -- that is the
    reproducibility finding in the task doc, made visible instead of silent.
    """
    rows = JS.load_rows(Path(path))
    if not rows:
        raise SystemExit(f"no rows in {path}")

    dropped = set(exclude_losses or ())
    if dropped:
        before = len(rows)
        rows = [r for r in rows if r["loss_name"] not in dropped]
        if not rows:
            raise SystemExit(
                f"exclude_losses {sorted(dropped)} removed every row in {path}"
            )
        if not quiet and len(rows) < before:
            print(f"  excluded {before - len(rows)} rows from "
                  f"{', '.join(sorted(dropped))} (exclude_losses)")

    by_budget: dict[tuple, list[dict]] = {}
    for row in rows:
        by_budget.setdefault(_budget(row), []).append(row)

    if budget is not None:
        chosen = tuple(budget)
        if chosen not in by_budget:
            raise SystemExit(
                f"no rows with budget {chosen}; present: {sorted(by_budget)}"
            )
    else:
        chosen = max(by_budget, key=lambda b: len(by_budget[b]))

    kept = by_budget[chosen]
    if len(by_budget) > 1 and not quiet:
        print(
            f"  {len(by_budget)} budgets present "
            f"(steps, ideal_seeds, init_seeds, init_bounds, loss_set_version); "
            f"using {chosen} ({len(kept)} rows). The rest are a different "
            "budget and are NOT comparable, so they are excluded: "
            + ", ".join(f"{b} ({len(v)})" for b, v in by_budget.items()
                        if b != chosen)
        )

    versions = {
        tuple(sorted(((row.get("meta") or {}).get("versions") or {}).items()))
        for row in kept
    }
    if len(versions) > 1 and not quiet:
        print(
            f"  WARNING {len(versions)} distinct library-version sets among the "
            "kept rows. A 100-step trajectory amplifies kernel differences into "
            "an O(1) change in the final distance, so these are not strictly "
            "one experiment."
        )
    if not quiet:
        opts = sorted({r.get("optimizer") for r in kept})
        losses = sorted({r["loss_name"] for r in kept})
        print(
            f"  {len(kept)} rows | optimizers {opts} | losses {losses} | "
            f"{len({r['set'] for r in kept})} parameter sets"
        )
    return kept


def arms(rows) -> list[tuple[str, str]]:
    """The ``(optimizer, loss)`` cells present, in a stable order."""
    seen = {(r.get("optimizer"), r["loss_name"]) for r in rows}
    return sorted(seen)


def optimizers(rows) -> list[str]:
    """Present optimizers, with `polyak` first: it is the inherited setup, and
    every figure reads as "the result, then the control"."""
    present = {r.get("optimizer") for r in rows}
    order = ["polyak", "adam", "rmsprop"]
    return [o for o in order if o in present] + sorted(present - set(order))


def losses(rows) -> list[str]:
    """Present losses, masked first so the contrast reads left to right."""
    present = {r["loss_name"] for r in rows}
    order = ["encoder-zscore-mask", "encoder-zscore-2", "encoder-zscore"]
    return [x for x in order if x in present] + sorted(present - set(order))


def table(rows, field, *, optimizer) -> dict[str, dict[str, float]]:
    """``{loss: {set: value}}`` for one optimizer arm. Dotted fields allowed."""
    return _pivot(rows, optimizer, field)


def index(rows) -> dict[tuple[str, str, str], dict]:
    """``(optimizer, loss, set) -> row``, asserting the design has no duplicates."""
    out = {}
    for r in rows:
        key = (r.get("optimizer"), r["loss_name"], r["set"])
        if key in out:
            raise SystemExit(
                f"duplicate row for {key}; dedup the score file with "
                "`gen_region_success_scores.py --merge` before plotting"
            )
        out[key] = r
    return out


def set_order(rows, *, field="region_score", optimizer=None,
              matrix=None) -> list[str]:
    """The parameter sets, hardest first, by median ``field`` across losses.

    One shared order for every figure: panels that do not share an x order
    cannot be read against each other, and the whole point of the overview is
    that polyak and adam agree about which triplets are hard.

    ``matrix`` orders by the score recomputed off that distance matrix instead
    of by a stored field, so a best-iterate figure gets a best-iterate order.
    Passing it makes ``field`` irrelevant.
    """
    pool = [r for r in rows if optimizer is None or r.get("optimizer") == optimizer]
    if not pool:
        pool = rows
    by_set: dict[str, list[float]] = {}
    for r in pool:
        v = rescore(r, SUCCESS_RADII, matrix=matrix) if matrix else _dig(r, field)
        if v is not None and np.isfinite(v):
            by_set.setdefault(r["set"], []).append(float(v))
    return sorted(by_set, key=lambda s: np.median(by_set[s]))


# ---------------------------------------------------------------------------
# rescoring: any ladder, any distance matrix, no GPU
# ---------------------------------------------------------------------------

RAW_FIELDS = ("raw_d_final", "raw_d_best", "raw_d0")


def raw(row, matrix="raw_d_final") -> np.ndarray:
    """The stored ``(n_ideal, n_init)`` distance matrix, as float64."""
    m = row.get(matrix)
    if m is None:
        raise SystemExit(
            f"row {row.get('optimizer')}/{row['loss_name']}/{row['set']} has no "
            f"'{matrix}'; it was written without --save_raw and cannot be "
            "rescored. Re-run that unit."
        )
    return np.asarray(m, dtype=float)


def rescore(row, radii, *, matrix="raw_d_final") -> float:
    """The ladder score for one row under an arbitrary radius ladder.

    ``ladder_score`` is the generator's own function, so ``rescore(row,
    SUCCESS_RADII)`` reproduces the stored ``region_score`` exactly -- which is
    the assertion :func:`check_rescore_matches_stored` pins.
    """
    return ladder_score(raw(row, matrix), radii=tuple(radii))[1]


def chance(row, radii) -> float:
    """The same ladder applied to the START distances: the random-guess floor."""
    return ladder_score(raw(row, "raw_d0"), radii=tuple(radii))[1]


def check_rescore_matches_stored(rows, *, tol=None) -> None:
    """Recomputing the default ladder must return the stored score.

    A one-line guard over the whole rescoring path: if it holds, every other
    ladder in these figures was computed the same way the published one was.

    It is *not* an exact-equality check, because ``--save_raw`` stores
    ``np.round(raw, 5)`` while the stored score was computed on the unrounded
    matrix. A distance sitting within 5e-6 of a radius can therefore land on the
    other side of ``d < r`` after the round trip, moving the score by exactly one
    quantum, ``1 / (n_radii * n_entries)``. Measured on the 336 rows on disk:
    334 reproduce bit-for-bit and 2 differ by exactly one quantum (2e-4), and
    there is indeed a stored distance of exactly 0.750000 against the 0.75 rung.
    The tolerance is a few quanta -- enough to absorb that, far too tight to hide
    a wrong ladder, a wrong matrix or a misaligned row.
    """
    diffs, where = [], None
    worst = 0.0
    for r in rows:
        d = abs(rescore(r, SUCCESS_RADII) - float(r["region_score"]))
        diffs.append(d)
        if d > worst:
            worst, where = d, (r.get("optimizer"), r["loss_name"], r["set"])
    n_entries = raw(rows[0]).size
    quantum = 1.0 / (len(SUCCESS_RADII) * n_entries)
    limit = 4 * quantum if tol is None else tol
    n_off = int(sum(d > 0 for d in diffs))
    if worst > limit:
        raise SystemExit(
            f"rescoring the default ladder does not reproduce the stored "
            f"region_score (max |diff| = {worst:.3e} at {where}, limit "
            f"{limit:.3e} = 4 rounding quanta). That is too large to be the "
            "--save_raw rounding; the plots would disagree with the score "
            "file. Fix before drawing."
        )
    print(
        f"  rescore check: {len(rows) - n_off}/{len(rows)} rows exact, "
        f"max |diff| = {worst:.3e} ({worst / quantum:.0f} rounding quanta of "
        f"{quantum:.1e})"
    )


def success_curve(rows, radii, *, matrix="raw_d_final") -> np.ndarray:
    """Mean success rate at each radius, pooled over the given rows."""
    return np.array([
        float(np.mean([(raw(r, matrix) < x).mean() for r in rows])) for x in radii
    ])


# The radius grid the success curves are drawn on, shared so the crossing
# quoted on `score_vs_chance` is the one marked on `success_vs_radius`. Spans
# from below the tightest rung to past the escape cut; geometric, because that
# is how the rungs themselves are spaced.
CURVE_GRID = np.geomspace(0.03, 6.0, 60)


def chance_curve(rows, radii) -> np.ndarray:
    """:func:`success_curve` on the START distances: the random-guess floor."""
    return np.array([
        float(np.mean([(raw(r, "raw_d0") < x).mean() for r in rows]))
        for x in radii
    ])


def crossing_radius(rows, radii=CURVE_GRID, *, chance_rows=None,
                    matrix="raw_d_final") -> float | None:
    """The radius at which an arm's success curve falls back to chance.

    The one number that reconciles ``distance_distributions`` with
    ``score_vs_chance``. Below it the optimizer beats a random start; above it
    it does not, because the runs that diverged end further out than a uniform
    draw ever would. The four tight rungs of :data:`SUCCESS_RADII` sit well
    below the crossing, which is why the score is positive; the median final
    distance sits above it, which is why the quantile figure looks like a loss.
    Measured on this study the crossing lands at r = 1.2-1.9, i.e. right at the
    loosest rung (1.5) -- which is exactly why the ``single_loosest`` ladder in
    ``ladder_table.md`` is the one that flips a contrast's sign.

    Interpolated log-linearly, matching the geometric grid the curves are drawn
    on. ``None`` if the two curves never meet over ``radii``. ``chance_rows``
    defaults to ``rows``; pass the whole arm to match the floor drawn in
    :func:`plot_region_ladder.fig_success_curve` (identical either way, since
    ``raw_d0`` is shared across the losses -- the C1 invariant).
    """
    radii = np.asarray(radii, dtype=float)
    gap = success_curve(rows, radii, matrix=matrix) - chance_curve(
        rows if chance_rows is None else chance_rows, radii
    )
    # Only a fall-back counts. As r -> 0 both curves go to 0 and their
    # difference is rounding, not a crossing, so start from the first radius
    # where the optimizer is genuinely ahead.
    ahead = np.flatnonzero(gap > 0)
    if ahead.size == 0:
        return None
    behind = np.flatnonzero(gap[ahead[0]:] <= 0)
    if behind.size == 0:
        return None
    i = int(ahead[0] + behind[0])  # >= 1: gap[ahead[0]] > 0 by construction
    lo, hi = gap[i - 1], gap[i]
    t = lo / (lo - hi)
    log_r = np.log(radii[i - 1]) + t * (np.log(radii[i]) - np.log(radii[i - 1]))
    return float(np.exp(log_r))


# ---------------------------------------------------------------------------
# paired contrasts -- the doc's headline table, computed once
# ---------------------------------------------------------------------------


def contrast(rows, optimizer, loss_a, loss_b, *, field="region_score",
             matrix=None, radii=None, bootstrap_n=10000, seed=0) -> dict | None:
    """Paired loss-vs-loss difference over the parameter sets they share.

    Positive means ``loss_b`` scores higher. Pairing is on the parameter set,
    which is exact here: both arms face the identical target neurons and the
    identical starts (``d0_hash`` is checked to be shared, the C1 invariant), so
    a signed-rank test over the 56 sets is the right instrument.

    ``field`` reads a stored value; passing ``matrix`` and ``radii`` instead
    recomputes the score from the raw distances, which is how the best-iterate
    and alternative-ladder versions of the same contrast are produced.
    """
    from scipy.stats import wilcoxon

    if matrix is not None or radii is not None:
        rad = tuple(radii) if radii is not None else SUCCESS_RADII
        mat = matrix or "raw_d_final"
        pick = {}
        for r in rows:
            if r.get("optimizer") != optimizer:
                continue
            pick.setdefault(r["loss_name"], {})[r["set"]] = rescore(
                r, rad, matrix=mat
            )
        a_map, b_map = pick.get(loss_a, {}), pick.get(loss_b, {})
    else:
        pivot = _pivot(rows, optimizer, field)
        a_map, b_map = pivot.get(loss_a, {}), pivot.get(loss_b, {})

    if not a_map or not b_map:
        return None
    sets, a, b = _paired(a_map, b_map)
    ok = np.isfinite(a) & np.isfinite(b)
    sets = [s for s, k in zip(sets, ok) if k]
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return None

    diff = b - a
    try:
        p = float(wilcoxon(diff).pvalue)
    except ValueError:  # every difference is zero
        p = float("nan")

    lo, hi = _boot_ci(diff, bootstrap_n, seed)
    return {
        "optimizer": optimizer,
        "loss_a": loss_a,
        "loss_b": loss_b,
        "field": field if matrix is None and radii is None else
                 f"{matrix or 'raw_d_final'}@{list(radii or SUCCESS_RADII)}",
        "sets": sets,
        "a": a,
        "b": b,
        "diff": diff,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "ci_lo": lo,
        "ci_hi": hi,
        "n": int(len(diff)),
        "n_pos": int((diff > 0).sum()),
        "wilcoxon_p": p,
    }


def _boot_ci(diff, n_boot, seed, level=95.0) -> tuple[float, float]:
    """Percentile bootstrap over the paired differences.

    Bootstrap rather than a t interval because the differences here are bounded
    sums of success indicators and visibly non-normal, and because resampling
    the SETS is the right unit -- the pairing within a set is exact, the sample
    of sets is what generalizes.
    """
    diff = np.asarray(diff, float)
    if len(diff) < 2 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(int(n_boot), len(diff)))
    means = diff[idx].mean(axis=1)
    half = (100.0 - level) / 2.0
    lo, hi = np.percentile(means, [half, 100.0 - half])
    return float(lo), float(hi)


def spread_over_noise(rows, optimizer, radii, *, matrix="raw_d_final") -> dict:
    """How well a ladder separates the losses, per the task doc's table 3.

    ``spread`` is the between-loss sd of the mean score; ``noise`` is the mean
    within-loss sd across parameter sets. Their ratio is scale-free, so ladders
    with different absolute levels stay comparable -- which is the whole point,
    since a looser ladder raises every score at once.
    """
    per_loss: dict[str, list[float]] = {}
    for r in rows:
        if r.get("optimizer") != optimizer:
            continue
        per_loss.setdefault(r["loss_name"], []).append(
            rescore(r, radii, matrix=matrix))
    if len(per_loss) < 2:
        return {}
    means = np.array([np.mean(v) for v in per_loss.values()])
    noise = float(np.mean([np.std(v, ddof=1) for v in per_loss.values()]))
    base = float(np.mean([
        chance(r, radii) for r in rows if r.get("optimizer") == optimizer
    ]))
    return {
        "means": {k: float(np.mean(v)) for k, v in per_loss.items()},
        "spread": float(means.std(ddof=1)),
        "noise": noise,
        "spread_over_noise": float(means.std(ddof=1) / noise) if noise else
                             float("nan"),
        "chance": base,
    }


def ladders_from_config(cfg) -> dict[str, tuple[float, ...]]:
    """The configured ladders, with ``default`` asserted to be the real one."""
    out = {k: tuple(float(x) for x in v) for k, v in cfg["ladders"].items()}
    if out.get("default") != tuple(SUCCESS_RADII):
        raise SystemExit(
            "configs/region_scores.yaml:ladders.default "
            f"{list(out.get('default', ()))} is not "
            f"gen_region_success_scores.SUCCESS_RADII {list(SUCCESS_RADII)}. "
            "Every stored region_score was computed on the latter; a figure "
            "labelled 'default' must mean it."
        )
    return out


def sets_containing(set_keys, *params) -> list[str]:
    """Sets holding every named short parameter -- for annotating an outlier group."""
    want = set(params)
    return [s for s in set_keys if want <= set(s.split("+"))]


# ---------------------------------------------------------------------------
# camps: one question, one small group of figures
# ---------------------------------------------------------------------------


def camp(cfg, name) -> dict:
    """One camp from the config, with its optional keys defaulted.

    Returned as a plain dict with ``name`` added, so a figure function can be
    handed a camp and need nothing else from the config.
    """
    camps = cfg.get("camps") or {}
    if name not in camps:
        raise SystemExit(
            f"no camp {name!r} in configs/region_scores.yaml; "
            f"have {sorted(camps)}"
        )
    c = dict(camps[name])
    c["name"] = name
    c.setdefault("title", name)
    c.setdefault("axis", "loss")
    c.setdefault("losses", [])
    c.setdefault("contrasts", [])
    c.setdefault("member_labels", {})
    c.setdefault("color_ramp", None)
    # A camp may name the arm it must be drawn under, overriding
    # `best_optimizer`. Only for a camp where the globally-best arm is
    # demonstrably the wrong one to read it on -- see `beta` in
    # configs/region_scores.yaml. None means "use the global choice".
    c.setdefault("optimizer", None)
    if c["axis"] == "optimizer":
        c.setdefault("optimizers", [])
    if not c["losses"]:
        raise SystemExit(f"camp {name!r} lists no losses")
    return c


def camp_names(cfg) -> list[str]:
    return list((cfg.get("camps") or {}).keys())


def camp_rows(rows, c, *, optimizer=None) -> list[dict]:
    """The rows belonging to one camp.

    This is what replaces a global ``exclude_losses``: every figure sees only
    its own arms, so ``set_order`` and ``spread_over_noise`` -- both properties
    of whichever losses are present -- cannot be moved by an unrelated study
    appending to the same store.

    A ``loss``-axis camp is drawn under one optimizer, so pass it. An
    ``optimizer``-axis camp spans the optimizers it names and ignores
    ``optimizer``.
    """
    losses = set(c["losses"])
    if c.get("axis") == "optimizer":
        opts = set(c.get("optimizers") or [])
        return [r for r in rows
                if r["loss_name"] in losses and r.get("optimizer") in opts]
    return [r for r in rows
            if r["loss_name"] in losses
            and (optimizer is None or r.get("optimizer") == optimizer)]


def camp_members(rows, c, *, optimizer=None) -> list[str]:
    """The camp's losses that actually have rows, in the configured order."""
    have = {r["loss_name"] for r in camp_rows(rows, c, optimizer=optimizer)}
    return [x for x in c["losses"] if x in have]


def member_label(c, loss) -> str | None:
    """The camp's own name for an arm, or None to fall back to the global one."""
    return (c.get("member_labels") or {}).get(loss)


def baseline_rows(rows, cfg, optimizer) -> list[dict]:
    """The reference bar every camp draws: ``baseline_loss`` under one arm."""
    name = cfg.get("baseline_loss")
    if not name:
        return []
    return [r for r in rows
            if r["loss_name"] == name and r.get("optimizer") == optimizer]


# ---------------------------------------------------------------------------
# scoring under the configured statistic
# ---------------------------------------------------------------------------


def score(row, cfg, radii=None) -> float:
    """The one score every camp figure is drawn on.

    A single funnel for ``cfg["score_matrix"]`` so no figure can end up drawing
    a different iterate than its neighbour. The stored ``region_score`` field is
    deliberately NOT used here: it was computed on ``raw_d_final``, and the
    headline moved to the best iterate on 2026-08-30.
    """
    return rescore(row, tuple(radii) if radii is not None else SUCCESS_RADII,
                   matrix=cfg["score_matrix"])


def arm_scores(rows, cfg, *, optimizer=None, radii=None) -> dict:
    """``{loss: {set: score}}`` under ``cfg["score_matrix"]``.

    The rescoring counterpart of :func:`table`, which can only read a stored
    field.
    """
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        if optimizer is not None and r.get("optimizer") != optimizer:
            continue
        out.setdefault(r["loss_name"], {})[r["set"]] = score(r, cfg, radii)
    return out


# ---------------------------------------------------------------------------
# which optimizer the camps are drawn under
# ---------------------------------------------------------------------------


def best_optimizer(rows, cfg) -> tuple[str, dict]:
    """The arm every loss-axis camp is drawn under, plus the full ranking.

    One implementation, called by all three figure scripts, because a figure
    labelled "under adam" beside one labelled "under polyak" is unreadable and
    the mistake is silent.

    The rule, from ``best_optimizer_rule`` in the config: highest mean score
    pooled over ``over_losses`` (the spine every arm is run on -- ranking over
    "whatever rows exist" would reward the arm that happened to be run on the
    easier losses), among arms whose between-loss ``escape_frac`` spread is
    within ``max_escape_spread``.

    That eligibility gate is not decoration. C5 in ``--check`` exists because an
    arm that escapes far more on one loss than another is not facing them
    equally, so its RANKING of them says as much about the optimizer as about
    the losses -- the 2026-08-29 activation study failed exactly this on polyak
    (spread 0.135, relu 0.625 against mse-zscore 0.489). An arm can therefore
    top the score table and still be disqualified.

    A literal ``best_optimizer`` in the config pins the choice and skips the
    rule; the ranking is still returned, so a pinned choice that the rule would
    not have made is visible in the sidecar rather than hidden.
    """
    rule = cfg.get("best_optimizer_rule") or {}
    over = list(rule.get("over_losses") or [])
    limit = float(rule.get("max_escape_spread", 1.0))

    ranking: dict[str, dict] = {}
    for opt in optimizers(rows):
        sub = [r for r in rows if r.get("optimizer") == opt
               and (not over or r["loss_name"] in over)]
        present = {r["loss_name"] for r in sub}
        missing = sorted(set(over) - present)
        per_loss: dict[str, list[float]] = {}
        per_escape: dict[str, list[float]] = {}
        for r in sub:
            per_loss.setdefault(r["loss_name"], []).append(score(r, cfg))
            e = _dig(r, "diag.escape_frac")
            if e is not None and np.isfinite(e):
                per_escape.setdefault(r["loss_name"], []).append(float(e))
        means = {k: float(np.mean(v)) for k, v in per_loss.items()}
        escapes = {k: float(np.mean(v)) for k, v in per_escape.items()}
        spread = (max(escapes.values()) - min(escapes.values())
                  if len(escapes) > 1 else 0.0)
        ranking[opt] = {
            # Each loss weighted equally, not each row: the arms need not have
            # the same number of parameter sets on every loss.
            "mean_score": float(np.mean(list(means.values()))) if means else None,
            "per_loss": means,
            "escape_frac": escapes,
            "escape_spread": spread,
            "missing_losses": missing,
            "eligible": bool(not missing and means and spread <= limit),
            "n_rows": len(sub),
        }

    pinned = cfg.get("best_optimizer", "auto")
    if pinned and pinned != "auto":
        if pinned not in ranking:
            raise SystemExit(
                f"best_optimizer is pinned to {pinned!r} but the store has no "
                f"such rows; present: {sorted(ranking)}"
            )
        return pinned, ranking

    eligible = [o for o, v in ranking.items() if v["eligible"]]
    if not eligible:
        raise SystemExit(
            "no optimizer arm is eligible to be the reference. Needed: rows for "
            f"every loss in {over} and an escape_frac spread <= {limit}. Got "
            + "; ".join(
                f"{o}: missing={v['missing_losses']} spread={v['escape_spread']:.3f}"
                for o, v in ranking.items()
            )
            + ". Run the missing arms, or pin `best_optimizer` in "
            "configs/region_scores.yaml."
        )
    return max(eligible, key=lambda o: ranking[o]["mean_score"]), ranking


# ---------------------------------------------------------------------------
# the noise floor, and the multiplicity gate
# ---------------------------------------------------------------------------


def noise_floor(rows, cfg, optimizer, *, names=None) -> dict:
    """Spread of the mean score over config-identical encoders.

    Retraining one configuration twice moves the score; ``noise_floor_group``
    names encoders that differ ONLY in that non-determinism, so the spread of
    their means is the smallest difference this study can distinguish from
    nothing. An effect smaller than it is not a finding however small its
    p-value, because the paired test is over 56 subsets and resolves far below
    the floor.
    """
    names = list(names if names is not None else (cfg.get("noise_floor_group") or []))
    per: dict[str, list[float]] = {}
    for r in rows:
        if r.get("optimizer") == optimizer and r["loss_name"] in names:
            per.setdefault(r["loss_name"], []).append(score(r, cfg))
    means = {k: float(np.mean(v)) for k, v in per.items()}
    spread = (max(means.values()) - min(means.values())
              if len(means) > 1 else float("nan"))
    return {"spread": spread, "means": means, "n": len(means),
            "missing": sorted(set(names) - set(means))}


def escape_gap(rows, optimizer, loss_a, loss_b) -> float:
    """|mean escape_frac(a) - mean escape_frac(b)| under one arm.

    Check C5's statistic, per PAIR rather than per arm. Two losses whose runs
    leave the box at very different rates are not facing the same effective
    optimizer, so the difference between their scores is partly a statement
    about the optimizer -- which is exactly what a loss-vs-loss figure must not
    quietly claim.

    Per-pair rather than per-arm because the arm-level number is dominated by
    its most extreme loss: under polyak the seven-loss spread is 0.135, driven
    by mse-zscore (0.489) against encoder-act-relu (0.625), while the three
    activation losses sit within 0.064 of each other. The activation contrasts
    are clean; the encoder-vs-MSE ones are not. One number for the whole arm
    cannot say that.

    NaN if either loss has no usable rows.
    """
    per: dict[str, list[float]] = {}
    for r in rows:
        if r.get("optimizer") != optimizer or r["loss_name"] not in (loss_a, loss_b):
            continue
        v = _dig(r, "diag.escape_frac")
        if v is not None and np.isfinite(v):
            per.setdefault(r["loss_name"], []).append(float(v))
    if loss_a not in per or loss_b not in per:
        return float("nan")
    return abs(float(np.mean(per[loss_a])) - float(np.mean(per[loss_b])))


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Step-down rather than plain Bonferroni: same familywise error control, more
    power, and it is the standard choice for a family of this size. Non-finite
    inputs pass through as NaN and are excluded from the family size, since a
    degenerate comparison is not a test that was performed.
    """
    p = np.asarray(pvalues, dtype=float)
    adj = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size == 0:
        return adj
    m = ok.size
    order = ok[np.argsort(p[ok], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * float(p[i]))
        adj[i] = min(running, 1.0)
    return adj


__all__ = [
    "PATH", "SUCCESS_RADII", "LEGACY_TOLS", "ladder_score", "success_table",
    "ausc", "quantiles", "_dig", "_pivot", "_paired", "pg",
    "load", "arms", "optimizers", "losses", "table", "index", "set_order",
    "raw", "rescore", "chance", "check_rescore_matches_stored",
    "success_curve", "chance_curve", "crossing_radius", "CURVE_GRID",
    "contrast", "spread_over_noise", "ladders_from_config", "sets_containing",
    "camp", "camp_names", "camp_rows", "camp_members", "member_label",
    "baseline_rows", "score", "arm_scores", "best_optimizer", "noise_floor",
    "escape_gap", "holm",
]
