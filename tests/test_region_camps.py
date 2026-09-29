"""Pins the camp layer: the gates that decide what reaches a figure.

`scripts/plotting/region_scores.py` grew three pieces of judgement that are not
arithmetic on a stored field, and each of them silently changes what the report
says:

* :func:`best_optimizer` -- which arm every loss-axis camp is drawn under. A
  wrong answer here relabels every camp figure rather than breaking one.
* :func:`noise_floor` and :func:`holm` -- the two gates in
  `plot_region_cross.py`. Too loose and the cross-camp figure fills with
  retraining noise; too tight and a real effect is dropped without a trace.
* :func:`camp_rows` -- what "this camp" means when one encoder belongs to two
  camps, which the beta ladder's top rung does.

None of these need a simulator, a GPU or the real store, so they are checked
against hand-built rows whose answers are known by construction.

    ../.venv/bin/python tests/test_region_camps.py
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENCODING = os.path.dirname(_HERE)
sys.path.insert(0, _ENCODING)
sys.path.insert(0, os.path.join(_ENCODING, "scripts", "plotting"))

import region_scores as R  # noqa: E402

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print("FAIL:", msg)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

RADII = R.SUCCESS_RADII


def _row(optimizer, loss, sset, d_best, *, escape=0.0, d_final=None):
    """One row whose ladder score is known: every entry at the same distance.

    With all distances equal to `d`, the ladder score is exactly the fraction of
    the five radii that exceed `d` -- so the expected score is countable rather
    than simulated.
    """
    d_fin = float(d_final if d_final is not None else d_best)
    mat = [[float(d_best)] * 4 for _ in range(2)]
    fin = [[d_fin] * 4 for _ in range(2)]
    return {
        "optimizer": optimizer, "loss_name": loss, "set": sset,
        "steps": 100, "ideal_seeds": [0, 2], "init_seeds": [0, 4],
        "init_bounds": [-1.5, 1.5],
        "raw_d_best": mat, "raw_d_final": fin,
        "raw_d0": [[2.0] * 4 for _ in range(2)],
        # The stored field is the FINAL-iterate ladder, as on a real row -- that
        # is what makes "score() must not read it" a meaningful assertion.
        "region_score": float(np.mean([d_fin < r for r in RADII])),
        "diag": {"escape_frac": escape},
        "meta": {"loss_set_version": 2},
    }


def _expected(d):
    return float(np.mean([d < r for r in RADII]))


CFG = {
    "score_matrix": "raw_d_best",
    "baseline_loss": "mse-zscore",
    "best_optimizer": "auto",
    "best_optimizer_rule": {
        "over_losses": ["a-loss", "b-loss"],
        "max_escape_spread": 0.10,
    },
    "noise_floor_group": ["a-loss", "b-loss"],
    "camps": {
        "one": {"title": "One", "losses": ["a-loss", "b-loss"],
                "contrasts": [["a-loss", "b-loss", "a vs b"]]},
        "two": {"title": "Two", "losses": ["b-loss", "c-loss"],
                "member_labels": {"b-loss": "shared rung"}},
        "opt": {"title": "Opt", "axis": "optimizer",
                "optimizers": ["fast", "slow"],
                "losses": ["a-loss", "b-loss"]},
    },
}


def _store(escape_fast_b=0.0):
    """`fast` beats `slow` on both losses; `escape_fast_b` can veto `fast`."""
    rows = []
    for s in ("s1", "s2", "s3"):
        rows.append(_row("fast", "a-loss", s, 0.10))
        rows.append(_row("fast", "b-loss", s, 0.20, escape=escape_fast_b))
        rows.append(_row("slow", "a-loss", s, 0.50))
        rows.append(_row("slow", "b-loss", s, 0.60))
        rows.append(_row("fast", "c-loss", s, 0.30))
        rows.append(_row("fast", "mse-zscore", s, 0.40))
    return rows


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_score_reads_the_configured_matrix():
    """`score` must follow cfg["score_matrix"], never the stored region_score."""
    row = _row("fast", "a-loss", "s1", 0.10, d_final=1.4)
    if not np.isclose(R.score(row, CFG), _expected(0.10)):
        fail(f"score read the wrong matrix: {R.score(row, CFG)}")
    other = dict(CFG, score_matrix="raw_d_final")
    if not np.isclose(R.score(row, other), _expected(1.4)):
        fail("score ignores a changed score_matrix")
    # The stored field is the final-iterate one; confusing them is the exact
    # mistake the raw_d_best switch is exposed to.
    if np.isclose(R.score(row, CFG), row["region_score"]):
        fail("score returned the stored region_score, which is d_final's")


def test_best_optimizer_picks_the_top_eligible_arm():
    name, ranking = R.best_optimizer(_store(), CFG)
    if name != "fast":
        fail(f"best_optimizer chose {name}, want 'fast'")
    # Each loss weighted equally, not each row.
    want = (_expected(0.10) + _expected(0.20)) / 2
    if not np.isclose(ranking["fast"]["mean_score"], want):
        fail(f"mean_score {ranking['fast']['mean_score']} != {want}")
    # c-loss and mse-zscore are outside `over_losses` and must not count.
    if set(ranking["fast"]["per_loss"]) != {"a-loss", "b-loss"}:
        fail(f"ranking pooled over {sorted(ranking['fast']['per_loss'])}, "
             "want only the over_losses spine")


def test_escape_spread_vetoes_the_top_arm():
    """C5's gate: an arm that faces its losses unequally cannot be the reference.

    `fast` still scores higher; it is disqualified anyway. If this ever stops
    holding, every camp is silently drawn under an arm whose ranking of the
    losses is partly a statement about the optimizer.
    """
    name, ranking = R.best_optimizer(_store(escape_fast_b=0.5), CFG)
    if ranking["fast"]["eligible"]:
        fail("an escape spread of 0.5 did not veto the arm (limit 0.10)")
    if name != "slow":
        fail(f"vetoed arm still chosen: got {name}")
    if not ranking["fast"]["mean_score"] > ranking["slow"]["mean_score"]:
        fail("the fixture no longer tests a veto -- 'fast' must score higher")


def test_pinned_optimizer_overrides_the_rule():
    pinned = dict(CFG, best_optimizer="slow")
    name, ranking = R.best_optimizer(_store(), pinned)
    if name != "slow":
        fail(f"a pinned best_optimizer was ignored: got {name}")
    if not ranking["fast"]["eligible"]:
        fail("pinning must still report the full ranking, so a pinned choice "
             "the rule would not have made is visible")

    # A missing arm on the spine is not eligible, and if nothing is, that is a
    # hard error rather than a quiet fallback to whatever has rows.
    partial = [r for r in _store() if r["loss_name"] != "b-loss"]
    try:
        R.best_optimizer(partial, CFG)
        fail("best_optimizer accepted a store where no arm has the full spine")
    except SystemExit:
        pass


def test_noise_floor_is_the_spread_of_the_group():
    floor = R.noise_floor(_store(), CFG, "fast")
    want = abs(_expected(0.10) - _expected(0.20))
    if not np.isclose(floor["spread"], want):
        fail(f"noise floor {floor['spread']} != {want}")
    if floor["missing"]:
        fail(f"unexpected missing members: {floor['missing']}")
    lonely = R.noise_floor(_store(), CFG, "fast", names=["a-loss"])
    if not np.isnan(lonely["spread"]):
        fail("a one-member group must give NaN, not 0 -- 0 would disable the "
             "cross-camp gate silently")


def test_camp_rows_respect_a_shared_member():
    rows = _store()
    one, two = R.camp(CFG, "one"), R.camp(CFG, "two")
    if R.camp_members(rows, one, optimizer="fast") != ["a-loss", "b-loss"]:
        fail("camp 'one' members wrong")
    if R.camp_members(rows, two, optimizer="fast") != ["b-loss", "c-loss"]:
        fail("camp 'two' members wrong")
    # b-loss is in BOTH camps, which is the beta ladder's top rung situation.
    if R.member_label(two, "b-loss") != "shared rung":
        fail("member_label did not override the shared arm's name")
    if R.member_label(one, "b-loss") is not None:
        fail("member_label invented a label the camp does not define")

    # An optimizer-axis camp spans its own arms and ignores `optimizer`.
    opt = R.camp(CFG, "opt")
    got = {r.get("optimizer") for r in R.camp_rows(rows, opt, optimizer="fast")}
    if got != {"fast", "slow"}:
        fail(f"an axis:optimizer camp collapsed to one arm: {got}")

    try:
        R.camp(CFG, "nope")
        fail("an unknown camp name was accepted")
    except SystemExit:
        pass


def test_baseline_rows_follow_the_arm():
    rows = _store()
    if len(R.baseline_rows(rows, CFG, "fast")) != 3:
        fail("baseline rows not found under the chosen arm")
    if R.baseline_rows(rows, CFG, "slow"):
        fail("baseline rows leaked from another arm")


def test_holm_is_step_down_and_nan_safe():
    got = R.holm([0.01, 0.02, 0.03])
    if not np.allclose(got, [0.03, 0.04, 0.04]):
        fail(f"holm([.01,.02,.03]) = {got}, want [.03,.04,.04]")

    p = np.array([0.001, 0.04, 0.2, 0.9, 0.03])
    adj = R.holm(p)
    if not np.all(adj >= p - 1e-12):
        fail("an adjusted p-value came out below its raw value")
    order = np.argsort(p)
    if not np.all(np.diff(adj[order]) >= -1e-12):
        fail("holm is not monotone in the p-order (not step-down)")
    if not np.all(adj <= 1.0):
        fail("holm returned a p-value above 1")

    # A degenerate comparison is not a test that was performed, so it must not
    # inflate the family size and weaken every other correction.
    with_nan = R.holm([0.01, float("nan"), 0.02])
    if not (np.isnan(with_nan[1]) and np.allclose(with_nan[[0, 2]], [0.02, 0.02])):
        fail(f"nan handling wrong: {with_nan}")
    if R.holm([]).size != 0:
        fail("holm of an empty family should be empty")


def test_set_order_can_use_the_rescored_statistic():
    """The shared x order must be a best-iterate order, not a stored-field one."""
    rows = [_row("fast", "a-loss", "hard", 1.0, d_final=0.01),
            _row("fast", "a-loss", "easy", 0.01, d_final=1.0)]
    if R.set_order(rows, matrix="raw_d_best") != ["hard", "easy"]:
        fail("set_order ignored the matrix and used the stored field")
    if R.set_order(rows, matrix="raw_d_final") != ["easy", "hard"]:
        fail("set_order does not follow the matrix it is given")


if __name__ == "__main__":
    test_score_reads_the_configured_matrix()
    test_best_optimizer_picks_the_top_eligible_arm()
    test_escape_spread_vetoes_the_top_arm()
    test_pinned_optimizer_overrides_the_rule()
    test_noise_floor_is_the_spread_of_the_group()
    test_camp_rows_respect_a_shared_member()
    test_baseline_rows_follow_the_arm()
    test_holm_is_step_down_and_nan_safe()
    test_set_order_can_use_the_rescored_statistic()

    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES[:25]:
            print("  ", f)
        sys.exit(1)
    print("all camp-layer tests passed")
