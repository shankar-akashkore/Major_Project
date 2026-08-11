"""Evaluation-harness tests.

The most important test in this file is
:func:`test_the_ceiling_formula_matches_a_simulation_end_to_end`.  Everything the
report claims about being "near ceiling" depends on one algebraic step, and an
algebraic step that nothing checks is a claim, not a measurement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from adml.evaluate import (
    Evaluation,
    ablate,
    ceiling_from_agreement,
    credit,
    mcnemar_exact,
    measure_ceiling,
    paired_difference,
    set_metrics,
    stage_agreement,
    within_set_outcomes,
)
from adml.split import Observation
from adschema.annotation import PairKind


def evaluation(name: str, values: list[float], observations: list[Observation]) -> Evaluation:
    return Evaluation(name=name, observations=observations, credit=np.array(values, dtype=float))


# --- The ceiling ------------------------------------------------------------


def test_perfect_agreement_implies_a_ceiling_of_one():
    assert ceiling_from_agreement(1.0) == pytest.approx(1.0)


def test_chance_agreement_carries_no_information_about_the_ceiling():
    assert math.isnan(ceiling_from_agreement(0.5))
    assert math.isnan(ceiling_from_agreement(0.3))


def test_the_documented_worked_example_holds():
    """0.60 test-retest implies a ceiling near 0.72 — the number in the docstring."""
    assert ceiling_from_agreement(0.60) == pytest.approx(0.7236, abs=1e-3)


def test_the_ceiling_rises_monotonically_with_agreement():
    values = [ceiling_from_agreement(a) for a in (0.55, 0.65, 0.75, 0.85, 0.95)]
    assert values == sorted(values)


def test_the_ceiling_formula_matches_a_simulation_end_to_end():
    """Simulate noisy judges, measure agreement, and check the implied ceiling.

    A judge expresses the true preference with probability ``q``.  Repeated
    judgements of the same pair give an observed test-retest agreement; the formula
    must invert that back to ``q``, and ``q`` must equal the accuracy an oracle
    actually achieves against single judgements.  All three are measured here, so
    the derivation is checked rather than asserted.
    """
    rng = np.random.default_rng(2026)
    for q in (0.65, 0.75, 0.85, 0.95):
        n = 40000
        first = rng.random(n) < q
        second = rng.random(n) < q
        agreement = float(np.mean(first == second))
        implied = ceiling_from_agreement(agreement)
        oracle_accuracy = float(np.mean(first))  # an oracle matches a judgement iff correct
        assert implied == pytest.approx(q, abs=0.01)
        assert implied == pytest.approx(oracle_accuracy, abs=0.01)


def test_the_ceiling_is_measured_from_repeats_and_from_multiple_annotators():
    ceiling = measure_ceiling([1.0, 1.0, 0.0, 1.0], [["a", "a"], ["a", "b"], ["b", "b"]])
    assert ceiling.n_repeats == 4
    assert ceiling.test_retest_agreement == pytest.approx(0.75)
    assert ceiling.n_multi_judged_pairs == 3
    assert ceiling.inter_annotator_agreement == pytest.approx(2 / 3)


def test_a_pair_judged_once_contributes_no_agreement():
    ceiling = measure_ceiling([], [["a"]])
    assert ceiling.n_multi_judged_pairs == 0
    assert math.isnan(ceiling.inter_annotator_ceiling)


def test_no_repeats_means_no_ceiling_rather_than_a_ceiling_of_one():
    ceiling = measure_ceiling([], [])
    assert math.isnan(ceiling.test_retest_ceiling)
    assert "n/a" in ceiling.summary()


# --- Credit ----------------------------------------------------------------


def test_credit_is_one_half_or_zero():
    obs = [Observation("a", "b"), Observation("b", "a")]
    values = credit({"a": 1.0, "b": 0.0}, obs)
    assert list(values) == [1.0, 0.0]


def test_equal_scores_earn_a_coin_flip():
    assert list(credit({"a": 1.0, "b": 1.0}, [Observation("a", "b")])) == [0.5]


def test_a_tie_observation_is_neither_rewarded_nor_punished():
    obs = [Observation("a", "b", is_tie=True)]
    assert list(credit({"a": 9.0, "b": -9.0}, obs)) == [0.5]


def test_an_unscored_item_yields_nan_and_keeps_the_index_aligned():
    obs = [Observation("a", "b"), Observation("c", "d")]
    values = credit({"a": 1.0, "b": 0.0}, obs)
    assert len(values) == 2
    assert math.isnan(values[1])


def test_accuracy_ignores_the_comparisons_the_scorer_had_no_opinion_on():
    obs = [Observation("a", "b"), Observation("c", "d")]
    result = evaluation("m", [1.0, float("nan")], obs)
    assert result.n == 1
    assert result.accuracy == pytest.approx(1.0)


def test_accuracy_is_reported_separately_per_comparison_kind():
    obs = [
        Observation("a", "b", PairKind.WITHIN_SET),
        Observation("c", "d", PairKind.CROSS_SET),
        Observation("e", "f", PairKind.CROSS_SET),
    ]
    by_kind = evaluation("m", [1.0, 1.0, 0.0], obs).accuracy_by_kind()
    assert by_kind["within_set"] == (1.0, 1)
    assert by_kind["cross_set"] == (0.5, 2)


def test_the_summary_reports_position_against_a_ceiling():
    obs = [Observation("a", "b")] * 4
    text = evaluation("m", [1.0, 1.0, 1.0, 0.0], obs).summary(ceiling=0.80)
    assert "94% of ceiling" in text


def test_stand_in_features_are_flagged_in_the_summary():
    result = Evaluation("m", [Observation("a", "b")], np.array([1.0]), is_real=False)
    assert "STAND-IN" in result.summary()


# --- Paired comparison -----------------------------------------------------


def test_a_paired_difference_ignores_the_comparisons_both_models_agree_on():
    """Agreement contributes no variance, which is why pairing is powerful."""
    obs = [Observation(f"a{i}", f"b{i}") for i in range(200)]
    shared = [1.0] * 190
    a = evaluation("a", shared + [1.0] * 10, obs)
    b = evaluation("b", shared + [0.0] * 10, obs)
    difference = paired_difference(a, b)
    assert difference.delta == pytest.approx(0.05)
    assert difference.a_only_correct == 10
    assert difference.b_only_correct == 0
    assert difference.resolved


def test_two_identical_models_are_never_reported_as_different():
    obs = [Observation(f"a{i}", f"b{i}") for i in range(100)]
    values = list(np.random.default_rng(0).integers(0, 2, size=100).astype(float))
    difference = paired_difference(evaluation("a", values, obs), evaluation("b", values, obs))
    assert difference.delta == pytest.approx(0.0)
    assert difference.p_value == 1.0
    assert not difference.resolved


def test_pairing_resolves_a_difference_that_independent_intervals_would_not():
    """The methodological claim in the module docstring, asserted.

    Both models sit near 0.70 with overlapping marginal intervals, but one is
    strictly better on every comparison where they differ.
    """
    n = 300
    rng = np.random.default_rng(1)
    shared = list(rng.integers(0, 2, size=n - 30).astype(float))
    obs = [Observation(f"a{i}", f"b{i}") for i in range(n)]
    a = evaluation("a", shared + [1.0] * 30, obs)
    b = evaluation("b", shared + [0.0] * 30, obs)

    a_lo, a_hi = a.ci()
    b_lo, b_hi = b.ci()
    assert a_lo < b_hi, "marginal intervals should overlap, or the test proves nothing"
    assert paired_difference(a, b).resolved


def test_mismatched_observation_lists_are_refused():
    a = evaluation("a", [1.0], [Observation("a", "b")])
    b = evaluation("b", [1.0, 0.0], [Observation("a", "b"), Observation("c", "d")])
    with pytest.raises(ValueError, match="same observation list"):
        paired_difference(a, b)


def test_mcnemar_uses_only_the_discordant_comparisons():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)


def test_mcnemar_is_less_certain_with_fewer_discordant_comparisons():
    assert mcnemar_exact(3, 0) > mcnemar_exact(10, 0)


# --- Set-level metrics -----------------------------------------------------


def test_a_transitive_triple_yields_a_determinate_order():
    obs = [
        Observation("a", "b", PairKind.WITHIN_SET),
        Observation("b", "c", PairKind.WITHIN_SET),
        Observation("a", "c", PairKind.WITHIN_SET),
    ]
    outcome = within_set_outcomes(obs, dict.fromkeys("abc", "s0"))[0]
    assert outcome.order == ["a", "b", "c"]
    assert not outcome.is_intransitive


def test_a_cycle_is_marked_rather_than_ordered():
    """A > B > C > A has no best candidate and must not be given one."""
    obs = [
        Observation("a", "b", PairKind.WITHIN_SET),
        Observation("b", "c", PairKind.WITHIN_SET),
        Observation("c", "a", PairKind.WITHIN_SET),
    ]
    outcome = within_set_outcomes(obs, dict.fromkeys("abc", "s0"))[0]
    assert outcome.is_intransitive


def test_intransitive_sets_are_excluded_from_top_one_and_counted():
    set_of = {"a": "s0", "b": "s0", "c": "s0", "x": "s1", "y": "s1", "z": "s1"}
    obs = [
        Observation("a", "b", PairKind.WITHIN_SET),
        Observation("b", "c", PairKind.WITHIN_SET),
        Observation("c", "a", PairKind.WITHIN_SET),
        Observation("x", "y", PairKind.WITHIN_SET),
        Observation("y", "z", PairKind.WITHIN_SET),
        Observation("x", "z", PairKind.WITHIN_SET),
    ]
    outcomes = within_set_outcomes(obs, set_of)
    metrics = set_metrics({"x": 3.0, "y": 2.0, "z": 1.0, "a": 1.0, "b": 2.0, "c": 3.0}, outcomes)
    assert metrics.n_intransitive == 1
    assert metrics.n_sets == 1
    assert metrics.top1_retention == pytest.approx(1.0)


def test_cross_set_comparisons_do_not_contribute_within_set_outcomes():
    obs = [Observation("a", "x", PairKind.CROSS_SET)]
    assert within_set_outcomes(obs, {"a": "s0", "x": "s1"}) == []


def test_a_perfect_scorer_gets_ndcg_of_one_and_a_worst_pick_gets_less():
    set_of = dict.fromkeys("abc", "s0")
    obs = [
        Observation("a", "b", PairKind.WITHIN_SET),
        Observation("b", "c", PairKind.WITHIN_SET),
        Observation("a", "c", PairKind.WITHIN_SET),
    ]
    outcomes = within_set_outcomes(obs, set_of)
    good = set_metrics({"a": 3.0, "b": 2.0, "c": 1.0}, outcomes)
    bad = set_metrics({"a": 1.0, "b": 2.0, "c": 3.0}, outcomes)
    assert good.ndcg_at_1 == pytest.approx(1.0)
    assert bad.ndcg_at_1 < good.ndcg_at_1
    assert good.mean_spearman > bad.mean_spearman


# --- Stage agreement -------------------------------------------------------


def test_identical_stage_orders_agree_perfectly():
    orders = {"s0": ["a", "b", "c"], "s1": ["x", "y", "z"]}
    result = stage_agreement(orders, orders)
    assert result.mean_spearman == pytest.approx(1.0)
    assert result.top1_retention == pytest.approx(1.0)
    assert result.n_sets == 2


def test_a_reversed_stage_order_disagrees_perfectly():
    result = stage_agreement({"s0": ["a", "b", "c"]}, {"s0": ["c", "b", "a"]})
    assert result.mean_spearman == pytest.approx(-1.0)
    assert result.top1_retention == pytest.approx(0.0)


def test_sets_ranked_at_only_one_stage_are_skipped():
    result = stage_agreement({"s0": ["a", "b"], "s1": ["c", "d"]}, {"s0": ["a", "b"]})
    assert result.n_sets == 1


# --- Ablation --------------------------------------------------------------


def test_the_ablation_table_pairs_every_row_against_the_full_model():
    obs = [Observation(f"a{i}", f"b{i}") for i in range(120)]
    base = list(np.random.default_rng(3).integers(0, 2, size=120).astype(float))
    full = evaluation("full", base, obs)
    worse = evaluation("minus x", [0.0] * 120, obs)
    rows = ablate(full, [("minus x", 4, worse)])
    assert rows[0].label == "full"
    assert rows[1].delta_vs_full is not None
    assert rows[1].delta_vs_full.delta < 0
    assert rows[1].delta_vs_full.resolved
    assert "*" in rows[1].summary()


def test_an_unresolved_ablation_row_is_not_marked_as_a_finding():
    obs = [Observation(f"a{i}", f"b{i}") for i in range(40)]
    values = list(np.random.default_rng(7).integers(0, 2, size=40).astype(float))
    rows = ablate(evaluation("full", values, obs), [("minus x", 2, evaluation("v", values, obs))])
    assert not rows[1].delta_vs_full.resolved
    assert "*" not in rows[1].summary()
