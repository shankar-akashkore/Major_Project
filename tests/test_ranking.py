"""Bradley-Terry and the evaluation metrics.

The metrics are checked against values computed by hand rather than against
another implementation of the same formula, because the failure mode worth
catching is "the formula is subtly wrong", and a second copy of a wrong formula
agrees with the first.
"""

from __future__ import annotations

import math
import random

import pytest
from adml import ranking as R

# --- Bradley-Terry -----------------------------------------------------------


def _synthetic_comparisons(strengths: dict[str, float], n: int, seed: int = 0):
    """Comparisons drawn from the Bradley-Terry model itself."""
    rng = random.Random(seed)
    ids = list(strengths)
    out = []
    for _ in range(n):
        a, b = rng.sample(ids, 2)
        p_a = 1.0 / (1.0 + math.exp(-(strengths[a] - strengths[b])))
        out.append((a, b) if rng.random() < p_a else (b, a))
    return out


def test_recovers_a_known_ranking():
    truth = {f"i{k}": k * 0.8 for k in range(6)}
    fit = R.fit_bradley_terry(_synthetic_comparisons(truth, 4000, seed=1))

    assert fit.converged
    assert fit.is_connected
    assert fit.ranking() == ["i5", "i4", "i3", "i2", "i1", "i0"]

    ids = list(fit.strengths)
    rho = R.spearman_rho([truth[i] for i in ids], [fit.strengths[i] for i in ids])
    assert rho == pytest.approx(1.0)


def test_recovers_the_spacing_not_only_the_order():
    """Strength differences carry meaning, so check them, not just the sort."""
    truth = {"a": 2.0, "b": 0.0, "c": -2.0}
    fit = R.fit_bradley_terry(_synthetic_comparisons(truth, 6000, seed=2))
    # Tolerance, not equality: these are 6000 sampled comparisons, so the fit
    # carries sampling noise. The point is that the *gap* is recovered, not only
    # the order — a fit that got the order right and the gap wrong would pass a
    # ranking assertion and still mispredict every margin.
    assert fit.strengths["a"] - fit.strengths["c"] == pytest.approx(4.0, abs=0.2)
    assert fit.probability("a", "c") == pytest.approx(1 / (1 + math.exp(-4.0)), abs=0.01)


def test_strengths_are_centred_on_zero():
    fit = R.fit_bradley_terry([("a", "b"), ("b", "c"), ("c", "a"), ("a", "c")])
    assert abs(sum(fit.strengths.values())) < 1e-9


def test_an_undefeated_item_stays_finite():
    """The unregularised MLE diverges here. The prior is what stops it."""
    fit = R.fit_bradley_terry([("a", "b"), ("a", "c"), ("a", "d"), ("a", "b")])

    assert fit.converged
    assert math.isfinite(fit.strengths["a"])
    assert fit.strengths["a"] < 5.0  # pulled toward the middle, not run away
    assert fit.ranking()[0] == "a"


def test_a_stronger_prior_shrinks_harder():
    weak = R.fit_bradley_terry([("a", "b")] * 5, prior_strength=0.1)
    strong = R.fit_bradley_terry([("a", "b")] * 5, prior_strength=10.0)
    assert strong.strengths["a"] < weak.strengths["a"]


def test_ties_are_counted_as_half_a_win_each_way():
    """Two items that only ever tie must come out equal."""
    fit = R.fit_bradley_terry([], ties=[("a", "b")] * 10)
    assert abs(fit.strengths["a"] - fit.strengths["b"]) < 1e-9
    assert fit.n_comparisons == 10


def test_a_tie_pulls_toward_equality():
    losses = [("a", "b")] * 6
    without = R.fit_bradley_terry(losses)
    with_ties = R.fit_bradley_terry(losses, ties=[("a", "b")] * 6)
    assert with_ties.strengths["a"] < without.strengths["a"]


def test_items_with_no_comparisons_are_reported_at_the_prior():
    fit = R.fit_bradley_terry([("a", "b")], items=["a", "b", "lonely"])
    assert fit.strengths["lonely"] == 0.0 or abs(fit.strengths["lonely"]) < 0.6
    assert not fit.is_connected  # the isolated item is its own component


def test_disconnection_is_reported_rather_than_hidden():
    """Two islands: computable because of the prior, not comparable in fact."""
    fit = R.fit_bradley_terry([("a", "b"), ("c", "d")])
    assert fit.converged
    assert not fit.is_connected


def test_a_self_comparison_is_refused():
    try:
        R.fit_bradley_terry([("a", "a")])
    except ValueError as exc:
        assert "two different items" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a self-comparison should not be accepted")


# --- Rank correlation --------------------------------------------------------


def test_spearman_hand_computed():
    # Perfectly reversed ranks over 4 items is exactly -1.
    assert R.spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert R.spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    # One adjacent swap out of 4: rho = 1 - 6*sum(d^2)/(n(n^2-1)) = 1 - 6*2/60 = 0.8
    assert R.spearman_rho([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(0.8)


def test_spearman_is_nan_for_a_constant_vector():
    """A vector with no rank information has no correlation, and 0.0 would read
    as 'measured, and independent'."""
    assert math.isnan(R.spearman_rho([1, 1, 1], [1, 2, 3]))


def test_kendall_tau_b_hand_computed():
    assert R.kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert R.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # 6 pairs, one discordant: (5-1)/6
    assert R.kendall_tau([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(4 / 6)


def test_kendall_tau_b_corrects_for_ties():
    """tau-a would understate this; tau-b divides by the untied pair counts."""
    # x: 3 pairs, all untied. y ties the first two.
    tau = R.kendall_tau([1, 2, 3], [1, 1, 2])
    # C=2 (pairs (1,3),(2,3)), D=0, tie_y=1 → 2/sqrt(3*2)
    assert tau == pytest.approx(2 / math.sqrt(6))


# --- Predictive accuracy -----------------------------------------------------


def test_pairwise_accuracy_scores_ties_as_a_coin_flip():
    scores = {"a": 1.0, "b": 0.0, "c": 0.0}
    assert R.pairwise_accuracy(scores, [("a", "b")]) == 1.0
    assert R.pairwise_accuracy(scores, [("b", "a")]) == 0.0
    assert R.pairwise_accuracy(scores, [("b", "c")]) == 0.5
    assert R.pairwise_accuracy(scores, [("a", "b"), ("b", "a")]) == 0.5


def test_pairwise_accuracy_skips_items_the_scorer_never_saw():
    """Unknown items are not failures — the model was not asked about them."""
    scores = {"a": 1.0, "b": 0.0}
    assert R.pairwise_accuracy(scores, [("a", "b"), ("x", "y")]) == 1.0
    assert math.isnan(R.pairwise_accuracy(scores, [("x", "y")]))


def test_top1_retention_is_the_counterfactual():
    sets = [
        (["a", "b", "c"], ["a", "c", "b"]),  # kept the human's favourite
        (["b", "a", "c"], ["a", "b", "c"]),  # lost it
    ]
    assert R.top1_retention(sets) == 0.5


def test_ndcg_at_1_penalises_by_how_wrong_the_pick_was():
    relevance = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert R.ndcg_at_k(["a", "b", "c"], relevance, k=1) == 1.0
    assert R.ndcg_at_k(["b", "a", "c"], relevance, k=1) == 2 / 3
    assert R.ndcg_at_k(["c", "a", "b"], relevance, k=1) == 1 / 3


# --- Agreement ---------------------------------------------------------------


def test_cohen_kappa_hand_computed():
    a = ["x", "x", "y", "y"]
    assert R.cohen_kappa(a, a) == 1.0
    # Observed 0.5; expected = 0.5*0.5 + 0.5*0.5 = 0.5 → kappa 0
    assert R.cohen_kappa(["x", "x", "y", "y"], ["x", "y", "x", "y"]) == pytest.approx(0.0)


def test_cohen_kappa_is_nan_when_everyone_always_says_the_same_thing():
    assert math.isnan(R.cohen_kappa(["x", "x"], ["x", "x"]))


def test_krippendorff_alpha_bounds():
    assert R.krippendorff_alpha_nominal([["a", "a"], ["b", "b"]]) == pytest.approx(1.0)
    # Complete disagreement over 2 units is negative; the exact value carries the
    # (n-1) small-sample correction, so bound it rather than pinning it.
    alpha = R.krippendorff_alpha_nominal([["a", "b"], ["b", "a"]])
    assert alpha < 0.0


def test_krippendorff_alpha_ignores_units_with_one_rating():
    """A pair only one person judged carries no agreement information."""
    with_singles = R.krippendorff_alpha_nominal([["a", "a"], ["b", "b"], ["a"], ["b"]])
    without = R.krippendorff_alpha_nominal([["a", "a"], ["b", "b"]])
    assert with_singles == without


def test_krippendorff_alpha_handles_uneven_rater_counts():
    """The reason it is used instead of kappa: 3 raters here, 2 there."""
    alpha = R.krippendorff_alpha_nominal([["a", "a", "a"], ["b", "b"], ["a", "b", "a"]])
    assert -1.0 <= alpha <= 1.0


# --- Reliability -------------------------------------------------------------


def test_split_half_splits_annotators_not_comparisons():
    """Agreeing annotators correlate; the split must be on people."""
    truth = {f"i{k}": k * 1.0 for k in range(8)}
    rows = []
    for annotator in range(6):
        for winner, loser in _synthetic_comparisons(truth, 400, seed=annotator):
            rows.append((f"a{annotator}", winner, loser))

    split = R.split_half_reliability(rows, seed=0)
    assert split.left_annotators == 3
    assert split.right_annotators == 3
    assert split.n_shared_items == 8
    assert split.rho > 0.8


def test_split_half_needs_two_annotators():
    result = R.split_half_reliability([("only", "a", "b")])
    assert math.isnan(result.rho)


def test_bootstrap_ci_brackets_the_mean():
    values = [0.7] * 50 + [0.9] * 50
    lo, hi = R.bootstrap_ci(values, resamples=500, seed=0)
    assert lo < 0.8 < hi
    assert hi - lo < 0.1  # 100 samples is a tight-ish interval


def test_bootstrap_ci_is_wider_for_less_data():
    rng = random.Random(0)
    small = [rng.random() for _ in range(10)]
    large = [rng.random() for _ in range(1000)]
    lo_s, hi_s = R.bootstrap_ci(small, seed=1)
    lo_l, hi_l = R.bootstrap_ci(large, seed=1)
    assert (hi_s - lo_s) > (hi_l - lo_l)
