"""Splitting tests.

The point of every test here is that a split can be wrong in a way that makes the
*results* look better.  A leak does not raise; it reports a higher number.  So these
tests assert the structural properties that make the number honest, and one of them
measures the size of the leak the design exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest
from adml.predictor import PairwiseRanker, TrainConfig, encode
from adml.split import (
    Observation,
    cross_validate_folds,
    holdout,
    observations_from_pairs,
    partition,
    set_folds,
    split_report,
)
from adschema.annotation import ComparisonPair, PairKind


def make_corpus(n_sets: int = 10, per_set: int = 3):
    items = [f"s{s:03d}-i{i}" for s in range(n_sets) for i in range(per_set)]
    return items, {i: i.split("-")[0] for i in items}


def within_and_cross(items, set_of, *, rounds: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    obs = []
    by_set: dict[str, list[str]] = {}
    for item in items:
        by_set.setdefault(set_of[item], []).append(item)
    for members in by_set.values():
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                obs.append(Observation(a, b, PairKind.WITHIN_SET))
    for _ in range(rounds):
        shuffled = list(items)
        rng.shuffle(shuffled)
        for k in range(0, len(shuffled) - 1, 2):
            a, b = shuffled[k], shuffled[k + 1]
            if set_of[a] != set_of[b]:
                obs.append(Observation(a, b, PairKind.CROSS_SET))
    return obs


# --- Fold construction -------------------------------------------------------


def test_folds_are_disjoint_and_cover_everything():
    _, set_of = make_corpus(23)
    sets = sorted(set(set_of.values()))
    folds = set_folds(sets, k=5, seed=3)
    flat = [s for fold in folds for s in fold]
    assert sorted(flat) == sets
    assert len(flat) == len(set(flat))


def test_an_uneven_remainder_is_spread_not_dumped_in_the_last_fold():
    # 23 sets over 5 folds: sizes must differ by at most one, or the last fold
    # would carry a systematically different sample size from the others.
    _, set_of = make_corpus(23)
    sizes = [len(f) for f in set_folds(sorted(set(set_of.values())), k=5, seed=0)]
    assert max(sizes) - min(sizes) <= 1


def test_folds_are_deterministic_for_a_seed():
    _, set_of = make_corpus(12)
    sets = sorted(set(set_of.values()))
    assert set_folds(sets, k=4, seed=9) == set_folds(sets, k=4, seed=9)
    assert set_folds(sets, k=4, seed=9) != set_folds(sets, k=4, seed=10)


def test_too_few_sets_for_the_requested_folds_is_an_error():
    _, set_of = make_corpus(3)
    with pytest.raises(ValueError, match="cannot be split"):
        set_folds(sorted(set(set_of.values())), k=5)


# --- The core guarantee ------------------------------------------------------


def test_no_item_appears_on_both_sides_of_a_fold():
    """The property the whole module exists for."""
    items, set_of = make_corpus(20)
    obs = within_and_cross(items, set_of)
    for fold in cross_validate_folds(obs, set_of, k=5):
        train_items = {i for o in fold.train for i in o.items}
        test_items = {i for o in fold.test for i in o.items}
        assert not (train_items & test_items)


def test_a_generation_set_is_never_divided():
    items, set_of = make_corpus(20)
    obs = within_and_cross(items, set_of)
    for fold in cross_validate_folds(obs, set_of, k=5):
        for o in fold.test:
            assert set_of[o.winner] in set(fold.test_sets)
            assert set_of[o.loser] in set(fold.test_sets)


def test_straddling_comparisons_are_discarded_not_assigned():
    items, set_of = make_corpus(10)
    obs = within_and_cross(items, set_of, rounds=6)
    fold = partition(obs, set_of, {"s000", "s001"})
    for o in fold.discarded:
        in_test = [set_of[o.winner] in fold.test_sets, set_of[o.loser] in fold.test_sets]
        assert any(in_test) and not all(in_test)
    assert len(fold.train) + len(fold.test) + len(fold.discarded) == len(obs)


def test_within_set_comparisons_never_straddle():
    """Both items share a set by construction, so they always land together."""
    items, set_of = make_corpus(15)
    obs = within_and_cross(items, set_of)
    for fold in cross_validate_folds(obs, set_of, k=5):
        assert all(o.kind is not PairKind.WITHIN_SET for o in fold.discarded)


def test_an_item_with_no_known_set_is_discarded_rather_than_guessed():
    items, set_of = make_corpus(6)
    obs = [*within_and_cross(items, set_of), Observation("orphan", items[0])]
    fold = partition(obs, set_of, {"s000"})
    assert any(o.winner == "orphan" for o in fold.discarded)


def test_the_inner_holdout_only_ever_sees_training_sets():
    items, set_of = make_corpus(20)
    obs = within_and_cross(items, set_of)
    outer = cross_validate_folds(obs, set_of, k=5)[0]
    train_items = [i for i in items if set_of[i] in set(outer.train_sets)]
    inner = holdout(outer.train, {i: set_of[i] for i in train_items}, test_fraction=0.25)
    assert not set(inner.test_sets) & set(outer.test_sets)


# --- Accounting --------------------------------------------------------------


def test_the_report_counts_every_observation_exactly_once_per_fold():
    items, set_of = make_corpus(25)
    obs = within_and_cross(items, set_of, rounds=8)
    report = split_report(obs, set_of, k=5)
    assert report.trainable + report.testable + report.discarded == len(obs) * 5


def test_the_discard_fraction_matches_the_predicted_2pq():
    """~2 * p * (1-p) of cross-set comparisons straddle a p-sized holdout.

    Asserted because it is the arithmetic the annotation budget was set against: an
    honest 20% holdout throws away about a third of the cross-set comparisons, and
    that has to be planned for rather than discovered.
    """
    items, set_of = make_corpus(100)
    cross = [o for o in within_and_cross(items, set_of, rounds=8) if o.kind is PairKind.CROSS_SET]
    fold = partition(cross, set_of, sorted(set(set_of.values()))[:20])
    predicted = 2 * 0.2 * 0.8
    assert fold.n_discarded / len(cross) == pytest.approx(predicted, abs=0.05)


def test_the_margin_widens_as_the_test_set_shrinks():
    small = split_report(*_corpus_of(10), k=5)
    large = split_report(*_corpus_of(60), k=5)
    assert small.accuracy_margin() > large.accuracy_margin()


def _corpus_of(n_sets: int):
    items, set_of = make_corpus(n_sets)
    return within_and_cross(items, set_of, rounds=6), set_of


# --- The leak this all prevents ---------------------------------------------


def test_a_random_split_over_comparisons_reports_accuracy_that_is_not_there():
    """Uninformative features, so any held-out accuracy above 0.5 is leakage.

    True quality is idiosyncratic per item and unrelated to the features, and the
    features are high-dimensional enough to identify individual items.  A random
    split over comparisons therefore lets the model memorise item strengths and
    report them as generalisation; a set-wise split cannot.

    This is the measurement the module's design rests on, so it is asserted rather
    than described.
    """
    rng = np.random.default_rng(11)
    items, set_of = make_corpus(60)
    row = {item: i for i, item in enumerate(items)}
    x = rng.normal(size=(len(items), 200)) / np.sqrt(200)
    quality = rng.normal(size=len(items))  # nothing to do with x

    obs = []
    for template in within_and_cross(items, set_of, rounds=6, seed=2):
        a, b = template.items
        d = 1.5 * (quality[row[a]] - quality[row[b]])
        winner, loser = (a, b) if rng.random() < 1 / (1 + np.exp(-d)) else (b, a)
        obs.append(Observation(winner, loser, template.kind))

    config = TrainConfig(l2=1e-2, max_steps=800)

    def accuracy(train, test):
        model = PairwiseRanker.fit(x, encode(train, row), config=config)
        return model.accuracy(x, encode(test, row))

    shuffled = list(obs)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * 0.8)
    leaked = accuracy(shuffled[:cut], shuffled[cut:])
    honest = float(
        np.mean([accuracy(f.train, f.test) for f in cross_validate_folds(obs, set_of, k=5)])
    )

    assert honest == pytest.approx(0.5, abs=0.06), "set-wise split must find no signal"
    assert leaked > honest + 0.10, "the leak this module prevents did not reproduce"


# --- Building observations ---------------------------------------------------


def test_unjudged_pairs_are_skipped_rather_than_given_an_outcome():
    pairs = [
        ComparisonPair(pair_id="p1", item_a="a", item_b="b", kind=PairKind.WITHIN_SET),
        ComparisonPair(pair_id="p2", item_a="c", item_b="d", kind=PairKind.CROSS_SET),
    ]
    out = observations_from_pairs(pairs, {"p1": "b"})
    assert len(out) == 1
    assert (out[0].winner, out[0].loser) == ("b", "a")


def test_a_tie_keeps_the_observation_without_a_direction():
    pairs = [ComparisonPair(pair_id="p1", item_a="a", item_b="b")]
    out = observations_from_pairs(pairs, {"p1": None})
    assert out[0].is_tie


def test_catch_pairs_never_become_observations():
    pairs = [
        ComparisonPair(
            pair_id="p1", item_a="real", item_b="decoy", kind=PairKind.CATCH, expected_winner="real"
        )
    ]
    assert observations_from_pairs(pairs, {"p1": "real"}) == []
