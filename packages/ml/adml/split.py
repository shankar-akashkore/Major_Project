"""Train/test splitting for pairwise data, where the obvious split is wrong.

The tempting split is a random one over *comparisons*: shuffle the 1,500 pairs,
hold back 20%.  It leaks badly.  Every item appears in about ten comparisons, so
an item held out in one pair is almost certainly present in the training set
through nine others, and the model can learn that specific item's strength
directly instead of learning what makes an image good.  Reported accuracy goes up
and generalisation to a new job goes down — the exact opposite of what the number
is supposed to certify.

The honest unit is the **generation set**: three candidates rendered from one
pair of reference photographs.  A set goes entirely to train or entirely to test,
because that is the deployment question — *given references the model has never
seen, rank the three candidates.*

That costs real data, and the cost is the thing worth knowing before designing
the annotation budget rather than after.  A comparison whose two items land on
opposite sides of the boundary is unusable in either half: it cannot train
(one item is held out) and it cannot test (one item was trained on).  With a
20% test fraction, roughly ``2 x 0.8 x 0.2 = 32%`` of *cross-set* comparisons
straddle and are discarded.  Within-set comparisons never straddle, since both
items share a set by construction.

Two consequences follow, and both are load-bearing for the evaluation:

* **The test fold is small.**  Measured on the planned corpus — 100 sets, 1,500
  designed comparisons — a single 20% fold yields about 105 testable comparisons.
  A point estimate on 105 observations has a +-9 point interval, which cannot
  separate a good model from a mediocre one.  So results are **pooled across
  folds** (:mod:`adml.evaluate`) rather than averaged as per-fold means.
* **The test fold is mostly within-set comparisons**, which is a feature, not a
  bug: within-set ranking *is* the deployment task.  Cross-set comparisons earn
  their place by making the strength scale mutually comparable, and they still
  contribute test observations, just proportionally fewer.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from adschema.annotation import ComparisonPair, PairKind

#: Comparison kinds that are never modelled.  Catch pairs have a correct answer
#: rather than a preference, and a decoy's strength would distort the scale it
#: shares with real items.
EXCLUDED_KINDS = (PairKind.CATCH,)


@dataclass(frozen=True)
class Observation:
    """One modelled comparison: ``winner`` was preferred to ``loser``.

    ``is_tie`` observations carry no direction.  They are kept rather than dropped
    because a recorded tie is data — the loss treats them as half a win each way,
    matching :func:`adml.ranking.fit_bradley_terry`.
    """

    winner: str
    loser: str
    kind: PairKind = PairKind.CROSS_SET
    annotator_id: str | None = None
    is_tie: bool = False

    @property
    def items(self) -> tuple[str, str]:
        return (self.winner, self.loser)


@dataclass
class Fold:
    """One train/test division, with the discards accounted for explicitly."""

    index: int
    train_sets: list[str]
    test_sets: list[str]
    train: list[Observation]
    test: list[Observation]
    discarded: list[Observation] = field(default_factory=list)

    @property
    def n_discarded(self) -> int:
        return len(self.discarded)

    def test_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for obs in self.test:
            counts[obs.kind.value] = counts.get(obs.kind.value, 0) + 1
        return counts

    def summary(self) -> str:
        by_kind = ", ".join(f"{k} {v}" for k, v in sorted(self.test_counts().items()))
        return (
            f"fold {self.index}: {len(self.train_sets)}/{len(self.test_sets)} sets  "
            f"train {len(self.train):5d}  test {len(self.test):4d} ({by_kind or 'none'})  "
            f"discarded {self.n_discarded:5d}"
        )


def observations_from_pairs(
    pairs: Iterable[ComparisonPair],
    winners: dict[str, str | None],
    *,
    annotators: dict[str, str] | None = None,
) -> list[Observation]:
    """Turn stored pairs plus their judged outcomes into modelling observations.

    ``winners`` maps ``pair_id`` to the winning item id, or ``None`` for a tie.
    Pairs absent from the mapping were never judged and are skipped: an unjudged
    pair is missing data, and inventing an outcome for it would be fabrication.
    """
    annotators = annotators or {}
    out: list[Observation] = []
    for pair in pairs:
        if pair.kind in EXCLUDED_KINDS:
            continue
        if pair.pair_id not in winners:
            continue
        winner = winners[pair.pair_id]
        if winner is None:
            out.append(
                Observation(
                    winner=pair.item_a,
                    loser=pair.item_b,
                    kind=pair.kind,
                    annotator_id=annotators.get(pair.pair_id),
                    is_tie=True,
                )
            )
            continue
        loser = pair.other(winner)
        out.append(
            Observation(
                winner=winner,
                loser=loser,
                kind=pair.kind,
                annotator_id=annotators.get(pair.pair_id),
            )
        )
    return out


def set_folds(
    set_ids: Sequence[str],
    *,
    k: int = 5,
    seed: int = 0,
) -> list[list[str]]:
    """Partition set ids into ``k`` disjoint groups of roughly equal size.

    Deterministic given ``seed`` so a reported number can be reproduced exactly.
    """
    if k < 2:
        raise ValueError(f"need at least 2 folds, got {k}")
    unique = sorted(set(set_ids))
    if len(unique) < k:
        raise ValueError(f"{len(unique)} sets cannot be split into {k} folds")
    shuffled = unique[:]
    random.Random(seed).shuffle(shuffled)
    # Strided rather than sliced, so an uneven remainder is spread across folds
    # instead of landing entirely in the last one.
    return [shuffled[i::k] for i in range(k)]


def partition(
    observations: Iterable[Observation],
    set_of: dict[str, str],
    test_sets: Iterable[str],
    *,
    index: int = 0,
) -> Fold:
    """Split observations by which side of the set boundary their items fall on.

    An observation naming an item with no known set is discarded rather than
    guessed at — that is a data-integrity problem, and silently training on it
    would hide the problem behind a plausible number.
    """
    held = set(test_sets)
    rows = list(observations)
    train: list[Observation] = []
    test: list[Observation] = []
    discarded: list[Observation] = []

    for obs in rows:
        a, b = set_of.get(obs.winner), set_of.get(obs.loser)
        if a is None or b is None:
            discarded.append(obs)
            continue
        in_a, in_b = a in held, b in held
        if in_a and in_b:
            test.append(obs)
        elif not in_a and not in_b:
            train.append(obs)
        else:
            discarded.append(obs)

    all_sets = sorted(set(set_of.values()))
    return Fold(
        index=index,
        train_sets=[s for s in all_sets if s not in held],
        test_sets=sorted(held),
        train=train,
        test=test,
        discarded=discarded,
    )


def cross_validate_folds(
    observations: Iterable[Observation],
    set_of: dict[str, str],
    *,
    k: int = 5,
    seed: int = 0,
) -> list[Fold]:
    """``k`` folds over sets, each holding out one group."""
    rows = list(observations)
    groups = set_folds(sorted(set(set_of.values())), k=k, seed=seed)
    return [partition(rows, set_of, group, index=i) for i, group in enumerate(groups)]


def holdout(
    observations: Iterable[Observation],
    set_of: dict[str, str],
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Fold:
    """A single set-wise holdout, for the inner validation split during training.

    Used to decide when to stop fitting.  Nested inside the outer fold, and drawn
    from the outer fold's *training* sets only — a validation split that touches
    test sets is a test-set peek with extra steps.
    """
    sets = sorted(set(set_of.values()))
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    shuffled = sets[:]
    random.Random(seed).shuffle(shuffled)
    n = max(1, round(len(sets) * test_fraction))
    return partition(observations, set_of, shuffled[:n])


@dataclass
class SplitReport:
    """What an honest split costs, in the units the annotation budget is set in."""

    n_observations: int
    n_sets: int
    k: int
    trainable: int
    testable: int
    discarded: int
    testable_within_set: int
    testable_cross_set: int

    @property
    def discard_fraction(self) -> float:
        total = self.trainable + self.testable + self.discarded
        return self.discarded / total if total else float("nan")

    @property
    def testable_per_fold(self) -> float:
        return self.testable / self.k if self.k else float("nan")

    def accuracy_margin(self, p: float = 0.70) -> float:
        """Half-width of the 95% interval for a pooled accuracy of ``p``.

        The number that decides whether the evaluation can resolve anything.  A
        margin wider than the gap between two models means the comparison is
        unresolvable at this corpus size, however the models actually differ —
        which is an argument for more *sets*, not for more comparisons per set.
        """
        if self.testable < 2:
            return float("nan")
        return 1.96 * (p * (1.0 - p) / self.testable) ** 0.5

    def summary(self) -> str:
        return (
            f"{self.n_observations} observations over {self.n_sets} sets, {self.k} folds\n"
            f"  trainable (pooled)   {self.trainable}\n"
            f"  testable  (pooled)   {self.testable}"
            f"   [within-set {self.testable_within_set}, cross-set {self.testable_cross_set}]\n"
            f"  discarded (straddle) {self.discarded}  ({self.discard_fraction:.0%})\n"
            f"  per fold: ~{self.testable_per_fold:.0f} test comparisons\n"
            f"  pooled 95% margin at p=0.70: +-{self.accuracy_margin():.3f}"
        )


def split_report(
    observations: Iterable[Observation],
    set_of: dict[str, str],
    *,
    k: int = 5,
    seed: int = 0,
) -> SplitReport:
    """Measure the cost of the split before collecting labels against it."""
    rows = list(observations)
    folds = cross_validate_folds(rows, set_of, k=k, seed=seed)
    testable = [obs for fold in folds for obs in fold.test]
    return SplitReport(
        n_observations=len(rows),
        n_sets=len({s for s in set_of.values()}),
        k=k,
        trainable=sum(len(f.train) for f in folds),
        testable=len(testable),
        discarded=sum(f.n_discarded for f in folds),
        testable_within_set=sum(1 for o in testable if o.kind is PairKind.WITHIN_SET),
        testable_cross_set=sum(1 for o in testable if o.kind is not PairKind.WITHIN_SET),
    )
