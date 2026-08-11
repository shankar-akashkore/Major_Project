"""Designing *which* pairs to show annotators.

Human comparison time is the scarcest resource in this project — scarcer than the
$35 — so which pairs get shown decides how much the labels are worth.  Three
things drive the design, and the first is the one that is easy to get wrong.

**The comparison graph must be connected.**  Bradley-Terry infers a single latent
strength per item from who beat whom.  If the graph of "who was compared to whom"
splits into disconnected components, the strengths within each component are
estimable and *strengths across components are not comparable at all* — there is
no data linking the scales.  This matters here specifically because the obvious
design is "compare the 3 candidates within each job", and that produces exactly
the pathological case: 100 disconnected triangles, 100 incomparable scales.
:func:`design_pairs` therefore adds cross-set comparisons and then *verifies*
connectivity with union-find, stitching any remaining components with explicit
``BRIDGE`` pairs.  A probabilistic argument that random matchings are usually
connected is not the same as a guarantee, and this is cheap to guarantee.

**Degree must be balanced.**  Sampling pairs uniformly at random leaves some
items with one comparison and others with fifteen, and an item nobody compared
has no estimable strength.  Cross-set pairs are generated as a sequence of random
near-perfect matchings, so every item gains one comparison per round by
construction.

**Within-set pairs are the ones that answer the research question.**  They hold
the product, the model and the brand constant, so the comparison isolates the
design point — which is both the deployment task (rank 3 candidates for one job)
and what the sampler ablation needs.  Cross-set pairs are the connective tissue
that makes them jointly analysable.

Sizing: recovering a reliable ranking over ``n`` items by pairwise comparison
needs on the order of ``n log n`` comparisons.  For the planned 300-image corpus
that is ~1,700 pairs, which :func:`estimate_effort` turns into human-hours so the
design can be checked for feasibility before anyone is asked to sit down with it.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from adschema import ComparisonPair, CorpusItem, ItemKind, PairKind

#: Cross-set matchings per item.  8 rounds over 300 items in sets of 3 gives
#: 300 within-set + ~1,200 cross-set pairs = ~1,500, close to the n log n target
#: while keeping every item at ~10 comparisons.
DEFAULT_CROSS_SET_ROUNDS = 8

#: Observed seconds per comparison in this kind of task, used only for effort
#: estimation.  Measured for real by `scripts/annotation_report.py` once the first
#: session has run — this is a planning figure, not a finding.
ASSUMED_SECONDS_PER_PAIR = 4.0


class _UnionFind:
    """Disjoint-set forest with path compression.  Small and exact."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self._parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # compress
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> bool:
        """Join two sets.  Returns True when they were previously separate."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._parent[rb] = ra
        return True


def connected_components(
    pairs: Iterable[ComparisonPair], *, include: Iterable[str] | None = None
) -> list[list[str]]:
    """Components of the comparison graph, largest first.

    ``include`` adds items that must appear even if no pair mentions them — an
    item with zero comparisons is its own component and is exactly the case a
    connectivity check needs to surface rather than hide.

    ``CATCH`` pairs are excluded: a decoy is not a corpus item and linking
    through one would report a connectivity the real data does not have.
    """
    uf = _UnionFind()
    for item in include or ():
        uf.add(item)
    for pair in pairs:
        if pair.kind is PairKind.CATCH:
            continue
        uf.add(pair.item_a)
        uf.add(pair.item_b)
        uf.union(pair.item_a, pair.item_b)

    groups: dict[str, list[str]] = {}
    for item in list(uf._parent):
        groups.setdefault(uf.find(item), []).append(item)
    return sorted((sorted(g) for g in groups.values()), key=len, reverse=True)


@dataclass
class PairDesign:
    """A designed set of comparisons, with the diagnostics needed to trust it."""

    pairs: list[ComparisonPair] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)
    seed: int = 0

    def count(self, kind: PairKind) -> int:
        return sum(1 for p in self.pairs if p.kind is kind)

    @property
    def degree(self) -> dict[str, int]:
        """Comparisons per item, counting only pairs that feed the model."""
        out = dict.fromkeys(self.item_ids, 0)
        for pair in self.pairs:
            if pair.kind is PairKind.CATCH:
                continue
            for item in (pair.item_a, pair.item_b):
                if item in out:
                    out[item] += 1
        return out

    @property
    def components(self) -> list[list[str]]:
        return connected_components(self.pairs, include=self.item_ids)

    @property
    def is_connected(self) -> bool:
        """True when every item's strength is comparable with every other's."""
        return len(self.components) <= 1

    @property
    def isolated_items(self) -> list[str]:
        return [item for item, d in self.degree.items() if d == 0]

    def estimate_effort(
        self,
        *,
        judgements_per_pair: float = 1.0,
        annotators: int = 1,
        seconds_per_pair: float = ASSUMED_SECONDS_PER_PAIR,
        repeat_rate: float = 0.0,
    ) -> dict[str, float]:
        """How much human time this design costs, before asking anyone for it."""
        judgements = len(self.pairs) * judgements_per_pair * (1.0 + repeat_rate)
        total_minutes = judgements * seconds_per_pair / 60.0
        return {
            "judgements": round(judgements),
            "total_minutes": round(total_minutes, 1),
            "minutes_each": round(total_minutes / max(annotators, 1), 1),
        }

    def summary(self) -> str:
        deg = self.degree
        lo = min(deg.values()) if deg else 0
        hi = max(deg.values()) if deg else 0
        mean = sum(deg.values()) / len(deg) if deg else 0.0
        parts = [
            f"{len(self.pairs)} pairs over {len(self.item_ids)} items",
            f"within-set {self.count(PairKind.WITHIN_SET)}",
            f"cross-set {self.count(PairKind.CROSS_SET)}",
            f"bridge {self.count(PairKind.BRIDGE)}",
            f"catch {self.count(PairKind.CATCH)}",
            f"degree min/mean/max {lo}/{mean:.1f}/{hi}",
            "connected" if self.is_connected else f"DISCONNECTED ({len(self.components)} parts)",
        ]
        return " | ".join(parts)


def design_pairs(
    items: Sequence[CorpusItem],
    *,
    seed: int = 0,
    cross_set_rounds: int = DEFAULT_CROSS_SET_ROUNDS,
    include_within_set: bool = True,
    kind: ItemKind | None = None,
) -> PairDesign:
    """Design the comparison set for a corpus.

    Decoys are skipped — they only ever appear in catch pairs, built separately by
    :func:`design_catch_pairs`.  ``kind`` restricts the design to image or video
    items; mixing the two into one Bradley-Terry fit would produce a strength
    scale spanning two different stages, which answers no question the project asks.
    """
    pool = [it for it in items if not it.is_decoy and (kind is None or it.kind is kind)]
    if len(pool) < 2:
        return PairDesign(pairs=[], item_ids=[it.item_id for it in pool], seed=seed)

    rng = random.Random(seed)
    by_id = {it.item_id: it for it in pool}
    ids = list(by_id)
    pairs: list[ComparisonPair] = []
    used: set[tuple[str, str]] = set()

    def emit(a: str, b: str, pair_kind: PairKind) -> bool:
        key = (a, b) if a <= b else (b, a)
        if key in used:
            return False
        used.add(key)
        pairs.append(ComparisonPair(item_a=a, item_b=b, kind=pair_kind))
        return True

    # --- Within-set: every pair inside each generation set ---
    if include_within_set:
        sets: dict[str, list[str]] = {}
        for item in pool:
            sets.setdefault(item.set_id, []).append(item.item_id)
        for members in sets.values():
            members = sorted(members)
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    emit(a, b, PairKind.WITHIN_SET)

    # --- Cross-set: one random near-perfect matching per round ---
    # Each round gives every item one more comparison, so degree stays balanced
    # by construction rather than by luck.
    for _ in range(max(cross_set_rounds, 0)):
        unmatched = ids[:]
        rng.shuffle(unmatched)
        while len(unmatched) >= 2:
            a = unmatched.pop()
            partner_at = None
            for idx in range(len(unmatched) - 1, -1, -1):
                b = unmatched[idx]
                key = (a, b) if a <= b else (b, a)
                if by_id[a].set_id != by_id[b].set_id and key not in used:
                    partner_at = idx
                    break
            if partner_at is None:
                # No legal partner left this round. Dropping `a` is correct: the
                # alternative is a duplicate pair, which buys no information.
                continue
            emit(a, unmatched.pop(partner_at), PairKind.CROSS_SET)

    # --- Guarantee connectivity ---
    # Random matchings are *usually* connected. "Usually" is not a property you
    # want a headline measurement to rest on, so any remaining components are
    # joined explicitly and counted as BRIDGE so the report can say how many.
    design = PairDesign(pairs=pairs, item_ids=ids, seed=seed)
    components = design.components
    while len(components) > 1:
        first, second = components[0], components[1]
        a = rng.choice(first)
        b = rng.choice(second)
        if not emit(a, b, PairKind.BRIDGE):
            # That exact pair already exists, so the two must be joined via some
            # other member; pick deterministically to guarantee progress.
            joined = False
            for x in first:
                for y in second:
                    if emit(x, y, PairKind.BRIDGE):
                        joined = True
                        break
                if joined:
                    break
            if not joined:  # pragma: no cover - impossible: components are disjoint
                raise RuntimeError("could not bridge two disjoint components")
        design = PairDesign(pairs=pairs, item_ids=ids, seed=seed)
        components = design.components

    return design


def design_catch_pairs(
    items: Sequence[CorpusItem],
    decoys: Sequence[CorpusItem],
    *,
    seed: int = 0,
) -> list[ComparisonPair]:
    """Pair each decoy against the item it was degraded from.

    A catch trial has a correct answer, which is what makes it a screening tool
    rather than another preference.  Decoys whose source is missing from ``items``
    are skipped rather than paired against something arbitrary — a catch trial
    whose "correct" answer is a guess is worse than no catch trial.
    """
    del seed  # kept for signature symmetry with design_pairs; nothing is sampled
    known = {it.item_id for it in items if not it.is_decoy}
    out: list[ComparisonPair] = []
    for decoy in decoys:
        source = decoy.degraded_from
        if source is None or source not in known:
            continue
        out.append(
            ComparisonPair(
                item_a=source,
                item_b=decoy.item_id,
                kind=PairKind.CATCH,
                expected_winner=source,
            )
        )
    return out
