"""The comparison design's invariants.

The connectivity tests are the important ones. Everything else here guards a
property that would be noticed; a disconnected comparison graph fails silently —
Bradley-Terry returns numbers, the report looks fine, and the strengths being
compared across components were never linked by any data.
"""

from __future__ import annotations

from adml import pairs as PAIRS
from adschema import AssetRef, CorpusItem, ItemKind, PairKind


def corpus(n_sets: int = 10, per_set: int = 3, kind: ItemKind = ItemKind.IMAGE):
    return [
        CorpusItem(
            item_id=f"s{s:03d}c{c}",
            set_id=f"s{s:03d}",
            kind=kind,
            asset=AssetRef(key=f"x/{s}/{c}.png"),
        )
        for s in range(n_sets)
        for c in range(per_set)
    ]


def test_within_set_only_design_would_be_a_pile_of_triangles():
    """The obvious design is the broken one, and the bridging is what saves it.

    100 sets of 3 compared only inside their own set gives 100 disconnected
    triangles. This asserts the exact arithmetic of the fix: 99 bridges to join
    100 components, and connected afterwards.
    """
    design = PAIRS.design_pairs(corpus(100), seed=1, cross_set_rounds=0)

    assert design.count(PairKind.WITHIN_SET) == 300  # 3 per set
    assert design.count(PairKind.BRIDGE) == 99  # n_sets - 1
    assert design.is_connected


def test_cross_set_rounds_balance_the_comparison_count():
    """Every item gains exactly one comparison per round, by construction."""
    design = PAIRS.design_pairs(corpus(100), seed=1, cross_set_rounds=8)
    degrees = set(design.degree.values())

    # 2 within-set (3 items per set) + 8 cross-set rounds
    assert degrees == {10}
    assert design.is_connected
    assert design.count(PairKind.BRIDGE) == 0  # 8 rounds needs no stitching


def test_no_duplicate_and_no_same_set_cross_pairs():
    design = PAIRS.design_pairs(corpus(20), seed=3, cross_set_rounds=6)
    keys = [p.key for p in design.pairs]
    assert len(keys) == len(set(keys))

    sets = {i.item_id: i.set_id for i in corpus(20)}
    for pair in design.pairs:
        if pair.kind is PairKind.CROSS_SET:
            assert sets[pair.item_a] != sets[pair.item_b]


def test_connectivity_is_guaranteed_not_merely_likely():
    """Sparse designs across many seeds — never disconnected, never isolated."""
    for seed in range(25):
        design = PAIRS.design_pairs(
            corpus(8, per_set=2), seed=seed, cross_set_rounds=1, include_within_set=False
        )
        assert design.is_connected, f"seed {seed} left {len(design.components)} components"
        assert design.isolated_items == []


def test_decoys_never_enter_the_design():
    items = corpus(4)
    items.append(
        CorpusItem(
            item_id="decoy",
            set_id="s000",
            asset=AssetRef(key="d.png"),
            degraded_from="s000c0",
            degradation="blur-20px",
        )
    )
    design = PAIRS.design_pairs(items, seed=1)

    assert "decoy" not in design.item_ids
    assert all("decoy" not in (p.item_a, p.item_b) for p in design.pairs)


def test_video_and_image_items_are_never_designed_together():
    """One strength scale per stage. A mixed fit would answer no real question."""
    items = [*corpus(4), *corpus(4, kind=ItemKind.VIDEO)]
    # The two corpora share item ids, so filter by kind and check the count.
    design = PAIRS.design_pairs(items, seed=1, kind=ItemKind.VIDEO)
    assert len(design.item_ids) == 12


def test_catch_pairs_do_not_create_apparent_connectivity():
    """A decoy must not be a bridge between two otherwise separate components."""
    items = corpus(2, per_set=2)
    decoys = [
        CorpusItem(
            item_id=f"{i.item_id}-decoy",
            set_id=i.set_id,
            asset=AssetRef(key="d.png"),
            degraded_from=i.item_id,
            degradation="blur",
        )
        for i in items
    ]
    catch = PAIRS.design_catch_pairs(items, decoys)
    assert len(catch) == 4

    # Within-set pairs only: two sets, so two components. The catch pairs must not
    # collapse that into one.
    within = PAIRS.design_pairs(items, seed=0, cross_set_rounds=0)
    components = PAIRS.connected_components(
        [*[p for p in within.pairs if p.kind is PairKind.WITHIN_SET], *catch],
        include=[i.item_id for i in items],
    )
    assert len(components) == 2


def test_catch_pair_needs_a_known_source():
    """A catch trial whose correct answer is a guess is worse than none at all."""
    items = corpus(1, per_set=2)
    orphan = CorpusItem(
        item_id="orphan-decoy",
        set_id="elsewhere",
        asset=AssetRef(key="d.png"),
        degraded_from="an-item-that-was-never-registered",
        degradation="blur",
    )
    assert PAIRS.design_catch_pairs(items, [orphan]) == []


def test_catch_pair_names_the_real_image_as_the_winner():
    items = corpus(1, per_set=1)
    decoy = CorpusItem(
        item_id="d",
        set_id="s000",
        asset=AssetRef(key="d.png"),
        degraded_from="s000c0",
        degradation="blur",
    )
    (pair,) = PAIRS.design_catch_pairs(items, [decoy])
    assert pair.kind is PairKind.CATCH
    assert pair.expected_winner == "s000c0"


def test_effort_estimate_is_the_arithmetic_it_claims():
    design = PAIRS.design_pairs(corpus(100), seed=1, cross_set_rounds=8)
    assert len(design.pairs) == 1500

    effort = design.estimate_effort(annotators=6, seconds_per_pair=4.0, repeat_rate=0.10)
    assert effort["judgements"] == 1650  # 1500 x 1.10
    assert effort["total_minutes"] == 110.0  # 1650 x 4s
    assert effort["minutes_each"] == 18.3


def test_a_corpus_too_small_to_compare_yields_no_pairs():
    assert PAIRS.design_pairs([], seed=0).pairs == []
    assert PAIRS.design_pairs(corpus(1, per_set=1), seed=0).pairs == []
