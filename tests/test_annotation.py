"""The annotation store and its HTTP surface.

The tests that matter most are the ones about what the *client* controls. A
judgement is only useful if the record describes what was actually on screen, so
the side served and the showing index are derived server-side — and these assert
that a client cannot influence either, because a client that could would corrupt
the position-bias analysis silently.
"""

from __future__ import annotations

import random

import pytest
from adapi import annotate
from adapi.annotation_store import (
    CATCH_CHECKPOINTS,
    AnnotationStore,
    presentation_side,
)
from adschema import (
    MIN_CATCH_TRIALS,
    AnnotatorProfile,
    AssetRef,
    Choice,
    ComparisonPair,
    CorpusItem,
    ItemKind,
    PairKind,
)


@pytest.fixture
async def store() -> AnnotationStore:
    """An in-memory corpus: 8 sets of 3, plus decoys and their catch pairs."""
    store = AnnotationStore.from_url("sqlite+aiosqlite://")
    await store.create_all()

    items = [
        CorpusItem(
            item_id=f"s{s}c{c}",
            set_id=f"s{s}",
            kind=ItemKind.IMAGE,
            asset=AssetRef(key=f"img/{s}/{c}.png"),
        )
        for s in range(8)
        for c in range(3)
    ]
    decoys = [
        CorpusItem(
            item_id=f"s{s}c0-decoy",
            set_id=f"s{s}",
            asset=AssetRef(key=f"img/{s}/decoy.png"),
            degraded_from=f"s{s}c0",
            degradation="blur-20px",
        )
        for s in range(8)
    ]
    await store.add_items([*items, *decoys])

    pairs = [
        ComparisonPair(item_a=f"s{s}c{a}", item_b=f"s{s}c{b}", kind=PairKind.WITHIN_SET)
        for s in range(8)
        for a in range(3)
        for b in range(a + 1, 3)
    ]
    pairs += [
        ComparisonPair(item_a=f"s{s}c0", item_b=f"s{s + 1}c1", kind=PairKind.CROSS_SET)
        for s in range(7)
    ]
    pairs += [
        ComparisonPair(
            item_a=f"s{s}c0",
            item_b=f"s{s}c0-decoy",
            kind=PairKind.CATCH,
            expected_winner=f"s{s}c0",
        )
        for s in range(8)
    ]
    await store.add_pairs(pairs)
    return store


@pytest.fixture
async def annotator(store) -> str:
    profile = await store.enrol(AnnotatorProfile(label="tester", agreed_to_research_use=True))
    return profile.annotator_id


# --- Storage and idempotency -------------------------------------------------


async def test_items_and_pairs_are_idempotent(store):
    """The corpus builder is meant to be re-run as more jobs complete."""
    again = CorpusItem(item_id="s0c0", set_id="s0", asset=AssetRef(key="dup.png"))
    assert await store.add_items([again]) == 0

    dup = ComparisonPair(item_a="s0c1", item_b="s0c0")  # reversed order, same pair
    assert await store.add_pairs([dup]) == 0


async def test_decoys_are_excluded_from_the_item_list_by_default(store):
    assert len(await store.list_items()) == 24
    assert len(await store.list_items(include_decoys=True)) == 32


# --- The presentation side ---------------------------------------------------


def test_presentation_side_is_deterministic_and_mirrors_on_repeat():
    """The serving and recording endpoints must agree without sharing state, and a
    repeat must swap sides so test-retest measures preference, not motor memory."""
    first = presentation_side("annotator-1", "pair-1", 0)
    assert presentation_side("annotator-1", "pair-1", 0) is first
    assert presentation_side("annotator-1", "pair-1", 1) is (not first)
    assert presentation_side("annotator-1", "pair-1", 2) is first


def test_presentation_side_differs_across_annotators():
    """Otherwise every annotator sees the same item on the same side, and a shared
    left-side bias would look like a property of the images."""
    sides = [presentation_side(f"a{k}", "pair-1", 0) for k in range(40)]
    assert 10 < sum(sides) < 30


def test_presentation_side_is_balanced_across_pairs_for_one_annotator():
    """The axis that matters in a single session.

    If one annotator always got ``item_a`` on the left, then within a set the same
    candidate would always be on the same side for them, and their side preference
    and their content preference would be inseparable.
    """
    sides = [presentation_side("just-one-annotator", f"pair-{k}", 0) for k in range(200)]
    assert 80 < sum(sides) < 120


async def test_recorded_side_matches_what_was_served(store, annotator):
    served = await store.next_pair(annotator, rng=random.Random(0))
    assert served is not None
    pair, left_item, showing = served

    judgement = await store.record(
        annotator_id=annotator, pair_id=pair.pair_id, choice=Choice.LEFT, latency_ms=1500
    )
    assert judgement.shown_left == left_item
    assert judgement.showing == showing
    assert judgement.winner(pair) == left_item


async def test_the_client_cannot_choose_the_side_or_the_showing(store, annotator):
    """`record` takes only the side clicked; everything else is server-derived."""
    pair = (await store.list_pairs(include_catch=False))[0]

    first = await store.record(
        annotator_id=annotator, pair_id=pair.pair_id, choice=Choice.LEFT, latency_ms=900
    )
    second = await store.record(
        annotator_id=annotator, pair_id=pair.pair_id, choice=Choice.LEFT, latency_ms=900
    )

    assert (first.showing, second.showing) == (0, 1)
    # Same click, mirrored sides, so the two showings name different winners —
    # which is exactly what makes a repeat a real test of preference.
    assert first.shown_left != second.shown_left
    assert first.winner(pair) != second.winner(pair)


async def test_a_judgement_on_an_unknown_pair_is_refused(store, annotator):
    with pytest.raises(KeyError):
        await store.record(
            annotator_id=annotator, pair_id="no-such-pair", choice=Choice.LEFT, latency_ms=100
        )


# --- Serving policy ----------------------------------------------------------


async def test_serving_prefers_the_least_judged_pair(store):
    """Coverage first: an unjudged pair is worth far more than a sixth opinion."""
    pairs = await store.list_pairs(include_catch=False)
    hot = pairs[0]
    for k in range(5):
        other = await store.enrol(AnnotatorProfile(label=f"other{k}", agreed_to_research_use=True))
        await store.record(
            annotator_id=other.annotator_id,
            pair_id=hot.pair_id,
            choice=Choice.LEFT,
            latency_ms=1000,
        )

    fresh = await store.enrol(AnnotatorProfile(label="fresh", agreed_to_research_use=True))
    for _ in range(10):
        served = await store.next_pair(fresh.annotator_id, rng=random.Random(1))
        assert served is not None
        assert served[0].pair_id != hot.pair_id


async def test_serving_stops_when_every_pair_has_been_seen(store, annotator):
    rng = random.Random(2)
    seen = 0
    while (served := await store.next_pair(annotator, rng=rng)) is not None:
        pair, _left, _showing = served
        await store.record(
            annotator_id=annotator, pair_id=pair.pair_id, choice=Choice.LEFT, latency_ms=1200
        )
        seen += 1
        if seen > 200:  # pragma: no cover - guards an infinite loop in a bug
            raise AssertionError("serving never ran out of pairs")
    assert seen >= 31  # 24 within-set + 7 cross-set, plus catch trials


async def test_catch_trials_arrive_early_and_reach_the_screening_minimum(store, annotator):
    """A screen you only reach after 50 trials has not screened anything.

    Enough catch trials must land inside a plausible session for the binomial
    screen in `AnnotatorQuality` to be allowed to fire at all.
    """
    catch_ids = {p.pair_id for p in await store.list_pairs() if p.kind is PairKind.CATCH}
    rng = random.Random(3)
    positions = []
    for index in range(60):
        served = await store.next_pair(annotator, rng=rng)
        if served is None:
            break
        pair, _left, _showing = served
        if pair.pair_id in catch_ids:
            positions.append(index)
        await store.record(
            annotator_id=annotator, pair_id=pair.pair_id, choice=Choice.LEFT, latency_ms=1500
        )

    assert positions[0] <= CATCH_CHECKPOINTS[0] + 1
    assert len([p for p in positions if p < 25]) >= 3


async def test_a_catch_pair_is_never_re_served_as_a_repeat(store, annotator):
    """A repeat of a question with a right answer measures no preference, and would
    add a correlated second observation to a screen that assumes independence."""
    catch_ids = {p.pair_id for p in await store.list_pairs() if p.kind is PairKind.CATCH}
    rng = random.Random(5)
    for _ in range(80):
        served = await store.next_pair(annotator, rng=rng)
        if served is None:
            break
        pair, _left, showing = served
        assert not (showing > 0 and pair.pair_id in catch_ids)
        await store.record(
            annotator_id=annotator, pair_id=pair.pair_id, choice=Choice.LEFT, latency_ms=1500
        )


# --- Quality measurement -----------------------------------------------------


async def test_a_left_clicker_is_caught_on_every_signal(store):
    """Fast, always-left, and wrong on the decoys — all three should fire."""
    clicker = await store.enrol(AnnotatorProfile(label="clicker", agreed_to_research_use=True))
    rng = random.Random(7)
    for _ in range(70):
        served = await store.next_pair(clicker.annotator_id, rng=rng)
        if served is None:
            break
        pair, _left, _showing = served
        await store.record(
            annotator_id=clicker.annotator_id,
            pair_id=pair.pair_id,
            choice=Choice.LEFT,
            latency_ms=200,
        )

    (quality,) = await store.annotator_quality()
    assert quality.side_bias == 1.0
    assert quality.side_bias_z > 3.0
    assert quality.fast_click_rate == 1.0
    # Sides are mirrored on a repeat, so always pressing left is always inconsistent.
    assert quality.n_repeats > 0
    assert quality.repeat_consistency == 0.0

    flags = " ".join(quality.flags)
    assert "left side" in flags
    assert "600 ms" in flags
    # The repeat flag needs 5 repeats before it fires, and this fixture's 39 pairs
    # cannot accumulate that many at an 8% repeat rate — see the focused test below.


async def test_the_repeat_flag_needs_five_repeats_before_it_fires():
    """Small-sample guard, tested directly because the corpus fixture is too small
    to accumulate enough repeats through the serving policy."""
    from adschema import AnnotatorQuality

    few = AnnotatorQuality(annotator_id="x", n_judgements=100, n_repeats=4, repeat_agreements=0)
    assert few.repeat_consistency == 0.0
    assert not any("repeat" in f for f in few.flags)

    enough = AnnotatorQuality(annotator_id="x", n_judgements=100, n_repeats=5, repeat_agreements=0)
    assert any("repeat" in f for f in enough.flags)


async def test_an_attentive_annotator_is_not_flagged(store):
    """The false-positive case. A careful person must survive the screen."""
    careful = await store.enrol(AnnotatorProfile(label="careful", agreed_to_research_use=True))
    pairs = {p.pair_id: p for p in await store.list_pairs()}
    rng = random.Random(11)
    strength = {i.item_id: rng.random() for i in await store.list_items(include_decoys=True)}
    for item_id in list(strength):
        if item_id.endswith("-decoy"):
            strength[item_id] = -1.0

    for _ in range(80):
        served = await store.next_pair(careful.annotator_id, rng=rng)
        if served is None:
            break
        pair, left_item, _showing = served
        right_item = pairs[pair.pair_id].other(left_item)
        choice = Choice.LEFT if strength[left_item] > strength[right_item] else Choice.RIGHT
        await store.record(
            annotator_id=careful.annotator_id,
            pair_id=pair.pair_id,
            choice=choice,
            latency_ms=2400,
        )

    (quality,) = await store.annotator_quality()
    assert quality.repeat_consistency == 1.0
    assert quality.catch_accuracy == 1.0
    assert quality.n_catch >= 3
    assert abs(quality.side_bias_z) < 3.0
    assert quality.flags == []
    assert quality.is_trustworthy


async def test_catch_screening_needs_enough_trials_before_it_fires(store):
    """Below MIN_CATCH_TRIALS the number is reported and never acted on — a couple
    of mis-clicks must not exclude someone's whole session."""
    from adschema import AnnotatorQuality

    marginal = AnnotatorQuality(
        annotator_id="x", n_judgements=100, n_catch=MIN_CATCH_TRIALS - 1, catch_correct=2
    )
    assert marginal.catch_accuracy is not None
    assert not any("catch" in f for f in marginal.flags)

    enough = AnnotatorQuality(
        annotator_id="x", n_judgements=100, n_catch=MIN_CATCH_TRIALS, catch_correct=5
    )
    assert any("catch" in f for f in enough.flags)


async def test_observations_exclude_catch_pairs(store, annotator):
    """A decoy's strength is meaningless and would distort the shared scale."""
    catch = next(p for p in await store.list_pairs() if p.kind is PairKind.CATCH)
    real = next(p for p in await store.list_pairs() if p.kind is not PairKind.CATCH)
    for pair in (catch, real):
        await store.record(
            annotator_id=annotator, pair_id=pair.pair_id, choice=Choice.LEFT, latency_ms=1500
        )

    wins, ties = await store.observations()
    assert len(wins) == 1
    assert ties == []
    assert all("decoy" not in item for _a, *items in wins for item in items)


async def test_observations_can_drop_a_flagged_annotator(store):
    good = await store.enrol(AnnotatorProfile(label="good", agreed_to_research_use=True))
    bad = await store.enrol(AnnotatorProfile(label="bad", agreed_to_research_use=True))
    pair = (await store.list_pairs(include_catch=False))[0]
    for who in (good, bad):
        await store.record(
            annotator_id=who.annotator_id,
            pair_id=pair.pair_id,
            choice=Choice.LEFT,
            latency_ms=1500,
        )

    assert len((await store.observations())[0]) == 2
    kept, _ = await store.observations(exclude_annotators={bad.annotator_id})
    assert len(kept) == 1
    assert kept[0][0] == good.annotator_id


async def test_ties_are_returned_separately(store, annotator):
    pair = (await store.list_pairs(include_catch=False))[0]
    await store.record(
        annotator_id=annotator, pair_id=pair.pair_id, choice=Choice.TIE, latency_ms=3000
    )
    wins, ties = await store.observations()
    assert wins == []
    assert ties == [(annotator, pair.item_a, pair.item_b)]


# --- The HTTP surface --------------------------------------------------------


async def test_enrolment_is_refused_without_consent(store):
    """Classmates' judgements are human-subject data going into a report."""
    from fastapi import HTTPException

    ann = annotate.Annotating(store, random.Random(0))
    with pytest.raises(HTTPException) as caught:
        await annotate.enrol(ann, annotate.EnrolRequest(label="nope", agreed_to_research_use=False))
    assert caught.value.status_code == 422
    assert "Consent is required" in caught.value.detail


async def test_enrolment_returns_an_id_that_can_annotate(store):
    ann = annotate.Annotating(store, random.Random(0))
    profile = await annotate.enrol(
        ann, annotate.EnrolRequest(label="Ana", cohort="classmate", agreed_to_research_use=True)
    )
    assert profile.may_annotate
    assert await store.get_annotator(profile.annotator_id) is not None


async def test_served_pair_reveals_nothing_about_itself(store):
    """No pair kind, no provenance, no hint that a trial is a screen or a repeat.

    An annotator who can tell which trials are catch trials can pass the screen
    while clicking through everything else.
    """
    ann = annotate.Annotating(store, random.Random(0))
    profile = await annotate.enrol(
        ann, annotate.EnrolRequest(label="Ana", agreed_to_research_use=True)
    )
    served = await annotate.next_pair(ann, profile.annotator_id)
    assert served is not None

    payload = served.model_dump()
    assert set(payload) == {"pair_id", "left", "right", "judged_by_you"}
    assert set(payload["left"]) == {"item_id", "url"}
    text = str(payload)
    for leak in ("catch", "decoy", "showing", "kind", "expected_winner", "degrad"):
        assert leak not in text


async def test_judging_records_and_advances(store):
    ann = annotate.Annotating(store, random.Random(0))
    profile = await annotate.enrol(
        ann, annotate.EnrolRequest(label="Ana", agreed_to_research_use=True)
    )
    first = await annotate.next_pair(ann, profile.annotator_id)
    assert first is not None
    assert first.judged_by_you == 0

    result = await annotate.judge(
        ann,
        annotate.JudgeRequest(
            annotator_id=profile.annotator_id,
            pair_id=first.pair_id,
            choice=Choice.RIGHT,
            latency_ms=1800,
        ),
    )
    assert "recorded" in result
    # The showing index is deliberately absent, so no future UI can reveal a repeat.
    assert "showing" not in result

    second = await annotate.next_pair(ann, profile.annotator_id)
    assert second is not None
    assert second.judged_by_you == 1
    assert second.pair_id != first.pair_id


async def test_an_unknown_annotator_is_a_404(store):
    from fastapi import HTTPException

    ann = annotate.Annotating(store, random.Random(0))
    with pytest.raises(HTTPException) as caught:
        await annotate.next_pair(ann, "not-enrolled")
    assert caught.value.status_code == 404


async def test_stats_count_decoys_apart_from_the_corpus(store, annotator):
    stats = await store.stats()
    assert (stats.n_items, stats.n_decoys, stats.n_sets) == (24, 8, 8)
    assert stats.n_pairs == 39  # 24 within-set + 7 cross-set + 8 catch
    assert stats.n_judgements == 0

    pair = (await store.list_pairs(include_catch=False))[0]
    await store.record(
        annotator_id=annotator, pair_id=pair.pair_id, choice=Choice.LEFT, latency_ms=1000
    )
    stats = await store.stats()
    assert (stats.n_judgements, stats.pairs_judged, stats.n_annotators) == (1, 1, 1)


# --- Coverage targets --------------------------------------------------------


async def test_within_set_pairs_are_targeted_twice_and_nothing_overshoots():
    """The deficit ordering that makes ``TARGET_JUDGEMENTS`` mean anything.

    Ordering the coverage query on the raw judgement count would take every pair to
    one and stop, so no within-set pair would ever reach two.  Ordering on
    ``judged - target`` serves whichever pair is furthest below its own target.

    The overshoot assertion is the real content: a pair must not collect a third
    judgement while other pairs are still at zero.  That is what would happen if the
    ``LIMIT`` window were ordered by anything other than the deficit, and it wastes
    a volunteer's time on a comparison that is already settled.
    """
    from collections import Counter

    from adapi.annotation_store import TARGET_JUDGEMENTS
    from adml.pairs import design_pairs

    store = AnnotationStore.from_url("sqlite+aiosqlite://")
    await store.create_all()
    items = [
        CorpusItem(
            item_id=f"t{s:02d}-i{c}",
            set_id=f"t{s:02d}",
            kind=ItemKind.IMAGE,
            asset=AssetRef(key=f"img/{s}/{c}.png"),
        )
        for s in range(40)
        for c in range(3)
    ]
    await store.add_items(items)
    design = design_pairs(items, seed=1, cross_set_rounds=6)
    await store.add_pairs(design.pairs)

    for n in range(8):
        profile = await store.enrol(
            AnnotatorProfile(annotator_id=f"cov{n}", agreed_to_research_use=True)
        )
        rng = random.Random(n)
        for _ in range(60):
            served = await store.next_pair(profile.annotator_id, rng=rng)
            if served is None:
                break
            await store.record(
                annotator_id=profile.annotator_id,
                pair_id=served[0].pair_id,
                choice=Choice.LEFT,
                latency_ms=1500,
            )

    counts: Counter[str] = Counter()
    for judgement in await store.list_judgements():
        if judgement.showing == 0:
            counts[judgement.pair_id] += 1

    for kind, target in TARGET_JUDGEMENTS.items():
        pairs = [p for p in design.pairs if p.kind is kind]
        if not pairs:
            continue
        got = [counts.get(p.pair_id, 0) for p in pairs]
        assert max(got) <= target, f"{kind.value} overshot its target of {target}"

    within = [counts.get(p.pair_id, 0) for p in design.pairs if p.kind is PairKind.WITHIN_SET]
    assert sum(1 for c in within if c >= 2) / len(within) > 0.9


async def test_a_second_judgement_always_comes_from_a_different_annotator(store, annotator):
    """Otherwise it would measure test-retest, not inter-annotator agreement."""
    rng = random.Random(0)
    served: list[tuple[str, int]] = []
    for _ in range(30):
        nxt = await store.next_pair(annotator, rng=rng)
        if nxt is None:
            break
        pair, _left, showing = nxt
        served.append((pair.pair_id, showing))
        await store.record(
            annotator_id=annotator, pair_id=pair.pair_id, choice=Choice.LEFT, latency_ms=1500
        )

    first_showings = [pair_id for pair_id, showing in served if showing == 0]
    assert len(first_showings) == len(set(first_showings))
