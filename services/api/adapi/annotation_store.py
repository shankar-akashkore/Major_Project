"""Persistence for the annotation corpus, and the policy for what to show next.

Unlike :mod:`adapi.store`, which keeps jobs as JSON documents, this schema is
normalised.  That was a deliberate switch: jobs are only ever fetched by id or
listed by date, whereas the annotation data is *queried* — pairs by judgement
count, items by set, judgements grouped by annotator — and none of that works
against a blob.

The serving policy in :meth:`AnnotationStore.next_pair` is where most of the
thinking is, because it decides what the labels are worth:

* **Coverage first.**  The next pair is chosen from those with the *fewest*
  judgements so far.  Serving at random leaves some pairs judged five times and
  others never, and an unjudged pair contributes nothing while a sixth judgement
  on the same pair contributes almost nothing.
* **The presentation side is decided by the server**, from a hash of
  ``(annotator, pair, showing)``.  The client is never asked which side it drew,
  so a buggy or edited client cannot corrupt the position-bias analysis, and no
  extra server state is needed to remember what was sent.
* **A repeat is mirrored.**  The second showing of a pair always swaps the sides,
  so test-retest agreement measures a preference rather than a remembered
  keypress.
* **Repeats and catch trials are injected here, not baked into the design.**
  Test-retest needs the *same* annotator to see a pair twice, which the pair
  design cannot know in advance because it does not know who will be shown what.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from statistics import median

from adschema import (
    MIN_PLAUSIBLE_LATENCY_MS,
    AnnotatorProfile,
    AnnotatorQuality,
    AssetRef,
    Choice,
    ComparisonPair,
    CorpusItem,
    CorpusStats,
    ItemKind,
    Judgement,
    PairKind,
)
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    case,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

METADATA = MetaData()

#: Fraction of servings that re-show an already-judged pair, for test-retest.
#: 8% gives ~8 repeats per 100 judgements — enough to estimate consistency
#: without spending much of a volunteer's goodwill on questions they answered.
REPEAT_RATE = 0.08

#: No repeats until the annotator has this many judgements behind them; a repeat
#: on the third trial is a memory test, not a reliability measure.
MIN_BEFORE_REPEAT = 15

#: Fraction of servings that are catch trials, once the checkpoints below are past.
CATCH_RATE = 0.06

#: Judgement counts at which a catch trial is served *unconditionally*.
#:
#: A pure rate does not work, and a simulated run showed exactly why: at 6% a
#: 45-judgement session expects 2.7 catch trials, far short of
#: ``adschema.MIN_CATCH_TRIALS`` — so the one mechanism that directly detects
#: clicking-through could not fire at all for a short session, which is the common
#: case for a volunteer. Worse, a screen you only reach after 50 trials has not
#: screened anything: someone who is going to click through does it from the start.
#:
#: These reach MIN_CATCH_TRIALS by judgement 84, with the first three inside the
#: opening 15. Note the corpus needs at least as many decoys as there are
#: checkpoints, since an annotator sees each catch pair only once.
CATCH_CHECKPOINTS = (3, 8, 14, 21, 29, 38, 48, 59, 71, 84)

#: How many independent judgements each kind of comparison is collected for.
#:
#: Within-set pairs are targeted twice and everything else once, which is not a
#: preference — it is arithmetic. Three candidates judged once per pair can cycle
#: (A>B, B>C, C>A), and such a set has no best candidate, so it is excluded from
#: top-1 retention entirely rather than resolved by a coin. Measured on 100
#: simulated sets, doubling only the within-set pairs takes intransitive sets from
#: 15% to 5%, grows the testable pool from 532 to 832 comparisons, and is what makes
#: inter-annotator agreement — and therefore the noise ceiling — computable at all.
#:
#: The cost is +20% of the total judgements, because within-set pairs are only 300
#: of 1,500: about 17 minutes each becomes about 20. Doubling *every* pair costs 33
#: minutes each and buys almost nothing more. See ``docs/prediction-protocol.md``.
TARGET_JUDGEMENTS = {
    PairKind.WITHIN_SET: 2,
    PairKind.CROSS_SET: 1,
    PairKind.BRIDGE: 1,
}

#: How many low-coverage candidates to draw before picking randomly among them.
#: Strict "fewest judgements" would serve every annotator the same pair at the
#: same time; a small random pool over the least-covered pairs avoids that
#: without giving up the coverage objective.
_COVERAGE_POOL = 24


corpus_items = Table(
    "corpus_items",
    METADATA,
    Column("item_id", String(32), primary_key=True),
    Column("set_id", String(32), nullable=False, index=True),
    Column("kind", String(16), nullable=False, index=True),
    Column("asset_key", String(512), nullable=False),
    Column("mime_type", String(64), nullable=False, default="image/png"),
    Column("width", Integer, nullable=True),
    Column("height", Integer, nullable=True),
    Column("tier", String(16), nullable=False, default="mock"),
    Column("provider", String(64), nullable=False, default=""),
    Column("model", String(64), nullable=False, default=""),
    Column("vertical", String(32), nullable=False, default="other"),
    Column("platform", String(32), nullable=False, default="instagram_reels"),
    Column("seed", Integer, nullable=False, default=0),
    Column("angle", String(32), nullable=True),
    Column("lighting", String(32), nullable=True),
    Column("composition", String(32), nullable=True),
    Column("motion", String(32), nullable=True),
    Column("degraded_from", String(32), nullable=True, index=True),
    Column("degradation", String(64), nullable=False, default=""),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

comparison_pairs = Table(
    "comparison_pairs",
    METADATA,
    Column("pair_id", String(32), primary_key=True),
    Column("item_a", String(32), nullable=False, index=True),
    Column("item_b", String(32), nullable=False, index=True),
    Column("kind", String(16), nullable=False, index=True),
    Column("expected_winner", String(32), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

annotators = Table(
    "annotators",
    METADATA,
    Column("annotator_id", String(32), primary_key=True),
    Column("label", String(64), nullable=False, default=""),
    Column("cohort", String(32), nullable=False, default=""),
    Column("agreed_to_research_use", Boolean, nullable=False, default=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# Note the absence of a unique constraint on (pair_id, annotator_id): a second
# judgement from the same annotator on the same pair is the test-retest
# observation, not a duplicate to be rejected.
judgements = Table(
    "judgements",
    METADATA,
    Column("judgement_id", String(32), primary_key=True),
    Column("pair_id", String(32), nullable=False, index=True),
    Column("annotator_id", String(32), nullable=False, index=True),
    Column("shown_left", String(32), nullable=False),
    Column("choice", String(8), nullable=False),
    Column("latency_ms", Integer, nullable=False, default=0),
    Column("showing", Integer, nullable=False, default=0),
    Column("at", DateTime(timezone=True), nullable=False),
)


def presentation_side(annotator_id: str, pair_id: str, showing: int) -> bool:
    """True when ``item_a`` goes on the left for this showing.

    Deterministic, so the serving endpoint and the recording endpoint agree
    without storing anything, and the client cannot influence it.  The parity of
    ``showing`` is XORed in so a repeat is always mirrored.
    """
    digest = hashlib.sha256(f"{annotator_id}:{pair_id}".encode()).digest()
    return bool((digest[0] & 1) ^ (showing & 1))


def _item_from_row(row) -> CorpusItem:
    return CorpusItem(
        item_id=row.item_id,
        set_id=row.set_id,
        kind=row.kind,
        asset=AssetRef(
            key=row.asset_key,
            mime_type=row.mime_type,
            width=row.width,
            height=row.height,
        ),
        tier=row.tier,
        provider=row.provider,
        model=row.model,
        vertical=row.vertical,
        platform=row.platform,
        seed=row.seed,
        angle=row.angle,
        lighting=row.lighting,
        composition=row.composition,
        motion=row.motion,
        degraded_from=row.degraded_from,
        degradation=row.degradation,
        created_at=row.created_at,
    )


def _pair_from_row(row) -> ComparisonPair:
    return ComparisonPair(
        pair_id=row.pair_id,
        item_a=row.item_a,
        item_b=row.item_b,
        kind=row.kind,
        expected_winner=row.expected_winner,
        created_at=row.created_at,
    )


def _judgement_from_row(row) -> Judgement:
    return Judgement(
        judgement_id=row.judgement_id,
        pair_id=row.pair_id,
        annotator_id=row.annotator_id,
        shown_left=row.shown_left,
        choice=row.choice,
        latency_ms=row.latency_ms,
        showing=row.showing,
        at=row.at,
    )


class AnnotationStore:
    """The corpus, the designed pairs, the annotators and their judgements."""

    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    @classmethod
    def from_url(cls, url: str) -> AnnotationStore:
        return cls(create_async_engine(url, future=True))

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(METADATA.create_all)

    # --- Corpus ------------------------------------------------------------

    async def add_items(self, items: list[CorpusItem]) -> int:
        """Insert items, skipping ids that already exist.

        Idempotent because the corpus builder is expected to be re-run as more
        jobs complete, and re-ingesting should extend the corpus rather than
        fail or duplicate it.
        """
        if not items:
            return 0
        async with self.engine.begin() as conn:
            existing = set(
                (
                    await conn.execute(
                        select(corpus_items.c.item_id).where(
                            corpus_items.c.item_id.in_([i.item_id for i in items])
                        )
                    )
                ).scalars()
            )
            fresh = [i for i in items if i.item_id not in existing]
            if fresh:
                await conn.execute(
                    corpus_items.insert(),
                    [
                        {
                            "item_id": i.item_id,
                            "set_id": i.set_id,
                            "kind": i.kind.value,
                            "asset_key": i.asset.key,
                            "mime_type": i.asset.mime_type,
                            "width": i.asset.width,
                            "height": i.asset.height,
                            "tier": i.tier.value,
                            "provider": i.provider,
                            "model": i.model,
                            "vertical": i.vertical.value,
                            "platform": i.platform.value,
                            "seed": i.seed,
                            "angle": i.angle.value if i.angle else None,
                            "lighting": i.lighting.value if i.lighting else None,
                            "composition": i.composition.value if i.composition else None,
                            "motion": i.motion.value if i.motion else None,
                            "degraded_from": i.degraded_from,
                            "degradation": i.degradation,
                            "created_at": i.created_at,
                        }
                        for i in fresh
                    ],
                )
        return len(fresh)

    async def list_items(
        self, *, kind: ItemKind | None = None, include_decoys: bool = False
    ) -> list[CorpusItem]:
        stmt = select(corpus_items)
        if kind is not None:
            stmt = stmt.where(corpus_items.c.kind == kind.value)
        if not include_decoys:
            stmt = stmt.where(corpus_items.c.degraded_from.is_(None))
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt.order_by(corpus_items.c.created_at))).all()
        return [_item_from_row(r) for r in rows]

    async def get_items(self, item_ids: list[str]) -> dict[str, CorpusItem]:
        if not item_ids:
            return {}
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(select(corpus_items).where(corpus_items.c.item_id.in_(item_ids)))
            ).all()
        return {r.item_id: _item_from_row(r) for r in rows}

    # --- Pairs -------------------------------------------------------------

    async def add_pairs(self, pairs: list[ComparisonPair]) -> int:
        """Insert pairs, skipping any whose two items are already paired.

        De-duplication is on the *unordered item pair*, not the pair id, so
        re-running the designer with a different seed extends the design instead
        of asking annotators the same question twice under a new id.
        """
        if not pairs:
            return 0
        async with self.engine.begin() as conn:
            rows = (
                await conn.execute(select(comparison_pairs.c.item_a, comparison_pairs.c.item_b))
            ).all()
            seen = {
                (r.item_a, r.item_b) if r.item_a <= r.item_b else (r.item_b, r.item_a) for r in rows
            }
            fresh: list[ComparisonPair] = []
            for pair in pairs:
                if pair.key in seen:
                    continue
                seen.add(pair.key)
                fresh.append(pair)
            if fresh:
                await conn.execute(
                    comparison_pairs.insert(),
                    [
                        {
                            "pair_id": p.pair_id,
                            "item_a": p.item_a,
                            "item_b": p.item_b,
                            "kind": p.kind.value,
                            "expected_winner": p.expected_winner,
                            "created_at": p.created_at,
                        }
                        for p in fresh
                    ],
                )
        return len(fresh)

    async def list_pairs(self, *, include_catch: bool = True) -> list[ComparisonPair]:
        stmt = select(comparison_pairs)
        if not include_catch:
            stmt = stmt.where(comparison_pairs.c.kind != PairKind.CATCH.value)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt.order_by(comparison_pairs.c.created_at))).all()
        return [_pair_from_row(r) for r in rows]

    async def get_pair(self, pair_id: str) -> ComparisonPair | None:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(
                    select(comparison_pairs).where(comparison_pairs.c.pair_id == pair_id)
                )
            ).one_or_none()
        return _pair_from_row(row) if row else None

    # --- Annotators --------------------------------------------------------

    async def enrol(self, profile: AnnotatorProfile) -> AnnotatorProfile:
        async with self.engine.begin() as conn:
            await conn.execute(
                annotators.insert().values(
                    annotator_id=profile.annotator_id,
                    label=profile.label,
                    cohort=profile.cohort,
                    agreed_to_research_use=profile.agreed_to_research_use,
                    created_at=profile.created_at,
                )
            )
        return profile

    async def get_annotator(self, annotator_id: str) -> AnnotatorProfile | None:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(
                    select(annotators).where(annotators.c.annotator_id == annotator_id)
                )
            ).one_or_none()
        if row is None:
            return None
        return AnnotatorProfile(
            annotator_id=row.annotator_id,
            label=row.label,
            cohort=row.cohort,
            agreed_to_research_use=row.agreed_to_research_use,
            created_at=row.created_at,
        )

    # --- Serving -----------------------------------------------------------

    async def next_pair(
        self, annotator_id: str, *, rng: random.Random | None = None
    ) -> tuple[ComparisonPair, str, int] | None:
        """Choose the next comparison: ``(pair, item_id_on_the_left, showing)``.

        Returns None when this annotator has judged every pair once — at which
        point more of their time buys repeats only, and the honest thing is to
        thank them and stop.
        """
        rng = rng or random.Random()
        async with self.engine.connect() as conn:
            mine = (
                await conn.execute(
                    select(
                        judgements.c.pair_id,
                        comparison_pairs.c.kind,
                        func.count().label("n"),
                    )
                    .join(comparison_pairs, comparison_pairs.c.pair_id == judgements.c.pair_id)
                    .where(judgements.c.annotator_id == annotator_id)
                    .group_by(judgements.c.pair_id, comparison_pairs.c.kind)
                )
            ).all()
            n_mine = sum(r.n for r in mine)
            seen = {r.pair_id for r in mine}
            # Catch pairs are excluded from repeat candidates: re-asking a question
            # that has a right answer spends a test-retest slot without measuring a
            # preference, and it would add a second, correlated observation to the
            # catch tally that the binomial screen assumes is independent.
            judged_once = [r.pair_id for r in mine if r.n == 1 and r.kind != PairKind.CATCH.value]

            # 1. A deliberate repeat, for test-retest.
            if judged_once and n_mine >= MIN_BEFORE_REPEAT and rng.random() < REPEAT_RATE:
                pair_id = rng.choice(judged_once)
                pair = await self.get_pair(pair_id)
                if pair is not None:
                    return pair, self._left_item(pair, annotator_id, 1), 1

            # 2. A catch trial: guaranteed at the early checkpoints, then at a rate.
            if n_mine in CATCH_CHECKPOINTS or rng.random() < CATCH_RATE:
                catch_stmt = select(comparison_pairs).where(
                    comparison_pairs.c.kind == PairKind.CATCH.value
                )
                if seen:
                    catch_stmt = catch_stmt.where(comparison_pairs.c.pair_id.notin_(seen))
                catch = (await conn.execute(catch_stmt)).all()
                if catch:
                    pair = _pair_from_row(rng.choice(catch))
                    return pair, self._left_item(pair, annotator_id, 0), 0

            # 3. Coverage first: the pair furthest below its target, that this
            # annotator has not seen. Ordering on the *deficit* rather than on the
            # raw count is what makes TARGET_JUDGEMENTS mean anything — with a raw
            # count, every pair would reach one judgement and no within-set pair
            # would ever reach two.
            #
            # The `notin_(seen)` clause matters here too: a second judgement always
            # comes from a different annotator, which is precisely what
            # inter-annotator agreement needs. A repeat by the same person measures
            # test-retest instead and is served by branch 1.
            counts = (
                select(judgements.c.pair_id, func.count().label("n"))
                .group_by(judgements.c.pair_id)
                .subquery()
            )
            judged = func.coalesce(counts.c.n, 0)
            target = case(
                *(
                    (comparison_pairs.c.kind == kind.value, value)
                    for kind, value in TARGET_JUDGEMENTS.items()
                ),
                else_=1,
            )
            stmt = (
                select(comparison_pairs, judged.label("judged"), (judged - target).label("deficit"))
                .outerjoin(counts, counts.c.pair_id == comparison_pairs.c.pair_id)
                .where(comparison_pairs.c.kind != PairKind.CATCH.value)
                .order_by(judged - target)
                .limit(_COVERAGE_POOL)
            )
            if seen:
                stmt = stmt.where(comparison_pairs.c.pair_id.notin_(seen))
            rows = (await conn.execute(stmt)).all()

        if not rows:
            return None
        neediest = min(r.deficit for r in rows)
        pair = _pair_from_row(rng.choice([r for r in rows if r.deficit == neediest]))
        return pair, self._left_item(pair, annotator_id, 0), 0

    @staticmethod
    def _left_item(pair: ComparisonPair, annotator_id: str, showing: int) -> str:
        a_left = presentation_side(annotator_id, pair.pair_id, showing)
        return pair.item_a if a_left else pair.item_b

    async def record(
        self, *, annotator_id: str, pair_id: str, choice: Choice, latency_ms: int
    ) -> Judgement:
        """Persist a judgement, deriving the showing index and the side served.

        Neither is taken from the client.  ``showing`` is how many times this
        annotator has already judged this pair, and the side is recomputed from
        the same hash the serving endpoint used, so the record cannot disagree
        with what was actually on screen.
        """
        pair = await self.get_pair(pair_id)
        if pair is None:
            raise KeyError(f"no pair {pair_id!r}")

        async with self.engine.connect() as conn:
            prior = int(
                (
                    await conn.execute(
                        select(func.count())
                        .select_from(judgements)
                        .where(
                            judgements.c.annotator_id == annotator_id,
                            judgements.c.pair_id == pair_id,
                        )
                    )
                ).scalar_one()
            )

        judgement = Judgement(
            pair_id=pair_id,
            annotator_id=annotator_id,
            shown_left=self._left_item(pair, annotator_id, prior),
            choice=choice,
            latency_ms=max(latency_ms, 0),
            showing=prior,
        )
        async with self.engine.begin() as conn:
            await conn.execute(
                judgements.insert().values(
                    judgement_id=judgement.judgement_id,
                    pair_id=judgement.pair_id,
                    annotator_id=judgement.annotator_id,
                    shown_left=judgement.shown_left,
                    choice=judgement.choice.value,
                    latency_ms=judgement.latency_ms,
                    showing=judgement.showing,
                    at=judgement.at,
                )
            )
        return judgement

    async def list_judgements(self, *, annotator_id: str | None = None) -> list[Judgement]:
        stmt = select(judgements)
        if annotator_id is not None:
            stmt = stmt.where(judgements.c.annotator_id == annotator_id)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt.order_by(judgements.c.at))).all()
        return [_judgement_from_row(r) for r in rows]

    async def count_judgements(self, annotator_id: str) -> int:
        async with self.engine.connect() as conn:
            return int(
                (
                    await conn.execute(
                        select(func.count())
                        .select_from(judgements)
                        .where(judgements.c.annotator_id == annotator_id)
                    )
                ).scalar_one()
            )

    # --- Aggregates --------------------------------------------------------

    async def stats(self) -> CorpusStats:
        async with self.engine.connect() as conn:
            n_items = int(
                (
                    await conn.execute(
                        select(func.count())
                        .select_from(corpus_items)
                        .where(corpus_items.c.degraded_from.is_(None))
                    )
                ).scalar_one()
            )
            n_decoys = int(
                (
                    await conn.execute(
                        select(func.count())
                        .select_from(corpus_items)
                        .where(corpus_items.c.degraded_from.isnot(None))
                    )
                ).scalar_one()
            )
            n_sets = int(
                (
                    await conn.execute(select(func.count(func.distinct(corpus_items.c.set_id))))
                ).scalar_one()
            )
            n_pairs = int(
                (
                    await conn.execute(select(func.count()).select_from(comparison_pairs))
                ).scalar_one()
            )
            n_judgements = int(
                (await conn.execute(select(func.count()).select_from(judgements))).scalar_one()
            )
            n_annotators = int(
                (await conn.execute(select(func.count()).select_from(annotators))).scalar_one()
            )
            pairs_judged = int(
                (
                    await conn.execute(select(func.count(func.distinct(judgements.c.pair_id))))
                ).scalar_one()
            )
        return CorpusStats(
            n_items=n_items,
            n_decoys=n_decoys,
            n_sets=n_sets,
            n_pairs=n_pairs,
            n_judgements=n_judgements,
            n_annotators=n_annotators,
            pairs_judged=pairs_judged,
        )

    async def annotator_quality(self) -> list[AnnotatorQuality]:
        """Reliability statistics per annotator.

        Aggregated in Python rather than SQL: the repeat and catch logic needs the
        *winning item* per judgement, which depends on the pair's membership and
        the side served, and expressing that as SQL would obscure it for no gain
        at a few thousand rows.
        """
        profiles = {p.annotator_id: p for p in await self._list_annotators()}
        pairs = {p.pair_id: p for p in await self.list_pairs()}
        rows = await self.list_judgements()

        by_annotator: dict[str, list[Judgement]] = defaultdict(list)
        for j in rows:
            by_annotator[j.annotator_id].append(j)

        out: list[AnnotatorQuality] = []
        for annotator_id, profile in profiles.items():
            js = by_annotator.get(annotator_id, [])
            quality = AnnotatorQuality(
                annotator_id=annotator_id,
                label=profile.label,
                cohort=profile.cohort,
                n_judgements=len(js),
                n_pairs=len({j.pair_id for j in js}),
            )
            if not js:
                out.append(quality)
                continue

            latencies = [j.latency_ms for j in js if j.latency_ms > 0]
            quality.median_latency_ms = int(median(latencies)) if latencies else 0
            quality.fast_click_rate = (
                sum(1 for latency in latencies if latency < MIN_PLAUSIBLE_LATENCY_MS)
                / len(latencies)
                if latencies
                else 0.0
            )
            quality.tie_rate = sum(1 for j in js if j.choice is Choice.TIE) / len(js)
            sided = [j for j in js if j.choice is not Choice.TIE]
            quality.n_sided = len(sided)
            quality.chose_left = sum(1 for j in sided if j.choice is Choice.LEFT)

            # Catch trials: a decoy chosen, or a tie called, is a failure. Only the
            # first showing counts — the binomial screen treats trials as
            # independent, and a re-shown pair is not a second independent look.
            for j in js:
                pair = pairs.get(j.pair_id)
                if pair is None or pair.kind is not PairKind.CATCH or j.showing != 0:
                    continue
                quality.n_catch += 1
                quality.catch_correct += int(j.winner(pair) == pair.expected_winner)

            # Test-retest: compare each later showing against the first, on the
            # winning *item* — the sides are deliberately mirrored, so comparing
            # the clicked side would report disagreement on every consistent answer.
            by_pair: dict[str, list[Judgement]] = defaultdict(list)
            for j in js:
                by_pair[j.pair_id].append(j)
            for pair_id, group in by_pair.items():
                pair = pairs.get(pair_id)
                if pair is None or len(group) < 2:
                    continue
                group.sort(key=lambda j: j.showing)
                first = group[0].winner(pair)
                for later in group[1:]:
                    quality.n_repeats += 1
                    quality.repeat_agreements += int(later.winner(pair) == first)

            out.append(quality)

        return sorted(out, key=lambda q: -q.n_judgements)

    async def _list_annotators(self) -> list[AnnotatorProfile]:
        async with self.engine.connect() as conn:
            rows = (await conn.execute(select(annotators))).all()
        return [
            AnnotatorProfile(
                annotator_id=r.annotator_id,
                label=r.label,
                cohort=r.cohort,
                agreed_to_research_use=r.agreed_to_research_use,
                created_at=r.created_at,
            )
            for r in rows
        ]

    async def observations(
        self, *, exclude_annotators: set[str] | None = None
    ) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
        """Judgements as ``(annotator, winner, loser)``, plus ties in the same shape.

        Catch pairs are excluded: a decoy's strength is meaningless and would
        distort the scale it would otherwise share with real items.  ``ties`` are
        returned separately so the caller decides how to treat them —
        :func:`adml.ranking.fit_bradley_terry` counts them as half a win each way.
        """
        skip = exclude_annotators or set()
        pairs = {p.pair_id: p for p in await self.list_pairs()}
        wins: list[tuple[str, str, str]] = []
        ties: list[tuple[str, str, str]] = []
        for j in await self.list_judgements():
            if j.annotator_id in skip:
                continue
            pair = pairs.get(j.pair_id)
            if pair is None or pair.kind is PairKind.CATCH:
                continue
            if j.choice is Choice.TIE:
                ties.append((j.annotator_id, pair.item_a, pair.item_b))
                continue
            winner = j.winner(pair)
            assert winner is not None
            wins.append((j.annotator_id, winner, pair.other(winner)))
        return wins, ties
