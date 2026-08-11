"""Turn completed jobs into an annotation corpus, and design the comparisons.

    .venv/bin/python scripts/build_corpus.py --plan          # nothing is written
    .venv/bin/python scripts/build_corpus.py --commit
    .venv/bin/python scripts/build_corpus.py --commit --decoys 12

This is free — no provider is called.  It reads the candidates the pipeline has
already generated, registers each as a :class:`CorpusItem`, builds catch-trial
decoys, and runs :func:`adml.pairs.design_pairs` over the result.

``--plan`` is the default, and prints what would be written along with the
comparison design's diagnostics: how balanced the per-item comparison count is,
whether the graph is connected, and how many hours of human time the design would
cost.  That last number is the one worth reading before committing — a design
nobody has time to complete produces a partially-observed graph, and a partially
observed graph is exactly the disconnection problem the design set out to avoid.

Re-running is safe and additive: items and pairs are de-duplicated, so this is
the command to run again each time another batch of jobs completes.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("schema", "providers", "ml"):
    sys.path.insert(0, str(ROOT / "packages" / package))
sys.path.insert(0, str(ROOT / "services" / "api"))

import adproviders as P  # noqa: E402
from adapi.annotation_store import TARGET_JUDGEMENTS, AnnotationStore  # noqa: E402
from adapi.store import JobStore  # noqa: E402
from adml import degrade as D  # noqa: E402
from adml import pairs as PAIRS  # noqa: E402
from adschema import (  # noqa: E402
    AssetRef,
    CorpusItem,
    ItemKind,
    JobState,
)

DECOY_PREFIX = "corpus/decoys"


def _items_from_jobs(records) -> list[CorpusItem]:
    """One corpus item per generated image that passed the quality gate.

    Gate failures are excluded deliberately.  The gate answers "is this usable at
    all" and the annotation corpus is for learning "how well will this perform" —
    asking someone to rank a candidate with a mangled face teaches the predictor
    to detect mangled faces, which the gate already does for free, and wastes the
    comparison on a distinction nobody needs a model for.
    """
    out: list[CorpusItem] = []
    for record in records:
        if record.state is not JobState.COMPLETED or record.result is None:
            continue
        request = record.request
        for candidate in record.result.images:
            if not candidate.passed_gate:
                continue
            point = candidate.brief.design_point
            out.append(
                CorpusItem(
                    item_id=f"{record.job_id}-i{candidate.index}",
                    set_id=record.job_id,
                    kind=ItemKind.IMAGE,
                    asset=candidate.asset,
                    tier=candidate.tier,
                    provider=candidate.provider,
                    vertical=request.vertical,
                    platform=request.platform,
                    seed=candidate.seed,
                    angle=point.angle,
                    lighting=point.lighting,
                    composition=point.composition,
                    motion=point.motion,
                )
            )
    return out


def _build_decoys(
    items: list[CorpusItem], storage, *, count: int, seed: int
) -> tuple[list[CorpusItem], list[str]]:
    """Degrade a sample of items into catch-trial decoys.

    A decoy whose measured sharpness barely dropped is rejected rather than
    shipped: a catch trial that is genuinely hard tests eyesight and flags careful
    annotators as careless, which is worse than having no catch trial.
    """
    rng = random.Random(seed)
    chosen = rng.sample(items, min(count, len(items)))
    decoys: list[CorpusItem] = []
    notes: list[str] = []
    for item in chosen:
        try:
            source = storage.get_bytes(item.asset.key)
        except (FileNotFoundError, ValueError):
            notes.append(f"  skipped {item.item_id}: source bytes missing")
            continue
        result = D.degrade(source)
        if not result.is_obvious:
            notes.append(
                f"  skipped {item.item_id}: sharpness only fell "
                f"{result.sharpness_before:.3f} -> {result.sharpness_after:.3f}"
            )
            continue
        key = f"{DECOY_PREFIX}/{item.item_id}.png"
        stored = storage.put_bytes(key, result.data, "image/png")
        decoys.append(
            CorpusItem(
                item_id=f"{item.item_id}-decoy",
                set_id=item.set_id,
                kind=item.kind,
                asset=AssetRef(
                    key=stored.key,
                    mime_type="image/png",
                    width=stored.width,
                    height=stored.height,
                ),
                tier=item.tier,
                provider=item.provider,
                vertical=item.vertical,
                platform=item.platform,
                degraded_from=item.item_id,
                degradation=result.label,
            )
        )
    return decoys, notes


async def _run(args: argparse.Namespace) -> int:
    settings = P.Settings()
    storage = P.get_storage(settings.storage_backend, settings.storage_root)
    jobs = JobStore.from_url(settings.database_url)
    corpus = AnnotationStore.from_url(settings.database_url)
    # Both, because either can be the first thing to touch a fresh database:
    # generate_corpus.py fills the corpus without ever creating a job.
    await jobs.create_all()
    await corpus.create_all()

    records = await jobs.list_recent(limit=args.max_jobs)
    fresh = _items_from_jobs(records)
    stored = await corpus.list_items(include_decoys=True)
    existing = {i.item_id for i in stored}
    new_items = [i for i in fresh if i.item_id not in existing]

    if not fresh and not stored:
        print(
            "Nothing to build from. Either run some jobs:\n"
            "  curl -X POST 'http://127.0.0.1:8000/api/jobs/demo'\n"
            "or bulk-generate a corpus:\n"
            "  .venv/bin/python scripts/generate_corpus.py --sets 20"
        )
        return 1

    print(f"jobs scanned        : {len(records)}")
    print(f"gate-passing images : {len(fresh)} ({len(new_items)} not yet in the corpus)")
    print(f"already in corpus   : {len(existing)} items")

    # Decoys are built for items not already covered by one, so re-running tops up
    # the catch trials as the corpus grows instead of re-degrading the same frames.
    decoys: list[CorpusItem] = []
    notes: list[str] = []
    already_decoyed = {i.degraded_from for i in stored if i.is_decoy}
    decoy_candidates = [
        i
        for i in [*new_items, *(s for s in stored if not s.is_decoy)]
        if i.item_id not in already_decoyed
    ]
    if args.decoys > 0 and decoy_candidates:
        decoys, notes = _build_decoys(decoy_candidates, storage, count=args.decoys, seed=args.seed)
        print(f"catch-trial decoys  : {len(decoys)} built ({len(already_decoyed)} already existed)")
        for note in notes:
            print(note)

    # Design over the whole corpus, not only the new items: a pair joining a new
    # item to an old one is the most informative kind, and designing over each
    # batch alone would leave every batch as its own disconnected component.
    unique: dict[str, CorpusItem] = {
        item.item_id: item for item in await corpus.list_items(kind=ItemKind.IMAGE)
    }
    for item in new_items:
        unique.setdefault(item.item_id, item)
    pool = list(unique.values())

    design = PAIRS.design_pairs(pool, seed=args.seed, cross_set_rounds=args.rounds)
    catch = PAIRS.design_catch_pairs(pool, [*decoys, *await _stored_decoys(corpus)])

    print()
    print("comparison design")
    print(f"  {design.summary()}")
    if design.isolated_items:
        print(f"  WARNING {len(design.isolated_items)} items have no comparisons")
    # The effort estimate reads the store's own collection targets rather than a
    # flag, so the number quoted to a volunteer is the number the serving policy
    # will actually ask of them. Within-set pairs are targeted twice; see
    # `TARGET_JUDGEMENTS` for the arithmetic behind that.
    per_pair = args.judgements_per_pair
    if per_pair is None:
        wanted = sum(TARGET_JUDGEMENTS.get(p.kind, 1) for p in design.pairs)
        per_pair = wanted / len(design.pairs) if design.pairs else 1.0
    effort = design.estimate_effort(
        judgements_per_pair=per_pair,
        annotators=args.annotators,
        repeat_rate=0.08,
    )
    targets = ", ".join(f"{k.value} x{v}" for k, v in TARGET_JUDGEMENTS.items())
    print(f"  targets: {targets}  ({per_pair:.2f} judgements per pair on average)")
    print(
        f"  {effort['judgements']:.0f} judgements = {effort['total_minutes']:.0f} min total, "
        f"{effort['minutes_each']:.0f} min each across {args.annotators} annotators"
    )
    print(f"  plus {len(catch)} catch trials")

    if not args.commit:
        print("\n--plan: nothing written. Re-run with --commit to persist.")
        return 0

    wrote_items = await corpus.add_items([*new_items, *decoys])
    wrote_pairs = await corpus.add_pairs([*design.pairs, *catch])
    stats = await corpus.stats()
    print(f"\nwrote {wrote_items} items and {wrote_pairs} pairs")
    print(
        f"corpus now: {stats.n_items} items ({stats.n_decoys} decoys) in "
        f"{stats.n_sets} sets, {stats.n_pairs} pairs, {stats.n_judgements} judgements "
        f"from {stats.n_annotators} annotators"
    )
    print("\nShare the tool:  http://<your-ip>:8000/api/annotate/ui")

    await corpus.engine.dispose()
    await jobs.engine.dispose()
    return 0


async def _stored_decoys(corpus: AnnotationStore) -> list[CorpusItem]:
    return [i for i in await corpus.list_items(include_decoys=True) if i.is_decoy]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", action="store_true", help="write to the database")
    parser.add_argument("--plan", action="store_true", help="default; print and write nothing")
    parser.add_argument(
        "--decoys",
        type=int,
        default=12,
        help="catch-trial decoys to build. An annotator sees at most one catch trial per "
        "decoy, so this is the ceiling on the screening evidence per person; keep it at or "
        "above adschema.MIN_CATCH_TRIALS.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=PAIRS.DEFAULT_CROSS_SET_ROUNDS,
        help="cross-set matchings per item; higher means more pairs and finer strengths",
    )
    parser.add_argument("--annotators", type=int, default=6, help="for the effort estimate")
    parser.add_argument(
        "--judgements-per-pair",
        type=float,
        default=None,
        help="override the estimate; by default it is derived from the store's "
        "TARGET_JUDGEMENTS so the quoted time matches what will be asked",
    )
    parser.add_argument("--max-jobs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
