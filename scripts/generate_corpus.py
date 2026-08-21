"""Bulk-generate the annotation corpus: many sets of 3 images, no videos.

    # free, and the default — exercises the whole path in mock mode
    .venv/bin/python scripts/generate_corpus.py --sets 100

    # what it would cost for real, without calling anything
    .venv/bin/python scripts/generate_corpus.py --sets 100 --price

    # the real thing
    AD_PROVIDER_MODE=live .venv/bin/python scripts/generate_corpus.py \\
        --sets 100 --references fixtures/references --confirm-spend

The predictor needs far more labelled images than the 8 affordable premium jobs
can produce, so the training corpus is generated separately and **without video**:
3 images at $0.04 is $0.12 a set, against $2.22 for a full job.  100 sets is $12
and yields 300 images — the budget line the spike allocated.

The sets go through the *same* sampler and brief compiler as product jobs, and
that is the point rather than convenience. A predictor trained on images from a
different prompt distribution than the one it will score is being asked to
transfer across a gap nobody measured.

**Corpus diversity is bounded by the reference photographs supplied.**  With four
reference pairs and 100 sets, every product appears 25 times under different
design points. That is exactly right for the within-set comparisons the sampler
ablation needs, and it means cross-set comparisons partly reflect which product
was photographed better rather than which creative decision was made. Supply as
many reference pairs as you can, and record how many the corpus was built from —
it is a limitation for the write-up, not a bug to hide.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("schema", "providers", "ml"):
    sys.path.insert(0, str(ROOT / "packages" / package))
sys.path.insert(0, str(ROOT / "services" / "api"))
sys.path.insert(0, str(ROOT / "services" / "worker"))

import adproviders as P  # noqa: E402
from adapi.annotation_store import AnnotationStore  # noqa: E402
from adschema import (  # noqa: E402
    DEFAULT_CANDIDATE_COUNT,
    AspectRatio,
    ConsentAttestation,
    CorpusItem,
    GateVerdict,
    ImageCandidate,
    ItemKind,
    Mood,
    Platform,
    ProductScale,
    ProviderMode,
    ThemeSpec,
    Vertical,
)
from adworker.briefs import compile_briefs  # noqa: E402
from adworker.gate import evaluate_image  # noqa: E402

CORPUS_PREFIX = "corpus/images"

#: Rotated across sets so the corpus spans categories rather than describing one.
#: The vertical is a real predictor feature, so a corpus drawn from a single
#: category would train a model that cannot condition on it.
VERTICALS = [
    Vertical.BEAUTY,
    Vertical.APPAREL,
    Vertical.FOOTWEAR,
    Vertical.ELECTRONICS,
    Vertical.JEWELLERY,
    Vertical.FOOD_BEVERAGE,
]
MOODS = list(Mood)

#: One platform for the whole corpus, deliberately — see `--platform`.
DEFAULT_PLATFORM = Platform.INSTAGRAM_REELS


def _load_reference_pairs(directory: Path | None, storage, count: int) -> list[tuple]:
    """Reference pairs to build sets from.

    With ``--references`` this reads ``<dir>/human/*`` and ``<dir>/product/*`` and
    pairs them by position, cycling the shorter list. Without it, synthetic mock
    references are used — which keeps the whole path runnable for free, and is why
    a dry run exercises real code rather than a separate code path.
    """
    if directory is None:
        return [
            (
                P.mock_portrait_asset(
                    storage,
                    f"{CORPUS_PREFIX}/ref/human-{i}.png",
                    AspectRatio.PORTRAIT_4_5,
                    seed=100 + i,
                ),
                P.mock_product_asset(
                    storage,
                    f"{CORPUS_PREFIX}/ref/product-{i}.png",
                    AspectRatio.SQUARE_1_1,
                    seed=200 + i,
                ),
            )
            for i in range(min(count, 4))
        ]

    humans = sorted(p for p in (directory / "human").glob("*") if p.is_file())
    products = sorted(p for p in (directory / "product").glob("*") if p.is_file())
    if not humans or not products:
        raise SystemExit(
            f"{directory} needs a human/ and a product/ subdirectory with images in each"
        )

    pairs = []
    for i in range(max(len(humans), len(products))):
        human_path = humans[i % len(humans)]
        product_path = products[i % len(products)]
        pairs.append(
            (
                storage.put_bytes(
                    f"{CORPUS_PREFIX}/ref/{human_path.name}",
                    human_path.read_bytes(),
                    "image/png",
                ),
                storage.put_bytes(
                    f"{CORPUS_PREFIX}/ref/{product_path.name}",
                    product_path.read_bytes(),
                    "image/png",
                ),
            )
        )
    return pairs


def _request_for_set(
    index: int, references: tuple, seed: int, platform: Platform, product_scale: str = ""
):
    from adschema import AdJobRequest

    human, product = references
    return AdJobRequest(
        job_id=f"corpus-{index:04d}-{uuid.uuid4().hex[:6]}",
        human_model_image=human,
        product_image=product,
        product_name=f"Corpus product {index}",
        caption="",
        cta_text="Shop now",
        vertical=VERTICALS[index % len(VERTICALS)],
        # Normally left to the vertical, which is what varies across the corpus.
        # Overridable so a single set can be generated against one specific
        # product's real size — which is how the scale instruction gets tested
        # against a real photograph without paying for a whole job.
        product_scale=product_scale or None,
        platform=platform,
        mood=MOODS[index % len(MOODS)],
        theme=ThemeSpec(palette=[]),
        duration_seconds=9.0,
        # Matched to the product's set size on purpose. The ranker is trained on
        # these sets and served on product jobs, and NDCG@1 over a set of three
        # is not the same quantity as over a set of five.
        candidate_count=DEFAULT_CANDIDATE_COUNT,
        seed=seed + index * 17,
        # The corpus references are synthetic or licensed shoot material the
        # operator supplies, and the attestation is the operator's, exactly as it
        # would be for a product job. There is no path here that skips it.
        consent=ConsentAttestation(has_model_release=True, not_a_public_figure=True),
    )


async def _run(args: argparse.Namespace) -> int:
    settings = P.Settings()
    storage = P.get_storage(settings.storage_backend, settings.storage_root)
    provider = P.get_image_provider(settings, storage)
    if isinstance(provider, P.MockImageProvider):
        # No design-point caption burned into the frame: these images go in front
        # of annotators, and a readable "rim backlit / seed 1053" in the corner
        # would tell them the condition they are supposed to be judging blind.
        provider.annotate_frames = False

    per_image = provider.estimate_cost(1)
    images = args.sets * 3
    total = round(per_image * images, 4)

    print(f"provider      : {provider.name} / {provider.model} ({provider.tier.value})")
    print(f"sets          : {args.sets} x 3 images = {images} images")
    print(f"platform      : {args.platform} ({Platform(args.platform).aspect_ratio.value})")
    print(f"cost estimate : {images} x ${per_image:.4f} = ${total:.2f}")

    live = settings.provider_mode is ProviderMode.LIVE
    if not live:
        # In mock mode the estimate above is $0, which is true and useless for
        # planning. Price the configured live model as well, so `--price` answers
        # "what would this cost for real" without needing live mode turned on.
        live_model = (
            settings.image_provider if settings.image_provider != "mock" else "seedream-4.5-edit"
        )
        try:
            live_total = P.estimate_image_cost(live_model, images)
            print(f"if live       : {images} x {live_model} = ${live_total:.2f}")
        except KeyError:
            print(f"if live       : no price on record for {live_model!r}")

    if live and not args.confirm_spend:
        print(
            f"\nAD_PROVIDER_MODE is 'live' and this would spend ${total:.2f}. "
            "Re-run with --confirm-spend."
        )
        return 1
    if args.price:
        print("\n--price: nothing generated.")
        return 0
    if not live:
        print("mode          : MOCK — nothing is spent, images are synthetic")

    ledger = P.SqlLedger.from_url(settings.database_url)
    await ledger.create_all()
    governor = P.CostGovernor(ledger, settings)
    corpus = AnnotationStore.from_url(settings.database_url)
    await corpus.create_all()
    llm = P.get_llm_provider(settings)

    before = await ledger.total_spent()
    print(f"spent so far  : ${before:.4f} of ${settings.budget_total_usd:.2f}\n")

    references = _load_reference_pairs(
        Path(args.references) if args.references else None, storage, args.sets
    )
    print(
        f"reference pairs: {len(references)} (each product recurs "
        f"~{args.sets / max(len(references), 1):.0f} times across the corpus)"
    )

    written = 0
    rejected = 0
    for index in range(args.sets):
        request = _request_for_set(
            index,
            references[index % len(references)],
            args.seed,
            Platform(args.platform),
            args.product_scale,
        )
        brief_set = await compile_briefs(request, llm)
        items: list[CorpusItem] = []

        for brief in brief_set.briefs:
            gen = P.ImageGenRequest(
                brief=brief,
                references=[request.human_model_image, request.product_image],
                aspect_ratio=request.platform.aspect_ratio,
                seed=brief.design_point.seed,
                output_key=f"{CORPUS_PREFIX}/{request.job_id}/i{brief.index}.png",
                reference_roles=["human model", "product"],
            )
            try:
                result = await governor.guarded_call(
                    job_id=request.job_id,
                    provider=provider.name,
                    model=provider.model,
                    operation="image",
                    estimated_usd=provider.estimate_cost(1),
                    quantity=1.0,
                    call=lambda gen=gen: provider.generate(gen),
                    actual_cost_of=lambda r: r.cost_usd,
                    note=f"corpus set {index}",
                )
            except P.BudgetExceeded as exc:
                print(f"\nSTOPPED at set {index}: {exc}")
                print(f"wrote {written} items before the cap was reached")
                await corpus.engine.dispose()
                await ledger.engine.dispose()
                return 1

            candidate = ImageCandidate(
                index=brief.index,
                brief=brief,
                asset=result.asset,
                tier=provider.tier,
                provider=provider.name,
                seed=result.seed,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
            )
            # Same gate as a product job: an unusable frame is not worth a
            # volunteer's attention, and the annotation corpus is for learning
            # performance, not for re-learning what the gate already rejects.
            candidate.gate = evaluate_image(candidate, request, storage)
            if candidate.gate.verdict is not GateVerdict.PASS:
                rejected += 1
                continue

            point = brief.design_point
            items.append(
                CorpusItem(
                    item_id=f"{request.job_id}-i{brief.index}",
                    set_id=request.job_id,
                    kind=ItemKind.IMAGE,
                    asset=result.asset,
                    tier=provider.tier,
                    provider=provider.name,
                    model=provider.model,
                    vertical=request.vertical,
                    platform=request.platform,
                    seed=result.seed,
                    angle=point.angle,
                    lighting=point.lighting,
                    composition=point.composition,
                    motion=point.motion,
                )
            )

        written += await corpus.add_items(items)
        if (index + 1) % max(1, args.sets // 10) == 0 or index + 1 == args.sets:
            spent = await ledger.total_spent()
            print(
                f"  set {index + 1:>4}/{args.sets}  items {written:>4}  "
                f"gate-rejected {rejected:>3}  spent ${spent:.4f}"
            )

    after = await ledger.total_spent()
    stats = await corpus.stats()
    print(f"\nspent this run: ${after - before:.4f}   total ${after:.4f}")
    print(f"corpus        : {stats.n_items} items in {stats.n_sets} sets")
    print("\nNext: design the comparisons over the new items")
    print("  .venv/bin/python scripts/build_corpus.py --commit")

    await corpus.engine.dispose()
    await ledger.engine.dispose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", type=int, default=10, help="sets of 3 images to generate")
    parser.add_argument(
        "--references",
        default=None,
        help="directory with human/ and product/ subdirectories; omit for synthetic mock refs",
    )
    parser.add_argument(
        "--price",
        action="store_true",
        help="print the cost and the plan, generate nothing",
    )
    parser.add_argument(
        "--confirm-spend",
        action="store_true",
        help="required in live mode; without it nothing is called",
    )
    parser.add_argument(
        "--platform",
        default=DEFAULT_PLATFORM.value,
        choices=[p.value for p in Platform],
        help="One platform for the whole corpus. Mixing them would put images of "
        "different aspect ratios into the same comparison, so a cross-set judgement "
        "would partly be about framing shape rather than about the creative decision — "
        "and a 4:5 frame letterboxed beside a 9:16 one is not an equal presentation.",
    )
    parser.add_argument(
        "--product-scale",
        default="",
        choices=["", *[s.value for s in ProductScale]],
        help="Override the per-vertical product size for every set. Use when "
        "generating against one real product rather than a mixed corpus.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
