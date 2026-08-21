"""One real call per live adapter, to check the docs were telling the truth.

    .venv/bin/python scripts/smoke_live.py --image --confirm-spend
    .venv/bin/python scripts/smoke_live.py --video --confirm-spend

Why this exists as a separate script rather than a test: the adapters in
``adproviders.fal`` were written against fal's published schemas, and the test
suite drives them with a stub transport. That proves the adapters implement the
documentation. It cannot prove the documentation is accurate — the parameter
names, the duration enum, the response shape and the price are all claims, and
the only way to check a claim about someone else's API is to call it.

So this spends real money, deliberately and in the smallest amount that settles
the question:

* image — one Seedream edit, **$0.04**
* video — one Kling clip at the **5 s** option, **$0.35**, not the 10 s one,
  because the point is to confirm the contract rather than to make an ad

Both go through the ``CostGovernor``, so the ledger reflects them and the budget
cap applies exactly as it would in a real job. Run each once, read the report,
and record what differed from the documentation in ``docs/provider-spike.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "schema"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "providers"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "worker"))

import adproviders as P  # noqa: E402
from adschema import (  # noqa: E402
    AdJobRequest,
    AspectRatio,
    AssetRef,
    ConsentAttestation,
    ProviderMode,
    ShotBrief,
    Vertical,
)
from adworker.briefs import compile_briefs  # noqa: E402

SMOKE_DIR = "smoke"


def _smoke_request() -> AdJobRequest:
    """The request the brief is compiled from.

    The references are placeholders — the smoke test hands the provider its own
    generated assets directly. What this request is for is the *prompt*, and the
    seed is fixed so the same design point comes out every run.
    """
    ref = AssetRef(key="smoke/placeholder.png", sha256="0" * 64)
    return AdJobRequest(
        job_id="smoke",
        product_name="a cosmetics bottle",
        vertical=Vertical.BEAUTY,
        caption="Glow that lasts",
        human_model_image=ref,
        product_image=ref,
        seed=7,
        consent=ConsentAttestation(has_model_release=True, not_a_public_figure=True),
    )


async def _brief() -> ShotBrief:
    """The first candidate's brief, straight out of the real compiler.

    This used to be a ShotBrief written out by hand here, including a motion prompt
    reading "slow smooth dolly in toward the subject" — a sentence that appears
    nowhere in the pipeline. That made the cheapest verification the project owns
    test something the project does not send, and after the motion axis was
    rewritten it would have gone on spending $0.35 to confirm the exact prompt that
    caused the zoom problem in the first place.

    Compiling through ``compile_briefs`` costs nothing (the mock LLM does not call
    out) and means a $0.35 clip verifies the prompt a real job would send.
    """
    briefs = await compile_briefs(_smoke_request(), P.MockLLMProvider())
    return briefs.briefs[0]


async def _run(kind: str) -> int:
    settings = P.Settings()
    if settings.provider_mode is not ProviderMode.LIVE:
        print("AD_PROVIDER_MODE is not 'live'. Set it in .env to make a real call.")
        return 1
    if not settings.has_fal_key:
        print("AD_FAL_API_KEY is not set. Add it to .env.")
        return 1

    storage = P.get_storage(settings.storage_backend, settings.storage_root)
    ledger = P.SqlLedger.from_url(settings.database_url)
    governor = P.CostGovernor(ledger, settings)

    print(f"before: ${await ledger.total_spent():.4f} of ${settings.budget_total_usd:.2f} spent")

    human = P.mock_portrait_asset(storage, f"{SMOKE_DIR}/human.png", AspectRatio.PORTRAIT_4_5)
    product = P.mock_product_asset(storage, f"{SMOKE_DIR}/product.png", AspectRatio.SQUARE_1_1)

    if kind == "image":
        provider = P.get_image_provider(settings, storage)
        request = P.ImageGenRequest(
            brief=await _brief(),
            references=[human, product],
            aspect_ratio=AspectRatio.VERTICAL_9_16,
            seed=7,
            output_key=f"{SMOKE_DIR}/out_image.png",
            reference_roles=["human model", "product"],
        )
        estimate = provider.estimate_cost(1)
        print(f"calling {provider.model}, estimated ${estimate:.4f} …")
        result = await governor.guarded_call(
            job_id="smoke",
            provider=provider.name,
            model=provider.model,
            operation="image",
            estimated_usd=estimate,
            quantity=1.0,
            call=lambda: provider.generate(request),
            actual_cost_of=lambda r: r.cost_usd,
            note="live smoke test",
        )
        print(f"  saved      : {result.asset.key}")
        print(f"  dimensions : {result.asset.width}x{result.asset.height}")
        print(f"  seed echoed: {result.seed} (requested 7)")
        print(f"  latency    : {result.latency_ms} ms")
    else:
        provider = P.get_video_provider(settings, storage)
        # 5 s, not the project's 9 s: half the price and it answers the same
        # question about the duration enum and the response shape.
        seconds = 5.0
        request = P.VideoGenRequest(
            # The compiled prompt says 9 s while this asks for 5, and that is
            # left alone deliberately. `AdJobRequest` enforces the project's own
            # 8-10 s window, so a 5 s brief is not expressible without weakening
            # the schema to save $0.35 on a contract test. Kling takes duration
            # as an explicit parameter; the sentence in the prompt is advisory.
            brief=await _brief(),
            start_image=product,
            duration_seconds=seconds,
            aspect_ratio=AspectRatio.VERTICAL_9_16,
            seed=7,
            output_key=f"{SMOKE_DIR}/out_video.mp4",
        )
        estimate = provider.estimate_cost(seconds)
        print(
            f"calling {provider.model} at {provider.deliverable_duration(seconds):.0f}s, "
            f"estimated ${estimate:.4f} …"
        )
        result = await governor.guarded_call(
            job_id="smoke",
            provider=provider.name,
            model=provider.model,
            operation="video",
            estimated_usd=estimate,
            quantity=seconds,
            call=lambda: provider.generate(request),
            actual_cost_of=lambda r: r.cost_usd,
            note="live smoke test",
        )
        print(f"  saved      : {result.asset.key}")
        print(f"  duration   : {result.duration_seconds}s reported")
        print(f"  seeded     : {result.seed_honoured}")
        print(f"  latency    : {result.latency_ms} ms")
        print("  CHECK: play the file and confirm it really is that many seconds.")

    # Keep the raw response. It cost real money and it is the only record of what
    # the API actually returned — which fields exist, which the documentation
    # promised and did not deliver. The first image call found two such gaps and
    # they were nearly lost to a terminal scrollback.
    raw_path = Path(settings.storage_root) / SMOKE_DIR / f"raw_{kind}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(result.raw, indent=2, default=str))
    print(f"  raw saved  : {raw_path}")

    print(f"after : ${await ledger.total_spent():.4f} of ${settings.budget_total_usd:.2f} spent")
    await ledger.engine.dispose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", action="store_true", help="one Seedream edit, ~$0.04")
    group.add_argument("--video", action="store_true", help="one 5 s Kling clip, ~$0.35")
    parser.add_argument(
        "--confirm-spend",
        action="store_true",
        help="required; without it nothing is called",
    )
    args = parser.parse_args()

    if not args.confirm_spend:
        print("This spends real money. Re-run with --confirm-spend if that is what you want.")
        return 1

    return asyncio.run(_run("image" if args.image else "video"))


if __name__ == "__main__":
    raise SystemExit(main())
