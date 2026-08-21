"""Freeze one job into the golden demo set, so it can be shown again for free.

    # free, and the default — exercises the whole path on synthetic references
    .venv/bin/python scripts/freeze_golden.py --slug aurora-reels

    # what it would cost for real, without calling anything
    .venv/bin/python scripts/freeze_golden.py --slug aurora-reels --price

    # the real thing: a premium job, kept forever
    AD_PROVIDER_MODE=live .venv/bin/python scripts/freeze_golden.py \\
        --slug aurora-reels --references fixtures/references \\
        --product-name "Aurora Serum" --confirm-spend

    # and the artefact that survives a laptop reinstall
    .venv/bin/python scripts/freeze_golden.py --slug aurora-reels --pack

This is budget rule 6.  A premium job costs $2.22 and the video stage is not
reproducible at any price — Kling's image-to-video endpoint has no seed — so a run
that produced good candidates cannot be asked for again.  Freezing is the only way
to be able to show a specific result twice.

What gets written:

    fixtures/golden/<slug>/golden.json     the manifest — committed
    fixtures/golden/<slug>/assets/*        the frozen bytes — not committed

The split is deliberate.  Media in git is the one rule this repo has held to
throughout, and eight bundles of three 10 s clips is tens of megabytes.  The
manifest records every asset's SHA-256, so a missing or truncated file is reported
by name rather than showing up as a strange demo; ``--pack`` writes the whole thing
as one zip for backup, which is the copy that goes on a USB stick before the viva.
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
from adschema import (  # noqa: E402
    DEFAULT_CANDIDATE_COUNT,
    DEFAULT_VIDEO_COUNT,
    AdJobRequest,
    AspectRatio,
    BackgroundTreatment,
    ConsentAttestation,
    Mood,
    Platform,
    StageEvent,
    ThemeSpec,
    Vertical,
)
from adworker import Pipeline  # noqa: E402


def _references(directory: Path | None, storage, job_id: str, seed: int):
    """The two uploads a job needs.

    Synthetic by default so the whole freeze path is exercisable for free, and so a
    mock bundle exists to test the replay machinery against before any money is
    spent on a real one.  ``--references`` takes the first file from ``human/`` and
    ``product/`` — a golden bundle is one specific job, so there is nothing to
    rotate through.
    """
    if directory is None:
        human = P.mock_portrait_asset(
            storage, f"uploads/{job_id}/human.png", AspectRatio.PORTRAIT_4_5, seed=seed + 4
        )
        product = P.mock_product_asset(
            storage, f"uploads/{job_id}/product.png", AspectRatio.SQUARE_1_1, seed=seed + 15
        )
        return human, product, True

    def first(kind: str) -> Path:
        folder = directory / kind
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not files:
            raise SystemExit(f"no images in {folder}")
        return files[0]

    out = []
    for kind, mime in (("human", "image/png"), ("product", "image/png")):
        path = first(kind)
        data = path.read_bytes()
        suffix = path.suffix.lower()
        out.append(
            storage.put_bytes(
                f"uploads/{job_id}/{kind}{suffix}",
                data,
                "image/jpeg" if suffix in {".jpg", ".jpeg"} else mime,
            )
        )
    return out[0], out[1], False


async def _run(args: argparse.Namespace) -> int:
    settings = P.get_settings()
    storage = P.get_storage(settings.storage_backend, settings.storage_root)

    slug = args.slug.strip().lower()
    if not slug or not all(c.isalnum() or c in "-_" for c in slug):
        print(f"--slug must be a short alphanumeric name, got {args.slug!r}")
        return 2

    manifest = P.golden_bundle_path(settings, slug) / "golden.json"
    if manifest.exists() and not (args.pack or args.force):
        print(
            f"{manifest} already exists. Freezing again would overwrite a bundle that "
            "cost money to make — pass --force if that is what you mean, or pick "
            "another --slug."
        )
        return 2

    # --pack does not run a job. It repackages what is already frozen, which is the
    # common case: the bundle is made once and copied about many times.
    if args.pack:
        bundle = P.GoldenBundle.load(P.golden_bundle_path(settings, slug))
        problems = bundle.verify(storage)
        if problems:
            print(f"refusing to pack {slug}: the bundle is incomplete")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        dest = Path(args.pack_to or (ROOT / "fixtures" / "golden" / f"{slug}.zip"))
        bundle.pack(dest, storage)
        print(f"packed {bundle.summary()}")
        print(f"  → {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return 0

    duration = float(args.duration)
    platform = Platform(args.platform)
    estimate = P.estimate_job_cost(
        settings.image_provider if settings.is_live else "mock",
        settings.video_provider if settings.is_live else "mock",
        settings.llm_provider if settings.is_live else "mock",
        args.candidates,
        duration,
    )

    print(settings.describe())
    print(
        f"freezing {slug!r}: {args.candidates} candidates, {duration:.0f}s, "
        f"{platform.value}, estimated ${estimate:.4f}"
    )
    if args.price:
        print("--price: nothing was called.")
        return 0
    if settings.is_live and not args.confirm_spend:
        print(
            f"live mode needs --confirm-spend. This would cost about ${estimate:.4f} "
            "and there is no way to un-spend it."
        )
        return 2
    if settings.is_replay:
        print(
            "AD_PROVIDER_MODE=replay cannot freeze: it would re-record a bundle from "
            "itself. Use mock or live mode."
        )
        return 2

    ledger = P.SqlLedger.from_url(settings.database_url)
    await ledger.create_all()
    before = await ledger.total_spent()

    job_id = uuid.uuid4().hex[:16]
    human, product, synthetic = _references(
        args.references and Path(args.references), storage, job_id, args.seed
    )
    request = AdJobRequest(
        job_id=job_id,
        human_model_image=human,
        product_image=product,
        product_name=args.product_name,
        caption=args.caption,
        cta_text=args.cta,
        vertical=Vertical(args.vertical),
        platform=platform,
        mood=Mood(args.mood),
        theme=ThemeSpec(
            palette=[c.strip() for c in args.palette.split(",") if c.strip()],
            background=BackgroundTreatment(args.background),
        ),
        duration_seconds=duration,
        candidate_count=args.candidates,
        video_count=args.videos,
        seed=args.seed,
        consent=ConsentAttestation(has_model_release=True, not_a_public_figure=True),
    )

    # The request as submitted. Stage 1 writes the extracted palette back into this
    # object, so the bundle needs a copy taken before the run or a replay will be
    # handed a palette it should have had to derive.
    submitted = request.model_copy(deep=True)

    recorder = P.GoldenRecorder(slug, storage)
    providers = P.get_providers(settings, storage)
    pipeline = Pipeline(
        storage=storage,
        governor=P.CostGovernor(ledger, settings),
        image_provider=P.RecordingImageProvider(providers.images, recorder),
        video_provider=P.RecordingVideoProvider(providers.videos, recorder),
        llm_provider=P.RecordingLLMProvider(providers.llm, recorder),
        on_progress=_printer(),
    )

    record = await pipeline.run(request)
    spent = await ledger.total_spent() - before
    await ledger.engine.dispose()

    print()
    if record.state.value != "completed":
        print(f"the job ended {record.state.value}: {record.error or record.refusal_reason}")
        print("not freezing a job that did not finish — there would be nothing to demo.")
        return 1

    bundle = P.build_golden_bundle(
        slug,
        request=submitted,
        record=record,
        recorder=recorder,
        provider_mode=settings.provider_mode.value,
        image_model=providers.images.model,
        video_model=providers.videos.model,
        llm_model=providers.llm.model,
        title=args.title or args.product_name,
    )
    path = bundle.save(settings.golden_root)

    problems = bundle.verify(storage)
    print(bundle.summary())
    print(f"  manifest  {path}")
    print(f"  spent     ${spent:.4f} freezing this bundle")
    expectation = bundle.expectation
    if expectation is not None:
        print(
            f"  outcome   image order {expectation.image_order} → video order "
            f"{expectation.video_order}, winner slot {expectation.winner_slot}"
        )
        print(f"  delivery  {expectation.delivery_summary}")
    cached = bundle.provenance.get("backfilled") or []
    if cached:
        print(
            f"  cache     {len(cached)} generation(s) were served from the governor's "
            "cache and frozen from the job record rather than from a provider call"
        )
    if synthetic:
        print(
            "  note      frozen from synthetic references, so this bundle demonstrates "
            "the machinery rather than the output. Re-freeze from real photographs "
            "once the premium budget is released."
        )
    for problem in problems:
        print(f"  PROBLEM   {problem}")
    print()
    print(f"replay it with:  .venv/bin/python scripts/replay_golden.py {slug}")
    return 1 if problems else 0


def _printer():
    async def on_progress(event: StageEvent) -> None:
        if event.state in {"started", "failed", "warning"} or event.progress >= 1.0:
            mark = {"failed": "✗", "warning": "!"}.get(event.state, "·")
            print(f"  {mark} {event.stage.value:<12} {event.message}")

    return on_progress


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", required=True, help="short name for the bundle directory")
    parser.add_argument("--title", default="", help="human-readable label, defaults to the product")
    parser.add_argument(
        "--references",
        default=None,
        help="directory with human/ and product/ subdirectories; omit for synthetic refs",
    )
    parser.add_argument("--product-name", default="Aurora Serum")
    parser.add_argument("--caption", default="Glow that lasts")
    parser.add_argument("--cta", default="Shop now")
    parser.add_argument(
        "--vertical", default=Vertical.BEAUTY.value, choices=[v.value for v in Vertical]
    )
    parser.add_argument(
        "--platform", default=Platform.INSTAGRAM_REELS.value, choices=[p.value for p in Platform]
    )
    parser.add_argument("--mood", default=Mood.CALM_PREMIUM.value, choices=[m.value for m in Mood])
    parser.add_argument(
        "--background",
        default=BackgroundTreatment.SOFT_GRADIENT.value,
        choices=[b.value for b in BackgroundTreatment],
    )
    parser.add_argument("--palette", default="", help="comma-separated hex; empty means extract it")
    parser.add_argument("--duration", type=float, default=9.0)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument("--videos", type=int, default=DEFAULT_VIDEO_COUNT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--price", action="store_true", help="print the cost, call nothing")
    parser.add_argument(
        "--confirm-spend",
        action="store_true",
        help="required in live mode; without it nothing runs",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing bundle")
    parser.add_argument(
        "--pack", action="store_true", help="zip an existing bundle instead of freezing a new one"
    )
    parser.add_argument("--pack-to", default=None, help="where to write the zip")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
