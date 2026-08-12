"""FastAPI service.

Jobs run as asyncio background tasks in this process, which is right for a
single-developer dev loop and explicitly not right for anything else — the ARQ
worker replaces this when concurrency matters.  What the API commits to now is the
contract the web app will consume: create a job, stream its stages, read the
result, and read the budget.

The budget endpoint is not an afterthought.  With $35 total, "how much is left"
belongs on screen at all times, so it is a first-class route from day one.

**Every dependency is injected, and that is why this file has tests.**  Until now
the settings, storage, ledger, job store and event bus were built at module scope,
so importing this module opened the real database and pointed at the real fixtures
root.  Nothing could be tested over HTTP without those side effects, and the result
was a product surface with no test at all behind it — the pipeline was covered, the
routes that expose it were not.  :class:`Services` now holds the lot and
:func:`create_app` takes one, so ``tests/test_api.py`` runs the whole API against a
temporary directory and an in-memory ledger.

The module-level ``app`` is still what ``uvicorn adapi.main:app`` loads.  Building
it constructs SQLAlchemy engines, which do not connect until the lifespan runs, so
an import is cheap and touches no database.

One sharp edge is left deliberately: the annotation router keeps its store in a
module global (``annotate.bind``), because its own tests call those route functions
directly rather than over HTTP.  ``create_app`` binds it, so the last app created
in a process owns it.  That is fine for one server and for a sequential test run,
and it is the next thing to fix if either of those stops being true.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

import adproviders as P
from adschema import (
    AdJobRequest,
    AppConfig,
    AspectRatio,
    AssetRef,
    BudgetStatus,
    ConsentAttestation,
    DeliveryResponse,
    GoldenSummary,
    JobRecord,
    JobState,
    JobSummary,
    Launched,
    LedgerResponse,
    Mood,
    Platform,
    PlatformProfile,
    Stage,
    StageEvent,
    ThemeSpec,
    Vertical,
)
from adworker import Pipeline
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from . import annotate
from .annotation_store import AnnotationStore
from .store import EventBus, JobStore

MEDIA_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webp": "image/webp",
}


# --- Wiring ----------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class Services:
    """Everything a request might need, in one injectable bundle.

    A dataclass rather than a set of ``Depends`` providers because these are all
    process-lifetime singletons: one storage root, one ledger, one event bus. The
    only reason they are not module globals is that a test needs a second set.
    """

    settings: P.Settings
    storage: P.Storage
    ledger: P.LedgerStore
    store: JobStore
    annotations: AnnotationStore
    bus: EventBus

    #: Keeps background task references alive so they are not garbage collected
    #: mid-run, which asyncio does not otherwise prevent.
    running: dict[str, asyncio.Task] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: P.Settings | None = None) -> Services:
        settings = settings or P.get_settings()
        return cls(
            settings=settings,
            storage=P.get_storage(settings.storage_backend, settings.storage_root),
            ledger=P.SqlLedger.from_url(settings.database_url),
            store=JobStore.from_url(settings.database_url),
            annotations=AnnotationStore.from_url(settings.database_url),
            bus=EventBus(),
        )

    async def create_all(self) -> None:
        """Create tables for whichever components are SQL-backed.

        ``InMemoryLedger`` has no schema, and putting that branch in the ledger
        protocol would give every implementation a method that does nothing so that
        this one loop could stay a line shorter.
        """
        for component in (self.ledger, self.store, self.annotations):
            create = getattr(component, "create_all", None)
            if create is not None:
                await create()

    def governor(self, *, use_cache: bool = True) -> P.CostGovernor:
        return P.CostGovernor(self.ledger, self.settings, use_cache=use_cache)

    async def wait_for_jobs(self) -> None:
        """Await every job this app started.

        A test that posts a job and then reads the record needs the background task
        to have finished, and polling the store for a state change would trade a
        deterministic wait for a flaky one.
        """
        while self.running:
            await asyncio.gather(*list(self.running.values()), return_exceptions=True)


def services_of(request: Request) -> Services:
    return request.app.state.services


Svc = Annotated[Services, Depends(services_of)]


def create_app(services: Services | None = None) -> FastAPI:
    """Build the application around one :class:`Services` bundle."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc: Services = app.state.services
        await svc.create_all()
        # The mode banner, on startup, loudly: the failure most worth avoiding is
        # spending real money while believing the app is in mock mode.
        print(svc.settings.describe())
        yield

    app = FastAPI(title="Multi-Candidate Ad Generation", version="0.1.0", lifespan=lifespan)
    app.state.services = services or Services.from_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    annotate.bind(app.state.services.annotations)
    app.include_router(annotate.router)
    app.include_router(router)
    return app


# --- Job execution ---------------------------------------------------------


async def _store_upload(svc: Services, job_id: str, name: str, upload: UploadFile) -> AssetRef:
    data = await upload.read()
    if not data:
        raise HTTPException(400, f"{name} upload was empty")
    suffix = Path(upload.filename or "").suffix.lower() or ".png"
    if suffix not in MEDIA_MIME:
        raise HTTPException(400, f"{name}: unsupported file type {suffix!r}")
    key = f"uploads/{job_id}/{name}{suffix}"
    return svc.storage.put_bytes(key, data, MEDIA_MIME[suffix])


async def _run(svc: Services, record: JobRecord, providers: P.ProviderSet | None = None) -> None:
    """Execute a job, persisting after every stage so a refresh shows progress."""

    async def on_progress(event: StageEvent) -> None:
        await svc.bus.publish(event)
        await svc.store.save(record)

    providers = providers or P.get_providers(svc.settings, svc.storage)
    replay = providers.golden
    pipeline = Pipeline(
        storage=svc.storage,
        # A replay reads its generations from the bundle, so the cache is turned
        # off: with it on, every fingerprint hits and the frozen bytes are never
        # opened. See CostGovernor.use_cache.
        governor=svc.governor(use_cache=replay is None),
        image_provider=providers.images,
        video_provider=providers.videos,
        llm_provider=providers.llm,
        on_progress=on_progress,
    )
    try:
        before = await svc.ledger.total_spent()
        completed = await pipeline.run(record.request)
        if replay is not None:
            spent = await svc.ledger.total_spent() - before
            await _report_drift(completed, replay, spent, on_progress)
        await svc.store.save(completed)
    finally:
        svc.bus.mark_finished(record.job_id)
        svc.running.pop(record.job_id, None)


async def _report_drift(record: JobRecord, replay: P.GoldenSession, spent: float, emit) -> None:
    """Put the replay's disagreement with its bundle into the job's own event log.

    A replay is a regression test, and the place a regression has to be visible is
    the run that found it — not a script's stdout that nobody reads during a demo.
    """
    drift = P.compare_replay(
        replay.bundle, record, spent_usd=spent, notes=replay.notes + replay.audit()
    )
    messages = (
        [(f"replay of {drift.slug!r} matches what was frozen", "progress")]
        if drift.ok
        else [(f"golden replay drift — {line}", "warning") for line in drift.lines]
    )
    for message, state in messages:
        event = StageEvent(
            job_id=record.job_id,
            stage=Stage.DELIVERY,
            state=state,
            message=message,
            progress=1.0,
        )
        record.events.append(event)
        await emit(event)


async def _launch(
    svc: Services, request: AdJobRequest, providers: P.ProviderSet | None = None
) -> JobRecord:
    record = JobRecord(request=request, state=JobState.QUEUED)
    await svc.store.save(record)
    svc.running[record.job_id] = asyncio.create_task(_run(svc, record, providers))
    return record


# --- Routes ----------------------------------------------------------------

#: A router rather than decorators on a module-level ``app``: the routes have to be
#: attachable to more than one application, which is the whole point of the factory.
router = APIRouter()


@router.get("/api/budget", response_model=BudgetStatus)
async def get_budget(svc: Svc) -> BudgetStatus:
    return await svc.governor().status()


@router.get("/api/config")
async def get_config(svc: Svc) -> AppConfig:
    """What the UI needs to render honestly: the mode, and what it costs."""
    settings = svc.settings
    return AppConfig(
        provider_mode=settings.provider_mode,
        is_live=settings.is_live,
        is_replay=settings.is_replay,
        image_provider=settings.image_provider,
        video_provider=settings.video_provider,
        golden_set=settings.golden_set,
        banner=settings.describe(),
        platforms={p: PlatformProfile.of(p) for p in Platform},
    )


@router.post("/api/jobs/demo")
async def create_demo_job(
    svc: Svc,
    platform: Platform = Platform.INSTAGRAM_REELS,
    duration_seconds: float = 9.0,
    seed: int = 7,
    brand_palette: bool = False,
) -> Launched:
    """Start a job against synthetic references.

    Exists so the whole pipeline can be exercised without hunting for a model
    photograph — which matters because the mock path is the one used all day.

    Note the job id is fresh per invocation rather than derived from the seed.
    Deriving it from the seed seemed tidy but collided: re-running the same seed
    reused a finished job's id, so the event bus replayed the previous run's
    history and the page rendered a stale result. Determinism belongs to the
    *seed*, which still fixes the design points and the generated pixels — the
    job id is just identity.
    """
    job_id = uuid.uuid4().hex[:16]
    # Role-specific synthetic uploads so the demo exercises the real intake paths:
    # the portrait is detectable by the face cascade and the product sits on a flat
    # sweep the flood-fill cutout can handle.
    human = P.mock_portrait_asset(
        svc.storage, f"uploads/{job_id}/human.png", AspectRatio.PORTRAIT_4_5, seed=11
    )
    product = P.mock_product_asset(
        svc.storage, f"uploads/{job_id}/product.png", AspectRatio.SQUARE_1_1, seed=22
    )
    request = AdJobRequest(
        job_id=job_id,
        human_model_image=human,
        product_image=product,
        product_name="Aurora Serum",
        caption="Glow that lasts",
        cta_text="Shop now",
        vertical=Vertical.BEAUTY,
        platform=platform,
        mood=Mood.CALM_PREMIUM,
        # Empty by default so intake extracts the palette from the product cutout,
        # which is the more interesting path to be able to look at. Pass
        # `brand_palette=true` to exercise the supplied-palette path instead.
        theme=ThemeSpec(
            palette=["#2b3a55", "#ce7777", "#f2e7d5"] if brand_palette else [],
        ),
        duration_seconds=duration_seconds,
        candidate_count=3,
        seed=seed,
        consent=ConsentAttestation(has_model_release=True, not_a_public_figure=True),
    )
    record = await _launch(svc, request)
    return Launched(job_id=record.job_id, state=record.state)


@router.get("/api/golden")
async def list_golden(svc: Svc) -> list[GoldenSummary]:
    """The frozen demo set: what can be shown without a network or a budget."""
    out: list[GoldenSummary] = []
    for bundle in P.list_bundles(svc.settings.golden_root):
        problems = bundle.verify(svc.storage)
        expectation = bundle.expectation
        out.append(
            GoldenSummary(
                slug=bundle.slug,
                title=bundle.title,
                created_at=bundle.created_at,
                summary=bundle.summary(),
                frames=len(bundle.frames),
                clips=len(bundle.clips),
                image_model=bundle.image_model,
                video_model=bundle.video_model,
                original_cost_usd=bundle.provenance.get("cost_usd", 0.0),
                replayable=not problems,
                problems=problems,
                winner_slot=expectation.winner_slot if expectation else None,
            )
        )
    return out


@router.post("/api/jobs/golden/{slug}")
async def replay_golden_job(svc: Svc, slug: str) -> Launched:
    """Replay a frozen job as a real job: no network, no credential, no spend.

    Works whatever mode the app is in, deliberately.  The demo has to be one click
    away without restarting the service into replay mode — and without arming the
    live image provider, which flipping the money switch would also do.
    """
    try:
        providers = P.get_providers(svc.settings, svc.storage, golden_set=slug)
    except (P.ProviderUnavailable, P.GoldenError) as exc:
        raise HTTPException(404, str(exc)) from exc

    bundle = providers.golden.bundle if providers.golden else None
    if bundle is None:  # pragma: no cover - get_providers always sets it for a slug
        raise HTTPException(500, "golden providers resolved without a session")
    problems = bundle.verify(svc.storage)
    if problems:
        raise HTTPException(
            409,
            f"golden bundle {slug!r} cannot be replayed: " + "; ".join(problems),
        )

    request = bundle.materialise_request(uuid.uuid4().hex[:16], svc.storage)
    record = await _launch(svc, request, providers)
    return Launched(
        job_id=record.job_id,
        state=record.state,
        golden_set=slug,
        replaying=bundle.summary(),
    )


@router.post("/api/jobs")
async def create_job(
    svc: Svc,
    human_model_image: UploadFile = File(...),
    product_image: UploadFile = File(...),
    logo_image: UploadFile | None = File(None),
    product_name: str = Form(...),
    caption: str = Form(""),
    cta_text: str = Form(""),
    additional_prompt: str = Form(""),
    negative_constraints: str = Form(""),
    vertical: str = Form("other"),
    platform: str = Form("instagram_reels"),
    mood: str = Form("warm_lifestyle"),
    palette: str = Form("", description="Comma-separated hex colours"),
    background: str = Form("soft_gradient"),
    duration_seconds: float = Form(9.0),
    candidate_count: int = Form(3),
    seed: int = Form(0),
    locked_angle: str | None = Form(None),
    has_model_release: bool = Form(False),
    not_a_public_figure: bool = Form(False),
) -> Launched:
    consent = ConsentAttestation(
        has_model_release=has_model_release, not_a_public_figure=not_a_public_figure
    )
    if not consent.is_valid:
        # Refused here rather than inside the pipeline so the user gets a 400 with
        # an actionable message instead of a failed job to inspect.
        raise HTTPException(
            422,
            "Both rights attestations are required: you must hold a model release "
            "for this person's likeness, and the image must not be of a public figure.",
        )

    job_id = uuid.uuid4().hex[:16]
    try:
        request = AdJobRequest(
            job_id=job_id,
            human_model_image=await _store_upload(svc, job_id, "human", human_model_image),
            product_image=await _store_upload(svc, job_id, "product", product_image),
            logo_image=(
                await _store_upload(svc, job_id, "logo", logo_image) if logo_image else None
            ),
            product_name=product_name,
            caption=caption,
            cta_text=cta_text,
            additional_prompt=additional_prompt,
            negative_constraints=negative_constraints,
            vertical=vertical,
            platform=platform,
            mood=mood,
            theme=ThemeSpec(
                palette=[c.strip() for c in palette.split(",") if c.strip()],
                background=background,
            ),
            duration_seconds=duration_seconds,
            candidate_count=candidate_count,
            seed=seed,
            locked_angle=locked_angle or None,
            consent=consent,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    record = await _launch(svc, request)
    return Launched(job_id=record.job_id, state=record.state)


@router.get("/api/jobs")
async def list_jobs(svc: Svc, limit: int = 25) -> list[JobSummary]:
    records = await svc.store.list_recent(limit)
    return [
        JobSummary(
            job_id=r.job_id,
            state=r.state,
            product_name=r.request.product_name,
            platform=r.request.platform,
            created_at=r.request.created_at.isoformat(),
            cost_usd=r.result.total_cost_usd if r.result else 0.0,
            winner=(
                r.result.winner.video.source_image_index if r.result and r.result.winner else None
            ),
        )
        for r in records
    ]


@router.get("/api/jobs/{job_id}", response_model=JobRecord)
async def get_job(svc: Svc, job_id: str) -> JobRecord:
    record = await svc.store.get(job_id)
    if record is None:
        raise HTTPException(404, f"no job {job_id!r}")
    return record


@router.get("/api/jobs/{job_id}/ledger")
async def get_job_ledger(svc: Svc, job_id: str) -> LedgerResponse:
    """Every charge attributed to one job.

    404 for an unknown id rather than an empty ledger. "No entries" and "no such
    job" are both zero dollars and mean opposite things — the first says the run
    was free, the second says the question was about nothing.
    """
    if await svc.store.get(job_id) is None:
        raise HTTPException(404, f"no job {job_id!r}")
    entries = await svc.ledger.entries(job_id)
    return LedgerResponse(
        job_id=job_id,
        total_usd=round(sum(e.cost_usd for e in entries), 6),
        entries=entries,
    )


@router.get("/api/jobs/{job_id}/delivery")
async def get_delivery(svc: Svc, job_id: str) -> DeliveryResponse:
    """What stage 8 produced, including how lossy each reframe was.

    Separate from the job record so a client can poll the expensive part of the
    result without re-fetching every candidate's brief and score breakdown.
    """
    record = await svc.store.get(job_id)
    if record is None:
        raise HTTPException(404, f"no job {job_id!r}")
    if record.result is None or record.result.delivery is None:
        raise HTTPException(
            409,
            f"job {job_id!r} has not reached delivery (state {record.state.value})",
        )
    report = record.result.delivery
    return DeliveryResponse(
        job_id=job_id,
        summary=report.summary(),
        delivery=report,
        download_url=f"/api/jobs/{job_id}/bundle" if report.bundle else None,
    )


@router.get("/api/jobs/{job_id}/bundle")
async def get_bundle(svc: Svc, job_id: str) -> Response:
    """Download the zip: every render, preview and report card for the winner.

    Served with a filename the user will recognise a week later, rather than
    ``bundle.zip`` — a downloads folder with four of those in it is a downloads
    folder with none.
    """
    record = await svc.store.get(job_id)
    if record is None:
        raise HTTPException(404, f"no job {job_id!r}")
    report = record.result.delivery if record.result else None
    if report is None or report.bundle is None:
        raise HTTPException(409, f"job {job_id!r} has no delivery bundle yet")
    try:
        data = svc.storage.get_bytes(report.bundle.key)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, "the bundle is recorded but missing from storage") from exc

    slug = "".join(
        c if c.isalnum() or c in "-_" else "-" for c in record.request.product_name.lower()
    ).strip("-")
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{slug or "ad"}-{job_id[:8]}.zip"'},
    )


@router.get("/api/jobs/{job_id}/events")
async def stream_events(svc: Svc, job_id: str) -> StreamingResponse:
    """SSE stream: buffered history first, then the live tail."""
    bus = svc.bus

    async def generator() -> AsyncIterator[str]:
        queue = bus.subscribe(job_id)
        try:
            for event in bus.history(job_id):
                yield f"data: {event.model_dump_json()}\n\n"
            if bus.is_finished(job_id):
                yield "event: done\ndata: {}\n\n"
                return
            while True:
                event = await queue.get()
                if event is None:
                    yield "event: done\ndata: {}\n\n"
                    return
                yield f"data: {event.model_dump_json()}\n\n"
        finally:
            bus.unsubscribe(job_id, queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/media/{key:path}")
async def get_media(svc: Svc, key: str) -> Response:
    """Serve generated media in dev. Object storage takes over in deployment."""
    try:
        data = svc.storage.get_bytes(key)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, f"no media {key!r}") from exc
    mime = MEDIA_MIME.get(Path(key).suffix.lower(), "application/octet-stream")
    return Response(content=data, media_type=mime)


@router.get("/", response_class=HTMLResponse)
async def dev_harness() -> str:
    """A single-file inspection page for the pipeline.

    Deliberately not the product UI — that is the Next.js app, built once the
    design work happens. This exists so the pipeline's output can be seen in a
    browser today without a frontend build step.
    """
    page = Path(__file__).parent / "dev.html"
    return page.read_text(encoding="utf-8")


app = create_app()
