"""FastAPI service.

Jobs run as asyncio background tasks in this process, which is right for a
single-developer dev loop and explicitly not right for anything else — the ARQ
worker replaces this when concurrency matters.  What the API commits to now is the
contract the web app will consume: create a job, stream its stages, read the
result, and read the budget.

The budget endpoint is not an afterthought.  With $35 total, "how much is left"
belongs on screen at all times, so it is a first-class route from day one.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

import adproviders as P
from adschema import (
    AdJobRequest,
    AssetRef,
    BudgetStatus,
    ConsentAttestation,
    JobRecord,
    JobState,
    Platform,
    StageEvent,
    ThemeSpec,
)
from adworker import Pipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from .store import EventBus, JobStore

MEDIA_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webp": "image/webp",
}

app = FastAPI(title="Multi-Candidate Ad Generation", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = P.get_settings()
storage = P.get_storage(settings.storage_backend, settings.storage_root)
ledger = P.SqlLedger.from_url(settings.database_url)
store = JobStore.from_url(settings.database_url)
bus = EventBus()

#: Keeps background task references alive so they are not garbage collected
#: mid-run, which asyncio does not otherwise prevent.
_running: dict[str, asyncio.Task] = {}


@app.on_event("startup")
async def _startup() -> None:
    await ledger.create_all()
    await store.create_all()
    print(settings.describe())


def _governor() -> P.CostGovernor:
    return P.CostGovernor(ledger, settings)


async def _store_upload(job_id: str, name: str, upload: UploadFile) -> AssetRef:
    data = await upload.read()
    if not data:
        raise HTTPException(400, f"{name} upload was empty")
    suffix = Path(upload.filename or "").suffix.lower() or ".png"
    if suffix not in MEDIA_MIME:
        raise HTTPException(400, f"{name}: unsupported file type {suffix!r}")
    key = f"uploads/{job_id}/{name}{suffix}"
    return storage.put_bytes(key, data, MEDIA_MIME[suffix])


async def _run(record: JobRecord) -> None:
    """Execute a job, persisting after every stage so a refresh shows progress."""

    async def on_progress(event: StageEvent) -> None:
        await bus.publish(event)
        await store.save(record)

    pipeline = Pipeline(
        storage=storage,
        governor=_governor(),
        image_provider=P.get_image_provider(settings, storage),
        video_provider=P.get_video_provider(settings, storage),
        llm_provider=P.get_llm_provider(settings),
        on_progress=on_progress,
    )
    try:
        completed = await pipeline.run(record.request)
        await store.save(completed)
    finally:
        bus.mark_finished(record.job_id)
        _running.pop(record.job_id, None)


async def _launch(request: AdJobRequest) -> JobRecord:
    record = JobRecord(request=request, state=JobState.QUEUED)
    await store.save(record)
    _running[record.job_id] = asyncio.create_task(_run(record))
    return record


# --- Routes ----------------------------------------------------------------


@app.get("/api/budget", response_model=BudgetStatus)
async def get_budget() -> BudgetStatus:
    return await _governor().status()


@app.get("/api/config")
async def get_config() -> dict:
    """What the UI needs to render honestly: the mode, and what it costs."""
    return {
        "provider_mode": settings.provider_mode.value,
        "is_live": settings.is_live,
        "image_provider": settings.image_provider,
        "video_provider": settings.video_provider,
        "banner": settings.describe(),
        "platforms": {
            p.value: {
                "aspect_ratio": p.aspect_ratio.value,
                "safe_area": {"top": p.safe_area.top, "bottom": p.safe_area.bottom},
            }
            for p in Platform
        },
    }


@app.post("/api/jobs/demo")
async def create_demo_job(
    platform: Platform = Platform.INSTAGRAM_REELS,
    duration_seconds: float = 9.0,
    seed: int = 7,
) -> dict:
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
    import uuid

    from adschema import AspectRatio, Mood, Vertical

    job_id = uuid.uuid4().hex[:16]
    human = P.mock_reference_asset(
        storage, f"uploads/{job_id}/human.png", AspectRatio.PORTRAIT_4_5, seed=11
    )
    product = P.mock_reference_asset(
        storage, f"uploads/{job_id}/product.png", AspectRatio.SQUARE_1_1, seed=22
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
        theme=ThemeSpec(palette=["#2b3a55", "#ce7777", "#f2e7d5"]),
        duration_seconds=duration_seconds,
        candidate_count=3,
        seed=seed,
        consent=ConsentAttestation(has_model_release=True, not_a_public_figure=True),
    )
    record = await _launch(request)
    return {"job_id": record.job_id, "state": record.state.value}


@app.post("/api/jobs")
async def create_job(
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
) -> dict:
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

    import uuid

    job_id = uuid.uuid4().hex[:16]
    try:
        request = AdJobRequest(
            job_id=job_id,
            human_model_image=await _store_upload(job_id, "human", human_model_image),
            product_image=await _store_upload(job_id, "product", product_image),
            logo_image=(await _store_upload(job_id, "logo", logo_image) if logo_image else None),
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

    record = await _launch(request)
    return {"job_id": record.job_id, "state": record.state.value}


@app.get("/api/jobs")
async def list_jobs(limit: int = 25) -> list[dict]:
    records = await store.list_recent(limit)
    return [
        {
            "job_id": r.job_id,
            "state": r.state.value,
            "product_name": r.request.product_name,
            "platform": r.request.platform.value,
            "created_at": r.request.created_at.isoformat(),
            "cost_usd": r.result.total_cost_usd if r.result else 0.0,
            "winner": (
                r.result.winner.video.source_image_index if r.result and r.result.winner else None
            ),
        }
        for r in records
    ]


@app.get("/api/jobs/{job_id}", response_model=JobRecord)
async def get_job(job_id: str) -> JobRecord:
    record = await store.get(job_id)
    if record is None:
        raise HTTPException(404, f"no job {job_id!r}")
    return record


@app.get("/api/jobs/{job_id}/ledger")
async def get_job_ledger(job_id: str) -> dict:
    entries = await ledger.entries(job_id)
    return {
        "job_id": job_id,
        "total_usd": round(sum(e.cost_usd for e in entries), 6),
        "entries": [e.model_dump(mode="json") for e in entries],
    }


@app.get("/api/jobs/{job_id}/events")
async def stream_events(job_id: str) -> StreamingResponse:
    """SSE stream: buffered history first, then the live tail."""

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


@app.get("/media/{key:path}")
async def get_media(key: str) -> Response:
    """Serve generated media in dev. Object storage takes over in deployment."""
    try:
        data = storage.get_bytes(key)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, f"no media {key!r}") from exc
    mime = MEDIA_MIME.get(Path(key).suffix.lower(), "application/octet-stream")
    return Response(content=data, media_type=mime)


@app.get("/", response_class=HTMLResponse)
async def dev_harness() -> str:
    """A single-file inspection page for the pipeline.

    Deliberately not the product UI — that is the Next.js app, built once the
    design work happens. This exists so the pipeline's output can be seen in a
    browser today without a frontend build step.
    """
    page = Path(__file__).parent / "dev.html"
    return page.read_text(encoding="utf-8")


@contextlib.asynccontextmanager
async def lifespan_for_tests():  # pragma: no cover - test helper
    await ledger.create_all()
    await store.create_all()
    yield
