"""The HTTP surface.

Everything the product does, a client does over these routes, and until now none of
them had a test.  The pipeline was covered end to end and the honesty layer in the
web app was covered too — the boundary between them was the gap, which is exactly
where a renamed field or a swallowed refusal survives a green suite.

Three things are being pinned here and they are worth naming, because they are the
reasons this file exists rather than a list of status codes:

**A refusal is a feature.**  The consent gate returns 422 with text a person can
act on, and it does so *before* a job record exists.  A version that created the
job and then failed it would look almost identical in a browser and would leave
unauthorised likeness processing in the store.

**A mock job costs nothing, and the route says so.**  The ledger is asserted at
zero through the API, not just in the governor's own tests, because the number the
user sees comes through this layer.

**Two apps do not share state.**  That is what the factory bought, and it is the
test that stops the module-level singletons coming back.

Every test runs against a temporary storage root and an in-memory ledger, so the
suite cannot touch ``./adgen.db``, ``./fixtures`` or the network.
"""

from __future__ import annotations

import adproviders as P
import httpx
import pytest
from adapi.annotation_store import AnnotationStore
from adapi.main import Services, create_app
from adapi.store import EventBus, JobStore
from adschema import (
    MAX_DURATION_S,
    MIN_DURATION_S,
    AppConfig,
    AspectRatio,
    CorpusStats,
    DeliveryResponse,
    GoldenSummary,
    JobPage,
    JobRecord,
    Launched,
    LedgerResponse,
    Platform,
)


def _services(root) -> Services:
    """An app's worth of dependencies, entirely inside ``root``.

    The ledger is in-memory rather than SQL on purpose: a test asserting that a
    mock job spent nothing should not be able to read a stale row from a previous
    run, and there is nowhere for one to come from here.
    """
    url = f"sqlite+aiosqlite:///{root / 'api.db'}"
    settings = P.Settings(
        provider_mode="mock",
        storage_backend="local",
        storage_root=root,
        database_url=url,
        # Named so an assertion can tell the configured cap from the default.
        budget_total_usd=5.0,
        budget_per_job_usd=1.25,
    )
    return Services(
        settings=settings,
        storage=P.LocalStorage(root),
        ledger=P.InMemoryLedger(),
        store=JobStore.from_url(url),
        annotations=AnnotationStore.from_url(url),
        bus=EventBus(),
    )


@pytest.fixture
def services(tmp_path) -> Services:
    return _services(tmp_path)


@pytest.fixture
async def client(services):
    """A client wired straight to the ASGI app, lifespan and all.

    The lifespan is entered explicitly because ``ASGITransport`` does not run
    startup — and skipping it would leave the tables uncreated *and* leave the
    factory's own startup path untested.
    """
    app = create_app(services)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://api.test") as http:
            yield http


@pytest.fixture
def uploads(services) -> tuple[bytes, bytes]:
    """Real PNG bytes for the multipart path.

    The same synthetic references the demo route uses, read back out of storage:
    the point is to exercise the upload path with something intake can actually
    measure — a face to find and a product on a flat sweep to cut out.
    """
    storage = services.storage
    human = P.mock_portrait_asset(storage, "seed/human.png", AspectRatio.PORTRAIT_4_5, seed=11)
    product = P.mock_product_asset(storage, "seed/product.png", AspectRatio.SQUARE_1_1, seed=22)
    return storage.get_bytes(human.key), storage.get_bytes(product.key)


def _form(**overrides) -> dict[str, str]:
    payload = {
        "product_name": "Aurora Serum",
        "caption": "Glow that lasts",
        "cta_text": "Shop now",
        "vertical": "beauty",
        "platform": "instagram_reels",
        "mood": "calm_premium",
        "duration_seconds": "9.0",
        "candidate_count": "3",
        "seed": "7",
        "has_model_release": "true",
        "not_a_public_figure": "true",
    }
    payload.update({k: str(v) for k, v in overrides.items()})
    return payload


def _files(uploads: tuple[bytes, bytes]) -> dict:
    human, product = uploads
    return {
        "human_model_image": ("human.png", human, "image/png"),
        "product_image": ("product.png", product, "image/png"),
    }


# --- Configuration ---------------------------------------------------------


async def test_the_config_route_states_the_mode_and_derives_the_geometry(client):
    """The mode is the load-bearing field: mock, replay and live look identical."""
    body = (await client.get("/api/config")).json()
    config = AppConfig.model_validate(body)

    assert config.provider_mode.value == "mock"
    assert config.is_live is False
    assert "MOCK MODE" in config.banner

    reels = config.platforms[Platform.INSTAGRAM_REELS]
    assert reels.aspect_ratio is AspectRatio.VERTICAL_9_16
    # The right edge is the one the old dict dropped, and Reels chrome covers 14%
    # of it — a CTA placed there is under the share button.
    edges = (reels.safe_area.top, reels.safe_area.bottom, reels.safe_area.right)
    assert edges == (0.14, 0.20, 0.14)
    assert reels.safe_area.describe() == "14% top, 20% bottom, 14% right"


async def test_the_config_response_has_no_key_the_model_does_not_declare(client):
    """The drift check, at runtime.

    `scripts/export_types.py --check` compares the generated TypeScript with the
    models. This compares the models with what the route actually sends, which is
    the other half: a hand-assembled dict could satisfy the first and still ship a
    key the client never hears about.
    """
    body = (await client.get("/api/config")).json()
    assert set(body) == set(AppConfig.model_fields)


async def test_the_budget_route_reports_the_configured_caps(client):
    body = (await client.get("/api/budget")).json()
    assert body["total_budget_usd"] == 5.0
    assert body["per_job_cap_usd"] == 1.25
    assert body["spent_usd"] == 0.0
    # `remaining_usd` is a property, so it is computed client-side. Its absence
    # here is the contract `docs/ui-protocol.md` §1 documents, not an omission.
    assert "remaining_usd" not in body


# --- The consent gate ------------------------------------------------------


@pytest.mark.parametrize(
    "release,public_figure",
    [("false", "false"), ("true", "false"), ("false", "true")],
    ids=["neither", "release only", "not-a-public-figure only"],
)
async def test_consent_is_refused_before_a_job_exists(client, uploads, release, public_figure):
    """Both attestations, or nothing — and nothing means no record either.

    Refusing inside the pipeline instead would leave a row describing a person
    whose likeness there was no permission to process.
    """
    response = await client.post(
        "/api/jobs",
        data=_form(has_model_release=release, not_a_public_figure=public_figure),
        files=_files(uploads),
    )
    assert response.status_code == 422
    assert "model release" in response.json()["detail"]
    assert (await client.get("/api/jobs")).json()["jobs"] == []


# --- Upload validation -----------------------------------------------------


async def test_an_empty_upload_is_refused_by_name(client, uploads):
    _, product = uploads
    response = await client.post(
        "/api/jobs",
        data=_form(),
        files={
            "human_model_image": ("human.png", b"", "image/png"),
            "product_image": ("product.png", product, "image/png"),
        },
    )
    assert response.status_code == 400
    assert "human upload was empty" in response.json()["detail"]


async def test_an_unsupported_file_type_names_the_extension(client, uploads):
    human, _ = uploads
    response = await client.post(
        "/api/jobs",
        data=_form(),
        files={
            "human_model_image": ("human.png", human, "image/png"),
            "product_image": ("notes.txt", b"not an image", "text/plain"),
        },
    )
    assert response.status_code == 400
    assert "'.txt'" in response.json()["detail"]


async def test_a_duration_outside_the_supported_range_is_refused_with_the_bound(client, uploads):
    """The 8–10s window is a provider constraint, so the API states it."""
    response = await client.post(
        "/api/jobs",
        data=_form(duration_seconds=4.0),
        files=_files(uploads),
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "duration_seconds" in detail
    # Pydantic's own wording, which names the bound. Passed through verbatim rather
    # than replaced with a friendlier sentence: the number is the useful part, and a
    # rewritten message is one more thing that can disagree with the schema.
    assert f"greater than or equal to {MIN_DURATION_S:g}" in detail
    assert MIN_DURATION_S < MAX_DURATION_S  # the window this route is defending


# --- The whole thing, once -------------------------------------------------


async def test_a_posted_upload_runs_to_delivery_and_every_route_agrees(client, services, uploads):
    """One job, checked through every route a client uses.

    Deliberately one test rather than six: the pipeline run is the slow part, and
    what is being asserted is that these routes tell a *consistent* story about the
    same job — which is a single claim, not six.
    """
    launched = Launched.model_validate(
        (await client.post("/api/jobs", data=_form(), files=_files(uploads)))
        .raise_for_status()
        .json()
    )
    assert launched.state.value == "queued"
    assert launched.golden_set is None

    await services.wait_for_jobs()
    job_id = launched.job_id

    record = JobRecord.model_validate((await client.get(f"/api/jobs/{job_id}")).json())
    assert record.state.value == "completed", record.error
    assert len(record.result.images) == 3
    assert len(record.result.videos) == 3
    assert len(record.result.ranking) == 3
    assert record.result.intake is not None

    # The board row: a summary, not the record.
    page = JobPage.model_validate((await client.get("/api/jobs")).json())
    rows = page.jobs
    assert (page.total, page.offset, page.limit) == (1, 0, 25)
    assert [r.job_id for r in rows] == [job_id]
    assert rows[0].product_name == "Aurora Serum"
    assert rows[0].winner in {0, 1, 2}
    assert rows[0].cost_usd == 0.0

    # A mock job is free, and the route the UI reads says so.
    #
    # No entries at all, not entries that sum to zero: `CostGovernor.charge` returns
    # before it reserves anything in any non-live mode, so a free run cannot leave a
    # row behind. That makes an empty ledger a structural fact rather than an
    # arithmetic coincidence — a row here would mean a live provider had leaked in.
    ledger = LedgerResponse.model_validate((await client.get(f"/api/jobs/{job_id}/ledger")).json())
    assert ledger.total_usd == 0.0
    assert ledger.entries == []

    # Delivery, with its own envelope and a downloadable bundle.
    delivery = DeliveryResponse.model_validate(
        (await client.get(f"/api/jobs/{job_id}/delivery")).json()
    )
    assert delivery.delivery.renders, "stage 8 produced no renders"
    assert delivery.download_url == f"/api/jobs/{job_id}/bundle"

    bundle = await client.get(delivery.download_url)
    assert bundle.status_code == 200
    assert bundle.headers["content-type"] == "application/zip"
    # Named after the product, because four files called bundle.zip are four files
    # called nothing.
    assert "aurora-serum" in bundle.headers["content-disposition"]
    assert bundle.content.startswith(b"PK")

    # The stream replays the whole history and then closes, so a client that
    # connects after the job finished still gets the progress log.
    stream = await client.get(f"/api/jobs/{job_id}/events")
    assert stream.headers["content-type"].startswith("text/event-stream")
    # Counted on the payload, not on "data: ": the terminator is a `data: {}` frame
    # too, and counting both would have made an off-by-one look like a lost event.
    assert stream.text.count('data: {"job_id"') == len(services.bus.history(job_id))
    assert stream.text.rstrip().endswith("event: done\ndata: {}")


async def test_the_demo_route_needs_no_uploads(client, services):
    """The path used all day: exercise the pipeline without hunting for a photo."""
    launched = Launched.model_validate(
        (await client.post("/api/jobs/demo?seed=3&platform=tiktok")).json()
    )
    await services.wait_for_jobs()

    record = JobRecord.model_validate((await client.get(f"/api/jobs/{launched.job_id}")).json())
    assert record.state.value == "completed", record.error
    assert record.request.platform is Platform.TIKTOK
    assert record.request.aspect_ratio is AspectRatio.VERTICAL_9_16
    assert (await client.get("/api/budget")).json()["spent_usd"] == 0.0


# --- Absences --------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/api/jobs/nope", "/api/jobs/nope/ledger", "/api/jobs/nope/delivery", "/api/jobs/nope/bundle"],
)
async def test_an_unknown_job_is_a_404_on_every_sub_route(client, path):
    """Including the ledger.

    An empty ledger and a missing job are both zero dollars and mean opposite
    things: the first says the run was free, the second says the question was about
    nothing at all.
    """
    response = await client.get(path)
    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


async def test_a_job_before_delivery_is_a_conflict_not_a_404(client, services, make_request):
    """409 with the state, because the job exists and the answer is "not yet".

    A 404 here would send a client looking for a wrong id when the truth is that
    stage 8 has not run.
    """
    record = JobRecord(request=make_request(job_id="pending000000001"))
    await services.store.save(record)

    response = await client.get("/api/jobs/pending000000001/delivery")
    assert response.status_code == 409
    assert "has not reached delivery" in response.json()["detail"]
    assert "queued" in response.json()["detail"]


# --- Media -----------------------------------------------------------------


async def test_media_is_served_with_the_right_mime_type(client, services):
    services.storage.put_bytes("generations/j/img_0.png", b"\x89PNG\r\n\x1a\nnot-really")
    response = await client.get("/media/generations/j/img_0.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


async def test_a_key_that_escapes_the_storage_root_is_refused(client, services, tmp_path):
    """The storage boundary is real: keys can come from request data.

    Asserted at both levels — the store refuses the key, and the route turns that
    refusal into a 404 rather than a 500 traceback.
    """
    (tmp_path.parent / "outside.txt").write_bytes(b"secret")

    with pytest.raises(ValueError, match="escapes the storage root"):
        services.storage.get_bytes("../outside.txt")

    response = await client.get("/media/..%2Foutside.txt")
    assert response.status_code == 404


# --- The golden set --------------------------------------------------------


async def test_the_golden_route_is_empty_rather_than_absent_without_a_frozen_set(client):
    body = (await client.get("/api/golden")).json()
    assert body == []


async def test_replaying_a_bundle_that_is_not_there_is_a_404(client):
    response = await client.post("/api/jobs/golden/no-such-bundle")
    assert response.status_code == 404
    assert "no-such-bundle" in response.json()["detail"]


async def test_a_frozen_bundle_is_listed_with_what_is_wrong_with_it(client, services, make_request):
    """A bundle whose media is missing is listed as unplayable, not hidden.

    A demo set that quietly shrinks is worse than one that says what broke: the
    first is discovered in front of an examiner.
    """
    # Built through the real dataclasses rather than as hand-written JSON: a
    # manifest typed out here would pass for as long as the format stayed still and
    # then fail as a test bug rather than as the drift it is meant to catch.
    bundle = P.GoldenBundle(
        slug="broken-demo",
        request=make_request().model_dump(mode="json"),
        title="Broken demo",
        created_at="2026-08-01T00:00:00+00:00",
        frames=[
            P.FrozenFrame(
                slot=0,
                attempt=1,
                asset=P.FrozenAsset(
                    key="golden/broken-demo/assets/frame_0.png",
                    sha256="0" * 64,
                    size_bytes=1024,
                    mime_type="image/png",
                ),
                model="seedream-v4",
                tier="premium",
                seed=7,
            )
        ],
        provenance={"cost_usd": 2.22, "image_model": "seedream-v4", "video_model": "kling-v2.1"},
    )
    bundle.save(services.settings.golden_root)

    rows = [GoldenSummary.model_validate(r) for r in (await client.get("/api/golden")).json()]
    assert [r.slug for r in rows] == ["broken-demo"]
    assert rows[0].replayable is False
    assert rows[0].problems, "an unplayable bundle must say why"

    response = await client.post("/api/jobs/golden/broken-demo")
    assert response.status_code == 409
    assert "cannot be replayed" in response.json()["detail"]


# --- Paging ------------------------------------------------------------------


async def test_the_board_pages_and_says_what_it_left_out(client, services, make_request):
    """A list that silently stops is worse than a short one.

    The board returned a bare array capped at 25, so with more jobs than that the
    oldest were simply gone with nothing on screen saying so. `total` is the whole
    point of the envelope — the rows are the same rows.
    """
    for n in range(7):
        await services.store.save(JobRecord(request=make_request(job_id=f"job{n:012d}")))

    first = JobPage.model_validate((await client.get("/api/jobs?limit=3")).json())
    assert len(first.jobs) == 3
    assert first.total == 7
    assert first.has_more

    last = JobPage.model_validate((await client.get("/api/jobs?limit=3&offset=6")).json())
    assert len(last.jobs) == 1
    assert last.total == 7
    assert not last.has_more

    # No row appears twice and none is skipped, which is what the `job_id` tiebreak
    # in `JobStore.page` buys: these fixtures share a `created_at` to the microsecond.
    seen = []
    for offset in (0, 3, 6):
        got = await client.get(f"/api/jobs?limit=3&offset={offset}")
        seen.extend(row.job_id for row in JobPage.model_validate(got.json()).jobs)
    assert sorted(seen) == [f"job{n:012d}" for n in range(7)]


async def test_has_more_is_derived_and_not_on_the_wire(client):
    """A stored copy is a copy that can disagree with the three fields it sums up."""
    body = (await client.get("/api/jobs")).json()
    assert set(body) == set(JobPage.model_fields)
    assert "has_more" not in body


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1"])
async def test_the_page_size_is_capped(client, query):
    """Every row carries a rendered summary, so an uncapped limit is the whole store."""
    assert (await client.get(f"/api/jobs?{query}")).status_code == 422


# --- Concurrency -------------------------------------------------------------


async def test_sqlite_is_opened_in_wal_mode_with_a_busy_timeout(services):
    """Sixteen jobs at once, and ten of them died on `database is locked`.

    Not a pipeline failure — a persistence one. The API writes the whole job
    document after every progress event, so a few concurrent jobs are a stream of
    short write transactions, and SQLite's defaults answer that with a
    whole-file lock and a `busy_timeout` of *zero*: a contended write raises
    without waiting even a millisecond. On screen it looked like a job failing at
    intake.

    Asserted on the connection rather than on a stress test, which would be slow
    and would still pass on a fast enough machine.
    """
    async with services.store.engine.connect() as conn:
        journal = (await conn.exec_driver_sql("PRAGMA journal_mode")).scalar_one()
        timeout = (await conn.exec_driver_sql("PRAGMA busy_timeout")).scalar_one()
    assert str(journal).lower() == "wal"
    assert int(timeout) == P.BUSY_TIMEOUT_MS


async def test_concurrent_jobs_all_reach_a_terminal_state(client, services):
    """The regression itself: several jobs at once, none lost to a locked file.

    Eight rather than sixteen because the point is contention, not throughput, and
    every one of these runs the real pipeline.
    """
    for seed in range(8):
        assert (await client.post(f"/api/jobs/demo?seed={seed}")).status_code == 200
    await services.wait_for_jobs()

    page = JobPage.model_validate((await client.get("/api/jobs?limit=100")).json())
    assert page.total == 8
    assert [row.state.value for row in page.jobs] == ["completed"] * 8

    # And the reason a lock failure would have been so confusing: it surfaced as the
    # job's own `error`, written by the thing that was trying to write the job.
    for row in page.jobs:
        record = await services.store.get(row.job_id)
        assert record is not None and record.error is None


# --- Wiring ----------------------------------------------------------------


async def test_the_annotation_router_is_attached_by_the_factory(client):
    """The annotation store comes off `app.state`, so the factory has to attach it."""
    body = (await client.get("/api/annotate/progress")).json()
    assert set(body) == set(CorpusStats.model_fields)
    assert body["n_judgements"] == 0


async def test_each_app_owns_its_own_annotation_state(tmp_path):
    """The annotation store used to be a module global the last app overwrote.

    Two apps in one process shared an annotation database whatever their own
    settings said, and seeding the serving RNG in one test changed the trial order
    for every test after it. Asserting on identity rather than on behaviour because
    the failure mode is aliasing, and two empty databases look identical.
    """
    left, right = _services(tmp_path / "left"), _services(tmp_path / "right")
    app_left, app_right = create_app(left), create_app(right)

    assert app_left.state.annotating is not app_right.state.annotating
    assert app_left.state.annotating.store is left.annotations
    # Creating the second app must not have reached back into the first.
    assert app_left.state.annotating.store is not right.annotations
    assert app_left.state.annotating.rng is not app_right.state.annotating.rng


async def test_two_apps_do_not_share_state(tmp_path, make_request):
    """What the factory bought.

    With the dependencies at module scope this was impossible to write, which is
    why the API had no HTTP tests at all. If someone moves them back, this fails.
    """
    left, right = _services(tmp_path / "left"), _services(tmp_path / "right")
    (tmp_path / "left").mkdir(exist_ok=True)
    (tmp_path / "right").mkdir(exist_ok=True)

    app_left, app_right = create_app(left), create_app(right)
    async with app_left.router.lifespan_context(app_left):
        async with app_right.router.lifespan_context(app_right):
            await left.store.save(JobRecord(request=make_request(job_id="onlyinleft00001")))

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app_left), base_url="http://left"
            ) as client_left:
                rows = (await client_left.get("/api/jobs")).json()["jobs"]
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app_right), base_url="http://right"
            ) as client_right:
                empty = (await client_right.get("/api/jobs")).json()["jobs"]

    assert [r["job_id"] for r in rows] == ["onlyinleft00001"]
    assert empty == []
    assert left.storage is not right.storage


def test_the_default_services_read_the_environment_rather_than_a_hardcoded_path():
    """`Services.from_settings` is what uvicorn gets, so it is worth one assertion."""
    settings = P.Settings(provider_mode="mock", storage_backend="local", storage_root="./fixtures")
    services = Services.from_settings(settings)
    assert services.settings is settings
    assert services.governor().settings is settings
    assert services.running == {}
