"""The golden demo set: frozen jobs that replay offline.

Three properties carry the weight here, and each one was a bug before it was a
test.

**A replay has to actually read the bundle.**  The governor's generation cache is
consulted before the provider, and the frozen uploads hash to what they hashed to
at freeze time — so every fingerprint hits and a replay can complete, produce the
right ranking, and never open a frozen file.  Both halves of that are tested: the
cache hiding the bundle, and the bundle being read when the cache is off.

**A freeze has to capture cached generations too**, for the same reason from the
other side.  The first freeze written here produced a bundle with three ranked
candidates and zero frames.

**The frozen request must be the submitted one.**  Stage 1 writes the extracted
palette back into the request, so freezing the returned object would hand a replay
a palette it never had to derive — skipping the intake path the bundle exists to
exercise, and skipping it silently.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
import zipfile

import adproviders as P
import pytest
from adproviders.golden import MANIFEST_NAME
from adschema import (
    AdJobRequest,
    AspectRatio,
    ConsentAttestation,
    JobState,
    Mood,
    Platform,
    ProviderMode,
    ThemeSpec,
    Tier,
    Vertical,
)
from adworker import Pipeline

SLUG = "t-golden"


class _SceneLLM(P.LLMProvider):
    """An LLM that returns scene text, so the enrichment path is exercised.

    ``model`` is ``mock`` because it has to be a key in the price table — the
    pre-flight estimate looks all three models up before stage 1 runs.
    """

    name = "stub"
    model = "mock"

    def __init__(self, scenes: list[dict]):
        self.scenes = scenes

    async def complete_json(self, system: str, user: str, schema_hint: str) -> dict:
        return {"scenes": self.scenes}


def _request(storage, **overrides) -> AdJobRequest:
    """A job with no supplied palette, so intake has to extract one.

    That is the interesting path to freeze: it is where the difference between the
    submitted request and the returned one shows up.
    """
    human = P.mock_portrait_asset(storage, "uploads/g/human.png", AspectRatio.PORTRAIT_4_5, seed=11)
    product = P.mock_product_asset(
        storage, "uploads/g/product.png", AspectRatio.SQUARE_1_1, seed=22
    )
    payload = {
        "human_model_image": human,
        "product_image": product,
        "product_name": "Aurora Serum",
        "caption": "Glow that lasts",
        "vertical": Vertical.BEAUTY,
        "platform": Platform.INSTAGRAM_REELS,
        "mood": Mood.CALM_PREMIUM,
        "theme": ThemeSpec(palette=[]),
        "duration_seconds": 9.0,
        "candidate_count": 3,
        "seed": 7,
        "consent": ConsentAttestation(has_model_release=True, not_a_public_figure=True),
    }
    payload.update(overrides)
    return AdJobRequest(**payload)


async def _freeze(storage, settings, ledger, *, slug: str = SLUG, llm=None):
    """Run a job through recording providers and write the bundle."""
    recorder = P.GoldenRecorder(slug, storage)
    request = _request(storage)
    submitted = request.model_copy(deep=True)
    images = P.MockImageProvider(storage)
    videos = P.MockVideoProvider(storage)
    llm = llm or P.MockLLMProvider()

    pipeline = Pipeline(
        storage=storage,
        governor=P.CostGovernor(ledger, settings),
        image_provider=P.RecordingImageProvider(images, recorder),
        video_provider=P.RecordingVideoProvider(videos, recorder),
        llm_provider=P.RecordingLLMProvider(llm, recorder),
    )
    record = await pipeline.run(request)
    assert record.state is JobState.COMPLETED, record.error

    bundle = P.build_golden_bundle(
        slug,
        request=submitted,
        record=record,
        recorder=recorder,
        provider_mode="mock",
        image_model=images.model,
        video_model=videos.model,
        llm_model=llm.model,
    )
    bundle.save(storage.root)
    return bundle, record


async def _replay(bundle, storage, settings, ledger, *, use_cache: bool = False):
    session = P.GoldenSession(bundle, storage)
    request = bundle.materialise_request(uuid.uuid4().hex[:16], storage)
    before = await ledger.total_spent()
    pipeline = Pipeline(
        storage=storage,
        governor=P.CostGovernor(ledger, settings, use_cache=use_cache),
        image_provider=session.image_provider(),
        video_provider=session.video_provider(),
        llm_provider=session.llm_provider(),
    )
    record = await pipeline.run(request)
    spent = await ledger.total_spent() - before
    drift = P.compare_replay(bundle, record, spent_usd=spent, notes=session.notes + session.audit())
    return drift, record, session


# --- Fixtures ----------------------------------------------------------------
#
# The freeze runs the whole eight-stage pipeline, so it happens once per module and
# the tests work from copies. Synchronous on purpose: a module-scoped async fixture
# would need an event loop of matching scope, which is more machinery than calling
# asyncio.run once.


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("golden")
    storage = P.LocalStorage(root)
    settings = P.Settings(provider_mode="mock", storage_backend="local", storage_root=root)
    bundle, record = asyncio.run(_freeze(storage, settings, P.InMemoryLedger()))
    return root, bundle, record


@pytest.fixture(scope="module")
def frozen_with_llm(tmp_path_factory):
    root = tmp_path_factory.mktemp("golden-llm")
    storage = P.LocalStorage(root)
    settings = P.Settings(provider_mode="mock", storage_backend="local", storage_root=root)
    llm = _SceneLLM(
        [
            {"index": 0, "scene": "A sunlit marble bathroom.", "concept": "Morning ritual"},
            {"index": 1, "scene": "A linen-draped vanity.", "concept": "Quiet luxury"},
            {"index": 2, "scene": "A misted glass shelf.", "concept": "Clean slate"},
        ]
    )
    bundle, record = asyncio.run(_freeze(storage, settings, P.InMemoryLedger(), llm=llm))
    return root, bundle, record


@pytest.fixture
def replayable(frozen, tmp_path):
    """An isolated copy of the frozen bundle, safe to tamper with."""
    root, _, record = frozen
    shutil.copytree(root, tmp_path / "root")
    storage = P.LocalStorage(tmp_path / "root")
    settings = P.Settings(
        provider_mode="mock", storage_backend="local", storage_root=tmp_path / "root"
    )
    bundle = P.GoldenBundle.load(P.golden_bundle_path(settings, SLUG))
    return bundle, storage, settings, P.InMemoryLedger(), record


# --- The bundle format -------------------------------------------------------


def test_a_bundle_round_trips_through_its_manifest(frozen):
    root, bundle, _ = frozen
    reloaded = P.GoldenBundle.load(root / "golden" / SLUG)

    assert reloaded.to_dict() == bundle.to_dict()
    assert reloaded.slug == SLUG
    assert len(reloaded.frames) == 3
    assert len(reloaded.clips) == 3
    assert set(reloaded.uploads) == {"human", "product"}


def test_every_frozen_asset_is_present_and_hashes_to_the_manifest(frozen):
    root, bundle, _ = frozen
    storage = P.LocalStorage(root)

    assert bundle.verify(storage) == []
    # Two uploads, three frames, three clips — the whole of what cannot be recomputed.
    assert len(bundle.assets()) == 8


def test_the_frozen_request_is_the_one_that_was_submitted(frozen):
    """Not the one the pipeline returned, which intake has already written to."""
    _, bundle, record = frozen

    assert bundle.request["theme"]["palette"] == []
    assert bundle.request["theme"]["palette_auto_extracted"] is False
    # The run really did fill it in, so the assertion above is about the snapshot
    # rather than about a job where nothing happened.
    assert record.request.theme.palette
    assert record.request.theme.palette_auto_extracted is True


def test_an_incompatible_version_is_refused_rather_than_parsed(frozen, tmp_path):
    root, _, _ = frozen
    shutil.copytree(root / "golden" / SLUG, tmp_path / SLUG)
    path = tmp_path / SLUG / MANIFEST_NAME
    raw = json.loads(path.read_text())
    raw["version"] = P.GOLDEN_VERSION + 99
    path.write_text(json.dumps(raw))

    with pytest.raises(P.GoldenError, match="version"):
        P.GoldenBundle.load(path)


def test_a_bundle_with_nothing_frozen_is_reported_as_unreplayable(frozen):
    root, bundle, _ = frozen
    storage = P.LocalStorage(root)
    empty = P.GoldenBundle(slug="empty", request=bundle.request)

    assert any("nothing to replay" in problem for problem in empty.verify(storage))


def test_packing_carries_the_manifest_and_every_asset(frozen, tmp_path):
    root, bundle, _ = frozen
    dest = bundle.pack(tmp_path / "b.zip", P.LocalStorage(root))

    with zipfile.ZipFile(dest) as zf:
        names = set(zf.namelist())
        assert f"{SLUG}/{MANIFEST_NAME}" in names
        for asset in bundle.assets():
            assert asset.key in names
            assert len(zf.read(asset.key)) == asset.size_bytes


# --- Replay ------------------------------------------------------------------


async def test_a_replay_reproduces_the_frozen_ranking_exactly(replayable):
    bundle, storage, settings, ledger, _ = replayable
    drift, record, _ = await _replay(bundle, storage, settings, ledger)

    assert record.state is JobState.COMPLETED, record.error
    assert drift.ok, drift.lines
    assert list(record.result.video_stage_order) == bundle.expectation.video_order


async def test_a_replay_serves_the_frozen_bytes(replayable):
    bundle, storage, settings, ledger, _ = replayable
    _, _, session = await _replay(bundle, storage, settings, ledger)

    assert session.served_frames == len(bundle.frames)
    assert session.served_clips == len(bundle.clips)
    assert session.audit() == []


async def test_the_generation_cache_would_otherwise_hide_the_bundle(replayable):
    """The bug this module's cache-off default exists for.

    With the cache on, the pipeline is satisfied before the provider is reached:
    the fingerprints match the entries the freeze left behind, because the frozen
    uploads hash to the same digests.  The replay then passes while having read
    nothing — which is why ``GoldenSession.audit`` exists rather than trusting a
    clean result to mean the bundle was exercised.
    """
    bundle, storage, settings, _, _ = replayable
    ledger = P.InMemoryLedger()

    # Populate the cache the way a freeze in the same process would have.
    await _replay(bundle, storage, settings, ledger, use_cache=True)
    drift, record, session = await _replay(bundle, storage, settings, ledger, use_cache=True)

    assert record.state is JobState.COMPLETED
    assert session.served_frames == 0
    assert session.served_clips == 0
    assert any("did not exercise the bundle" in note for note in session.audit())
    assert not drift.ok


async def test_a_replay_spends_nothing(replayable):
    bundle, storage, settings, ledger, _ = replayable
    session = P.GoldenSession(bundle, storage)

    # Zero at the estimate, which is what the governor asserts on in a free run —
    # not merely zero after the fact.
    assert session.image_provider().estimate_cost(3) == 0.0
    assert session.video_provider().estimate_cost(9.0, 3) == 0.0

    drift, record, _ = await _replay(bundle, storage, settings, ledger)
    assert await ledger.total_spent() == 0.0
    assert record.result.total_cost_usd == 0.0
    assert drift.spend == []


async def test_a_replay_takes_a_fresh_job_id_and_its_own_storage_keys(replayable):
    bundle, storage, _, _, _ = replayable

    first = bundle.materialise_request("aaaa1111", storage)
    second = bundle.materialise_request("bbbb2222", storage)

    assert first.job_id != second.job_id
    assert first.human_model_image.key == "uploads/aaaa1111/human.png"
    assert second.human_model_image.key == "uploads/bbbb2222/human.png"
    # Same bytes on both, or the two replays are not replays of one job.
    assert first.human_model_image.sha256 == second.human_model_image.sha256


async def test_the_frozen_tier_survives_a_replay(replayable):
    """A premium clip replays as premium, not as whatever is serving it.

    The tier is per entry rather than per provider for the same reason the research
    tier records it that way: the bundle is evidence about a paid generation, and a
    report that labelled it mock would be describing the replay instead of the
    artefact.
    """
    bundle, storage, settings, ledger, _ = replayable
    bundle.frames = [
        P.FrozenFrame(**{**f.__dict__, "tier": Tier.PREMIUM.value}) for f in bundle.frames
    ]
    bundle.clips = [
        P.FrozenClip(**{**c.__dict__, "tier": Tier.PREMIUM.value}) for c in bundle.clips
    ]

    _, record, _ = await _replay(bundle, storage, settings, ledger)

    assert {c.tier for c in record.result.images} == {Tier.PREMIUM}
    assert {v.tier for v in record.result.videos} == {Tier.PREMIUM}


# --- Refusals ----------------------------------------------------------------


def test_altered_media_is_reported_by_name(replayable):
    bundle, storage, _, _, _ = replayable
    victim = bundle.frames[0].asset
    storage.put_bytes(victim.key, b"not a png at all", victim.mime_type)

    problems = bundle.verify(storage)
    assert len(problems) == 1
    assert victim.key in problems[0]


def test_missing_media_is_reported_by_name(replayable):
    bundle, storage, _, _, _ = replayable
    (storage.root / bundle.clips[0].asset.key).unlink()

    assert [f"missing: {bundle.clips[0].asset.key}"] == bundle.verify(storage)


async def test_altered_media_is_refused_at_serve_time_too(replayable):
    """Verification is a separate step, so serving cannot rely on it having run."""
    bundle, storage, settings, ledger, _ = replayable
    victim = bundle.frames[0].asset
    # Same length, different content: caught by the hash and by nothing else.
    storage.put_bytes(victim.key, b"\x00" * victim.size_bytes, victim.mime_type)

    _, record, _ = await _replay(bundle, storage, settings, ledger)
    assert record.state is JobState.FAILED
    assert "does not match the hash" in (record.error or "")


async def test_a_missing_slot_refuses_rather_than_substituting_a_generation(replayable):
    bundle, storage, settings, ledger, _ = replayable
    bundle.clips = [c for c in bundle.clips if c.slot != bundle.clips[0].slot]

    _, record, _ = await _replay(bundle, storage, settings, ledger)

    assert record.state is JobState.FAILED
    assert "Refusing to substitute a mock clip" in (record.error or "")


# --- Drift -------------------------------------------------------------------


async def test_a_changed_design_point_is_reported_but_still_served(replayable):
    """The demo keeps working; the mismatch is not silent."""
    bundle, storage, settings, ledger, _ = replayable
    stale = dict(bundle.frames[0].design_point)
    stale["lighting"] = "golden_hour" if stale.get("lighting") != "golden_hour" else "high_key"
    bundle.frames = [
        P.FrozenFrame(**{**bundle.frames[0].__dict__, "design_point": stale}),
        *bundle.frames[1:],
    ]

    drift, record, session = await _replay(bundle, storage, settings, ledger)

    assert record.state is JobState.COMPLETED
    assert session.served_frames == 3
    assert any("different design point" in note for note in session.notes)
    assert not drift.ok


async def test_a_changed_prompt_is_reported(replayable):
    bundle, storage, settings, ledger, _ = replayable
    bundle.frames = [
        P.FrozenFrame(**{**bundle.frames[0].__dict__, "prompt_sha256": "0" * 16}),
        *bundle.frames[1:],
    ]

    drift, record, session = await _replay(bundle, storage, settings, ledger)

    assert record.state is JobState.COMPLETED
    assert any("image prompt has changed" in note for note in session.notes)
    assert drift.serving


async def test_a_changed_ranking_is_reported_as_a_result_change(replayable):
    """Ordering drift is kept apart from score drift: they mean different things."""
    bundle, storage, settings, ledger, _ = replayable
    expectation = bundle.expectation
    bundle.expectation = P.GoldenExpectation(
        **{**expectation.__dict__, "video_order": list(reversed(expectation.video_order))}
    )

    drift, _, _ = await _replay(bundle, storage, settings, ledger)

    assert drift.ordering
    assert any("video-stage order" in line for line in drift.ordering)
    assert not drift.ok


async def test_a_changed_score_is_reported_without_claiming_the_result_moved(replayable):
    bundle, storage, settings, ledger, _ = replayable
    expectation = bundle.expectation
    nudged = {slot: value - 0.01 for slot, value in expectation.image_scores.items()}
    bundle.expectation = P.GoldenExpectation(**{**expectation.__dict__, "image_scores": nudged})

    drift, _, _ = await _replay(bundle, storage, settings, ledger)

    assert drift.numeric
    assert not drift.ordering
    assert not drift.ok


def test_a_replay_that_spent_money_is_drift_on_its_own(frozen):
    _, bundle, record = frozen
    drift = P.compare_replay(bundle, record, spent_usd=0.07)

    assert drift.spend
    assert "must cost exactly nothing" in drift.spend[0]


# --- Freezing ----------------------------------------------------------------


async def test_a_freeze_captures_generations_the_cache_served(tmp_path):
    """The bug that produced a bundle with three candidates and zero frames.

    The governor's cache is upstream of the provider, so wrapping the provider is
    not enough — a job whose generations are already cached calls nothing at all.
    """
    storage = P.LocalStorage(tmp_path)
    settings = P.Settings(provider_mode="mock", storage_backend="local", storage_root=tmp_path)
    ledger = P.InMemoryLedger()

    first, _ = await _freeze(storage, settings, ledger, slug="first")
    assert not first.provenance["backfilled"]

    # The same job again, against a ledger that now holds every fingerprint.
    second, _ = await _freeze(storage, settings, ledger, slug="second")

    assert len(second.frames) == 3
    assert len(second.clips) == 3
    assert len(second.provenance["backfilled"]) == 6
    assert second.verify(storage) == []


async def test_the_recording_wrapper_does_not_alter_what_the_provider_returned(
    storage, settings, ledger
):
    recorder = P.GoldenRecorder("wrap", storage)
    inner = P.MockImageProvider(storage)
    wrapped = P.RecordingImageProvider(inner, recorder)

    assert wrapped.model == inner.model
    assert wrapped.tier is inner.tier
    assert wrapped.estimate_cost(3) == inner.estimate_cost(3)


# --- LLM replay --------------------------------------------------------------


async def test_the_frozen_scene_text_is_replayed_into_the_prompt(frozen_with_llm, tmp_path):
    root, bundle, _ = frozen_with_llm
    shutil.copytree(root, tmp_path / "root")
    storage = P.LocalStorage(tmp_path / "root")
    settings = P.Settings(
        provider_mode="mock", storage_backend="local", storage_root=tmp_path / "root"
    )
    reloaded = P.GoldenBundle.load(P.golden_bundle_path(settings, SLUG))
    assert reloaded.completions, "the freeze should have recorded the enrichment"

    drift, record, session = await _replay(reloaded, storage, settings, P.InMemoryLedger())

    prompts = "\n".join(c.brief.image_prompt for c in record.result.images)
    assert "A sunlit marble bathroom." in prompts
    assert session.notes == []
    assert drift.ok, drift.lines


async def test_a_completion_miss_falls_back_to_the_template_and_says_so(frozen_with_llm, tmp_path):
    """A miss is survivable — the compiler has a template path — but not silent."""
    root, _, _ = frozen_with_llm
    shutil.copytree(root, tmp_path / "root")
    storage = P.LocalStorage(tmp_path / "root")
    settings = P.Settings(
        provider_mode="mock", storage_backend="local", storage_root=tmp_path / "root"
    )
    bundle = P.GoldenBundle.load(P.golden_bundle_path(settings, SLUG))
    bundle.completions = {"a-key-that-will-never-be-asked-for": {"scenes": []}}

    _, record, session = await _replay(bundle, storage, settings, P.InMemoryLedger())

    assert record.state is JobState.COMPLETED
    prompts = "\n".join(c.brief.image_prompt for c in record.result.images)
    assert "A sunlit marble bathroom." not in prompts
    assert any("different prompt than it did at freeze time" in n for n in session.notes)


def test_a_mock_freeze_records_no_completion_and_reports_no_miss(frozen):
    """The mock LLM returns nothing, so the template path *is* the frozen path.

    Recording ``{}`` would make a mock bundle look as though it had a frozen
    completion that happened to be empty, and every replay would then report a
    miss that is not one.
    """
    _, bundle, _ = frozen
    assert bundle.completions == {}


# --- Wiring ------------------------------------------------------------------


def test_replay_mode_refuses_to_resolve_without_a_named_bundle():
    settings = P.Settings(provider_mode="replay", storage_backend="local")

    assert settings.is_replay
    assert settings.is_live is False
    with pytest.raises(P.ProviderUnavailable, match="AD_GOLDEN_SET"):
        P.get_providers(settings)


def test_the_single_provider_getters_refuse_replay_mode():
    """Because the three golden providers have to share one session."""
    settings = P.Settings(provider_mode="replay", storage_backend="local", golden_set=SLUG)

    for getter in (P.get_image_provider, P.get_video_provider):
        with pytest.raises(P.ProviderUnavailable, match="get_providers"):
            getter(settings)
    with pytest.raises(P.ProviderUnavailable, match="get_providers"):
        P.get_llm_provider(settings)


def test_get_providers_hands_all_three_the_same_session(frozen):
    root, _, _ = frozen
    settings = P.Settings(
        provider_mode="replay", storage_backend="local", storage_root=root, golden_set=SLUG
    )
    providers = P.get_providers(settings)

    assert providers.golden is not None
    assert providers.images.session is providers.golden
    assert providers.videos.session is providers.golden
    assert providers.llm.session is providers.golden
    assert "replaying golden set" in providers.mode_note


def test_a_named_bundle_overrides_the_configured_mode(frozen):
    """The demo button must work while the app sits in mock mode.

    Requiring a restart into replay mode would mean the only way to show a frozen
    job is to reconfigure the service — and flipping to live mode to get there
    would arm the paid image provider at the same time.
    """
    root, _, _ = frozen
    settings = P.Settings(provider_mode="mock", storage_backend="local", storage_root=root)

    providers = P.get_providers(settings, golden_set=SLUG)

    assert providers.golden is not None
    assert P.get_providers(settings).golden is None


def test_replay_mode_is_free_by_construction():
    """``is_live`` is the only thing any budget check reads, and replay is not it."""
    replay = P.Settings(provider_mode="replay", storage_backend="local")

    assert replay.provider_mode is ProviderMode.REPLAY
    assert replay.is_live is False
    assert "no spend" in replay.describe().lower()


async def test_a_non_live_mode_still_refuses_a_priced_estimate(settings, ledger):
    """The assertion is written against 'not live', so replay inherits it.

    A golden bundle of a premium job carries model names that are priced in
    dollars. If a replay provider forgot to override ``estimate_cost``, this is
    what catches it.
    """
    governor = P.CostGovernor(ledger, P.Settings(provider_mode="replay", storage_backend="local"))

    async def call():
        return "should never run"

    with pytest.raises(AssertionError, match="replay mode must never estimate"):
        await governor.guarded_call(
            job_id="j",
            provider="golden",
            model="kling-2.5-turbo-pro",
            operation="video",
            estimated_usd=0.70,
            quantity=10.0,
            call=call,
        )
