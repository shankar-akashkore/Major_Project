"""End-to-end pipeline behaviour.

The headline assertion is the zero-spend one: a complete 3-image / 3-video job
must run start to finish without the ledger moving at all.  With a $35 lifetime
budget, "the pipeline is free to develop against" is a property worth a test
rather than a hope.
"""

from __future__ import annotations

import pytest
from adml import features as F
from adschema import (
    MAX_DURATION_S,
    MIN_DURATION_S,
    CameraAngle,
    ConsentAttestation,
    GateVerdict,
    JobState,
    Platform,
    Stage,
)
from adworker import Pipeline


async def test_full_job_completes_with_zero_spend(pipeline, make_request, ledger):
    record = await pipeline.run(make_request())

    assert record.state is JobState.COMPLETED, record.error or record.refusal_reason
    result = record.result
    assert len(result.images) == 3
    assert len(result.videos) == 3
    assert len(result.ranking) == 3

    # The point of the whole mock layer.
    assert await ledger.total_spent() == 0.0
    assert result.total_cost_usd == 0.0


async def test_all_eight_stages_report_completion(pipeline, make_request):
    record = await pipeline.run(make_request())
    completed = {e.stage for e in record.events if e.state == "completed"}
    assert completed == set(Stage), f"stages missing a completion event: {set(Stage) - completed}"


async def test_delivered_videos_are_in_the_committed_duration_window(
    pipeline, make_request, storage
):
    """Duration is read back from the file, not taken from the provider's word."""
    record = await pipeline.run(make_request(duration_seconds=9.0))
    for video in record.result.videos:
        actual = F.frame_durations_seconds(storage.get_bytes(video.asset.key))
        assert MIN_DURATION_S <= actual <= MAX_DURATION_S, f"{actual}s outside the window"
        assert abs(actual - video.duration_seconds) < 0.05, "reported duration disagrees with file"


@pytest.mark.parametrize("duration", [8.0, 9.0, 10.0])
async def test_duration_window_endpoints(pipeline, make_request, storage, duration):
    record = await pipeline.run(make_request(duration_seconds=duration))
    assert record.state is JobState.COMPLETED
    for video in record.result.videos:
        actual = F.frame_durations_seconds(storage.get_bytes(video.asset.key))
        assert abs(actual - duration) < 0.05


async def test_candidates_are_genuinely_diverse(pipeline, make_request):
    """Three candidates that differ on every design axis, not three near-duplicates."""
    record = await pipeline.run(make_request())
    briefs = record.result.briefs
    assert briefs.min_pairwise_distance == 4, briefs.min_pairwise_distance

    axes = [b.design_point.axes() for b in briefs.briefs]
    assert len(set(axes)) == 3


async def test_locked_angle_costs_exactly_one_axis(pipeline, make_request):
    record = await pipeline.run(make_request(locked_angle=CameraAngle.PROFILE))
    briefs = record.result.briefs
    assert briefs.min_pairwise_distance == 3
    assert {b.design_point.angle for b in briefs.briefs} == {CameraAngle.PROFILE}


async def test_image_ranking_is_recorded_before_video_generation(pipeline, make_request):
    """The cascade claim depends on the image-stage order being fixed before any
    video exists — otherwise agreement between the stages could be reconstructed
    after the fact and would mean nothing."""
    record = await pipeline.run(make_request())
    events = record.events
    rank_done = next(
        i for i, e in enumerate(events) if e.stage is Stage.IMAGE_RANK and e.state == "completed"
    )
    video_start = next(
        i for i, e in enumerate(events) if e.stage is Stage.VIDEO_GEN and e.state == "started"
    )
    assert rank_done < video_start

    result = record.result
    assert len(result.image_stage_order) == 3
    for ranked in result.ranking:
        assert ranked.image_stage_rank is not None
        assert ranked.rank_shift is not None


async def test_rankings_are_a_permutation_of_the_candidates(pipeline, make_request):
    record = await pipeline.run(make_request())
    result = record.result
    assert sorted(result.image_stage_order) == [0, 1, 2]
    assert sorted(result.video_stage_order) == [0, 1, 2]
    assert [r.rank for r in result.ranking] == [1, 2, 3]


async def test_missing_consent_is_refused_before_any_generation(pipeline, make_request, ledger):
    record = await pipeline.run(
        make_request(consent=ConsentAttestation(has_model_release=False, not_a_public_figure=True))
    )
    assert record.state is JobState.FAILED
    assert "rights attestation" in (record.error or "")
    assert record.result.images == []
    assert await ledger.total_spent() == 0.0


async def test_gate_reports_unimplemented_checks_as_pending(pipeline, make_request):
    """A check that cannot run must be visibly distinct from one that passed,
    or the write-up would claim identity verification that never happened."""
    record = await pipeline.run(make_request())
    for candidate in record.result.images:
        pending = {c.name for c in candidate.gate.pending_checks}
        assert pending == {"product_identity", "face_identity", "nsfw"}
        assert candidate.gate.verdict is GateVerdict.PASS
        assert not candidate.gate.verified_identity


async def test_scores_are_flagged_as_stubs(pipeline, make_request):
    record = await pipeline.run(make_request())
    for candidate in record.result.images:
        assert candidate.score.is_stub
    for video in record.result.videos:
        assert video.score.is_stub
        # CLIPScore is unavailable, so alignment must be absent rather than invented.
        assert video.score.prompt_alignment is None


async def test_explanations_name_a_differentiator_not_a_saturated_component(pipeline, make_request):
    record = await pipeline.run(make_request())
    ranking = record.result.ranking
    assert all(r.explanation for r in ranking)
    # With three distinct candidates at least one should have a real differentiator
    # rather than every explanation falling back to the "within noise" wording.
    assert any("best of the set" in r.explanation for r in ranking)


async def test_determinism_same_seed_same_ranking(storage, governor, make_request):
    """Reproducibility is what makes the planned ablations comparable."""
    import adproviders as P

    def build() -> Pipeline:
        return Pipeline(
            storage=storage,
            governor=P.CostGovernor(P.InMemoryLedger(), P.Settings(provider_mode="mock")),
            image_provider=P.MockImageProvider(storage),
            video_provider=P.MockVideoProvider(storage),
            llm_provider=P.MockLLMProvider(),
        )

    first = await build().run(make_request(seed=123))
    second = await build().run(make_request(seed=123))

    assert first.result.image_stage_order == second.result.image_stage_order
    assert first.result.video_stage_order == second.result.video_stage_order
    assert [round(c.score.overall, 6) for c in first.result.images] == [
        round(c.score.overall, 6) for c in second.result.images
    ]


async def test_platform_drives_geometry(pipeline, make_request, storage):
    record = await pipeline.run(make_request(platform=Platform.YOUTUBE_INSTREAM))
    assert record.state is JobState.COMPLETED
    expected = Platform.YOUTUBE_INSTREAM.aspect_ratio
    assert record.request.aspect_ratio is expected
    for video in record.result.videos:
        assert expected.value in video.platform_renders
    # A landscape placement must actually produce landscape frames.
    image = F.load_image(storage.get_bytes(record.result.images[0].asset.key))
    height, width = image.shape[:2]
    assert width > height
