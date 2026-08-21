"""End-to-end pipeline behaviour.

The headline assertion is the zero-spend one: a complete 3-image / 3-video job
must run start to finish without the ledger moving at all.  With a $35 lifetime
budget, "the pipeline is free to develop against" is a property worth a test
rather than a hope.
"""

from __future__ import annotations

import asyncio
import time

import adproviders as P
import pytest
from adml import features as F
from adml import video as V
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
from adworker.sampler import describe_diversity, max_achievable_distance


async def test_full_job_completes_with_zero_spend(pipeline, make_request, ledger):
    record = await pipeline.run(make_request())

    assert record.state is JobState.COMPLETED, record.error or record.refusal_reason
    result = record.result
    request = record.request

    # Five frames, two clips. Derived from the request rather than written out,
    # because the counts are now a product decision and a test that hard-codes
    # them fails for the wrong reason when that decision changes.
    assert len(result.images) == request.candidate_count
    assert len(result.videos) == request.video_count
    assert len(result.ranking) == request.video_count
    assert sum(c.promoted for c in result.images) == request.video_count

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
    """Duration is read back from the file, not taken from the provider's word.

    Probed through :mod:`adml.video`, which is the point: this test previously used
    a PIL helper and so could only ever have passed on a GIF. It was green while
    the pipeline was structurally unable to read the MP4s the paid provider returns.
    """
    record = await pipeline.run(make_request(duration_seconds=9.0))
    for video in record.result.videos:
        check = V.verify_duration(
            storage.get_bytes(video.asset.key),
            video.duration_seconds,
            min_seconds=MIN_DURATION_S,
            max_seconds=MAX_DURATION_S,
        )
        assert check.in_window, check.summary()
        assert check.matches_claim, check.summary()


async def test_the_mock_provider_writes_a_container_the_scorer_can_read(
    pipeline, make_request, storage
):
    """The mock's output must travel the same decode path as a paid clip.

    Where ffmpeg is available that means a real MP4. This is the guard on the
    defect this phase found: while the mock wrote GIF and the scorer used PIL, the
    entire video half of the pipeline was untested against the format every real
    provider returns.
    """
    record = await pipeline.run(make_request())
    for video in record.result.videos:
        data = storage.get_bytes(video.asset.key)
        container = V.container_of(data)
        assert container == ("mp4" if V.FFMPEG else "gif"), container
        # And it decodes to real frames with real motion, not to an empty list.
        clip, motion = V.measure(data)
        assert clip.n_sampled >= 2
        assert motion.motion_energy_mean > 0.0


@pytest.mark.parametrize("duration", [8.0, 9.0, 10.0])
async def test_duration_window_endpoints(pipeline, make_request, storage, duration):
    record = await pipeline.run(make_request(duration_seconds=duration))
    assert record.state is JobState.COMPLETED
    for video in record.result.videos:
        actual = V.probe(storage.get_bytes(video.asset.key)).duration_seconds
        assert abs(actual - duration) < 0.05


async def test_candidates_are_genuinely_diverse(pipeline, make_request):
    """Candidates that differ on every axis the design space can still separate.

    This asserted a flat 4 while the job asked for three candidates and every axis
    had at least three levels. Five candidates changed that: on Reels only four
    compositions are viable, because `negative_space_top` puts the subject under
    the platform's bottom chrome, so two of the five must share one and the best
    achievable minimum is 3.

    Asserting the computed bound rather than the remembered number is the fix. The
    guarantee was never "4" — it was "as separated as the axes allow", and that is
    now a claim the code can state.
    """
    record = await pipeline.run(make_request())
    request, briefs = record.request, record.result.briefs
    bound = max_achievable_distance(request.candidate_count, platform=request.platform)

    assert briefs.min_pairwise_distance == bound, describe_diversity(
        [b.design_point for b in briefs.briefs]
    )
    axes = [b.design_point.axes() for b in briefs.briefs]
    assert len(set(axes)) == request.candidate_count, "two candidates share every axis"


async def test_locked_angle_costs_exactly_one_axis(pipeline, make_request):
    """Locking the angle collapses one axis to a constant, and the sampler says so
    rather than hiding it — the user asked for it, and the report should show what
    it cost."""
    record = await pipeline.run(make_request(locked_angle=CameraAngle.PROFILE))
    request, briefs = record.request, record.result.briefs

    free = max_achievable_distance(request.candidate_count, platform=request.platform)
    locked = max_achievable_distance(
        request.candidate_count, platform=request.platform, locked_angle=CameraAngle.PROFILE
    )
    assert locked == free - 1, "locking an axis should cost exactly one, never more"
    assert briefs.min_pairwise_distance == locked
    assert {b.design_point.angle for b in briefs.briefs} == {CameraAngle.PROFILE}


async def test_a_platform_with_room_for_every_composition_keeps_the_full_guarantee(
    pipeline, make_request
):
    """The counterpart, and the reason the bound is computed rather than lowered.

    Reels loses a composition to its bottom chrome. Feed does not, so five
    candidates there still separate on all four axes. Without this, dropping the
    Reels expectation to 3 would look like the guarantee weakening everywhere.
    """
    record = await pipeline.run(make_request(platform=Platform.INSTAGRAM_FEED))
    briefs = record.result.briefs
    assert briefs.min_pairwise_distance == 4
    assert max_achievable_distance(5, platform=Platform.INSTAGRAM_FEED) == 4


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
    assert len(result.image_stage_order) == record.request.candidate_count
    for ranked in result.ranking:
        assert ranked.image_stage_rank is not None
        assert ranked.rank_shift is not None


async def test_rankings_are_a_permutation_of_the_candidates(pipeline, make_request):
    """The two stages no longer rank the same set, and that is the design.

    The image stage ranks everything that passed the gate. The video stage ranks
    only what was promoted, so its ordering is a *subset* — and specifically the
    top of the image stage's ordering, since that is what promotion means.
    """
    record = await pipeline.run(make_request())
    result, request = record.result, record.request

    assert sorted(result.image_stage_order) == list(range(request.candidate_count))
    assert set(result.video_stage_order) <= set(result.image_stage_order)
    assert len(result.video_stage_order) == request.video_count
    assert [r.rank for r in result.ranking] == list(range(1, request.video_count + 1))

    # Promotion is by image-stage rank, so the animated set is exactly the head of
    # the image ordering. If this ever fails, the cut has stopped being a decision
    # the predictor makes and become an accident of iteration order.
    promoted = set(result.image_stage_order[: request.video_count])
    assert set(result.video_stage_order) == promoted


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
    or the write-up would claim identity verification that never happened.

    ``product_scale`` is pending here for a different reason from the other three:
    they await model weights, while it awaits a frame the mock renderer cannot
    produce — it paints from a palette extracted from the product, so the
    product's own colours locate nothing. Same treatment either way, which is the
    point: the UI cannot tell the user a check ran when it did not.
    """
    record = await pipeline.run(make_request())
    for candidate in record.result.images:
        pending = {c.name for c in candidate.gate.pending_checks}
        assert pending == {"product_identity", "face_identity", "nsfw", "product_scale"}
        assert all(c.passed for c in candidate.gate.pending_checks)
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


async def test_every_motion_intent_asks_the_model_to_do_something(pipeline, make_request):
    """The guard on the mistake that cost a whole premium job.

    `MotionIntent` used to hold six levels and five of them named the camera and
    nothing else — `slow_dolly_in`, `orbit_left`, and a `static_subtle` that asked
    in as many words for a near-static frame. Image-to-video models do what they
    are told, so a $2.22 job came back as three slow pans across a still photo.

    This asserts the property that was missing rather than the wording that
    replaced it: every level in the axis must put the *model* in motion, so a
    level added later cannot quietly reintroduce a moving poster. The word list is
    the vocabulary the beat table actually uses, and a new beat that matches none
    of it is a beat that has stopped describing a performance.
    """
    from adschema import MotionIntent
    from adworker.briefs import _MOTION_BEATS

    assert set(_MOTION_BEATS) == set(MotionIntent), "every level needs a beat"

    verbs = ("raises", "turns", "uses", "extends", "reaches", "steps", "lifts", "holds")
    for intent, beat in _MOTION_BEATS.items():
        assert any(v in beat.action for v in verbs), f"{intent.value} has no human action"
        assert "model" in beat.opening, f"{intent.value}'s opening frame does not place the model"


async def test_the_users_own_direction_reaches_the_video_stage(pipeline, make_request):
    """It used to reach the image stage only.

    Someone who typed "the model picks the shoes up off the table" got that in
    their still and never once in their clip — the single most specific piece of
    creative information the job had, withheld from the stage that most needed it.
    """
    direction = "the shoes are picked up off the table and held toward the camera"
    record = await pipeline.run(make_request(additional_prompt=direction))

    briefs = record.result.briefs.briefs
    assert all(direction in b.image_prompt for b in briefs), "regression in the image stage"
    assert all(direction in b.motion_prompt for b in briefs), "the video stage cannot see it"


async def test_the_still_is_composed_as_the_first_frame_of_its_motion(pipeline, make_request):
    """The two stages had never been introduced.

    The image prompt asked for a finished advertising photograph — a settled pose,
    with nowhere left to go — and then the video stage was handed it and asked for
    movement. The only thing left available to a video model at that point is to
    move the camera.
    """
    from adworker.briefs import _MOTION_BEATS

    record = await pipeline.run(make_request())
    for b in record.result.briefs.briefs:
        assert _MOTION_BEATS[b.design_point.motion].opening in b.image_prompt


async def test_no_mood_prefers_a_pool_the_sampler_cannot_animate(make_request):
    """The quiet half of the same bug.

    The sampler draws preferred levels first, so a mood whose pool is three camera
    moves does not express a preference — it issues a guarantee. `warm_lifestyle`,
    the default, preferred `[slow_dolly_in, product_present, static_subtle]`: two of
    every three clips camera-only and one asked to hold still.
    """
    from adschema import Mood
    from adworker.sampler import _MOOD_MOTION

    assert set(_MOOD_MOTION) == set(Mood)
    for mood, pool in _MOOD_MOTION.items():
        assert len(set(pool)) >= 3, f"{mood.value} cannot fill three candidates distinctly"


async def test_the_cut_is_made_by_the_predictor_not_by_slot_order(pipeline, make_request):
    """The claim the whole 5-to-2 shape rests on.

    Promoting the first two candidates in slot order would produce videos, pass
    every count assertion above, and be worthless — the image-stage predictor would
    have had no effect on where the money went while appearing to. So this checks
    the promoted set against the *scores*, independently of the pipeline's own
    ordering helper.
    """
    record = await pipeline.run(make_request())
    result, request = record.result, record.request

    scored = [c for c in result.images if c.score is not None]
    by_score = sorted(scored, key=lambda c: -c.score.overall)
    expected = {c.index for c in by_score[: request.video_count]}

    assert {c.index for c in result.images if c.promoted} == expected
    assert {v.source_image_index for v in result.videos} == expected


async def test_a_candidate_that_fails_the_gate_is_never_promoted(pipeline, make_request):
    """Gate first, then rank.

    The gate is a hard pass/fail about whether a frame is usable; the score is a
    soft opinion about whether it will perform. A high score cannot rescue an
    unusable frame, and conflating the two is the usual way these systems get
    muddled.
    """
    record = await pipeline.run(make_request())
    for candidate in record.result.images:
        if not candidate.passed_gate:
            assert not candidate.promoted, f"slot {candidate.index} failed its gate but animated"


async def test_animating_everything_is_still_available_for_evaluation(pipeline, make_request):
    """The research claim needs complete observations, and the product does not.

    A 5-to-2 job yields a *truncated* paired observation: the three candidates the
    predictor cut have no video-stage outcome and never will, so a rank agreement
    computed over them would be range-restricted by construction. Setting
    video_count == candidate_count restores the complete observation the
    evaluation needs, at the cost of animating everything.
    """
    record = await pipeline.run(make_request(candidate_count=3, video_count=3))
    result = record.result

    assert record.request.animates_everything
    assert len(result.videos) == 3
    assert sorted(result.video_stage_order) == sorted(result.image_stage_order)
    assert all(c.promoted for c in result.images if c.passed_gate)


async def test_asking_for_more_videos_than_candidates_animates_what_exists(pipeline, make_request):
    """Incoherent rather than malicious — a leftover form value, or a script that
    changed one number and not the other. Clamping costs the user nothing; refusing
    would cost them their uploads to tell them something the clamp already says."""
    record = await pipeline.run(make_request(candidate_count=2, video_count=5))

    assert record.request.video_count == 2
    assert record.state is JobState.COMPLETED, record.error
    assert len(record.result.videos) == 2


async def test_the_preflight_reserves_what_a_five_to_two_job_will_actually_spend(make_request):
    """A governor that over-reserves refuses jobs the budget could have afforded.

    That is a quieter failure than overspending and just as wrong, and it was one
    line away: the video term is 94% of a default job, so pricing two clips as five
    reserves more than three times the real cost.
    """
    from adproviders.pricing import estimate_job_cost

    request = make_request()
    args = ("seedream-4.5-edit", "kling-2.5-turbo-pro", "mock")
    five_to_two = estimate_job_cost(*args, request.candidate_count, 9.0, request.video_count)
    animate_all = estimate_job_cost(*args, request.candidate_count, 9.0, request.candidate_count)

    assert five_to_two == pytest.approx(0.20 + 1.40)
    assert animate_all == pytest.approx(0.20 + 3.50)
    # Omitting video_count must keep meaning the 1:1 cascade, or every existing
    # caller silently starts under-reserving.
    assert estimate_job_cost(*args, 3, 9.0) == pytest.approx(estimate_job_cost(*args, 3, 9.0, 3))


# --- Concurrency ----------------------------------------------------------
#
# The first live job took eight minutes and the user's complaint was about the
# wait, not the output. The cause was in the stage loops rather than anywhere
# interesting: three image generations of 54.1 s, 46.9 s and 49.9 s occupied
# 152 s of wall clock, so nothing had overlapped with anything. At five
# candidates that is four minutes of watching a progress bar.
#
# Mock providers return instantly, so a test that just runs the pipeline cannot
# tell a fan-out from a queue. These give the provider a measurable duration and
# time the result, which is the only way the property is observable.


class _SlowImages(P.MockImageProvider):
    """A mock that takes a second, so overlap is something a clock can see."""

    async def generate(self, request):
        await asyncio.sleep(_SLOW_S)
        return await super().generate(request)


class _SlowVideos(P.MockVideoProvider):
    async def generate(self, request):
        await asyncio.sleep(_SLOW_S)
        return await super().generate(request)


_SLOW_S = 0.5


async def test_candidates_are_generated_concurrently_not_one_at_a_time(
    storage, governor, make_request
):
    """Seven half-second generations must not cost three and a half seconds.

    Timed as a *difference* rather than as a total. A job also does intake, gating,
    scoring, ffprobe and delivery, and on this machine that fixed work outweighs
    the generation it surrounds — the first version of this test measured the whole
    run and failed while the fan-out underneath it was working correctly. Running
    the same job twice, once with instant providers, isolates the part the change
    was about and makes the bound independent of how fast the laptop is.
    """
    request = make_request(candidate_count=5, video_count=2)

    def build(images, videos):
        return Pipeline(
            storage=storage,
            governor=governor,
            image_provider=images(storage),
            video_provider=videos(storage),
            llm_provider=P.MockLLMProvider(),
        )

    started = time.monotonic()
    await build(P.MockImageProvider, P.MockVideoProvider).run(request)
    overhead = time.monotonic() - started

    started = time.monotonic()
    record = await build(_SlowImages, _SlowVideos).run(request)
    generating = time.monotonic() - started - overhead

    assert record.state is JobState.COMPLETED, record.error
    sequential = (request.candidate_count + request.video_count) * _SLOW_S
    assert generating < sequential * 0.6, (
        f"{generating:.2f}s spent generating work that is {sequential:.2f}s "
        "sequentially — the stages are still running one slot at a time"
    )


async def test_the_fan_out_does_not_reorder_the_candidates(storage, governor, make_request):
    """Slot order is load-bearing, and completion order is not slot order.

    ``result.images`` is indexed by slot by the ranker, the golden replay and the
    letters shown in the UI. This provider finishes the last slot first, which is
    exactly what a real queue does and what appending-as-they-land would record.
    """

    class _ReverseOrder(P.MockImageProvider):
        async def generate(self, request):
            # Later slots return sooner: slot 0 waits longest.
            await asyncio.sleep(_SLOW_S * (5 - request.seed % 5) / 10)
            return await super().generate(request)

    pipeline = Pipeline(
        storage=storage,
        governor=governor,
        image_provider=_ReverseOrder(storage),
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
    )
    record = await pipeline.run(make_request(candidate_count=5, video_count=2))

    assert [c.index for c in record.result.images] == [0, 1, 2, 3, 4]
    assert [c.brief.index for c in record.result.images] == [0, 1, 2, 3, 4]


async def test_concurrency_is_bounded_by_the_configured_limit(storage, governor, make_request):
    """Fan-out is capped, because the thing being fanned out at has a rate limit.

    A 429 fails a slot that has already cleared the budget check, so an unbounded
    speed-up trades latency for lost generations.
    """
    live = 0
    peak = 0

    class _Counting(P.MockImageProvider):
        async def generate(self, request):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            try:
                await asyncio.sleep(_SLOW_S / 5)
                return await super().generate(request)
            finally:
                live -= 1

    governor.settings.max_concurrent_generations = 2
    pipeline = Pipeline(
        storage=storage,
        governor=governor,
        image_provider=_Counting(storage),
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
    )
    await pipeline.run(make_request(candidate_count=5, video_count=2))

    assert peak <= 2, f"{peak} generations were in flight against a limit of 2"
    assert peak == 2, "nothing ran concurrently at all"
