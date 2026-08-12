"""Tests for harvesting the two-stage rank agreement.

The three properties worth naming, because each one is a way the headline number
could be wrong while every individual component is correct:

* **A replay is not an observation.** Identical candidates under a new job id
  collapse onto the set they replay. Without this, the week-14 replay suite inflates
  n and drives the interval to zero width on a single job.
* **Nondeterminism is a defect, not data.** Same bytes, different ordering means the
  scorer moved; it is reported as a conflict rather than averaged into the result.
* **A number needs enough sets and a real scorer, independently.** A stub agreement
  over fifty sets is quotable and not a result. A trained agreement over one set is
  a result and not quotable.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from adml import stages as ST
from adschema import (
    AdJobRequest,
    AssetRef,
    CameraAngle,
    Composition,
    ConsentAttestation,
    DesignPoint,
    ImageCandidate,
    JobRecord,
    JobResult,
    JobState,
    Lighting,
    MotionIntent,
    RankedCandidate,
    ScoreBreakdown,
    ShotBrief,
    VideoCandidate,
)

EPOCH = datetime(2026, 8, 1, tzinfo=UTC)


def brief(index: int) -> ShotBrief:
    return ShotBrief(
        index=index,
        design_point=DesignPoint(
            index=index,
            angle=CameraAngle.EYE_LEVEL,
            lighting=Lighting.SOFT_DIFFUSED,
            composition=Composition.CENTERED_HERO,
            motion=MotionIntent.SLOW_DOLLY_IN,
            seed=index,
        ),
        image_prompt="a prompt",
        motion_prompt="some motion",
    )


def record(
    job_id: str,
    *,
    image_shas: dict[int, str | None],
    image_scores: dict[int, float],
    video_order: list[int],
    is_stub: bool = True,
    finished_offset: int = 0,
    state: JobState = JobState.COMPLETED,
) -> JobRecord:
    """A completed job carrying both stage orderings.

    ``image_scores`` drives the image-stage order (highest first) and
    ``video_order`` is the final ranking, best first — so a test can set the two
    orderings independently, which is the whole point of the measurement.
    """
    images = [
        ImageCandidate(
            index=i,
            brief=brief(i),
            asset=AssetRef(key=f"generations/{job_id}/img_{i}.png", sha256=sha),
            score=ScoreBreakdown(overall=image_scores[i], is_stub=is_stub),
        )
        for i, sha in sorted(image_shas.items())
    ]
    videos = [
        VideoCandidate(
            index=i,
            source_image_index=i,
            brief=brief(i),
            asset=AssetRef(key=f"generations/{job_id}/vid_{i}.mp4", sha256=f"v{i}"),
            duration_seconds=9.0,
            score=ScoreBreakdown(overall=1.0, is_stub=is_stub),
        )
        for i in video_order
    ]
    ranking = [
        RankedCandidate(
            rank=position + 1,
            video=next(v for v in videos if v.source_image_index == i),
            explanation="because",
        )
        for position, i in enumerate(video_order)
    ]

    request = AdJobRequest(
        job_id=job_id,
        human_model_image=AssetRef(key="uploads/h.png"),
        product_image=AssetRef(key="uploads/p.png"),
        product_name="Aurora Serum",
        consent=ConsentAttestation(has_model_release=True, not_a_public_figure=True),
    )
    return JobRecord(
        request=request,
        state=state,
        result=JobResult(job_id=job_id, images=images, videos=videos, ranking=ranking),
        finished_at=EPOCH + timedelta(minutes=finished_offset),
    )


def three(shas: tuple[str, str, str]) -> dict[int, str | None]:
    return {0: shas[0], 1: shas[1], 2: shas[2]}


DESCENDING = {0: 0.9, 1: 0.6, 2: 0.3}


# --- Deduplication by content ------------------------------------------------


def test_a_replay_of_the_same_bytes_is_not_a_second_observation():
    """The trap the week-14 replay machinery sets for this measurement.

    Every replay writes a fresh record with a new job id over bit-identical
    candidates. Counted per job, replaying the golden set ten times would report ten
    sets in perfect agreement.
    """
    first = record(
        "job-a", image_shas=three(("x", "y", "z")), image_scores=DESCENDING, video_order=[0, 1, 2]
    )
    replay = record(
        "job-b",
        image_shas=three(("x", "y", "z")),
        image_scores=DESCENDING,
        video_order=[0, 1, 2],
        finished_offset=5,
    )

    harvest = ST.harvest([first, replay])

    assert harvest.n_sets == 1
    assert harvest.n_completed == 2
    assert harvest.n_collapsed == 1
    assert harvest.observations[0].n_replays == 2
    assert not harvest.conflicts


def test_different_bytes_are_different_observations():
    a = record(
        "job-a", image_shas=three(("x", "y", "z")), image_scores=DESCENDING, video_order=[0, 1, 2]
    )
    b = record(
        "job-b",
        image_shas=three(("p", "q", "r")),
        image_scores=DESCENDING,
        video_order=[2, 1, 0],
        finished_offset=5,
    )

    harvest = ST.harvest([a, b])

    assert harvest.n_sets == 2
    assert harvest.n_collapsed == 0


def test_the_set_key_ignores_the_storage_key():
    """Two jobs' assets live under different prefixes and are still the same set."""
    a = record(
        "job-a", image_shas=three(("x", "y", "z")), image_scores=DESCENDING, video_order=[0, 1, 2]
    )
    b = record(
        "job-b", image_shas=three(("x", "y", "z")), image_scores=DESCENDING, video_order=[0, 1, 2]
    )

    assert a.result.images[0].asset.key != b.result.images[0].asset.key
    assert ST.set_key(a.result) == ST.set_key(b.result)


def test_candidate_order_does_not_change_the_key():
    """The key is over content, so it cannot depend on how the list was assembled."""
    a = record(
        "job-a", image_shas={0: "x", 1: "y", 2: "z"}, image_scores=DESCENDING, video_order=[0, 1, 2]
    )
    b = record(
        "job-b", image_shas={2: "z", 1: "y", 0: "x"}, image_scores=DESCENDING, video_order=[0, 1, 2]
    )

    assert ST.set_key(a.result)[0] == ST.set_key(b.result)[0]


def test_a_set_with_the_same_frames_in_different_slots_is_a_different_set():
    """Slot matters: the same three frames ranked in different slots is a different
    experiment, because the design point attached to each slot differs."""
    a = record(
        "job-a", image_shas={0: "x", 1: "y", 2: "z"}, image_scores=DESCENDING, video_order=[0, 1, 2]
    )
    b = record(
        "job-b", image_shas={0: "y", 1: "x", 2: "z"}, image_scores=DESCENDING, video_order=[0, 1, 2]
    )

    assert ST.set_key(a.result)[0] != ST.set_key(b.result)[0]


def test_missing_digests_fall_back_to_the_job_id_and_say_so():
    """An older record has no content hash, so dedup is impossible rather than wrong."""
    a = record(
        "job-a",
        image_shas={0: "x", 1: None, 2: "z"},
        image_scores=DESCENDING,
        video_order=[0, 1, 2],
    )

    key, by_content = ST.set_key(a.result)
    assert key == "job-a"
    assert by_content is False

    harvest = ST.harvest([a])
    assert harvest.dedup_incomplete
    assert "upper bound" in harvest.summary()


def test_the_job_kept_for_a_set_is_the_one_that_produced_it():
    """Oldest first, so the reported job id is the original and not the last replay."""
    replay = record(
        "job-late",
        image_shas=three(("x", "y", "z")),
        image_scores=DESCENDING,
        video_order=[0, 1, 2],
        finished_offset=99,
    )
    original = record(
        "job-early",
        image_shas=three(("x", "y", "z")),
        image_scores=DESCENDING,
        video_order=[0, 1, 2],
        finished_offset=1,
    )

    harvest = ST.harvest([replay, original])

    assert harvest.observations[0].job_ids[0] == "job-early"


# --- Nondeterminism ----------------------------------------------------------


def test_a_ranking_that_moved_over_identical_bytes_is_a_conflict():
    """Same content, different order. That is the scorer moving, not new evidence."""
    a = record(
        "job-a", image_shas=three(("x", "y", "z")), image_scores=DESCENDING, video_order=[0, 1, 2]
    )
    b = record(
        "job-b",
        image_shas=three(("x", "y", "z")),
        image_scores=DESCENDING,
        video_order=[2, 1, 0],
        finished_offset=5,
    )

    harvest = ST.harvest([a, b])

    assert harvest.n_sets == 1
    assert len(harvest.conflicts) == 1
    conflict = harvest.conflicts[0]
    assert conflict.stage == "video-stage"
    assert "not deterministic" in conflict.summary()
    assert "CONFLICT" in harvest.summary()


def test_an_image_stage_conflict_is_reported_separately():
    a = record(
        "job-a",
        image_shas=three(("x", "y", "z")),
        image_scores={0: 0.9, 1: 0.6, 2: 0.3},
        video_order=[0, 1, 2],
    )
    b = record(
        "job-b",
        image_shas=three(("x", "y", "z")),
        image_scores={0: 0.3, 1: 0.6, 2: 0.9},
        video_order=[0, 1, 2],
        finished_offset=5,
    )

    harvest = ST.harvest([a, b])

    assert [c.stage for c in harvest.conflicts] == ["image-stage"]


# --- What counts as a result --------------------------------------------------


def test_a_stub_scored_harvest_is_not_a_result():
    harvest = ST.harvest(
        [
            record(
                "job-a",
                image_shas=three(("x", "y", "z")),
                image_scores=DESCENDING,
                video_order=[0, 1, 2],
                is_stub=True,
            ),
        ]
    )

    assert harvest.all_stub
    assert harvest.n_real_sets == 0
    assert "none of this is a result" in harvest.summary()


def test_trained_sets_are_counted_apart_from_stub_ones():
    stub = record(
        "job-a",
        image_shas=three(("x", "y", "z")),
        image_scores=DESCENDING,
        video_order=[0, 1, 2],
        is_stub=True,
    )
    real = record(
        "job-b",
        image_shas=three(("p", "q", "r")),
        image_scores=DESCENDING,
        video_order=[0, 1, 2],
        is_stub=False,
        finished_offset=5,
    )

    harvest = ST.harvest([stub, real])

    assert harvest.n_sets == 2
    assert harvest.n_real_sets == 1
    assert not harvest.all_stub
    assert harvest.model_vs_model(real_only=True).n_sets == 1


def test_a_set_with_no_scores_at_all_counts_as_stub():
    """Absence of a flag must not read as a trained score."""
    a = record(
        "job-a",
        image_shas=three(("x", "y", "z")),
        image_scores=DESCENDING,
        video_order=[0, 1, 2],
        is_stub=False,
    )
    for candidate in a.result.images:
        candidate.score = None
    for ranked in a.result.ranking:
        ranked.video.score = None

    assert ST._is_stub(a.result) is True


def test_one_set_is_not_quotable_and_two_are():
    a = record(
        "job-a", image_shas=three(("x", "y", "z")), image_scores=DESCENDING, video_order=[0, 1, 2]
    )
    b = record(
        "job-b",
        image_shas=three(("p", "q", "r")),
        image_scores=DESCENDING,
        video_order=[0, 1, 2],
        finished_offset=5,
    )

    assert not ST.harvest([a]).quotable()
    assert ST.harvest([a, b]).quotable()


def test_a_single_set_still_reports_a_perfect_correlation_which_is_why_it_is_gated():
    """The arithmetic that motivates MIN_QUOTABLE_SETS.

    Over one set of three candidates ranked identically, rho is exactly +1 whatever
    the ranker did. The harness reports it faithfully; the gate is what stops it
    being quoted.
    """
    harvest = ST.harvest(
        [
            record(
                "job-a",
                image_shas=three(("x", "y", "z")),
                image_scores=DESCENDING,
                video_order=[0, 1, 2],
            ),
        ]
    )

    assert harvest.model_vs_model().mean_spearman == 1.0
    assert not harvest.quotable()


# --- Skipping and incomplete jobs --------------------------------------------


def test_an_unfinished_job_contributes_nothing():
    queued = record(
        "job-a",
        image_shas=three(("x", "y", "z")),
        image_scores=DESCENDING,
        video_order=[0, 1, 2],
        state=JobState.QUEUED,
    )

    harvest = ST.harvest([queued])

    assert harvest.n_records == 1
    assert harvest.n_completed == 0
    assert harvest.n_sets == 0


def test_a_gate_failure_leaves_a_shared_subset_that_is_still_measurable():
    """Two of three slots survived. The shared pair is a real observation."""
    a = record(
        "job-a", image_shas=three(("x", "y", "z")), image_scores=DESCENDING, video_order=[0, 1]
    )

    harvest = ST.harvest([a])

    assert harvest.n_sets == 1
    assert harvest.observations[0].video_order == ["0", "1"]


def test_a_job_with_one_surviving_candidate_is_skipped_with_a_reason():
    a = record("job-a", image_shas=three(("x", "y", "z")), image_scores=DESCENDING, video_order=[0])

    harvest = ST.harvest([a])

    assert harvest.n_sets == 0
    assert harvest.skipped
    reason = next(iter(harvest.skipped))
    assert "final ranking" in reason


# --- Rank movement -----------------------------------------------------------


def test_rank_shifts_count_movement_between_the_stages():
    """A reversal of three candidates moves the ends by two and holds the middle."""
    a = record(
        "job-a", image_shas=three(("x", "y", "z")), image_scores=DESCENDING, video_order=[2, 1, 0]
    )

    assert ST.harvest([a]).rank_shifts() == {-2: 1, 0: 1, 2: 1}


def test_no_movement_is_all_zeros():
    a = record(
        "job-a", image_shas=three(("x", "y", "z")), image_scores=DESCENDING, video_order=[0, 1, 2]
    )

    assert ST.harvest([a]).rank_shifts() == {0: 3}


# --- The claim itself --------------------------------------------------------


def test_the_claim_is_pending_without_video_judgements():
    result = ST.human_stage_agreement({"s0": ["0", "1", "2"]}, {}, n_video_judgements=0)

    assert not result.available
    assert "no video-stage pairwise judgements" in result.reason
    assert result.summary().startswith("PENDING")


def test_the_claim_needs_two_sets_even_when_judgements_exist():
    result = ST.human_stage_agreement(
        {"s0": ["0", "1", "2"]}, {"s0": ["1", "0", "2"]}, n_video_judgements=12
    )

    assert not result.available
    assert "two are needed" in result.reason


def test_the_claim_computes_once_two_sets_are_judged():
    image = {"s0": ["0", "1", "2"], "s1": ["0", "1", "2"]}
    human = {"s0": ["0", "1", "2"], "s1": ["2", "1", "0"]}

    result = ST.human_stage_agreement(image, human, n_video_judgements=24)

    assert result.available
    assert result.agreement is not None
    assert result.agreement.n_sets == 2
    # One set agrees perfectly and one is reversed, so the mean is zero.
    assert result.agreement.mean_spearman == 0.0
    assert result.agreement.top1_retention == 0.5


def test_a_judged_set_with_no_prediction_is_ignored_rather_than_guessed():
    image = {"s0": ["0", "1", "2"], "s1": ["0", "1", "2"]}
    human = {"s0": ["0", "1", "2"], "s9": ["2", "1", "0"]}

    result = ST.human_stage_agreement(image, human, n_video_judgements=24)

    assert not result.available
    assert "1 set(s)" in result.reason
