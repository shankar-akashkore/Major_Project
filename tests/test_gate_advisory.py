"""Why the first live Seedream job cost $0.20 and produced nothing.

Five candidates, all rejected, all on ``palette_adherence``: 0.427, 0.253, 0.412,
0.390, 0.278 against a threshold of 0.456.  All five were then retried, all five
failed again, and the pipeline correctly refused to spend on video.  The images
were fine.

The palette was the problem.  ``rembg`` was not installed, the flood-fill fallback
declined the photograph as too varied to matte (border uniformity 0.041 against a
0.35 floor), so intake fell back to reading the palette off the *whole* product
photo.  That palette came back **79% #eee9e2** — the off-white sweep the product
was photographed on.  The gate then required every advertisement to be 79%
off-white, which no brief in this system would ever ask a generator to produce.

Two things were wrong and each is fixed here.

The gate rejected on a reference it had already admitted it did not trust: the job
page prints "a materially weaker constraint" beside exactly this palette source.
A check whose *reference* is untrustworthy can still report — the number is real —
but it cannot veto a paid generation.  ``GateCheck.advisory`` is that distinction,
and it is deliberately not ``implemented=False``: this measurement ran.

And the retry paid twice for one answer.  When every candidate fails the same
check the cause is upstream of any prompt, and rewording five prompts buys five
more rejections at full price.

Worth being precise about what is *not* claimed here: ``THRESH_PALETTE`` is fine.
Scored against a palette that genuinely describes them, those same five frames
measure 0.789-0.911, comfortably clear of 0.456.  The threshold was the obvious
suspect and it was innocent.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from adproviders.mock import _render_frame
from adschema import (
    CameraAngle,
    Composition,
    DesignPoint,
    GateCheck,
    GateResult,
    GateVerdict,
    ImageCandidate,
    Lighting,
    MotionIntent,
    ShotBrief,
    Tier,
)
from adworker.gate import evaluate_image
from adworker.pipeline import _shared_blocking_failure
from PIL import Image

#: The palette intake extracted from the live job's product photograph. Not
#: invented: 79% of the frame was the studio sweep behind the headphones.
BACKDROP_PALETTE = ["#eee9e2", "#afa599", "#050505"]


def _frame(palette: list[str], size: tuple[int, int] = (270, 480)) -> bytes:
    """A structurally sound ad frame drawn in ``palette``.

    The mock renderer rather than a flat rectangle, and that is not decoration: the
    gate also measures focal clarity and safe-area attention, and a uniformly noisy
    square fails both for reasons that have nothing to do with colour. Using it here
    would prove the frame was rejected without proving *why*.
    """
    return _png(
        np.asarray(
            _render_frame(
                size,
                palette,
                CameraAngle.EYE_LEVEL,
                Lighting.SOFT_DIFFUSED,
                Composition.CENTERED_HERO,
                label="",
            ).convert("RGB")
        )
    )


def _png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _candidate(storage, key: str, palette: list[str]) -> ImageCandidate:
    return ImageCandidate(
        index=0,
        brief=ShotBrief(
            index=0,
            design_point=DesignPoint(
                index=0,
                angle=CameraAngle.EYE_LEVEL,
                lighting=Lighting.SOFT_DIFFUSED,
                composition=Composition.CENTERED_HERO,
                motion=MotionIntent.HERO_TURN,
                seed=1,
            ),
            image_prompt="p",
            negative_prompt="n",
            motion_prompt="m",
            concept="c",
        ),
        asset=storage.put_bytes(key, _frame(palette)),
        tier=Tier.PREMIUM,
    )


# --- An untrustworthy palette cannot veto ------------------------------------

#: Roughly what the live job's briefs asked for and got — a plum studio sweep
#: under warm light. Far from an off-white backdrop by any colour metric, which is
#: the entire point: this frame is exactly what the brief ordered.
#: Measured at 0.264 against ``BACKDROP_PALETTE`` — inside the 0.253-0.427 band
#: the five live candidates actually landed in.
PLUM_AD = ["#6e3c55", "#8d5d60", "#3a1f2c"]


def test_a_backdrop_palette_rejects_a_perfectly_good_frame(storage, make_request):
    """The bug, reproduced. Kept as a test so the fix has something to be a fix of."""
    request = make_request(theme={"palette": BACKDROP_PALETTE})
    candidate = _candidate(storage, "gate/plum.png", PLUM_AD)

    result = evaluate_image(candidate, request, storage, 1, None, palette_reliable=True)

    assert "palette_adherence" in {c.name for c in result.failures}
    assert result.verdict is not GateVerdict.PASS


def test_the_same_frame_passes_once_the_palette_is_known_to_be_unreliable(storage, make_request):
    """The fix. Same frame, same palette, same threshold — only the trust changes."""
    request = make_request(theme={"palette": BACKDROP_PALETTE})
    candidate = _candidate(storage, "gate/plum.png", PLUM_AD)

    result = evaluate_image(candidate, request, storage, 1, None, palette_reliable=False)

    assert result.verdict is GateVerdict.PASS


def test_an_advisory_failure_is_still_reported(storage, make_request):
    """Downgraded, not hidden.

    The number is a real measurement and it is the reason a frame looks off-brand.
    Suppressing it would trade a gate that rejects too much for a gate that explains
    nothing, which is the worse of the two.
    """
    request = make_request(theme={"palette": BACKDROP_PALETTE})
    candidate = _candidate(storage, "gate/plum.png", PLUM_AD)

    result = evaluate_image(candidate, request, storage, 1, None, palette_reliable=False)
    palette = next(c for c in result.checks if c.name == "palette_adherence")

    assert palette.advisory
    assert not palette.passed, "the measurement itself must not be faked into a pass"
    assert palette.implemented, "it ran; it is not a pending check"
    assert palette in result.failures
    assert palette not in result.blocking_failures
    assert "advisory" in palette.detail
    assert "whole product photograph" in palette.detail


def test_a_real_check_still_rejects_alongside_an_advisory_one(storage, make_request):
    """Advisory applies to one check, not to the gate.

    In the live job one candidate also failed ``safe_area`` at 0.492 — a genuine
    check against a trustworthy reference. That candidate should still be rejected
    after the fix, and it is.
    """
    request = make_request(theme={"palette": BACKDROP_PALETTE})
    candidate = _candidate(storage, "gate/black.png", ["#020202", "#010101", "#000000"])

    result = evaluate_image(candidate, request, storage, 2, None, palette_reliable=False)

    assert result.verdict is GateVerdict.REJECT
    assert {c.name for c in result.blocking_failures}, "something real must still bite"


# --- A job-wide failure is not five unlucky generations ----------------------


def _gate(*failures: str, verdict=GateVerdict.RETRY, advisory=()) -> GateResult:
    return GateResult(
        verdict=verdict,
        checks=[
            GateCheck(
                name=name,
                value=0.1,
                threshold=0.5,
                passed=False,
                advisory=name in advisory,
            )
            for name in failures
        ],
    )


def _with(gate: GateResult | None) -> ImageCandidate:
    return ImageCandidate.model_construct(index=0, gate=gate)


def test_a_check_every_candidate_failed_is_a_job_wide_cause():
    """The live job's shape: five candidates, one shared check."""
    cands = [_with(_gate("palette_adherence")) for _ in range(5)]
    assert _shared_blocking_failure(cands) == {"palette_adherence"}


def test_candidates_failing_different_checks_are_ordinary_failures():
    """Intersection, not union.

    Two candidates failing for different reasons are two ordinary failures, and a
    stricter prompt may well fix either. Treating that as structural would delete a
    retry that works.
    """
    cands = [_with(_gate("palette_adherence")), _with(_gate("safe_area"))]
    assert _shared_blocking_failure(cands) == set()


def test_one_survivor_means_nothing_is_job_wide():
    """A job with a viable candidate has nothing job-wide wrong with it."""
    cands = [
        _with(_gate("safe_area")),
        _with(_gate(verdict=GateVerdict.PASS)),
    ]
    assert _shared_blocking_failure(cands) == set()


def test_an_advisory_failure_is_never_a_reason_to_skip_the_retry():
    """It is not a reason to reject either, so it cannot be a shared blocking cause.

    Without this the two fixes would collide: the advisory downgrade would stop the
    palette rejecting, and then the *same* advisory check would suppress the retry
    that a genuine failure had earned.
    """
    cands = [_with(_gate("palette_adherence", advisory=("palette_adherence",))) for _ in range(3)]
    assert _shared_blocking_failure(cands) == set()


def test_a_slot_that_produced_nothing_is_not_evidence():
    """An absent candidate says nothing about any check, so it cannot vote."""
    assert _shared_blocking_failure([_with(_gate("safe_area")), _with(None)]) == set()
    assert _shared_blocking_failure([]) == set()


@pytest.mark.parametrize("n", [1, 2, 5])
def test_the_shared_cause_holds_at_any_candidate_count(n):
    cands = [_with(_gate("palette_adherence", "safe_area")) for _ in range(n)]
    assert _shared_blocking_failure(cands) == {"palette_adherence", "safe_area"}


# --- ...and the retry that is not bought -------------------------------------


async def _run_with_gate(storage, governor, request, gate_for, monkeypatch):
    """Run a real job with the gate's verdict stubbed per attempt.

    Stubbed rather than provoked, because making the mock renderer fail a *chosen*
    check for every candidate means rendering it in one palette and judging it
    against another — which tests the fixture more than the pipeline. What changed
    here is the orchestration, so that is what this drives.
    """
    import adproviders as P
    import adworker.pipeline as PL
    from adworker import Pipeline

    monkeypatch.setattr(PL, "evaluate_image", gate_for)
    pipeline = Pipeline(
        storage=storage,
        governor=governor,
        image_provider=P.MockImageProvider(storage),
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
    )
    return await pipeline.run(request)


def _retries_paid_for(storage, job_id: str) -> int:
    from pathlib import Path

    return len(list((Path(storage.root) / "generations" / job_id).glob("img_*_r2.png")))


async def test_a_job_wide_failure_does_not_buy_a_second_round(
    storage, governor, make_request, monkeypatch
):
    """The $0.10 this exists to save.

    Every candidate failing the same check is one cause, not five accidents. The
    live job paid for ten Seedream generations to learn what five had already said.
    """

    def always_the_same(*args, **kwargs):
        return _gate("safe_area")

    request = make_request()
    record = await _run_with_gate(storage, governor, request, always_the_same, monkeypatch)

    assert _retries_paid_for(storage, request.job_id) == 0, "a retry was bought anyway"
    assert record.state.value == "failed"
    assert "quality gate" in (record.error or "")


async def test_candidates_failing_differently_still_get_their_retry(
    storage, governor, make_request, monkeypatch
):
    """The other half of the fix, and the one that would go unnoticed if broken.

    Skipping every retry would also have saved the $0.10, and would have quietly
    deleted a mechanism that works. A candidate that failed for its own reasons is
    exactly what a stricter prompt is for.
    """
    seen: list[int] = []

    def alternating(candidate, request, storage_, attempt, *args, **kwargs):
        seen.append(attempt)
        if attempt > 1:
            return _gate(verdict=GateVerdict.PASS)
        # Different checks per slot, so nothing is shared and nothing is structural.
        name = "safe_area" if candidate.index % 2 == 0 else "focal_clarity"
        return _gate(name)

    request = make_request()
    await _run_with_gate(storage, governor, request, alternating, monkeypatch)

    assert _retries_paid_for(storage, request.job_id) > 0, "the retry stopped happening"
    assert 2 in seen, "the gate was never told it was looking at a retry"
