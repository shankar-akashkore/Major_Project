"""Face and product identity: real cosine similarity, still not a gate.

These two checks used to be pure placeholders — ``implemented=False`` with a
reason and nothing behind it, because ArcFace and DINOv2 both need torch, and
torch does not run on this laptop (see ``adml.embeddings``). They are no longer
placeholders: given an embedding archive, ``_identity_check`` computes a real
cosine similarity with plain numpy.

What has not changed, and is the point of every test below: nothing here ever
turns into a hard pass/fail. No calibration sweep has measured where
same-identity and different-identity pairs separate for this generator, so a
threshold would be an invented number, and ``implemented`` stays ``False`` even
once a real value comes out — otherwise ``GateResult.verified_identity`` would
claim a verification this check cannot back.
"""

from __future__ import annotations

import numpy as np
import pytest
from adml.embeddings import EmbeddingSet, EmbeddingSpec
from adschema import AssetRef, GateVerdict
from adworker.gate import _identity_check, evaluate_image


def _asset(key: str, sha256: str | None = None) -> AssetRef:
    return AssetRef(key=key, url=None, mime_type="image/png", sha256=sha256)


def _archive(block_name: str, vectors: dict[str, np.ndarray]) -> dict[str, EmbeddingSet]:
    dim = next(iter(vectors.values())).shape[0]
    spec = EmbeddingSpec(name=block_name, dim=dim, source="test-stand-in")
    return {block_name: EmbeddingSet(spec, vectors)}


# --- No archive: unchanged from before this existed -------------------------


def test_with_no_archive_the_check_is_unimplemented():
    check = _identity_check("face_identity", _asset("cand.png"), _asset("ref.png"), None)
    assert not check.implemented
    assert check.passed
    assert check.value == 0.0
    assert "arcface" in check.detail


def test_with_no_reference_the_check_is_unimplemented():
    archive = _archive("arcface", {"cand.png": np.array([1.0, 0.0])})
    check = _identity_check("face_identity", _asset("cand.png"), None, archive)
    assert not check.implemented
    assert "reference image was available" in check.detail


def test_a_block_the_archive_lacks_is_unimplemented():
    """The archive exists but ran a different encoder than this check needs."""
    archive = _archive("dinov2", {"cand.png": np.array([1.0, 0.0])})
    check = _identity_check("face_identity", _asset("cand.png"), _asset("ref.png"), archive)
    assert not check.implemented
    assert "needs insightface/buffalo_l" in check.detail


def test_a_missing_vector_is_unimplemented_and_names_the_gap():
    """The encoder ran over the corpus but never saw this particular reference."""
    archive = _archive("arcface", {"cand.png": np.array([1.0, 0.0])})
    check = _identity_check("face_identity", _asset("cand.png"), _asset("ref.png"), archive)
    assert not check.implemented
    assert "missing" in check.detail


# --- Archive present: a real number, still never a verdict ------------------


def test_identical_vectors_report_cosine_one():
    vec = np.array([0.6, 0.8])
    archive = _archive("arcface", {"cand.png": vec, "ref.png": vec})
    check = _identity_check("face_identity", _asset("cand.png"), _asset("ref.png"), archive)
    assert check.value == pytest.approx(1.0)
    assert check.passed, "a real number must not veto — nothing calibrated this yet"
    assert not check.implemented, "a real value is not the same as a verified check"
    assert "0.456" not in check.detail  # sanity: no gate threshold's been smuggled in


def test_orthogonal_vectors_report_cosine_zero():
    vectors = {"cand.png": np.array([1.0, 0.0]), "ref.png": np.array([0.0, 1.0])}
    archive = _archive("dinov2", vectors)
    check = _identity_check("product_identity", _asset("cand.png"), _asset("ref.png"), archive)
    assert check.value == pytest.approx(0.0)
    assert check.passed
    assert not check.implemented


def test_the_measured_value_is_in_the_detail_for_a_human_to_read():
    vec_a, vec_b = np.array([1.0, 0.0]), np.array([0.7071, 0.7071])
    archive = _archive("arcface", {"cand.png": vec_a, "ref.png": vec_b})
    check = _identity_check("face_identity", _asset("cand.png"), _asset("ref.png"), archive)
    assert f"{check.value:.3f}" in check.detail


def test_keys_are_content_addressed_not_job_scoped():
    """The same reference photo across two jobs must hit the same vector.

    Matches the fingerprint convention in ``adworker.pipeline`` (``sha256 or
    key``) — an embedding archive is expensive to produce and should not need
    recomputing per job for a reference that has not changed.
    """
    vec = np.array([1.0, 0.0])
    archive = _archive("arcface", {"deadbeef": vec, "cand.png": vec})
    ref = _asset("uploads/job-a/human.png", sha256="deadbeef")
    check = _identity_check("face_identity", _asset("cand.png"), ref, archive)
    assert check.value == pytest.approx(1.0)


# --- Wiring: evaluate_image actually calls this ------------------------------


def test_the_gate_runs_both_identity_checks_by_name(storage, make_request):
    from adschema import (
        CameraAngle,
        Composition,
        DesignPoint,
        ImageCandidate,
        Lighting,
        MotionIntent,
        ShotBrief,
        Tier,
    )

    request = make_request()
    asset = storage.put_bytes("gate/frame.png", _blank_png())
    point = DesignPoint(
        index=0,
        angle=CameraAngle.EYE_LEVEL,
        lighting=Lighting.SOFT_DIFFUSED,
        composition=Composition.CENTERED_HERO,
        motion=MotionIntent.HERO_TURN,
        seed=1,
    )
    candidate = ImageCandidate(
        index=0,
        brief=ShotBrief(
            index=0,
            design_point=point,
            image_prompt="p",
            negative_prompt="n",
            motion_prompt="m",
            concept="c",
        ),
        asset=asset,
        tier=Tier.MOCK,
    )

    result = evaluate_image(candidate, request, storage)
    names = {c.name for c in result.checks}
    assert {"face_identity", "product_identity"} <= names
    assert result.verdict is not GateVerdict.REJECT, "unimplemented checks must not block"


def test_verified_identity_stays_false_without_an_archive(storage, make_request):
    """The default, live-job behaviour: unchanged by any of this.

    Nothing in the synchronous worker can run an encoder, so ``embeddings`` is
    ``None`` on every real job today, and ``verified_identity`` must keep saying
    so rather than claiming a verification that used a real number under the
    hood but no calibrated threshold.
    """
    from adschema import (
        CameraAngle,
        Composition,
        DesignPoint,
        ImageCandidate,
        Lighting,
        MotionIntent,
        ShotBrief,
    )

    request = make_request()
    asset = storage.put_bytes("gate/frame2.png", _blank_png())
    point = DesignPoint(
        index=0,
        angle=CameraAngle.EYE_LEVEL,
        lighting=Lighting.SOFT_DIFFUSED,
        composition=Composition.CENTERED_HERO,
        motion=MotionIntent.HERO_TURN,
        seed=1,
    )
    candidate = ImageCandidate(
        index=0,
        brief=ShotBrief(
            index=0,
            design_point=point,
            image_prompt="p",
            negative_prompt="n",
            motion_prompt="m",
            concept="c",
        ),
        asset=asset,
    )

    result = evaluate_image(candidate, request, storage, human_reference=request.human_model_image)
    assert not result.verified_identity


def test_an_archive_surfaces_a_real_value_through_the_gate(storage, make_request):
    """The one test that proves the wiring, not just the function.

    Everything above calls ``_identity_check`` directly, which proves the
    mapping and nothing about whether ``evaluate_image`` actually threads
    ``embeddings`` through — a refactor that stopped passing it would leave
    every test above green.
    """
    from adschema import (
        CameraAngle,
        Composition,
        DesignPoint,
        ImageCandidate,
        Lighting,
        MotionIntent,
        ShotBrief,
    )

    request = make_request()
    asset = storage.put_bytes("gate/frame3.png", _blank_png())
    vec = np.array([1.0, 0.0, 0.0])
    archive = _archive(
        "arcface",
        {
            asset.sha256 or asset.key: vec,
            request.human_model_image.sha256 or request.human_model_image.key: vec,
        },
    )
    point = DesignPoint(
        index=0,
        angle=CameraAngle.EYE_LEVEL,
        lighting=Lighting.SOFT_DIFFUSED,
        composition=Composition.CENTERED_HERO,
        motion=MotionIntent.HERO_TURN,
        seed=1,
    )
    candidate = ImageCandidate(
        index=0,
        brief=ShotBrief(
            index=0,
            design_point=point,
            image_prompt="p",
            negative_prompt="n",
            motion_prompt="m",
            concept="c",
        ),
        asset=asset,
    )

    result = evaluate_image(
        candidate,
        request,
        storage,
        human_reference=request.human_model_image,
        embeddings=archive,
    )
    face = next(c for c in result.checks if c.name == "face_identity")
    assert face.value == pytest.approx(1.0)
    assert not face.implemented, "still uncalibrated, even with a real archive wired in"


def _blank_png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8) + 128).save(buf, format="PNG")
    return buf.getvalue()
