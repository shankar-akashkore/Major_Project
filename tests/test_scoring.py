"""Image-stage scoring, and the component that was pointing the wrong way.

``product_salience`` is not a product measurement.  It is
:func:`adml.features.focal_concentration` — the share of the saliency map held by
its most salient tenth — and for a composite of a person and a product that rises
with the *size of the product*.  Attributing attention to the product specifically
needs the product located in the frame, which is the pending ``product_identity``
work; until then this is a proxy, and it carried a quarter of the image score.

The first live iPhone job is the evidence.  Its three scored candidates measured
0.45, 0.40 and 0.32, in exactly descending order of how oversized the handset was,
and the ranker promoted the two worst frames and paid to animate them.  The most
physically plausible image on the page came last.

The fix is not to delete the feature — "does this frame have a clear subject at
all" is worth something — but to stop an unbounded reward for concentration
acting as an unbounded reward for drawing the product bigger.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from adschema import (
    CameraAngle,
    Composition,
    DesignPoint,
    ImageCandidate,
    Lighting,
    MotionIntent,
    ShotBrief,
)
from adworker.scoring import _FOCUS_SWEET_SPOT, IMAGE_WEIGHTS, _band_score, score_image
from PIL import Image

#: Concentration values measured off real Seedream frames: 0.332-0.377 in the job
#: recorded under fixtures/evidence, 0.32-0.45 in the iPhone job. The mock
#: generator draws a synthetic hero blob and sits far higher, at 0.58-0.73, which
#: is why a *ceiling* on this number cannot serve as a scale check: any ceiling
#: loose enough to pass the mock corpus passes every real frame ever generated.
REAL_FRAME_RANGE = (0.32, 0.45)


def _frame_with_blob(fraction: float, size: int = 512) -> bytes:
    """A dark frame containing one bright square covering ``fraction`` of the area.

    A crude stand-in for "product occupies this much of the shot", which is
    exactly the axis the feature must stop rewarding.
    """
    rgb = np.full((size, size, 3), 40, dtype=np.uint8)
    side = max(1, int(round(size * fraction**0.5)))
    top = (size - side) // 2
    rgb[top : top + side, top : top + side] = 235

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _candidate(storage, index: int, fraction: float) -> ImageCandidate:
    asset = storage.put_bytes(f"score/{index}.png", _frame_with_blob(fraction))
    point = DesignPoint(
        index=index,
        angle=CameraAngle.EYE_LEVEL,
        lighting=Lighting.SOFT_DIFFUSED,
        composition=Composition.CENTERED_HERO,
        motion=MotionIntent.HERO_TURN,
        seed=index,
    )
    return ImageCandidate(
        index=index,
        brief=ShotBrief(
            index=index,
            design_point=point,
            image_prompt="p",
            negative_prompt="n",
            motion_prompt="m",
            concept="c",
        ),
        asset=asset,
    )


def test_the_measured_values_from_the_failing_job_map_to_one_score():
    """The regression, using the numbers the live job actually produced.

    Deliberately *not* tested by synthesising a big bright rectangle and a small
    one. Saliency tracks edges and texture rather than area, so a larger flat
    block does not concentrate attention the way a larger photographed product
    does — a synthetic version of this test passes against the broken scoring and
    proves nothing. These three values are measurements off real Seedream frames,
    recorded in the job the user reported, ordered by how oversized the handset
    was in each.

    This exercises the mapping, not the wiring — it reads ``_band_score`` directly,
    so it would still pass if ``score_image`` stopped calling it. That the scorer
    actually uses it is what ``test_the_component_is_flat_across_plausible_frames``
    below is for; the two are only sound together.
    """
    biggest_phone, middling, correct_size = 0.45, 0.40, 0.32

    scores = [_band_score(v, *_FOCUS_SWEET_SPOT) for v in (biggest_phone, middling, correct_size)]
    assert len(set(scores)) == 1, "size must not be a gradient across real frames"

    # And the contribution it makes to the blended score is now negligible either
    # way. Under the old weights the spread between these three was 0.0325 of the
    # final score — enough to reorder a candidate set, and it did.
    contributions = [IMAGE_WEIGHTS["product_salience"] * s for s in scores]
    assert max(contributions) - min(contributions) == pytest.approx(0.0)


@pytest.mark.parametrize("fraction", [0.06, 0.10, 0.20, 0.35, 0.55])
def test_the_component_is_flat_across_plausible_frames(storage, make_request, fraction):
    """Flat, not merely non-increasing.

    A feature that still crept upward would keep the same bias, just slower. Over
    the range real generations occupy there must be no gradient at all — any
    ordering between candidates has to come from the components that are not
    proxies for object size.
    """
    request = make_request()
    score = score_image(_candidate(storage, 9, fraction), request, storage)
    assert score.product_salience == pytest.approx(1.0)


def test_smeared_attention_is_still_penalised(storage, make_request):
    """The floor survives — banding it must not make it toothless.

    A frame with no focal subject at all is a real defect: the viewer cannot tell
    what is being sold. That is what this component was always for, and it is the
    only thing it can honestly claim to measure.

    This one guards the floor rather than the fix — it passes against both the old
    and the new scoring, and is here so that a future attempt to neutralise this
    feature entirely has something to trip over.
    """
    noise = (np.random.default_rng(0).random((512, 512, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(noise).save(buf, format="PNG")
    asset = storage.put_bytes("score/noise.png", buf.getvalue())

    candidate = _candidate(storage, 5, 0.2)
    candidate.asset = asset
    score = score_image(candidate, make_request(), storage)

    assert score.product_salience is not None
    assert score.product_salience < 0.9


def test_the_size_proxy_is_no_longer_a_quarter_of_the_score():
    """A proxy this weak should not have outweighed safe-area compliance.

    Kept as an explicit assertion rather than left to the weights dict, because
    the number is the finding: at 0.25 this feature could reorder a candidate set
    on its own, and it did.
    """
    assert IMAGE_WEIGHTS["product_salience"] <= 0.10
    assert IMAGE_WEIGHTS["product_salience"] < IMAGE_WEIGHTS["aesthetic"]
    assert IMAGE_WEIGHTS["product_salience"] < IMAGE_WEIGHTS["safe_area_compliance"]
    assert sum(IMAGE_WEIGHTS.values()) == pytest.approx(1.0)
