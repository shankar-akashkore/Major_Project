"""The quality gate's scale ceiling, and what it can honestly claim to measure.

Every other threshold in :mod:`adworker.gate` is a floor calibrated against the
mock renderer.  ``product_scale`` is neither: it is a ceiling, and it is the only
one calibrated against real generations, because it exists to catch a failure the
mock corpus cannot produce — a product drawn far larger than the person holding
it, which both of the first two live jobs returned.

The measurement is colour, not detection, and the tests below are as interested
in the cases where it declines to answer as in the ones where it fires.  A check
that quietly passed every white trainer and black phone would be claiming to have
verified scale on exactly the products it cannot see.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from adml import features as F
from adschema import AssetRef, GateResult, GateVerdict, Tier
from adworker.gate import (
    THRESH_PRODUCT_AREA,
    UNDISCRIMINATIVE_SHARE,
    _product_scale_check,
    stricter_prompt,
)
from PIL import Image

#: The measurements the threshold was set from — eleven frames, two live Seedream
#: jobs, labelled by looking at them.  Kept here so the calibration is pinned by a
#: test rather than living only in a comment that can drift from the constant.
MEASURED_LEGITIMATE = (0.0114, 0.0131, 0.0282, 0.0361, 0.0840, 0.0923, 0.0941, 0.1133)
MEASURED_OVERSIZED = (0.1723, 0.2850, 0.2938)


def _png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _cutout(colour: tuple[int, int, int], size: int = 64) -> bytes:
    """A product cutout: a solid colour on a fully transparent surround."""
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[8:-8, 8:-8, :3] = colour
    rgba[8:-8, 8:-8, 3] = 255
    return _png(rgba)


def _frame(colour: tuple[int, int, int], fraction: float, size: int = 512) -> bytes:
    """A neutral frame with ``fraction`` of its area painted the product's colour."""
    rgb = np.full((size, size, 3), 128, dtype=np.uint8)
    side = int(round(size * fraction**0.5))
    rgb[:side, :side] = colour
    return _png(rgb)


class _Store:
    """Just enough Storage for the check: it only ever reads the reference."""

    def __init__(self, blobs: dict[str, bytes]) -> None:
        self._blobs = blobs

    def get_bytes(self, key: str) -> bytes:
        return self._blobs[key]


def _check(product_png: bytes | None, frame_png: bytes, tier: Tier = Tier.PREMIUM):
    store = _Store({"p.png": product_png} if product_png else {})
    ref = AssetRef(key="p.png", url=None, mime_type="image/png") if product_png else None
    return _product_scale_check(F.load_image(frame_png), ref, store, tier)


TEAL = (20, 140, 150)


# --- The ceiling ----------------------------------------------------------


def test_a_product_that_swallows_the_frame_is_rejected():
    check = _check(_cutout(TEAL), _frame(TEAL, 0.25))
    assert not check.passed
    assert check.implemented
    assert check.higher_is_better is False, "this is a ceiling, and the UI has to say so"
    assert check.value > THRESH_PRODUCT_AREA


def test_a_product_at_a_plausible_size_passes():
    check = _check(_cutout(TEAL), _frame(TEAL, 0.05))
    assert check.passed
    assert check.implemented
    assert check.value < THRESH_PRODUCT_AREA


def test_the_threshold_separates_the_two_measured_classes():
    """Pins the calibration to the frames it was derived from.

    Moving the constant without re-measuring fails here, which is the point: the
    number is only defensible while it still sits in the gap those eleven frames
    left.
    """
    assert max(MEASURED_LEGITIMATE) < THRESH_PRODUCT_AREA < min(MEASURED_OVERSIZED)


def test_the_ceiling_favours_the_legitimate_class():
    """Set nearer the oversized class, deliberately.

    A false reject costs a paid retry on a usable frame; a miss costs a frame that
    is merely ranked.  The module's floors are all placed by that same asymmetry
    and the ceiling has to be placed by it too, or it quietly becomes the most
    expensive check here.
    """
    headroom = THRESH_PRODUCT_AREA - max(MEASURED_LEGITIMATE)
    margin = min(MEASURED_OVERSIZED) - THRESH_PRODUCT_AREA
    assert headroom > margin


@pytest.mark.parametrize("fraction", [0.02, 0.05, 0.09])
def test_plausible_sizes_are_not_a_gradient_toward_rejection(fraction):
    """Nothing in the legitimate range approaches the ceiling."""
    assert _check(_cutout(TEAL), _frame(TEAL, fraction)).value < THRESH_PRODUCT_AREA


# --- What it will not claim ------------------------------------------------


def test_a_neutral_product_is_reported_as_unmeasurable_not_as_passing():
    """A white trainer or a black phone has no colour to be found by.

    This is the common case, not an edge case, and the check has to be visibly
    distinguishable from one that genuinely measured something — otherwise the
    write-up claims a scale check that never ran.
    """
    check = _check(_cutout((242, 242, 242)), _frame((242, 242, 242), 0.60))
    assert check.passed, "an unmeasurable frame must not be failed"
    assert not check.implemented
    assert "chromatic" in check.detail


def test_a_missing_product_reference_does_not_fail_the_frame():
    check = _check(None, _frame(TEAL, 0.25))
    assert check.passed
    assert not check.implemented


def test_an_unreadable_reference_degrades_rather_than_raising():
    """A missing blob is a misconfiguration, not a reason to fail a paid frame."""
    store = _Store({})
    ref = AssetRef(key="absent.png", url=None, mime_type="image/png")
    check = _product_scale_check(F.load_image(_frame(TEAL, 0.25)), ref, store, Tier.PREMIUM)
    assert check.passed
    assert not check.implemented


def test_an_unmeasurable_check_does_not_count_as_a_gate_failure():
    """The pending checks pass, so a frame is not rejected for being unmeasurable."""
    check = _check(None, _frame(TEAL, 0.25))
    result = GateResult(verdict=GateVerdict.PASS, checks=[check], attempt=1)
    assert result.failures == []
    assert check in result.pending_checks


# --- The retry -------------------------------------------------------------


def test_the_retry_prompt_names_the_scale_problem():
    """A retry that repeats the prompt is a wasted paid generation."""
    failed = GateResult(
        verdict=GateVerdict.RETRY,
        attempt=1,
        checks=[
            _product_scale_check(
                F.load_image(_frame(TEAL, 0.25)),
                AssetRef(key="p.png", url=None, mime_type="image/png"),
                _Store({"p.png": _cutout(TEAL)}),
                Tier.PREMIUM,
            )
        ],
    )
    amended = stricter_prompt("shoot the thing", failed)
    assert "true physical size" in amended
    assert "never enlarged to fill the frame" in amended
    assert amended.startswith("shoot the thing")


def test_a_passing_scale_check_leaves_the_prompt_alone():
    passing = GateResult(
        verdict=GateVerdict.PASS,
        attempt=1,
        checks=[
            _product_scale_check(
                F.load_image(_frame(TEAL, 0.03)),
                AssetRef(key="p.png", url=None, mime_type="image/png"),
                _Store({"p.png": _cutout(TEAL)}),
                Tier.PREMIUM,
            )
        ],
    )
    assert stricter_prompt("shoot the thing", passing) == "shoot the thing"


# --- Wiring ----------------------------------------------------------------


def test_the_scale_check_is_actually_run_by_the_gate(storage, make_request):
    """Everything above reads ``_product_scale_check`` directly.

    That proves the mapping and nothing about the wiring — a refactor that stopped
    calling it would leave every test above green. This is the one that fails.
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
    from adworker.gate import evaluate_image

    request = make_request()
    asset = storage.put_bytes("gate/frame.png", _frame(TEAL, 0.25))
    product = storage.put_bytes("gate/product.png", _cutout(TEAL))
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
        tier=Tier.PREMIUM,
    )

    result = evaluate_image(candidate, request, storage, 1, product)
    names = {c.name for c in result.checks}
    assert "product_scale" in names, "the gate must run the scale check"
    assert "product_scale" in {c.name for c in result.failures}
    assert result.verdict is GateVerdict.RETRY


def test_the_gate_still_works_without_a_product_reference(storage, make_request):
    """The parameter is optional, and omitting it must not fail a job."""
    from adschema import (
        CameraAngle,
        Composition,
        DesignPoint,
        ImageCandidate,
        Lighting,
        MotionIntent,
        ShotBrief,
    )
    from adworker.gate import evaluate_image

    request = make_request()
    asset = storage.put_bytes("gate/frame2.png", _frame(TEAL, 0.02))
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

    result = evaluate_image(candidate, request, storage)
    scale = next(c for c in result.checks if c.name == "product_scale")
    assert not scale.implemented
    assert scale.passed


def test_a_product_photograph_is_refused_where_a_cutout_would_be_measured():
    """The commonest undefined case, and the one that nearly shipped as a bug.

    ``product_reference`` hands over the original upload when rembg produced
    nothing usable. A product photograph's dominant colours are its backdrop's as
    much as the product's, so its signature matches most of any frame — measured,
    59% of a frame whose product covered 3% of it. Declining is the only honest
    answer; failing the frame would reject good work for the intake stage's
    shortcoming, and passing it silently would claim a check that never ran.
    """
    opaque = np.zeros((64, 64, 4), dtype=np.uint8)
    opaque[:, :, :3] = TEAL
    opaque[:, :, 3] = 255  # a photograph: nothing was cut away

    photograph = _check(_png(opaque), _frame(TEAL, 0.25))
    assert photograph.passed
    assert not photograph.implemented

    # The same product, actually cut out, is measured and rejected — so the refusal
    # is about the missing cutout, not about the number being inconvenient.
    assert not _check(_cutout(TEAL), _frame(TEAL, 0.25)).passed


def test_a_colour_that_covers_the_frame_is_declined_rather_than_rejected():
    """Past a point the number stops being about the product.

    The measurement finds the product by colour and cannot tell it from anything
    else wearing the same colour. The mock pipeline makes that unmissable — intake
    reads the brand palette off the product cutout and the renderer then paints
    the backdrop from it — and the right answer there is "could not measure", not
    "your product is enormous".
    """
    check = _check(_cutout(TEAL), _frame(TEAL, 0.70))
    assert check.value > UNDISCRIMINATIVE_SHARE
    assert check.passed, "declining must not fail the frame"
    assert not check.implemented
    assert "matching the backdrop" in check.detail


def test_the_decline_band_sits_above_every_real_failure():
    """A genuinely oversized frame must never fall into the decline band.

    Real oversized frames reach 0.294; if the band started below that the check
    would excuse the exact failures it exists to catch.
    """
    assert max(MEASURED_OVERSIZED) < UNDISCRIMINATIVE_SHARE


def test_a_synthetic_frame_is_not_measured_for_scale():
    """The mock closes a loop no real generator closes.

    Intake reads the brand palette off the product cutout; the mock renderer then
    paints backdrop, wardrobe and product from that palette. The reading is
    meaningless across its whole range rather than merely high, and which end a
    given seed lands on is arbitrary — so this cannot be left to the decline band
    without the dev pipeline rejecting candidates by seed.
    """
    check = _check(_cutout(TEAL), _frame(TEAL, 0.25), tier=Tier.MOCK)
    assert check.passed
    assert not check.implemented
    assert "mock renderer" in check.detail

    # The same frame from a paid tier is still rejected, so the exclusion is about
    # the renderer rather than about the number being inconvenient.
    assert not _check(_cutout(TEAL), _frame(TEAL, 0.25), tier=Tier.PREMIUM).passed
