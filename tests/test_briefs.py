"""The brief compiler, and specifically the scale instruction.

Everything here descends from one live job.  A user submitted a photograph of a
man and a photograph of an iPhone, and got back five frames in which the handset
ranged from "implausibly large" to "taller than the person holding it".  The
generator was not at fault: the prompt described the face, the label text, the
palette, the platform safe area and the pose, and never once said how big the
product was.  A reference-composition model handed two full-frame images has no
reason to believe they are not the same size.

Three phrases then made it worse by asking for the product to be *prominent*,
and the cheapest way to make something prominent is to draw it enormous.

These tests are the fence around that.
"""

from __future__ import annotations

import adproviders as P
import pytest
from adschema import (
    CameraAngle,
    Composition,
    DesignPoint,
    Lighting,
    MotionIntent,
    ProductScale,
    PromptStyle,
    Vertical,
)
from adworker.briefs import (
    _ANGLE_PHRASE,
    _LIGHTING_PHRASE,
    _MOTION_BEATS,
    _build_image_prompt,
    _build_negative_prompt,
    _lead,
    compile_briefs,
)


def _point(
    angle: CameraAngle = CameraAngle.EYE_LEVEL,
    composition: Composition = Composition.CENTERED_HERO,
) -> DesignPoint:
    return DesignPoint(
        index=0,
        angle=angle,
        lighting=Lighting.SOFT_DIFFUSED,
        composition=composition,
        motion=MotionIntent.HERO_TURN,
        seed=0,
    )


def test_a_known_product_size_is_stated_in_the_prompt(make_request):
    """The size has to be *said*. Nothing else in the job carries it."""
    request = make_request(vertical=Vertical.ELECTRONICS, product_name="iPhone")
    prompt = _build_image_prompt(request, _point(), "")

    assert "SCALE" in prompt
    assert "held comfortably in one hand" in prompt


def test_the_size_comes_from_the_vertical_when_it_is_not_given(make_request):
    """Beauty implies palm-sized; the user should not have to say so."""
    request = make_request(vertical=Vertical.BEAUTY)
    assert request.effective_scale is ProductScale.PALM
    assert "close a fist around" in _build_image_prompt(request, _point(), "")


def test_an_explicit_size_overrides_the_vertical(make_request):
    """A tripod is 'electronics' and is not hand-held. The override is the escape."""
    request = make_request(vertical=Vertical.ELECTRONICS, product_scale=ProductScale.FLOOR_STANDING)
    prompt = _build_image_prompt(request, _point(), "")

    assert "resting on the floor" in prompt
    assert "held comfortably in one hand" not in prompt


def test_an_unknown_size_still_forbids_enlarging_the_product(make_request):
    """The half of the instruction that does not need a measurement.

    This is the case the live job was in — vertical ``other``, no size given. It
    would be easy to conclude there is nothing to say here and emit no SCALE block
    at all, but the failure was never a wrong measurement. It was that three
    separate phrases asked for prominence and none of them said where prominence
    comes from. That sentence is just as true when the size is unknown, and it is
    the sentence that does the work.
    """
    request = make_request(vertical=Vertical.OTHER)
    assert request.effective_scale is None

    prompt = _build_image_prompt(request, _point(), "")
    assert "SCALE" in prompt
    assert "Do not enlarge the product." in prompt
    assert "true physical size relative to the model's hands and body" in prompt
    # ...but it must not invent a size it was never told.
    assert "held comfortably in one hand" not in prompt


@pytest.mark.parametrize("vertical", list(Vertical))
def test_every_vertical_gets_the_no_enlarging_rule(make_request, vertical):
    """No category may be exempt — the rule costs nothing and the bug was silent."""
    request = make_request(vertical=vertical)
    assert "Do not enlarge the product." in _build_image_prompt(request, _point(), "")


def test_prominence_is_never_requested_as_size(make_request):
    """The two design levels that used to ask for a bigger product.

    ``close_up_product`` said "tight close-up centred on the product" and
    ``product_foreground`` said "product prominent in the foreground". Both are
    legitimate art direction and both were read by the generator as *scale it up*.
    They now name the camera and perspective as the mechanism instead, which is
    how a photographer would actually achieve either shot.
    """
    request = make_request()
    for angle, composition in [
        (CameraAngle.CLOSE_UP_PRODUCT, Composition.CENTERED_HERO),
        (CameraAngle.EYE_LEVEL, Composition.PRODUCT_FOREGROUND),
    ]:
        prompt = _build_image_prompt(request, _point(angle, composition), "")
        assert "product prominent in the foreground" not in prompt
        assert "tight close-up centred on the product" not in prompt


def test_a_hand_across_the_product_is_allowed(make_request):
    """The constraint that made gigantism the cheapest way out.

    "The product must be fully visible and unobstructed" is unsatisfiable for a
    phone held in a hand — fingers obstruct it, that is what holding is. Given an
    impossible constraint the model found the loophole: draw the product too big
    for a hand to cover. The constraint now asks for the product to be *readable*,
    and says outright that a grip is expected.
    """
    prompt = _build_image_prompt(make_request(), _point(), "")

    assert "fully visible and unobstructed" not in prompt
    assert "natural hand grip" in prompt
    assert "do not resize, float or reposition" in prompt


def test_scale_failures_are_in_the_negative_prompt(make_request):
    """Belt and braces: the positive prompt asks, the negative forbids.

    Listed separately from the artefact negatives because they are not artefacts.
    Every frame of the failing job was clean, well-lit and anatomically sound. It
    passed all five implemented gate checks. It was simply the wrong size.
    """
    negatives = _build_negative_prompt(make_request())

    assert "product out of scale" in negatives
    assert "giant oversized product" in negatives
    assert "miniature person" in negatives
    # The artefact negatives are still there.
    assert "distorted face" in negatives


# --- Prompt verbosity ------------------------------------------------------
#
# A competing implementation reached visibly better composition on the same two
# reference photographs with a 37-word prompt, against our 343. Two of its five
# lines — "make the person naturally hold or use the product" and "match
# lighting, shadows, scale and perspective" — did in eleven words what our
# version was not doing at all. Whether the rest of the gap is verbosity or the
# model is a measurable question, so verbosity became a switch.


async def test_the_compact_prompt_is_dramatically_shorter(make_request):
    """The point of the exercise, stated as a number."""
    request = make_request()

    full = await compile_briefs(request, P.MockLLMProvider(), PromptStyle.FULL)
    compact = await compile_briefs(request, P.MockLLMProvider(), PromptStyle.COMPACT)

    full_words = sum(len(b.image_prompt.split()) for b in full.briefs) / len(full.briefs)
    compact_words = sum(len(b.image_prompt.split()) for b in compact.briefs) / len(compact.briefs)

    assert compact_words < full_words * 0.5


async def test_the_compact_prompt_keeps_every_design_axis(make_request):
    """Shortening must not become "generate one safe image".

    The competing implementation's prompt is 37 words because it carries no
    design point at all — one fixed brief, one image, no candidates. Ours cannot
    go that low without deleting the multi-candidate claim, and a comparison
    between a five-candidate system and a one-candidate one would not be about
    verbosity any more.
    """
    request = make_request()
    briefs = await compile_briefs(request, P.MockLLMProvider(), PromptStyle.COMPACT)

    for brief in briefs.briefs:
        dp = brief.design_point
        prompt = brief.image_prompt.lower()
        # Each axis has to be recognisable in the text, not merely in the record.
        assert _lead(_LIGHTING_PHRASE[dp.lighting]).lower() in prompt
        assert _MOTION_BEATS[dp.motion].opening.lower() in prompt
        assert dp.angle is CameraAngle.CLOSE_UP_PRODUCT or (
            _lead(_ANGLE_PHRASE[dp.angle]).lower() in prompt
        )


async def test_the_compact_prompt_still_forbids_enlarging_the_product(make_request):
    """The scale fix must survive the compression, including in the risky levels.

    ``close_up_product`` and ``product_foreground`` carry their scale-protective
    clause *after* the first comma, and compacting by taking the leading clause
    would silently drop exactly the two qualifications that matter most. They get
    explicit short forms for that reason.
    """
    request = make_request(vertical=Vertical.ELECTRONICS)
    briefs = await compile_briefs(request, P.MockLLMProvider(), PromptStyle.COMPACT)

    for brief in briefs.briefs:
        assert "never enlarge it for emphasis" in brief.image_prompt
        assert "held comfortably in one hand" in brief.image_prompt
        if brief.design_point.angle is CameraAngle.CLOSE_UP_PRODUCT:
            assert "hands in shot for scale" in brief.image_prompt
        if brief.design_point.composition is Composition.PRODUCT_FOREGROUND:
            assert "larger only by perspective" in brief.image_prompt


async def test_the_style_is_recorded_on_the_brief_set(make_request):
    """Provenance, so an ablation is attributable rather than reconstructed."""
    for style in PromptStyle:
        briefs = await compile_briefs(make_request(), P.MockLLMProvider(), style)
        assert briefs.prompt_style == style.value
