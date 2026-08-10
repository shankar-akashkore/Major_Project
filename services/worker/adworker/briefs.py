"""Brief compilation: design point → prompt-ready shot brief.

The division of labour here is deliberate.

A **template** builds the structural parts of the prompt: the explicit reference
roles, the camera angle, the lighting, the composition, the palette, the safe-area
constraint.  These are the parts that must not be dropped or paraphrased — naming
the references positionally ("Image 1 is the model … preserve her face exactly")
is the single most effective thing that stops identity drift, and an LLM asked to
be creative will happily rewrite it away.

An **LLM** then optionally enriches only the scene description and the one-line
concept.  If the LLM is unavailable, returns malformed JSON, or is the mock, the
template output stands on its own and the job proceeds.  That fallback is not an
edge case — it is the path every mock run takes, so it is continuously tested.
"""

from __future__ import annotations

import json

from adproviders import LLMProvider
from adschema import (
    AdJobRequest,
    BackgroundTreatment,
    BriefSet,
    CameraAngle,
    Composition,
    DesignPoint,
    Lighting,
    Mood,
    MotionIntent,
    ShotBrief,
)

from .sampler import attach_diversity, sample_design_points

# --- Vocabulary: enum level → prompt language -------------------------------

_ANGLE_PHRASE: dict[CameraAngle, str] = {
    CameraAngle.EYE_LEVEL: "shot at eye level, camera square to the subject",
    CameraAngle.LOW_ANGLE: "shot from a low angle looking slightly up, making the subject dominant",
    CameraAngle.HIGH_ANGLE: "shot from a slightly high angle looking down",
    CameraAngle.THREE_QUARTER: "three-quarter view, subject turned about 45 degrees to camera",
    CameraAngle.PROFILE: "side profile view of the subject",
    CameraAngle.CLOSE_UP_PRODUCT: "tight close-up centred on the product, subject partially framed",
}

_LIGHTING_PHRASE: dict[Lighting, str] = {
    Lighting.SOFT_DIFFUSED: "soft diffused studio lighting, gentle even shadows",
    Lighting.HARD_DIRECTIONAL: "hard directional key light, crisp defined shadows, high contrast",
    Lighting.RIM_BACKLIT: (
        "rim backlighting separating the subject from the background, glowing edges"
    ),
    Lighting.GOLDEN_HOUR: "warm golden-hour light, long soft shadows, amber cast",
    Lighting.HIGH_KEY: "bright high-key lighting, minimal shadow, airy and clean",
}

_COMPOSITION_PHRASE: dict[Composition, str] = {
    Composition.CENTERED_HERO: "subject centred, symmetrical hero framing",
    Composition.RULE_OF_THIRDS_LEFT: "subject on the left third, open space to the right",
    Composition.RULE_OF_THIRDS_RIGHT: "subject on the right third, open space to the left",
    Composition.NEGATIVE_SPACE_TOP: (
        "subject low in frame with generous negative space above for text"
    ),
    Composition.PRODUCT_FOREGROUND: "product prominent in the foreground, subject softly behind it",
}

_BACKGROUND_PHRASE: dict[BackgroundTreatment, str] = {
    BackgroundTreatment.STUDIO_WHITE: "clean seamless white studio backdrop",
    BackgroundTreatment.SEAMLESS_COLOR: "seamless solid colour backdrop in {colour}",
    BackgroundTreatment.SOFT_GRADIENT: "smooth soft colour gradient backdrop",
    BackgroundTreatment.LIFESTYLE_SCENE: "tasteful lifestyle interior setting, softly out of focus",
    BackgroundTreatment.OUTDOOR_NATURAL: "natural outdoor setting with shallow depth of field",
}

_MOOD_PHRASE: dict[Mood, str] = {
    Mood.CALM_PREMIUM: "calm, premium, understated and expensive-looking",
    Mood.WARM_LIFESTYLE: "warm, inviting, natural and relatable",
    Mood.BOLD_CONFIDENT: "bold, confident, striking and graphic",
    Mood.HIGH_ENERGY: "high-energy, dynamic, youthful and vivid",
}

_MOTION_PHRASE: dict[MotionIntent, str] = {
    MotionIntent.SLOW_DOLLY_IN: "slow smooth dolly in toward the subject",
    MotionIntent.SLOW_DOLLY_OUT: "slow smooth dolly out revealing more of the scene",
    MotionIntent.ORBIT_LEFT: "gentle orbit of the camera to the left around the subject",
    MotionIntent.PRODUCT_PRESENT: (
        "the model raises and turns the product toward the camera, presenting it clearly"
    ),
    MotionIntent.HANDHELD_DRIFT: "subtle handheld camera drift, natural and unpolished",
    MotionIntent.STATIC_SUBTLE: "near-static frame with only subtle natural movement",
}

#: Baked into every negative prompt.  These are the failure modes that make an ad
#: frame unusable regardless of taste, so they are not left to the user to think of.
_BASE_NEGATIVES = (
    "distorted face, extra fingers, malformed hands, warped product label, "
    "duplicated product, unreadable text, watermark, logo artefacts, "
    "blurry product, cropped product"
)


def _reference_roles(request: AdJobRequest) -> list[str]:
    """Ordered reference descriptions.  Order is load-bearing — the prompt refers
    to these positionally, so it must match the order references are uploaded in."""
    roles = ["the human model", "the product"]
    if request.logo_image is not None:
        roles.append("the brand logo")
    return roles


def _build_image_prompt(request: AdJobRequest, dp: DesignPoint, scene: str) -> str:
    """Assemble the composition prompt.

    Structure over prose: identity instructions first (they matter most and
    models weight early tokens more heavily), then the shot, then the look.
    """
    palette = (
        ", ".join(request.theme.palette) if request.theme.palette else "the product's own colours"
    )
    background = _BACKGROUND_PHRASE[request.theme.background].format(
        colour=request.theme.background_color or "#f2f2f2"
    )

    lines = [
        f"Advertising photograph for {request.product_name}.",
        "",
        "REFERENCES — preserve these exactly:",
        "Image 1 is the human model. Preserve the model's face, skin tone, hair and body "
        "proportions exactly as shown. Do not restyle the face.",
        "Image 2 is the product. Preserve its exact shape, colour, proportions, label "
        "text and finish. The product must be immediately recognisable as the same item.",
    ]
    if request.logo_image is not None:
        lines.append(
            "Image 3 is the brand logo. Reproduce it cleanly and unaltered; do not "
            "distort, recolour or regenerate the letterforms."
        )

    lines += [
        "",
        "SHOT:",
        f"- {_ANGLE_PHRASE[dp.angle]}",
        f"- {_LIGHTING_PHRASE[dp.lighting]}",
        f"- {_COMPOSITION_PHRASE[dp.composition]}",
        f"- {background}",
        "",
        "LOOK:",
        f"- Mood: {_MOOD_PHRASE[request.mood]}",
        f"- Colour palette: {palette}",
        f"- Format: {request.aspect_ratio.value} vertical-safe framing for "
        f"{request.platform.value.replace('_', ' ')}",
    ]

    if scene:
        lines += ["", f"SCENE: {scene}"]

    safe = request.platform.safe_area
    lines += [
        "",
        "CONSTRAINTS:",
        f"- Keep the product and the model's face clear of the top {safe.top:.0%} and "
        f"bottom {safe.bottom:.0%} of the frame, which platform UI covers.",
        "- The product must be fully visible and unobstructed.",
        "- Photorealistic commercial photography. No text overlays, captions or "
        "graphic elements — those are composited later.",
    ]

    # The user writes negative constraints as prohibitions ("no hands covering the
    # label"). Those belong here, in natural language, rather than appended to the
    # comma-separated negative prompt — "avoid: no hands covering the label" is a
    # double negative that can invert the user's intent.
    if request.negative_constraints:
        lines.append(f"- Additionally: {request.negative_constraints}")

    if request.additional_prompt:
        lines += ["", f"ADDITIONAL DIRECTION: {request.additional_prompt}"]

    return "\n".join(lines)


def _build_negative_prompt(request: AdJobRequest) -> str:
    """Keyword-style negatives only.

    Kept free of the user's free-text constraints on purpose — see the note in
    :func:`_build_image_prompt`. This field is a list of artefacts to suppress,
    not a place for sentences.
    """
    return _BASE_NEGATIVES


def _build_motion_prompt(request: AdJobRequest, dp: DesignPoint) -> str:
    return (
        f"{_MOTION_PHRASE[dp.motion]}. "
        f"The product stays clearly visible and in focus throughout. "
        f"Preserve the model's identity and the product's exact appearance across every frame. "
        f"Overall feel: {_MOOD_PHRASE[request.mood]}. "
        f"Duration {request.duration_seconds:.0f} seconds, single continuous shot, no cuts."
    )


def _template_concept(request: AdJobRequest, dp: DesignPoint) -> str:
    return (
        f"{_ANGLE_PHRASE[dp.angle].split(',')[0].capitalize()} with "
        f"{_LIGHTING_PHRASE[dp.lighting].split(',')[0]}, "
        f"{_COMPOSITION_PHRASE[dp.composition].split(',')[0]}."
    )


# --- LLM enrichment ---------------------------------------------------------

_LLM_SYSTEM = (
    "You are an advertising art director. You will be given a product, an audience "
    "and a fixed shot specification (camera angle, lighting, composition). The shot "
    "specification is already decided — do not change it or comment on it. Your only "
    "job is to invent the setting and a one-line creative concept that suit it."
)


def _llm_user_prompt(request: AdJobRequest, points: list[DesignPoint]) -> str:
    shots = [
        {
            "index": dp.index,
            "angle": dp.angle.value,
            "lighting": dp.lighting.value,
            "composition": dp.composition.value,
        }
        for dp in points
    ]
    return json.dumps(
        {
            "product_name": request.product_name,
            "vertical": request.vertical.value,
            "caption": request.caption,
            "mood": request.mood.value,
            "audience": request.audience.model_dump(),
            "platform": request.platform.value,
            "background_treatment": request.theme.background.value,
            "additional_direction": request.additional_prompt,
            "shots": shots,
        },
        indent=2,
    )


_LLM_SCHEMA_HINT = (
    '{"scenes": [{"index": 0, "scene": "one or two sentences describing the setting", '
    '"concept": "a short creative one-liner"}]}'
)


async def _enrich(
    llm: LLMProvider, request: AdJobRequest, points: list[DesignPoint]
) -> dict[int, dict[str, str]]:
    """Ask the LLM for scene text.  Any failure degrades to the template.

    Deliberately broad exception handling: brief enrichment is a nice-to-have,
    and a flaky LLM call must never take down a job that has already been paid
    for downstream.
    """
    try:
        raw = await llm.complete_json(
            _LLM_SYSTEM, _llm_user_prompt(request, points), _LLM_SCHEMA_HINT
        )
    except Exception:
        return {}

    out: dict[int, dict[str, str]] = {}
    for item in (raw or {}).get("scenes", []):
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if not isinstance(idx, int):
            continue
        out[idx] = {
            "scene": str(item.get("scene", ""))[:400],
            "concept": str(item.get("concept", ""))[:200],
        }
    return out


# --- Public entry point -----------------------------------------------------


async def compile_briefs(request: AdJobRequest, llm: LLMProvider) -> BriefSet:
    """Produce one :class:`ShotBrief` per candidate slot."""
    points = sample_design_points(
        request.candidate_count,
        mood=request.mood,
        vertical=request.vertical,
        seed=request.seed,
        platform=request.platform,
        locked_angle=request.locked_angle,
    )
    enrichment = await _enrich(llm, request, points)

    briefs = []
    for dp in points:
        extra = enrichment.get(dp.index, {})
        scene = extra.get("scene", "")
        briefs.append(
            ShotBrief(
                index=dp.index,
                design_point=dp,
                image_prompt=_build_image_prompt(request, dp, scene),
                negative_prompt=_build_negative_prompt(request),
                motion_prompt=_build_motion_prompt(request, dp),
                concept=extra.get("concept") or _template_concept(request, dp),
            )
        )

    return attach_diversity(
        BriefSet(
            job_id=request.job_id,
            briefs=briefs,
            sampler="latin_hypercube",
        )
    )
