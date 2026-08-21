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
from dataclasses import dataclass

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
    ProductScale,
    PromptStyle,
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
    CameraAngle.CLOSE_UP_PRODUCT: (
        "tight close-up with the camera moved near the product, the model's hand kept "
        "in shot at matching scale so the product's true size stays readable"
    ),
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
    Composition.PRODUCT_FOREGROUND: (
        "product in the near foreground, larger only through perspective, with the model "
        "a little softer behind it — the size relationship must stay physically possible"
    ),
}

_BACKGROUND_PHRASE: dict[BackgroundTreatment, str] = {
    BackgroundTreatment.STUDIO_WHITE: "clean seamless white studio backdrop",
    BackgroundTreatment.SEAMLESS_COLOR: "seamless solid colour backdrop in {colour}",
    BackgroundTreatment.SOFT_GRADIENT: "smooth soft colour gradient backdrop",
    BackgroundTreatment.LIFESTYLE_SCENE: "tasteful lifestyle interior setting, softly out of focus",
    BackgroundTreatment.OUTDOOR_NATURAL: "natural outdoor setting with shallow depth of field",
}

#: Physical size, in the only terms a generator can act on.
#:
#: A measurement in millimetres is not renderable — "147 mm" constrains nothing a
#: diffusion model can check.  A *body relation* is, because the model is already
#: drawing the body: it can compare an object to a hand.  So every level names the
#: hand or the frame it fits, and the numbers that appear are approximations meant
#: to anchor the relation rather than to be measured against.
_SCALE_PHRASE: dict[ProductScale, str] = {
    ProductScale.PALM: (
        "small enough to close a fist around — roughly the length of a finger. It should "
        "read as a small object, handled delicately"
    ),
    ProductScale.ONE_HAND: (
        "held comfortably in one hand — roughly the length of a hand from wrist to "
        "fingertip. A phone, a bottle or a paperback is this size"
    ),
    ProductScale.TWO_HANDS: (
        "big enough to want both hands, but no wider than the model's shoulders. A "
        "laptop, a boxed appliance or a handbag is this size"
    ),
    ProductScale.WORN: (
        "worn on the body, so its size is fixed by the part of the body it fits, and it "
        "is shown at the size it genuinely is on a person"
    ),
    ProductScale.FLOOR_STANDING: (
        "a large object resting on the floor, at most about the model's own height"
    ),
}

#: The half of the scale instruction that does not depend on knowing the size.
#:
#: Worth separating, because it is the half that actually fixes the bug.  The first
#: live iPhone job did not fail because the model guessed 147 mm wrong; it failed
#: because three separate phrases asked for the product to be *prominent* and the
#: cheapest way to make something prominent is to make it enormous.  Naming a
#: different route to prominence is what removes the incentive, and that sentence
#: is just as true for a product whose size we were never told.
_SCALE_RULE = (
    "- Render the product at its true physical size relative to the model's hands and "
    "body. A viewer must be able to tell how big the real object is simply by looking "
    "at the photograph.\n"
    "- Prominence comes from placement, focus, lighting and how near the camera is — "
    "never from drawing the object larger than it is. Do not enlarge the product."
)


_MOOD_PHRASE: dict[Mood, str] = {
    Mood.CALM_PREMIUM: "calm, premium, understated and expensive-looking",
    Mood.WARM_LIFESTYLE: "warm, inviting, natural and relatable",
    Mood.BOLD_CONFIDENT: "bold, confident, striking and graphic",
    Mood.HIGH_ENERGY: "high-energy, dynamic, youthful and vivid",
}


@dataclass(frozen=True)
class _Beat:
    """One motion intent, decomposed into the three things a prompt has to say.

    ``action`` is what the *model* does, and it is always stated first.  It is the
    half that simply did not exist while this axis named camera moves, and its
    absence is why the first live job returned three slow zooms over a still
    frame.

    ``camera`` supports the action.  It is deliberately demoted to a subordinate
    clause — it is a modifier of the performance, never the performance itself.

    ``opening`` is handed to the *image* stage.  The generated still is literally
    frame 1 of this action, and the two stages were never told that about each
    other: the image prompt asked for a finished advertising photograph, which is
    a settled pose with nowhere to go, and then the video stage was asked to
    animate it.  A body caught mid-movement gives the video model somewhere to
    take the shot.
    """

    action: str
    camera: str
    opening: str


_MOTION_BEATS: dict[MotionIntent, _Beat] = {
    MotionIntent.PRODUCT_REVEAL: _Beat(
        action=(
            "raises the product up into frame and rotates it so its front face and label "
            "turn squarely toward the camera, then holds it there"
        ),
        camera="the camera pushes in slowly as the product comes up",
        opening=(
            "the model is holding the product low, only just entering the bottom of the "
            "frame and clearly already on its way up"
        ),
    ),
    MotionIntent.HERO_TURN: _Beat(
        action=(
            "turns their shoulders and head around to face the camera and settles into a "
            "confident hold, the product raised at chest height and angled to the lens"
        ),
        camera="the camera holds steady and lets the turn play out",
        opening=(
            "the model is turned partly away from the camera and has just started to "
            "rotate back — caught mid-turn rather than settled"
        ),
    ),
    MotionIntent.IN_USE: _Beat(
        action=(
            "actually uses the product — putting it on, opening it, applying it or lifting "
            "it into use — hands moving deliberately, eyes following what they are doing"
        ),
        camera="the camera drifts gently closer to the hands",
        opening=(
            "the model's hands are already on the product, caught in the middle of the "
            "gesture rather than posed with it"
        ),
    ),
    MotionIntent.OFFER_TO_CAMERA: _Beat(
        action=(
            "extends the product toward the lens in an unhurried offering gesture, arm "
            "straightening, then looks up to camera and holds"
        ),
        camera="the camera stays put and lets the product grow in frame as it comes forward",
        opening=(
            "the model holds the product close to the body with the elbow bent and their "
            "weight already shifting forward, about to extend"
        ),
    ),
    MotionIntent.PICK_UP: _Beat(
        action=(
            "reaches for the product where it is resting, takes hold of it, lifts it clear "
            "of the surface and draws it in toward their body, turning it to face the camera"
        ),
        camera="the camera eases down and in to follow the hands",
        opening=(
            "the product is resting on a surface and the model's hand is reaching toward "
            "it, caught just short of contact"
        ),
    ),
    MotionIntent.WALK_IN: _Beat(
        action=(
            "steps forward through the scene toward the camera carrying the product, weight "
            "shifting from one foot to the other, hair and clothing moving with the stride"
        ),
        camera="the camera pulls back a little to keep pace",
        opening="the model is mid-stride with their weight on the front foot, not standing still",
    ),
}

#: Baked into every negative prompt.  These are the failure modes that make an ad
#: frame unusable regardless of taste, so they are not left to the user to think of.
_BASE_NEGATIVES = (
    "distorted face, extra fingers, malformed hands, warped product label, "
    "duplicated product, unreadable text, watermark, logo artefacts, "
    "blurry product, cropped product, "
    # Scale failures. Named separately because they are not artefacts — every one
    # of these frames is well-formed, and the first live job passed every gate
    # check while shipping a phone taller than the man holding it.
    "product out of scale, giant oversized product, product larger than the model's head, "
    "product floating unsupported, miniature person, inconsistent scale between subjects"
)

#: Baked into every *video* negative prompt.  Deliberately a different list from
#: the image one above, because the two stages fail differently: the image
#: negatives suppress *artefacts* (extra fingers, a warped label), while these
#: suppress the *absence of a performance*.
#:
#: The video stage was reusing the image list, which never once mentions motion.
#: The terms below are the ones that name what the first live job actually
#: produced — a still photograph with a camera moving across it.
_BASE_VIDEO_NEGATIVES = (
    "static image, still photo, frozen subject, motionless model, no movement, "
    "slideshow, ken burns effect, zooming into a photograph, camera-only movement, "
    "morphing face, changing identity, warping hands, extra fingers, "
    "product changing shape or size, label warping, flicker, jitter, "
    "text overlay, watermark, subtitles, captions"
)


#: Short forms for the two levels whose scale-protective clause sits *after* the
#: first comma.  Everything else compacts safely by taking the leading clause, but
#: these two are precisely where the oversizing happened, so their qualification
#: is not optional and has to survive the compression.
_COMPACT_ANGLE: dict[CameraAngle, str] = {
    CameraAngle.CLOSE_UP_PRODUCT: (
        "camera close to the product, one of the model's hands in shot for scale"
    ),
}
_COMPACT_COMPOSITION: dict[Composition, str] = {
    Composition.PRODUCT_FOREGROUND: (
        "product nearer the camera than the model, larger only by perspective"
    ),
}


def _lead(phrase: str) -> str:
    """The clause before the first comma — the shot, without its qualifications."""
    return phrase.split(",")[0]


def _build_image_prompt_compact(request: AdJobRequest, dp: DesignPoint, scene: str) -> str:
    """The same job, said in a third of the words.

    Written after seeing a competing implementation get visibly better composition
    from a 37-word prompt. Its whole instruction set was five imperatives, two of
    which — "make the person naturally hold or use the product" and "match
    lighting, shadows, scale and perspective" — did the work our 343-word version
    was failing to do at all.

    The structural difference is not length for its own sake. Our full template
    speaks in labelled blocks (REFERENCES / SCALE / SHOT / LOOK / CONSTRAINTS),
    which reads as five separate briefs to satisfy. This states one photograph and
    then qualifies it, which is how the instruction is actually meant to be read.

    Every design axis still appears. Dropping them to save words would delete the
    diversity the multi-candidate claim rests on, and the comparison would no
    longer be about verbosity.
    """
    scale = request.effective_scale
    size = f"{_SCALE_PHRASE[scale].split('—')[0].strip()}" if scale is not None else ""
    safe = request.platform.safe_area

    background = _lead(
        _BACKGROUND_PHRASE[request.theme.background].format(
            colour=request.theme.background_color or "#f2f2f2"
        )
    )
    angle = _COMPACT_ANGLE.get(dp.angle) or _lead(_ANGLE_PHRASE[dp.angle])
    composition = _COMPACT_COMPOSITION.get(dp.composition) or _lead(
        _COMPOSITION_PHRASE[dp.composition]
    )

    lines = [
        f"Photorealistic advertising photograph for {request.product_name}: the person in "
        "Image 1 naturally holding and using the product in Image 2.",
        "Preserve the person's face and body exactly, and the product's exact shape, "
        "colour, proportions and label.",
        "Match lighting, shadows, scale and perspective. "
        + (f"The product is {size}. " if size else "")
        + "Render it at its true size next to the model — never enlarge it for emphasis.",
        f"{angle}. {_lead(_LIGHTING_PHRASE[dp.lighting])}. {composition}. {background}.",
        f"Catch the model mid-movement, not posed: {_MOTION_BEATS[dp.motion].opening}.",
    ]
    if scene:
        lines.append(scene)
    if request.negative_constraints:
        lines.append(f"Also: {request.negative_constraints}.")
    if request.additional_prompt:
        lines.append(request.additional_prompt)

    lines.append(
        f"{_lead(_MOOD_PHRASE[request.mood])}, {request.aspect_ratio.value} for "
        f"{request.platform.value.replace('_', ' ')}. Keep the product and face out of the "
        f"top {safe.top:.0%} and bottom {safe.bottom:.0%}. No text, captions or graphics."
    )
    return "\n".join(lines)


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

    # Placed directly after the references and before the shot, because it is a
    # property *of* the references and because early tokens carry more weight. The
    # generator is being handed a full-frame photograph of a person and a
    # full-frame cutout of a phone; without this block it has no reason to believe
    # they are not the same size, and every downstream instruction about making the
    # product prominent pushes it to resolve that ambiguity the wrong way.
    scale = request.effective_scale
    lines += ["", "SCALE — the most commonly botched part of this shot:"]
    if scale is not None:
        lines.append(f"- The product is {_SCALE_PHRASE[scale]}.")
    lines.append(_SCALE_RULE)

    lines += [
        "",
        "SHOT:",
        f"- {_ANGLE_PHRASE[dp.angle]}",
        f"- {_LIGHTING_PHRASE[dp.lighting]}",
        f"- {_COMPOSITION_PHRASE[dp.composition]}",
        f"- {background}",
        # This frame is about to become frame 1 of a clip, and the two stages
        # were never told that about each other. A settled, finished-looking
        # pose has nowhere to go, so the video model does the only thing left
        # available to it and moves the camera instead.
        f"- Catch the model mid-movement rather than posed: {_MOTION_BEATS[dp.motion].opening}",
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
        "- The whole product is in frame with its front face readable. A natural hand "
        "grip across it is expected and correct — do not resize, float or reposition "
        "the product to keep the model's fingers off it.",
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


def _build_motion_prompt(request: AdJobRequest, dp: DesignPoint, action_note: str = "") -> str:
    """Assemble the image-to-video instruction.

    The ordering here is the fix, and it is worth stating why each part sits
    where it does.

    **The action comes first.**  Video models weight early tokens heavily, and the
    previous version opened with a camera move and closed with "preserve the
    model's identity and the product's exact appearance across every frame" — a
    sentence a video model can only read as *change nothing*.  Identity still has
    to hold, but it is now phrased as continuity of a person who is moving rather
    than as sameness between frames.

    **The user's own direction is included.**  It used to reach the image stage
    only, so someone who wrote "the model picks the shoes up off the table" got
    that in their still and never once in their clip.  That is the single most
    specific piece of creative information the job has, and the stage that most
    needed it was the stage that never saw it.

    **The still is named as frame 1.**  Saying so out loud stops the model
    treating the input as a finished poster to pan across.
    """
    beat = _MOTION_BEATS[dp.motion]

    lines = [
        f"Live-action television commercial for {request.product_name}. "
        "A real person performing on camera, filmed as one continuous take. "
        "The supplied image is the first frame of that take, not a photograph to pan across.",
        "",
        f"ACTION: The model {beat.action}. Meanwhile {beat.camera}.",
    ]

    direction = " ".join(t.strip() for t in (request.additional_prompt, action_note) if t).strip()
    if direction:
        lines.append(f"Follow this direction for the performance: {direction}")

    if request.negative_constraints:
        lines.append(f"Avoid, throughout: {request.negative_constraints}")

    lines += [
        "",
        "The model moves continuously and naturally for the whole shot — body, arms, "
        "hands, head, hair and clothing all in motion. This is a filmed performance, "
        "not a still image with a moving camera.",
        f"{request.product_name} stays sharp, fully in frame and the right way up "
        "throughout, and the model stays recognisably the same person from the first "
        "frame to the last.",
        "",
        f"Mood: {_MOOD_PHRASE[request.mood]}. "
        f"{request.duration_seconds:.0f} seconds, one continuous shot, no cuts, no text.",
    ]
    return "\n".join(lines)


def _build_video_negative_prompt(request: AdJobRequest) -> str:
    """Keyword-style negatives for the video stage.

    Separate from the image negatives for the reason given at
    :data:`_BASE_VIDEO_NEGATIVES`.  The user's free-text constraints stay out of
    it and go into the positive prompt instead, on the same double-negative
    reasoning as :func:`_build_image_prompt`.
    """
    return _BASE_VIDEO_NEGATIVES


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
    "job is to invent the setting, one sentence of extra physical detail for what the "
    "model physically does on camera, and a one-line creative concept that suit it."
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
    '"action": "one sentence of extra physical detail for what the model does on camera", '
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
            # Additive only. The beat table owns the structure of the motion
            # prompt; the LLM may add physical detail to it and may not replace
            # it, so a bad completion costs a job some texture, never its motion.
            "action": str(item.get("action", ""))[:300],
            "concept": str(item.get("concept", ""))[:200],
        }
    return out


# --- Public entry point -----------------------------------------------------


async def compile_briefs(
    request: AdJobRequest, llm: LLMProvider, style: PromptStyle = PromptStyle.FULL
) -> BriefSet:
    """Produce one :class:`ShotBrief` per candidate slot.

    ``style`` selects the image-prompt template. It is a system setting rather
    than a request field because it is an ablation axis — the person submitting a
    job has no view on prompt verbosity — but it is recorded on the returned
    :class:`BriefSet` so a result can be attributed to it afterwards.
    """
    build = _build_image_prompt if style is PromptStyle.FULL else _build_image_prompt_compact
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
                image_prompt=build(request, dp, scene),
                negative_prompt=_build_negative_prompt(request),
                motion_prompt=_build_motion_prompt(request, dp, extra.get("action", "")),
                video_negative_prompt=_build_video_negative_prompt(request),
                concept=extra.get("concept") or _template_concept(request, dp),
            )
        )

    return attach_diversity(
        BriefSet(
            job_id=request.job_id,
            briefs=briefs,
            sampler="latin_hypercube",
            prompt_style=style.value,
        )
    )
