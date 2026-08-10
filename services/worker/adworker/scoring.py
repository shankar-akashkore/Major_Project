"""Performance prediction — the learned part of the system, currently a baseline.

What exists today is a **documented heuristic ensemble** over the real numpy
features: a fixed linear blend with hand-set weights.  Every score it produces is
stamped ``is_stub=True`` and ``model_version="heuristic-0"``.

That honesty matters for two reasons.  The obvious one is that presenting
hand-tuned weights as a learned predictor would misrepresent the contribution.
The less obvious one is that this heuristic is itself one of the baselines the
trained model has to beat in the evaluation — so building it properly now is not
throwaway scaffolding, it is the control condition.

Components left as ``None`` are ones that genuinely cannot be computed yet
(``prompt_alignment`` needs CLIPScore).  A null is more useful than a fabricated
number: it shows up as absent in the ablation table instead of quietly diluting
a feature group's apparent contribution.
"""

from __future__ import annotations

import math

from adml import features as F
from adproviders import Storage
from adschema import (
    AdJobRequest,
    ImageCandidate,
    RankedCandidate,
    ScoreBreakdown,
    VideoCandidate,
)

MODEL_VERSION = "heuristic-0"

#: Image-stage blend.  Weights sum to 1.0.  These are priors, not fitted values —
#: the whole point of Stage B calibration is to replace them with weights learned
#: from pairwise human preference.
IMAGE_WEIGHTS = {
    "aesthetic": 0.25,
    "composition": 0.20,
    "product_salience": 0.25,
    "palette_adherence": 0.15,
    "safe_area_compliance": 0.15,
}

#: Video-stage blend.  Hook strength is weighted heavily on purpose: short-form ads
#: are won or lost in the first second, and averaging that away across ten seconds
#: is the most common way a naive video metric misranks creative.
VIDEO_WEIGHTS = {
    "hook_strength": 0.30,
    "temporal_consistency": 0.20,
    "motion_quality": 0.15,
    "product_screen_time": 0.15,
    "aesthetic": 0.10,
    "safe_area_compliance": 0.10,
}

#: RMS contrast band that reads as well-exposed commercial photography.
_CONTRAST_SWEET_SPOT = (0.16, 0.30)
#: Colourfulness band; too flat looks cheap, too saturated looks like a scam ad.
_COLOR_SWEET_SPOT = (0.25, 0.65)


def _band_score(value: float, lo: float, hi: float, falloff: float = 0.18) -> float:
    """1.0 inside ``[lo, hi]``, decaying smoothly outside it.

    Used for features where more is not better — exposure and saturation both have
    an optimum rather than a direction.
    """
    if lo <= value <= hi:
        return 1.0
    distance = lo - value if value < lo else value - hi
    return float(math.exp(-((distance / falloff) ** 2)))


def _aesthetic_proxy(contrast: float, colour: float) -> float:
    """Stand-in for the LAION aesthetic predictor.

    Replaced by the real thing once torch is available in Colab; kept as a
    baseline afterwards.
    """
    return 0.5 * _band_score(contrast, *_CONTRAST_SWEET_SPOT) + 0.5 * _band_score(
        colour, *_COLOR_SWEET_SPOT
    )


def _blend(parts: dict[str, float | None], weights: dict[str, float]) -> float:
    """Weighted mean over the components that are actually available.

    Renormalises over present components, so a missing feature reduces confidence
    rather than silently scoring zero.
    """
    num = 0.0
    den = 0.0
    for name, weight in weights.items():
        value = parts.get(name)
        if value is None:
            continue
        num += weight * value
        den += weight
    return float(min(1.0, max(0.0, num / den))) if den > 0 else 0.0


def score_image(
    candidate: ImageCandidate, request: AdJobRequest, storage: Storage
) -> ScoreBreakdown:
    """Image-stage prediction: how well will this frame perform once animated?

    This is the score the project's headline claim rests on — whether a ranking
    computed here survives to the video stage.
    """
    rgb = F.load_image(storage.get_bytes(candidate.asset.key))
    sal = F.saliency_map(rgb)
    safe = request.platform.safe_area

    contrast = F.rms_contrast(rgb)
    colour = F.colorfulness(rgb)
    palette_score, _ = F.palette_adherence(rgb, request.theme.palette)

    parts: dict[str, float | None] = {
        "aesthetic": _aesthetic_proxy(contrast, colour),
        "composition": F.thirds_alignment(sal),
        "product_salience": F.focal_concentration(sal),
        "palette_adherence": palette_score,
        "safe_area_compliance": F.region_saliency_share(sal, safe.top, safe.bottom),
    }

    return ScoreBreakdown(
        overall=round(_blend(parts, IMAGE_WEIGHTS), 4),
        aesthetic=round(parts["aesthetic"], 4),
        composition=round(parts["composition"], 4),
        product_salience=round(parts["product_salience"], 4),
        palette_adherence=round(parts["palette_adherence"], 4),
        safe_area_compliance=round(parts["safe_area_compliance"], 4),
        # Needs CLIPScore; deliberately absent rather than invented.
        prompt_alignment=None,
        model_version=MODEL_VERSION,
        is_stub=True,
    )


def score_video(
    candidate: VideoCandidate, request: AdJobRequest, storage: Storage
) -> ScoreBreakdown:
    """Video-stage scoring: the features that only exist once the clip moves."""
    data = storage.get_bytes(candidate.asset.key)
    frames = F.load_frames(data)
    safe = request.platform.safe_area

    consistency = F.temporal_consistency(frames)
    energies = F.motion_energy(frames)
    mean_energy = sum(energies) / len(energies) if energies else 0.0
    hook = F.hook_strength(frames, fps=candidate.fps)

    # Motion quality rewards visible-but-controlled movement. Both extremes are
    # failures: a frozen clip wastes the format, a thrashing one is unwatchable.
    motion_quality = _band_score(mean_energy, 0.008, 0.045, falloff=0.03)

    mid = frames[len(frames) // 2]
    sal_mid = F.saliency_map(mid)
    contrast = F.rms_contrast(mid)
    colour = F.colorfulness(mid)

    # Product screen time needs the product mask to be exact. Until then, the
    # share of frames retaining a clear focal subject is the honest proxy.
    focal_per_frame = [
        F.focal_concentration(F.saliency_map(f)) for f in frames[:: max(1, len(frames) // 8)]
    ]
    screen_time = sum(1 for v in focal_per_frame if v >= 0.12) / max(1, len(focal_per_frame))

    parts: dict[str, float | None] = {
        "hook_strength": hook,
        "temporal_consistency": consistency,
        "motion_quality": motion_quality,
        "product_screen_time": screen_time,
        "aesthetic": _aesthetic_proxy(contrast, colour),
        "safe_area_compliance": F.region_saliency_share(sal_mid, safe.top, safe.bottom),
    }

    return ScoreBreakdown(
        overall=round(_blend(parts, VIDEO_WEIGHTS), 4),
        hook_strength=round(hook, 4),
        temporal_consistency=round(consistency, 4),
        motion_quality=round(motion_quality, 4),
        product_screen_time=round(screen_time, 4),
        aesthetic=round(parts["aesthetic"], 4),
        safe_area_compliance=round(parts["safe_area_compliance"], 4),
        prompt_alignment=None,
        model_version=MODEL_VERSION,
        is_stub=True,
    )


_LABELS = {
    "hook_strength": "opens strongly in the first second",
    "temporal_consistency": "holds together frame to frame",
    "motion_quality": "moves at a controlled, watchable pace",
    "product_screen_time": "keeps the product on screen",
    "aesthetic": "reads as polished commercial photography",
    "safe_area_compliance": "keeps key content clear of platform UI",
    "composition": "sits well against rule-of-thirds framing",
    "product_salience": "gives the product a clear focal point",
    "palette_adherence": "stays close to the brand palette",
}


#: Weighted deviation from the peer mean below which a component is treated as
#: indistinguishable between candidates and left out of the explanation.
_DIFFERENTIATOR_EPSILON = 0.002


def _differentiators(
    score: ScoreBreakdown,
    peers: list[ScoreBreakdown],
    weights: dict[str, float],
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Components where this candidate departs most from its peers.

    Reporting a candidate's highest-scoring components sounds like an explanation
    but usually is not: in a set of three variants generated from the same
    references, several components saturate near 1.0 for everybody, so naming them
    tells the user nothing about why *this* one won. What matters is where the
    candidate differs from the others, scaled by how much the ranker weights that
    component.

    Returns ``(strengths, weaknesses)`` as ``(component, weighted_deviation)``.
    """
    own = score.components()
    if len(peers) < 2:
        return [(k, 0.0) for k, _ in score.top_drivers(3)], []

    deviations: list[tuple[str, float]] = []
    for name, value in own.items():
        peer_values = [p.components().get(name) for p in peers]
        present = [v for v in peer_values if v is not None]
        if len(present) < 2:
            continue
        mean = sum(present) / len(present)
        deviations.append((name, weights.get(name, 0.0) * (value - mean)))

    strengths = sorted(
        (d for d in deviations if d[1] > _DIFFERENTIATOR_EPSILON),
        key=lambda kv: -kv[1],
    )
    weaknesses = sorted(
        (d for d in deviations if d[1] < -_DIFFERENTIATOR_EPSILON),
        key=lambda kv: kv[1],
    )
    return strengths, weaknesses


def explain(
    score: ScoreBreakdown,
    rank: int,
    rank_shift: int | None,
    peers: list[ScoreBreakdown] | None = None,
    weights: dict[str, float] | None = None,
) -> str:
    """Build the human-readable justification shown next to a ranked result."""
    weights = weights or VIDEO_WEIGHTS
    own = score.components()
    strengths, weaknesses = _differentiators(score, peers or [], weights)

    parts: list[str] = []
    if strengths:
        parts += [
            f"{_LABELS.get(name, name)} ({own[name]:.0%}, best of the set)"
            if index == 0
            else f"{_LABELS.get(name, name)} ({own[name]:.0%})"
            for index, (name, _) in enumerate(strengths[:2])
        ]
    if not parts:
        # Genuinely indistinguishable on the measured components — say so rather
        # than dressing up the highest number as a reason.
        parts.append("scored within noise of the other candidates on every measured component")

    text = f"Ranked #{rank} — " + "; ".join(parts) + "."

    if weaknesses:
        name = weaknesses[0][0]
        text += f" Weakest on {_LABELS.get(name, name)} ({own[name]:.0%})."

    if rank_shift is not None and rank_shift != 0:
        direction = "up" if rank_shift > 0 else "down"
        text += (
            f" Moved {direction} {abs(rank_shift)} place(s) from the image-stage "
            f"prediction once animated."
        )
    elif rank_shift == 0:
        text += " The image-stage prediction placed it here too."

    if score.is_stub:
        text += " (Heuristic baseline — the trained predictor has not replaced it yet.)"
    return text


def rank_videos(videos: list[VideoCandidate], image_order: list[int]) -> list[RankedCandidate]:
    """Order videos by predicted performance and attach explanations.

    ``image_order`` is the image-stage ranking, best first, given as source image
    indices.  Carrying it through is what lets the report measure agreement
    between the two stages — the project's headline number.
    """
    image_rank_of = {idx: pos + 1 for pos, idx in enumerate(image_order)}
    ordered = sorted(
        videos,
        key=lambda v: (-(v.score.overall if v.score else 0.0), v.index),
    )
    # The whole peer set, so each explanation can say what set this candidate apart
    # rather than just which of its own components happened to be highest.
    peers = [v.score for v in ordered if v.score is not None]

    ranked: list[RankedCandidate] = []
    for position, video in enumerate(ordered, start=1):
        prior = image_rank_of.get(video.source_image_index)
        shift = None if prior is None else prior - position
        score = video.score or ScoreBreakdown(overall=0.0, model_version=MODEL_VERSION)
        ranked.append(
            RankedCandidate(
                rank=position,
                video=video,
                explanation=explain(score, position, shift, peers, VIDEO_WEIGHTS),
                image_stage_rank=prior,
            )
        )
    return ranked
