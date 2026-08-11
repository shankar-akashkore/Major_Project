"""Performance prediction: the trained ranker when one exists, the baseline when not.

Two scorers live here and which one runs is decided by whether a trained model is
on disk.

**The heuristic ensemble** (``heuristic-0``) is a fixed linear blend with hand-set
weights over the real numpy features.  It is not throwaway scaffolding — it is one
of the baselines the trained model has to beat in the evaluation, so it is the
control condition and it stays.  Everything it produces is stamped
``is_stub=True``.

**The trained ranker** is loaded through :mod:`adml.serving` and replaces only the
``overall`` figure.  The component breakdown stays heuristic on purpose: the
trained head is a single score over z-scored features with no per-component
decomposition, and inventing one by attributing its weights back to named
components would be a plausible-looking fiction.  So a served result carries the
model's ordering and the measured components that explain it, and says which is
which.

Two behaviours are deliberate.

*A trained model does not make a score non-stub by itself.*  ``is_stub`` follows
the model card: a model fitted on stand-in embeddings, or one whose held-out
accuracy was never measured, is still a stub however confident its numbers look.

*The ranking is set-relative.*  The pairwise objective fixes no origin, so the
0-1 ``overall`` for a trained model is a within-set position, not a calibrated
score. ``score_image_set`` therefore takes the whole candidate set at once; there
is no way to score one candidate in isolation and no pretence that there is.

Components left as ``None`` are ones that genuinely cannot be computed yet
(``prompt_alignment`` needs CLIPScore).  A null is more useful than a fabricated
number: it shows up as absent in the ablation table instead of quietly diluting
a feature group's apparent contribution.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from adml import features as F
from adml import featureset as FS
from adml import serving as S
from adml import video as V
from adproviders import Storage
from adschema import (
    AdJobRequest,
    ImageCandidate,
    ItemKind,
    RankedCandidate,
    ScoreBreakdown,
    VideoCandidate,
)
from adschema.annotation import CorpusItem

MODEL_VERSION = "heuristic-0"

#: Where the pipeline looks for a trained ranker. Overridable so a test can point
#: at a temporary one, and so an experiment can serve a specific fold's model.
MODEL_PATH_ENV = "AD_RANKER_MODEL"


def model_path() -> Path:
    return Path(os.environ.get(MODEL_PATH_ENV) or S.DEFAULT_MODEL_PATH)


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

    NaN is treated as missing, exactly like ``None``.  It arrives for real:
    :func:`adml.features.temporal_consistency` returns NaN for a single-frame clip
    because a still has no temporal behaviour.  Without this guard one NaN
    component would propagate through the sum and make the whole ``overall`` NaN,
    which pydantic then rejects at the field bound — a validation error several
    frames away from the degenerate input that caused it.
    """
    num = 0.0
    den = 0.0
    for name, weight in weights.items():
        value = parts.get(name)
        if value is None or not math.isfinite(value):
            continue
        num += weight * value
        den += weight
    return float(min(1.0, max(0.0, num / den))) if den > 0 else 0.0


def _clean(value: float | None) -> float | None:
    """Round for display, mapping non-finite values to ``None``.

    ``ScoreBreakdown`` bounds every component to 0-1, so a NaN would fail
    validation. Absent is the correct report for a measurement that could not be
    taken, and it is what the ablation table already knows how to show.
    """
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), 4)


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
        aesthetic=_clean(parts["aesthetic"]),
        composition=_clean(parts["composition"]),
        product_salience=_clean(parts["product_salience"]),
        palette_adherence=_clean(parts["palette_adherence"]),
        safe_area_compliance=_clean(parts["safe_area_compliance"]),
        # Needs CLIPScore; deliberately absent rather than invented.
        prompt_alignment=None,
        model_version=MODEL_VERSION,
        is_stub=True,
    )


def score_video(
    candidate: VideoCandidate, request: AdJobRequest, storage: Storage
) -> ScoreBreakdown:
    """Video-stage scoring: the features that only exist once the clip moves.

    Decodes through :mod:`adml.video` rather than PIL.  That is the fix for a real
    defect: this function previously called ``F.load_frames``, which is PIL, and
    PIL cannot open an MP4.  Because the mock provider wrote GIFs, nothing caught
    it — the first paid Kling clip would have raised ``UnidentifiedImageError``
    here, after the clip had been generated and billed.
    """
    data = storage.get_bytes(candidate.asset.key)
    clip, motion = V.measure(data)
    frames = clip.frames
    safe = request.platform.safe_area

    # Motion quality rewards visible-but-controlled movement. Both extremes are
    # failures: a frozen clip wastes the format, a thrashing one is unwatchable.
    motion_quality = _band_score(motion.motion_energy_mean, 0.008, 0.045, falloff=0.03)

    mid = frames[len(frames) // 2]
    sal_mid = F.saliency_map(mid)
    contrast = F.rms_contrast(mid)
    colour = F.colorfulness(mid)

    parts: dict[str, float | None] = {
        "hook_strength": motion.hook_strength,
        "temporal_consistency": motion.temporal_consistency,
        "motion_quality": motion_quality,
        # Product screen time needs the product mask to be exact. Until then, the
        # share of frames retaining a clear focal subject is the honest proxy.
        "product_screen_time": motion.focal_persistence,
        "aesthetic": _aesthetic_proxy(contrast, colour),
        "safe_area_compliance": F.region_saliency_share(sal_mid, safe.top, safe.bottom),
    }

    return ScoreBreakdown(
        overall=round(_blend(parts, VIDEO_WEIGHTS), 4),
        hook_strength=_clean(parts["hook_strength"]),
        temporal_consistency=_clean(parts["temporal_consistency"]),
        motion_quality=_clean(motion_quality),
        product_screen_time=_clean(parts["product_screen_time"]),
        aesthetic=_clean(parts["aesthetic"]),
        safe_area_compliance=_clean(parts["safe_area_compliance"]),
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

    if score.model_version == MODEL_VERSION:
        text += " (Heuristic baseline — the trained predictor has not replaced it yet.)"
    elif score.is_stub:
        # A trained model can still be a stub: fitted on stand-in embeddings, or
        # never measured against a held-out fold. Saying "trained" without saying
        # that would be the misleading half of the truth.
        text += (
            f" (Ranked by {score.model_version}, which is not yet validated — "
            "stand-in features or unmeasured held-out accuracy.)"
        )
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


# --- The trained ranker ------------------------------------------------------


def _corpus_item(candidate: ImageCandidate, request: AdJobRequest, kind: ItemKind) -> CorpusItem:
    """Describe a live candidate the way the training corpus described its items.

    The trained model's context one-hots are built from :class:`CorpusItem` fields,
    so a live candidate has to be presented in the same shape or the design-axis
    columns land in the wrong places. Building the same object rather than a
    parallel adapter is what keeps that guarantee: if a field is added to the
    corpus, this fails to construct instead of silently omitting a column.
    """
    point = candidate.brief.design_point
    return CorpusItem(
        item_id=f"{request.job_id}-i{candidate.index}",
        set_id=request.job_id,
        kind=kind,
        asset=candidate.asset,
        tier=candidate.tier,
        provider=candidate.provider,
        vertical=request.vertical,
        platform=request.platform,
        seed=candidate.seed,
        angle=point.angle,
        lighting=point.lighting,
        composition=point.composition,
        motion=point.motion,
    )


def load_ranker(path: Path | None = None) -> S.ServedModel | None:
    """Load the trained ranker, or ``None`` when none has been trained yet.

    No model is the normal state for most of this project's life, so absence must
    not fail a job. A model that is *present but unloadable* still raises: that is
    a misconfiguration, and quietly serving the baseline instead would hide it
    behind results that look fine.
    """
    return S.try_load(path or model_path())


@dataclass(frozen=True)
class SetScores:
    """Scores for one candidate set, and which scorer produced them."""

    breakdowns: dict[int, ScoreBreakdown]
    scored_by: str
    #: Set when a trained model was available but could not be used. Carried out
    #: rather than logged, so the pipeline can put it in the job's event stream —
    #: a ranking silently produced by the baseline when a model was configured is
    #: exactly the kind of thing that goes unnoticed until the write-up.
    fallback_reason: str | None = None


def score_image_set(
    candidates: list[ImageCandidate],
    request: AdJobRequest,
    storage: Storage,
    *,
    model: S.ServedModel | None = None,
) -> SetScores:
    """Score a whole candidate set, keyed by candidate index.

    Set-at-once rather than one at a time, because that is what the trained model
    supports: the pairwise objective fixes no origin, so its ``overall`` is a
    position within the set being compared and cannot be computed for a candidate
    in isolation.

    Components always come from the heuristic measurements. Only ``overall`` and
    the provenance change when a model is served — see the module docstring on why
    a decomposition of the trained score would be invented rather than measured.

    **A model whose features this path cannot compute falls back, loudly.**  That
    is a real and permanent state, not a transient one: a model trained with the
    Colab embedding blocks needs SigLIP and DINOv2 columns, and the serving path
    has no torch to produce them. So the fallback is reported rather than fixed —
    the fix is to train a serving model on the features the pipeline can compute,
    which is a decision, not an error to swallow.
    """
    breakdowns = {c.index: score_image(c, request, storage) for c in candidates}
    if model is None or not candidates:
        return SetScores(breakdowns, scored_by=MODEL_VERSION)

    items = [_corpus_item(c, request, ItemKind.IMAGE) for c in candidates]
    measured = {
        item.item_id: FS.image_features(
            item, storage.get_bytes(item.asset.key), request.theme.palette
        )
        for item in items
    }
    table = FS.build_table(items, measured)
    if not table.item_ids:
        return SetScores(
            breakdowns,
            scored_by=MODEL_VERSION,
            fallback_reason="no candidate produced a complete feature row",
        )

    try:
        scores = S.rank_within_set(model, table, list(table.item_ids))
    except S.ModelMismatch as exc:
        return SetScores(breakdowns, scored_by=MODEL_VERSION, fallback_reason=str(exc))

    relative = scores.relative()
    by_index = {item.item_id: c.index for item, c in zip(items, candidates, strict=True)}
    for item_id, value in relative.items():
        index = by_index[item_id]
        breakdowns[index] = breakdowns[index].model_copy(
            update={
                "overall": round(value, 4),
                "model_version": model.card.model_version,
                "is_stub": model.card.is_stub,
            }
        )
    return SetScores(breakdowns, scored_by=model.card.model_version)
