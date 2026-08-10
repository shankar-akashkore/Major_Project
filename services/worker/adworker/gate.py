"""The quality gate — hard pass/fail, no scoring.

This stage answers one question: *is this candidate usable at all?*  It is not a
ranking, and it deliberately shares no code path with the performance predictor.
A candidate whose product is unrecognisable is not "slightly worse", it is
unusable, and expressing that as a low score would let it be ranked first on a bad
day.

Some checks are real today (palette, safe area, exposure, focal clarity — all
numpy, all running now).  Two are not: product identity needs DINOv2 and face
identity needs ArcFace, both of which arrive with the Colab feature-extraction
work.  Those return ``implemented=False`` and pass by default, so nothing claims
to have verified an identity it never looked at.
"""

from __future__ import annotations

from adml import features as F
from adproviders import Storage
from adschema import (
    AdJobRequest,
    GateCheck,
    GateResult,
    GateVerdict,
    ImageCandidate,
)

# Thresholds calibrated by ``scripts/calibrate_gate.py`` against the mock renderer
# (150 lighting x composition x angle combinations, matched vs deliberately
# mismatched palettes). Measured on-brand ranges are recorded beside each value so
# the week-6 recalibration against real generations has a baseline to compare to.
#
# Each threshold sits below the observed minimum of the legitimate class, not at
# its median: the gate exists to catch genuine failures, and a filter tuned to
# reject a third of usable candidates costs more than it saves.

#: on-brand 0.912-0.972, off-brand 0.000 — classes separate with a 0.912 gap.
THRESH_PALETTE = 0.456
#: on-brand 0.680-0.914 once the sampler stops picking compositions that
#: structurally collide with platform chrome (see ``sampler.compositions_for``).
THRESH_SAFE_AREA = 0.55
#: on-brand 0.578-0.728. Held well below that as a smeared-attention floor,
#: because real generations will spread wider than the mock's synthetic frames.
THRESH_FOCUS = 0.25
#: on-brand luminance 0.531-0.643; these bounds only catch black or blown frames.
MIN_LUMINANCE = 0.06
MAX_LUMINANCE = 0.96
#: on-brand contrast 0.180-0.248; this only catches flat/empty frames.
MIN_CONTRAST = 0.03

#: Checks that would need model weights this machine cannot host yet.
_PENDING = {
    "product_identity": (
        "DINOv2 cosine vs the rembg product cutout — lands with the Colab extractor"
    ),
    "face_identity": "ArcFace cosine vs the human reference — lands with the Colab extractor",
    "nsfw": "safety classifier — lands with the intake safety gate",
}


def _pending_check(name: str) -> GateCheck:
    return GateCheck(
        name=name,
        value=0.0,
        threshold=0.0,
        passed=True,
        detail=_PENDING[name],
        implemented=False,
    )


def evaluate_image(
    candidate: ImageCandidate,
    request: AdJobRequest,
    storage: Storage,
    attempt: int = 1,
) -> GateResult:
    """Run every hard filter against a generated frame."""
    rgb = F.load_image(storage.get_bytes(candidate.asset.key))
    sal = F.saliency_map(rgb)
    safe = request.platform.safe_area

    checks: list[GateCheck] = []

    palette_score, mean_de = F.palette_adherence(rgb, request.theme.palette)
    checks.append(
        GateCheck(
            name="palette_adherence",
            value=round(palette_score, 4),
            threshold=THRESH_PALETTE,
            passed=palette_score >= THRESH_PALETTE,
            detail=f"weighted mean ΔE76 {mean_de:.1f} against the brand palette",
        )
    )

    safe_share = F.region_saliency_share(sal, safe.top, safe.bottom)
    checks.append(
        GateCheck(
            name="safe_area",
            value=round(safe_share, 4),
            threshold=THRESH_SAFE_AREA,
            passed=safe_share >= THRESH_SAFE_AREA,
            detail=(
                f"{safe_share:.0%} of attention sits clear of the top {safe.top:.0%} / "
                f"bottom {safe.bottom:.0%} platform chrome"
            ),
        )
    )

    focus = F.focal_concentration(sal)
    checks.append(
        GateCheck(
            name="focal_clarity",
            value=round(focus, 4),
            threshold=THRESH_FOCUS,
            passed=focus >= THRESH_FOCUS,
            detail="attention is concentrated on a clear subject rather than smeared",
        )
    )

    lum = F.mean_luminance(rgb)
    checks.append(
        GateCheck(
            name="exposure",
            value=round(lum, 4),
            threshold=MIN_LUMINANCE,
            passed=MIN_LUMINANCE <= lum <= MAX_LUMINANCE,
            detail=f"mean luminance {lum:.2f}; rejects fully black or blown-out frames",
        )
    )

    contrast = F.rms_contrast(rgb)
    checks.append(
        GateCheck(
            name="contrast",
            value=round(contrast, 4),
            threshold=MIN_CONTRAST,
            passed=contrast >= MIN_CONTRAST,
            detail=f"RMS contrast {contrast:.3f}; rejects flat or empty frames",
        )
    )

    checks.extend(_pending_check(name) for name in _PENDING)

    failed = [c for c in checks if not c.passed]
    if not failed:
        verdict = GateVerdict.PASS
    elif attempt == 1:
        # One retry with a stricter prompt, then give up. The retry allowance is
        # enforced by the cost governor, not here — this only expresses intent.
        verdict = GateVerdict.RETRY
    else:
        verdict = GateVerdict.REJECT

    return GateResult(verdict=verdict, checks=checks, attempt=attempt)


def stricter_prompt(original: str, result: GateResult) -> str:
    """Amend a prompt in response to the specific checks that failed.

    Retrying with the identical prompt is a waste of a paid generation, so the
    retry says something new about what went wrong.
    """
    failed = {c.name for c in result.failures}
    additions = []
    if "palette_adherence" in failed:
        additions.append(
            "Adhere strictly to the specified colour palette — the backdrop and "
            "wardrobe must come from those colours."
        )
    if "safe_area" in failed:
        additions.append(
            "Move the product and the model's face toward the vertical centre of the "
            "frame, well clear of the top and bottom edges."
        )
    if "focal_clarity" in failed:
        additions.append(
            "Make the product the single unmistakable focal point; simplify the "
            "background and remove competing detail."
        )
    if "exposure" in failed or "contrast" in failed:
        additions.append(
            "Use balanced commercial exposure with clear tonal separation between "
            "the subject and the background."
        )
    if not additions:
        return original
    return (
        original
        + "\n\nCORRECTIONS (previous attempt failed quality control):\n"
        + "\n".join(f"- {a}" for a in additions)
    )
