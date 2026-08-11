"""Stage 1 — validate the uploads, cut out the product, derive the palette.

This is the only stage that looks at what the *user* supplied rather than what the
system produced, which makes it the cheapest place in the pipeline to stop a job.
An 800×600 motion-blurred phone photo cannot become a usable ad no matter how good
the generator is, and discovering that at the quality gate means three paid
generations have already been spent proving it.

Three jobs, in order of how much they change the output:

1. **Product cutout.**  Multi-reference composition holds a product's identity far
   better when the reference is the product alone on transparency than when it is
   the product plus whatever it was photographed against.  Preferred path is
   ``rembg`` (a trained matting model); the fallback is a flood fill that is only
   trusted when the backdrop measures flat enough for it to be safe.

2. **Palette extraction.**  Reading dominant colours from inside the product mask
   rather than off the whole frame is the difference between a brand palette and a
   description of the user's tablecloth.  Measured on the calibration fixtures: a
   lifestyle product shot yields ``['#6f614b', '#cabba6', '#e6e1d5', '#2c3b54',
   '#a15d4d']`` from the frame — four backdrop colours and the product fourth — and
   ``['#2b3a55', '#f0ece2', '#c6a05c']`` from the mask, which is exactly the
   product's three real colours.

3. **Reference validation.**  Resolution and focus, blocking only where the input
   is unambiguously unusable and advising everywhere else.

*Known risk to revisit in week 6.*  When no cutout is available the palette is read
from the whole frame, and that palette then constrains the quality gate.  A palette
made of backdrop colours would make the gate demand generated images match the
user's tablecloth.  ``palette_source`` records which case applied so the
recalibration against real generations can measure whether it matters; until then
the frame-derived case carries an advisory.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from adml import features as F
from adproviders import Storage
from adschema import (
    AdJobRequest,
    AssetRef,
    CutoutReport,
    FaceReport,
    GateCheck,
    IntakeReport,
    ReferenceReport,
)
from PIL import Image

# --- Thresholds ------------------------------------------------------------
#
# Calibrated by ``scripts/calibrate_intake.py`` against synthetic fixtures whose
# ground-truth product mask is known, so the flood-fill threshold is set from the
# segmentation's measured IoU rather than from how plausible its output looks.
# Measured ranges are recorded beside each value for the week-6 recalibration.

#: Below this short edge an upload cannot supply detail the generator can use.
#: Blocking. Measured: a 640 px fixture downscaled to 192 px still reads as
#: focused (0.47) — resolution has to be checked on its own terms, not inferred
#: from sharpness.
MIN_EDGE_PX = 256
#: Advisory below this: usable, but the generator will be working from little.
SOFT_EDGE_PX = 512

#: Focus floor — **advisory, not blocking**, and the reason is worth stating.
#:
#: The calibration fixtures are vector drawings, and hard vector edges produce far
#: more Laplacian variance than photographic detail does: the "usable" class scored
#: 0.961-1.000, which is not a range real photographs occupy. A threshold set
#: against that class looked safe and was not — a flat-shaded synthetic portrait
#: measures 0.108 and would have been refused. There are no real uploads to
#: calibrate against yet, so the value is measured and reported on every job and
#: blocks nothing. Promote it to a blocking check in week 6, once real photographs
#: have supplied an honest distribution.
#:
#: Reference points from the synthetic run: whole-frame blur σ=1 at 0.607, σ=2 at
#: 0.149, σ=4 at 0.026, a 128 px upload at 0.188.
#:
#: Known weakness to account for when that calibration happens: Laplacian variance
#: reads sensor grain as detail, so a grainy out-of-focus photo scores higher than
#: a clean one. It measures high-frequency energy, not focus as such.
MIN_SHARPNESS = 0.35
#: Second advisory band: soft enough to mention, well short of unusable.
SOFT_SHARPNESS = 0.75

#: Flood-fill is only trusted above this border uniformity. Measured: backdrops
#: that flood correctly score 0.498-1.000 and reach IoU 0.985 against the true
#: product mask; backdrops that do not score 0.208-0.217 and reach IoU 0.25.
#: The two classes do not overlap, which is what makes this fallback safe to ship.
MIN_BORDER_UNIFORMITY = 0.35
#: ΔE latitude for the flood. Measured: 8 leaves a gradient sweep at IoU 0.419,
#: 12 lifts it to 0.985, and 24 adds nothing while risking eating into the product.
FLOOD_TOLERANCE_DE = 12.0

#: A cutout retaining less than this is a failed matte, not a product.
MIN_CUTOUT_COVERAGE = 0.02
#: A cutout retaining more than this removed essentially nothing — which happens
#: legitimately when the product already fills the frame. Not an error; the
#: original is simply used instead.
MAX_CUTOUT_COVERAGE = 0.95

#: Palette slots. ``ThemeSpec.palette`` caps at 6; five distinct hues is already
#: more than most brand guidelines carry.
PALETTE_SIZE = 5

#: Per-role rules: ``(min_edge_px, min_sharpness | None)``. A logo is legitimately
#: small and legitimately flat — blocking it on either would refuse valid input.
_ROLE_RULES: dict[str, tuple[int, float | None]] = {
    "human_model": (MIN_EDGE_PX, MIN_SHARPNESS),
    "product": (MIN_EDGE_PX, MIN_SHARPNESS),
    "logo": (96, None),
}


# --- Optional dependencies -------------------------------------------------


def _remove_background_rembg(rgb: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Trained matting, when ``rembg`` is installed.

    An ``ImportError`` and a runtime failure are reported differently on purpose:
    the first is an expected environment state, the second is something gone wrong
    that should be visible rather than looking like an absent package.
    """
    try:
        from rembg import remove
    except ImportError:
        return None, "rembg is not installed (pip install '.[intake]')"
    try:
        out = remove(Image.fromarray(rgb[..., :3]))
    except Exception as exc:  # pragma: no cover - depends on an absent package
        return None, f"rembg failed: {type(exc).__name__}: {exc}"
    return np.asarray(out.convert("RGBA")), "rembg"


@lru_cache(maxsize=1)
def _face_cascade() -> tuple[Any | None, str]:
    """Load the Haar cascade once, or explain why it is unavailable."""
    try:
        import cv2
    except ImportError:
        return None, "OpenCV is not installed (pip install '.[intake]')"

    path = Path(getattr(cv2, "data", None).haarcascades) / "haarcascade_frontalface_default.xml"
    if not path.is_file():  # pragma: no cover - depends on the OpenCV build
        return None, f"OpenCV is installed but its Haar cascade is missing at {path}"
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():  # pragma: no cover - depends on the OpenCV build
        return None, f"the Haar cascade at {path} failed to load"
    return cascade, "opencv-haar"


# --- Reference validation ---------------------------------------------------


def _validate_reference(role: str, asset: AssetRef, rgb: np.ndarray) -> ReferenceReport:
    """Blocking checks and advisories for one upload."""
    height, width = rgb.shape[:2]
    short_edge = min(height, width)
    min_edge, min_sharpness = _ROLE_RULES.get(role, (MIN_EDGE_PX, MIN_SHARPNESS))

    # Resolution is the only blocking check: pixel count is a fact about the file,
    # needing no calibration to interpret. Everything else is measured and reported.
    checks = [
        GateCheck(
            name="resolution",
            value=float(short_edge),
            threshold=float(min_edge),
            passed=short_edge >= min_edge,
            detail=f"{width}x{height}; the short edge must be at least {min_edge}px",
        )
    ]
    measurements: list[GateCheck] = []
    advisories: list[str] = []

    focus = F.sharpness(rgb)
    if min_sharpness is not None:
        measurements.append(
            GateCheck(
                name="focus",
                value=round(focus, 4),
                threshold=min_sharpness,
                passed=focus >= min_sharpness,
                detail=(
                    f"focus {focus:.2f} of 1.0, measured on the sharpest regions so a "
                    "deliberately blurred background is not penalised. Advisory until "
                    "calibrated against real photographs — see MIN_SHARPNESS."
                ),
            )
        )
        if focus < min_sharpness:
            advisories.append(
                f"looks out of focus or heavily upscaled (focus {focus:.2f} of 1.0, below "
                f"{min_sharpness:.2f}); the generated frames will inherit the softness"
            )
        elif focus < SOFT_SHARPNESS:
            advisories.append(
                f"soft upload (focus {focus:.2f}); fine detail such as label text may not "
                "survive into the generated frames"
            )

    if min_edge <= short_edge < SOFT_EDGE_PX:
        advisories.append(
            f"small upload ({width}x{height}); {SOFT_EDGE_PX}px or more on the short edge "
            "gives the generator noticeably more to work from"
        )

    aspect = max(width, height) / max(1, min(width, height))
    if aspect > 3.0 and role != "logo":
        advisories.append(
            f"unusually elongated ({width}x{height}, {aspect:.1f}:1); expect heavy cropping "
            f"to reach the target frame"
        )

    return ReferenceReport(
        role=role,
        asset=asset,
        width=width,
        height=height,
        checks=checks,
        measurements=measurements,
        advisories=advisories,
    )


# --- Product cutout ---------------------------------------------------------


def _encode_rgba(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba.astype(np.uint8), "RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _cutout(rgb: np.ndarray, storage: Storage, key: str) -> CutoutReport:
    """Separate the product from its background, or explain why not.

    Order matters: the trained model first, the numpy fallback only if it is
    absent, and the fallback's own precondition checked before it runs.
    """
    uniformity = round(F.border_uniformity(rgb), 4)

    rgba, method = _remove_background_rembg(rgb)
    note = method
    if rgba is None:
        unavailable = note
        if uniformity < MIN_BORDER_UNIFORMITY:
            return CutoutReport(
                method="none",
                accepted=False,
                border_uniformity=uniformity,
                detail=(
                    f"{unavailable}; the flood-fill fallback was declined because the "
                    f"backdrop is too varied for it (uniformity {uniformity:.2f} < "
                    f"{MIN_BORDER_UNIFORMITY:.2f}) and would cut into the product. "
                    "The original product photo is used as the reference."
                ),
            )
        background = F.background_mask_by_flood(rgb, tolerance_de=FLOOD_TOLERANCE_DE)
        rgba = np.dstack([rgb[..., :3], (~background).astype(np.uint8) * 255])
        method = "flood-fill"
        note = (
            f"{unavailable}; used the flood-fill fallback, which the backdrop is flat "
            f"enough for (uniformity {uniformity:.2f})"
        )

    coverage = round(F.alpha_coverage(rgba), 4)
    if coverage < MIN_CUTOUT_COVERAGE:
        return CutoutReport(
            method=method,
            accepted=False,
            coverage=coverage,
            border_uniformity=uniformity,
            detail=(
                f"{note}, but it kept only {coverage:.1%} of the frame — that is a failed "
                "matte rather than a product. Using the original photo instead."
            ),
        )
    if coverage > MAX_CUTOUT_COVERAGE:
        return CutoutReport(
            method=method,
            accepted=False,
            coverage=coverage,
            border_uniformity=uniformity,
            detail=(
                f"{note}, but it kept {coverage:.1%} of the frame, so there was effectively "
                "no background to remove. Using the original photo, which is equivalent."
            ),
        )

    asset = storage.put_bytes(key, _encode_rgba(rgba), "image/png")
    asset.height, asset.width = rgba.shape[:2]
    return CutoutReport(
        method=method,
        accepted=True,
        asset=asset,
        coverage=coverage,
        border_uniformity=uniformity,
        detail=f"{note}; kept {coverage:.1%} of the frame as product",
    )


# --- Face presence ----------------------------------------------------------


def _detect_face(rgb: np.ndarray) -> FaceReport:
    """Locate the largest face in the human reference, if a detector is available.

    A miss is never fatal — see :class:`~adschema.intake.FaceReport` for why a
    detector with uneven error rates across faces must not gate a job.
    """
    cascade, note = _face_cascade()
    if cascade is None:
        return FaceReport(
            detector="unavailable",
            implemented=False,
            detail=f"{note}; face presence was not checked",
        )

    height, width = rgb.shape[:2]
    grey = np.asarray(Image.fromarray(rgb[..., :3]).convert("L"))
    minimum = max(24, round(min(height, width) * 0.08))
    faces = cascade.detectMultiScale(
        grey, scaleFactor=1.1, minNeighbors=5, minSize=(minimum, minimum)
    )

    if len(faces) == 0:
        return FaceReport(
            detector="opencv-haar",
            implemented=True,
            faces_found=0,
            detail=(
                "no frontal face found. This is advisory only: the detector misses "
                "profiles, occlusions and — unevenly — darker skin tones, so it is not "
                "allowed to refuse a job. Check the human reference is the right upload."
            ),
        )

    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    fraction = round(float(w * h) / float(width * height), 4)
    return FaceReport(
        detector="opencv-haar",
        implemented=True,
        faces_found=len(faces),
        box=[int(x), int(y), int(w), int(h)],
        box_area_fraction=min(1.0, fraction),
        detail=(
            f"{len(faces)} frontal face(s); largest covers {fraction:.1%} of the frame. "
            "Box retained for the ArcFace identity crop in the Colab work."
        ),
    )


# --- Palette ---------------------------------------------------------------


def _resolve_palette(
    request: AdJobRequest,
    product_rgb: np.ndarray,
    cutout: CutoutReport,
    storage: Storage,
) -> tuple[list[str], list[float], str, list[str]]:
    """``(palette, weights, source, advisories)``.

    A user-supplied palette always wins — it is a brand decision, and guessing
    over it would be presumptuous as well as wrong.
    """
    if request.theme.palette:
        return list(request.theme.palette), [], "user", []

    mask: np.ndarray | None = None
    source = "product-frame"
    advisories: list[str] = []

    if cutout.accepted and cutout.asset is not None:
        rgba = F.load_rgba(storage.get_bytes(cutout.asset.key))
        if rgba.shape[:2] == product_rgb.shape[:2]:
            mask = rgba[..., 3] > 8
            source = "product-cutout"
        else:  # pragma: no cover - only if a matting model rescales its output
            advisories.append(
                "the product cutout came back at a different size than the source, so the "
                "palette was read from the whole frame instead"
            )

    if mask is None:
        advisories.append(
            "no product cutout was available, so the brand palette was read from the whole "
            "product photo and may include its background — supply a palette explicitly if "
            "the extracted colours look wrong"
        )

    pairs = F.extract_palette(product_rgb, k=PALETTE_SIZE, mask=mask)
    if not pairs:  # pragma: no cover - requires a degenerate image
        return [], [], "none", [*advisories, "no palette could be extracted from the product"]

    return [c for c, _ in pairs], [w for _, w in pairs], source, advisories


# --- Entry point -----------------------------------------------------------


def preprocess(request: AdJobRequest, storage: Storage) -> IntakeReport:
    """Validate and preprocess the uploads for one job.

    Synchronous by design: it is all numpy and PIL, there is no I/O to overlap,
    and pretending otherwise would only add an await to every call site.
    """
    human_rgb = F.load_image(storage.get_bytes(request.human_model_image.key))
    product_rgb = F.load_image(storage.get_bytes(request.product_image.key))

    references = [
        _validate_reference("human_model", request.human_model_image, human_rgb),
        _validate_reference("product", request.product_image, product_rgb),
    ]
    if request.logo_image is not None:
        logo_rgb = F.load_image(storage.get_bytes(request.logo_image.key))
        references.append(_validate_reference("logo", request.logo_image, logo_rgb))

    report = IntakeReport(references=references)

    # Nothing below this point is worth doing on inputs the job will be refused
    # over — and the cutout is the most expensive part of the stage.
    if report.blocking_reason is not None:
        return report

    report.cutout = _cutout(product_rgb, storage, f"intake/{request.job_id}/product_cutout.png")
    report.face = _detect_face(human_rgb)

    palette, weights, source, advisories = _resolve_palette(
        request, product_rgb, report.cutout, storage
    )
    report.palette = palette
    report.palette_weights = weights
    report.palette_source = source
    report.advisories = advisories

    return report
