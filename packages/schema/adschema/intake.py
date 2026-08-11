"""What intake preprocessing found, and what it could not check.

Intake is the only stage that inspects the user's *uploads* rather than the
system's own output, so it is the only place that can catch an input no generator
could rescue: a thumbnail-sized product photo, a motion-blurred snap, a reference
whose colours will never yield a brand palette.  Catching those here costs
nothing; catching them at the quality gate costs three paid generations.

Two reporting conventions carry over from :mod:`adschema.candidates`, for the same
reason they exist there:

* **Blocking checks are structured, advisories are prose.**  A check that can
  refuse a job needs a measured value and a threshold so the refusal can be
  argued with.  A warning that the product photo is a little soft is guidance,
  and a number with no consequence attached only looks like rigour.
* **Anything that did not really run says so.**  Background removal and face
  detection both depend on optional packages this laptop may not have.  When they
  are absent the report records ``method="none"`` / ``implemented=False`` rather
  than a plausible-looking default, so the write-up cannot claim a cutout or an
  identity check that never happened.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .candidates import GateCheck
from .request import AssetRef


class ReferenceReport(BaseModel):
    """Per-upload validation outcome."""

    role: str = Field(description="'human_model' | 'product' | 'logo'")
    asset: AssetRef
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    checks: list[GateCheck] = Field(
        default_factory=list,
        description="Blocking checks only. A failure here refuses the job.",
    )
    measurements: list[GateCheck] = Field(
        default_factory=list,
        description="Measured and reported but never blocking. A measurement is here "
        "rather than in `checks` when the quantity is real but its threshold is not yet "
        "calibrated against real uploads — recording the value on every job is what "
        "makes that calibration possible later.",
    )
    advisories: list[str] = Field(
        default_factory=list,
        description="Non-blocking guidance shown to the user, e.g. a soft or small upload.",
    )

    @property
    def failures(self) -> list[GateCheck]:
        """Blocking failures only — deliberately not including ``measurements``."""
        return [c for c in self.checks if not c.passed]

    @property
    def usable(self) -> bool:
        return not self.failures

    @property
    def reason(self) -> str:
        if self.usable:
            return "accepted"
        return ", ".join(f"{c.name} ({c.value:.3f} vs {c.threshold:.3f})" for c in self.failures)


class CutoutReport(BaseModel):
    """Whether the product was successfully separated from its background.

    A clean cutout is the single cheapest way to improve product fidelity in
    multi-reference composition, which is why this is tracked explicitly rather
    than done silently: if the cutout was declined, the write-up needs to say the
    generator saw the original photograph instead.
    """

    method: str = Field(
        default="none",
        description="'rembg' (trained matting), 'flood-fill' (numpy fallback), or 'none'.",
    )
    accepted: bool = Field(
        default=False,
        description="False when no cutout was produced, or one was produced and rejected "
        "by its own sanity check. The original product image is used instead.",
    )
    asset: AssetRef | None = Field(default=None, description="Transparent PNG, when accepted.")
    coverage: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Opaque fraction of the cutout. Near 1.0 means nothing was removed; "
        "near 0.0 means the subject was removed along with the background.",
    )
    border_uniformity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How flat the source backdrop was. Gates the flood-fill fallback.",
    )
    detail: str = ""


class FaceReport(BaseModel):
    """Face presence in the human-model reference.

    Detection is a Haar cascade when OpenCV is installed and nothing at all when
    it is not.  The box is kept even though nothing uses it yet: ArcFace identity
    verification in the Colab work needs a face crop, and recording it now means
    that stage does not have to re-detect.

    Known limitation, stated because it belongs in the ethics section rather than
    a footnote: Haar cascades were trained on datasets skewed toward lighter skin
    and frontal poses, and their miss rate is not uniform across faces.  That is
    why a miss produces an advisory and never a refusal — a detector with uneven
    error rates must not be allowed to decide whose photograph is acceptable.
    """

    detector: str = Field(default="unavailable", description="'opencv-haar' or 'unavailable'.")
    implemented: bool = Field(
        default=False, description="False when no detector was available to run."
    )
    faces_found: int = Field(default=0, ge=0)
    box: list[int] | None = Field(
        default=None, description="Largest face as [x, y, w, h] in source pixels."
    )
    box_area_fraction: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Face area as a fraction of the frame."
    )
    detail: str = ""


class IntakeReport(BaseModel):
    """Everything stage 1 established about the uploads."""

    references: list[ReferenceReport] = Field(default_factory=list)

    palette: list[str] = Field(
        default_factory=list, description="Resolved brand palette, supplied or extracted."
    )
    palette_weights: list[float] = Field(
        default_factory=list, description="Coverage of each palette colour in its source region."
    )
    palette_source: str = Field(
        default="user",
        description="'user', 'product-cutout', 'product-frame', or 'none'. Recorded because "
        "a palette read off a whole photograph includes its backdrop, which is a "
        "materially weaker constraint than one read from inside the product mask.",
    )

    cutout: CutoutReport = Field(default_factory=CutoutReport)
    face: FaceReport = Field(default_factory=FaceReport)
    advisories: list[str] = Field(default_factory=list, description="Job-level guidance.")

    @property
    def unusable(self) -> list[ReferenceReport]:
        return [r for r in self.references if not r.usable]

    @property
    def blocking_reason(self) -> str | None:
        """Why the job cannot proceed, or ``None`` if it can."""
        bad = self.unusable
        if not bad:
            return None
        return "; ".join(f"{r.role}: {r.reason}" for r in bad)

    @property
    def all_advisories(self) -> list[str]:
        """Job-level plus per-reference guidance, flattened for display."""
        out = list(self.advisories)
        for ref in self.references:
            out.extend(f"{ref.role}: {note}" for note in ref.advisories)
        return out

    def product_reference(self, original: AssetRef) -> AssetRef:
        """The asset image generation should actually receive for the product.

        The cutout when there is a trustworthy one, the original otherwise.  Going
        through one accessor keeps the pipeline from having to know the difference,
        and keeps the fallback from being silent — ``cutout.accepted`` records it.
        """
        if self.cutout.accepted and self.cutout.asset is not None:
            return self.cutout.asset
        return original
