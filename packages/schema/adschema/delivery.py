"""What stage 8 produced, and what it cost to produce it.

The reason this is a schema rather than a pile of files: reframing an ad for a
platform it was not generated for is *lossy*, and the loss is a fact about the
deliverable that the person publishing it needs. A 9:16 clip reframed to 16:9 keeps
about a third of its height. Whether that is acceptable is a judgement, and it can
only be made by someone who is told.

So every render carries how it was produced and how much salient content survived,
and the platform variant that lost the most says so in prose.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .request import AssetRef


class ReframeReport(BaseModel):
    """One aspect-ratio variant, and the cost of producing it."""

    aspect_ratio: str = Field(description="e.g. '9:16'")
    asset: AssetRef
    mode: str = Field(description="'crop' | 'pad' | 'none'")
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    retained_salience: float = Field(
        ge=0.0,
        le=1.0,
        description="Share of attention-weighted content the reframe kept. 1.0 for a "
        "no-op; a padded variant keeps everything visible but smaller, so its figure "
        "describes what a crop *would* have kept and is why padding was chosen.",
    )
    centre_crop_salience: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="What a naive centre crop would have kept. Carried so the smart "
        "crop's benefit is a comparison rather than an unanchored number.",
    )
    tracking_gain: float = Field(
        default=0.0,
        description="Extra salience a per-frame tracking window would have kept. The "
        "price of holding the crop still, which is paid deliberately: a jittering "
        "frame is worse to watch than a slightly mis-framed steady one.",
    )
    note: str = ""

    @property
    def improvement_over_centre(self) -> float | None:
        if self.centre_crop_salience is None:
            return None
        return self.retained_salience - self.centre_crop_salience


class AudioReport(BaseModel):
    """The mix, with the provenance required to publish it.

    ``credit`` is not decoration. An ad is a commercial artefact and a submitted
    project is a published one, so audio whose licence is unrecorded cannot ship —
    ``adml.audio.AudioBed`` refuses to be constructed without it, and this is where
    it surfaces to whoever downloads the file.
    """

    attached: bool = False
    credit: str = ""
    licence: str = ""
    target_lufs: float | None = None
    measured_lufs: float | None = None
    has_voiceover: bool = False
    is_test_signal: bool = Field(
        default=False,
        description="True when the bed is a synthesised tone rather than music. Keeps "
        "a development placeholder from being described as a soundtrack.",
    )
    note: str = ""


class DeliveryReport(BaseModel):
    """Stage 8's output for one job."""

    renders: list[ReframeReport] = Field(default_factory=list)
    previews: dict[str, AssetRef] = Field(
        default_factory=dict,
        description="Platform mockup frames, keyed by platform value. The video frame "
        "with the platform's chrome overlaid, so a product sitting under the caption "
        "bar is visible before publishing rather than after.",
    )
    audio: AudioReport = Field(default_factory=AudioReport)
    bundle: AssetRef | None = Field(
        default=None, description="Zip of every render, preview and report card."
    )
    warnings: list[str] = Field(default_factory=list)

    @property
    def worst_reframe(self) -> ReframeReport | None:
        """The variant that lost the most, which is the one worth looking at."""
        lossy = [r for r in self.renders if r.mode != "none"]
        return min(lossy, key=lambda r: r.retained_salience) if lossy else None

    def summary(self) -> str:
        parts = [f"{len(self.renders)} renders", f"{len(self.previews)} previews"]
        worst = self.worst_reframe
        if worst is not None:
            parts.append(
                f"worst reframe {worst.aspect_ratio} at {worst.retained_salience:.0%} "
                f"salience via {worst.mode}"
            )
        if self.audio.attached:
            parts.append("audio attached")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return ", ".join(parts)
