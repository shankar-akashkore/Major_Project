"""Design points and shot briefs.

The pipeline separates *what to vary* from *how to say it*:

* A :class:`DesignPoint` is a coordinate in the discrete design space, chosen by
  the sampler.  Diversity is guaranteed here, geometrically — not by hoping an
  LLM returns three different ideas when asked nicely.  That distinction is what
  makes candidate diversity measurable, and lets the report ablate the sampler.
* A :class:`ShotBrief` is that coordinate expanded into prompt-ready language,
  plus the motion prompt the video stage will need later.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .enums import (
    CameraAngle,
    Composition,
    Lighting,
    MotionIntent,
)


class DesignPoint(BaseModel):
    """One sampled coordinate in the creative design space."""

    index: int = Field(ge=0, description="Candidate slot this point belongs to.")
    angle: CameraAngle
    lighting: Lighting
    composition: Composition
    motion: MotionIntent
    seed: int

    def axes(self) -> tuple[str, str, str, str]:
        """The categorical coordinate, for diversity measurement and logging."""
        return (self.angle.value, self.lighting.value, self.composition.value, self.motion.value)

    def distance(self, other: DesignPoint) -> int:
        """Hamming distance over the four axes.

        The sampler maximises the minimum pairwise distance, which is how three
        candidates end up genuinely distinct instead of three near-duplicates.
        """
        return sum(a != b for a, b in zip(self.axes(), other.axes(), strict=True))


class ShotBrief(BaseModel):
    """A single candidate's full creative instruction set."""

    index: int = Field(ge=0)
    design_point: DesignPoint

    # --- Prompts sent to the providers ---
    image_prompt: str = Field(
        description="Multi-reference composition prompt. Must name the reference "
        "roles explicitly ('Image 1 is the model...') — vague prompts cause identity drift."
    )
    negative_prompt: str = Field(default="")
    motion_prompt: str = Field(
        description="Image-to-video instruction applied to this candidate's frame."
    )
    video_negative_prompt: str = Field(
        default="",
        description="Negatives for the video stage. Separate from negative_prompt "
        "because the stages fail differently: the image stage produces artefacts, "
        "the video stage produces a still photograph with a moving camera, and the "
        "image list never mentions motion at all.",
    )

    # --- Human-readable rationale, surfaced in the UI and the report ---
    concept: str = Field(default="", description="One-line description of the creative idea.")

    @property
    def slot_label(self) -> str:
        """Short label for the UI, e.g. 'A · low angle / rim backlit'."""
        letter = chr(ord("A") + self.index)
        dp = self.design_point
        return (
            f"{letter} · {dp.angle.value.replace('_', ' ')} / {dp.lighting.value.replace('_', ' ')}"
        )


class BriefSet(BaseModel):
    """All briefs for one job, plus the diversity metric the sampler achieved."""

    job_id: str
    briefs: list[ShotBrief]
    min_pairwise_distance: int = Field(
        default=0,
        description="Minimum Hamming distance between any two design points. "
        "Reported as the sampler's diversity guarantee; 0 means duplicates exist.",
    )
    sampler: str = Field(default="latin_hypercube", description="Which sampler produced these.")
    prompt_style: str = Field(
        default="full",
        description="Which prompt template produced these. Recorded per job so an "
        "ablation over prompt verbosity can be attributed after the fact rather "
        "than reconstructed from timestamps.",
    )

    def __len__(self) -> int:
        return len(self.briefs)
