"""Candidates, gate results, scores and rankings.

The important structural idea in this module is the split between two things
that look similar and are not:

* :class:`GateResult` — a *hard* pass/fail. "Is this usable at all?"  Wrong
  product, deformed face, palette miles off, product buried under platform
  chrome.  Failures are rejected, not down-ranked.
* :class:`ScoreBreakdown` — a *learned* continuous prediction. "How well will
  this perform?"  Only ever computed for candidates that already passed the gate.

Keeping them apart means each can be tested on its own, and it stops a
quality defect from being laundered into a slightly lower performance score.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .brief import ShotBrief
from .enums import GateVerdict, Tier
from .request import AssetRef


class GateCheck(BaseModel):
    """One hard filter's outcome."""

    name: str = Field(description="e.g. 'product_identity', 'face_identity', 'palette_delta_e'")
    value: float
    threshold: float
    passed: bool
    higher_is_better: bool = True
    detail: str = ""
    implemented: bool = Field(
        default=True,
        description="False for checks whose model has not landed yet (DINOv2 product "
        "identity, ArcFace face identity). A pending check always passes, so it must "
        "be visibly distinguishable from one that genuinely verified something — "
        "otherwise the write-up would claim identity verification that never ran.",
    )


class GateResult(BaseModel):
    """The full quality-control verdict for a candidate."""

    verdict: GateVerdict
    checks: list[GateCheck] = Field(default_factory=list)
    attempt: int = Field(default=1, ge=1, description="1 = first try, 2 = after the one retry.")

    @property
    def failures(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def pending_checks(self) -> list[GateCheck]:
        """Checks that did not really run. Surfaced in the UI and the report."""
        return [c for c in self.checks if not c.implemented]

    @property
    def verified_identity(self) -> bool:
        """True only when identity checks actually executed.

        Guards against the pipeline reporting "passed all quality checks" when the
        identity models were not present to check anything.
        """
        names = {"product_identity", "face_identity"}
        ran = [c for c in self.checks if c.name in names and c.implemented]
        return len(ran) == len(names) and all(c.passed for c in ran)

    @property
    def reason(self) -> str:
        if self.verdict is GateVerdict.PASS:
            return "passed all quality checks"
        failed = ", ".join(f"{c.name} ({c.value:.3f} vs {c.threshold:.3f})" for c in self.failures)
        return failed or "no specific check recorded"


class ScoreBreakdown(BaseModel):
    """Predicted performance, decomposed.

    ``overall`` is what the ranker sorts on.  The component scores exist so the
    UI can explain a ranking and so the report can ablate feature groups — a
    single opaque number would make both impossible.
    """

    overall: float = Field(ge=0.0, le=1.0, description="Calibrated performance prediction.")

    # --- Shared image/video components ---
    aesthetic: float | None = Field(default=None, ge=0.0, le=1.0)
    prompt_alignment: float | None = Field(default=None, ge=0.0, le=1.0)
    product_salience: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Attention proxy: product's share of saliency."
    )
    composition: float | None = Field(default=None, ge=0.0, le=1.0)
    palette_adherence: float | None = Field(default=None, ge=0.0, le=1.0)

    # --- Video-only components ---
    temporal_consistency: float | None = Field(default=None, ge=0.0, le=1.0)
    motion_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    hook_strength: float | None = Field(
        default=None, ge=0.0, le=1.0, description="How arresting the first second is."
    )
    product_screen_time: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Fraction of frames the product is visible in."
    )
    safe_area_compliance: float | None = Field(default=None, ge=0.0, le=1.0)

    # --- Provenance ---
    model_version: str = Field(default="stub-0", description="Which predictor produced this.")
    is_stub: bool = Field(
        default=True,
        description="True until the trained head lands. Prevents stub numbers being "
        "mistaken for real predictions in the write-up.",
    )

    def components(self) -> dict[str, float]:
        """Non-null component scores, for explanation rendering."""
        skip = {"overall", "model_version", "is_stub"}
        return {
            k: v
            for k, v in self.model_dump().items()
            if k not in skip and isinstance(v, (int, float))
        }

    def top_drivers(self, n: int = 3) -> list[tuple[str, float]]:
        """Highest-scoring components — the 'why this ranked here' evidence."""
        return sorted(self.components().items(), key=lambda kv: kv[1], reverse=True)[:n]


class ImageCandidate(BaseModel):
    """One generated ad frame."""

    index: int = Field(ge=0)
    brief: ShotBrief
    asset: AssetRef
    tier: Tier = Tier.MOCK
    provider: str = "mock"
    seed: int = 0
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    gate: GateResult | None = None
    score: ScoreBreakdown | None = Field(
        default=None, description="Image-stage prediction. Only set once the gate passes."
    )

    @property
    def passed_gate(self) -> bool:
        return self.gate is not None and self.gate.verdict is GateVerdict.PASS


class VideoCandidate(BaseModel):
    """One generated ad video, animated from an :class:`ImageCandidate`."""

    index: int = Field(ge=0)
    source_image_index: int = Field(ge=0, description="Which image frame this was animated from.")
    brief: ShotBrief
    asset: AssetRef
    duration_seconds: float = Field(ge=0.0)
    fps: int = Field(default=24, ge=1)
    tier: Tier = Tier.MOCK
    provider: str = "mock"
    seed: int = 0
    cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    was_chained: bool = Field(
        default=False,
        description="True when 8-10 s was reached by concatenating two ~5 s clips "
        "rather than native long generation. Seam drift is measured when set.",
    )
    seam_consistency: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Temporal consistency across the concatenation seam; only when chained.",
    )

    score: ScoreBreakdown | None = None
    thumbnail: AssetRef | None = None
    platform_renders: dict[str, AssetRef] = Field(
        default_factory=dict, description="Aspect-ratio variants keyed by ratio, e.g. '9:16'."
    )


class RankedCandidate(BaseModel):
    """A video candidate placed in the final ordering, with its explanation."""

    rank: int = Field(ge=1)
    video: VideoCandidate
    explanation: str = Field(
        description="Human-readable justification built from the score components."
    )
    image_stage_rank: int | None = Field(
        default=None,
        ge=1,
        description="Where this candidate ranked before video generation. The "
        "agreement between this and `rank` is the project's headline measurement.",
    )

    @property
    def rank_shift(self) -> int | None:
        """How far the candidate moved between the image and video stages.

        Zero across the board would mean the image stage told us everything;
        large shifts mean animation matters. Either result is publishable.
        """
        if self.image_stage_rank is None:
            return None
        return self.image_stage_rank - self.rank
