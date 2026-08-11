"""Job lifecycle, progress events and the spend ledger."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .brief import BriefSet
from .candidates import ImageCandidate, RankedCandidate, VideoCandidate
from .delivery import DeliveryReport
from .enums import JobState, Stage
from .intake import IntakeReport
from .request import AdJobRequest


class StageEvent(BaseModel):
    """One progress tick, streamed to the UI over SSE."""

    job_id: str
    stage: Stage
    state: str = Field(description="'started' | 'progress' | 'completed' | 'failed'")
    message: str = ""
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction within the stage.")
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def overall_progress(self) -> float:
        """Fraction of the whole pipeline complete, for a single progress bar."""
        total = len(list(Stage))
        return min(1.0, (self.stage.index + self.progress) / total)


class SpendEntry(BaseModel):
    """One line in the cost ledger.

    Every external call that could cost money writes one of these *before* the
    call is made, so a crash mid-request cannot lose the charge. The governor
    reads the sum to decide whether the next call is allowed.
    """

    entry_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    job_id: str | None = None
    provider: str
    operation: str = Field(description="'image' | 'video' | 'llm'")
    model: str = ""
    quantity: float = Field(default=1.0, description="Images, or seconds of video.")
    unit_cost_usd: float = Field(default=0.0, ge=0.0)
    cost_usd: float = Field(ge=0.0)
    cache_hit: bool = Field(default=False, description="Cache hits are recorded at zero cost.")
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    note: str = ""


class BudgetStatus(BaseModel):
    """A snapshot the UI shows and the governor enforces."""

    total_budget_usd: float
    spent_usd: float
    per_job_cap_usd: float

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.total_budget_usd - self.spent_usd)

    @property
    def fraction_used(self) -> float:
        if self.total_budget_usd <= 0:
            return 1.0
        return min(1.0, self.spent_usd / self.total_budget_usd)

    def can_afford(self, amount_usd: float) -> bool:
        return self.spent_usd + amount_usd <= self.total_budget_usd


class JobResult(BaseModel):
    """Everything the pipeline produced for one job."""

    job_id: str
    intake: IntakeReport | None = Field(
        default=None,
        description="What stage 1 made of the uploads: which were usable, where the "
        "palette came from, whether a product cutout was produced.",
    )
    briefs: BriefSet | None = None
    images: list[ImageCandidate] = Field(default_factory=list)
    videos: list[VideoCandidate] = Field(default_factory=list)
    ranking: list[RankedCandidate] = Field(default_factory=list)
    delivery: DeliveryReport | None = Field(
        default=None,
        description="What stage 8 produced: platform renders and how lossy each "
        "reframe was, mockup previews, the audio mix's provenance, and the bundle.",
    )

    total_cost_usd: float = Field(default=0.0, ge=0.0)

    @property
    def winner(self) -> RankedCandidate | None:
        return self.ranking[0] if self.ranking else None

    @property
    def image_stage_order(self) -> list[int]:
        """Candidate indices ordered by image-stage prediction (best first)."""
        scored = [c for c in self.images if c.score is not None]
        return [c.index for c in sorted(scored, key=lambda c: -c.score.overall)]

    @property
    def video_stage_order(self) -> list[int]:
        """Candidate indices ordered by final video ranking (best first)."""
        return [r.video.source_image_index for r in sorted(self.ranking, key=lambda r: r.rank)]


class JobRecord(BaseModel):
    """The persisted job: request, state, progress history and results."""

    request: AdJobRequest
    state: JobState = JobState.QUEUED
    result: JobResult | None = None

    events: list[StageEvent] = Field(default_factory=list)
    error: str | None = None
    refusal_reason: str | None = Field(
        default=None, description="Set when the governor refused the job over budget."
    )

    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def job_id(self) -> str:
        return self.request.job_id

    @property
    def current_stage(self) -> Stage | None:
        return self.events[-1].stage if self.events else None

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()
