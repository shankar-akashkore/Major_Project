"""Shared job contract for the ad-generation platform.

This package is the single source of truth for every type that crosses a
boundary — API, worker, ML code and (via generated TypeScript) the web app.
Nothing here imports torch, a provider SDK, or a database driver, so it stays
importable on an 8 GB laptop and inside a Colab notebook alike.
"""

from .annotation import (
    MAX_CATCH_GUESS_PROBABILITY,
    MAX_SIDE_BIAS_Z,
    MIN_CATCH_ACCURACY,
    MIN_CATCH_TRIALS,
    MIN_PLAUSIBLE_LATENCY_MS,
    MIN_REPEAT_CONSISTENCY,
    MIN_TRIALS_TO_JUDGE_AN_ANNOTATOR,
    AnnotatorProfile,
    AnnotatorQuality,
    Choice,
    ComparisonPair,
    CorpusItem,
    CorpusStats,
    ItemKind,
    Judgement,
    PairKind,
)
from .brief import BriefSet, DesignPoint, ShotBrief
from .candidates import (
    GateCheck,
    GateResult,
    ImageCandidate,
    RankedCandidate,
    ScoreBreakdown,
    VideoCandidate,
)
from .delivery import AudioReport, DeliveryReport, ReframeReport
from .enums import (
    AspectRatio,
    BackgroundTreatment,
    CameraAngle,
    Composition,
    GateVerdict,
    JobState,
    Lighting,
    Mood,
    MotionIntent,
    Platform,
    ProviderMode,
    SafeArea,
    Stage,
    Tier,
    Vertical,
)
from .intake import CutoutReport, FaceReport, IntakeReport, ReferenceReport
from .job import BudgetStatus, JobRecord, JobResult, SpendEntry, StageEvent
from .request import (
    DEFAULT_CANDIDATE_COUNT,
    MAX_DURATION_S,
    MIN_DURATION_S,
    AdJobRequest,
    AssetRef,
    AudienceSpec,
    ConsentAttestation,
    ThemeSpec,
)

__all__ = [
    # enums
    "AspectRatio",
    "BackgroundTreatment",
    "CameraAngle",
    "Composition",
    "GateVerdict",
    "JobState",
    "Lighting",
    "Mood",
    "MotionIntent",
    "Platform",
    "ProviderMode",
    "SafeArea",
    "Stage",
    "Tier",
    "Vertical",
    # request
    "AdJobRequest",
    "AssetRef",
    "AudienceSpec",
    "ConsentAttestation",
    "ThemeSpec",
    "DEFAULT_CANDIDATE_COUNT",
    "MIN_DURATION_S",
    "MAX_DURATION_S",
    # brief
    "BriefSet",
    "DesignPoint",
    "ShotBrief",
    # candidates
    "GateCheck",
    "GateResult",
    "ImageCandidate",
    "RankedCandidate",
    "ScoreBreakdown",
    "VideoCandidate",
    # annotation
    "AnnotatorProfile",
    "AnnotatorQuality",
    "Choice",
    "ComparisonPair",
    "CorpusItem",
    "CorpusStats",
    "ItemKind",
    "Judgement",
    "PairKind",
    "MAX_CATCH_GUESS_PROBABILITY",
    "MAX_SIDE_BIAS_Z",
    "MIN_CATCH_TRIALS",
    "MIN_CATCH_ACCURACY",
    "MIN_PLAUSIBLE_LATENCY_MS",
    "MIN_REPEAT_CONSISTENCY",
    "MIN_TRIALS_TO_JUDGE_AN_ANNOTATOR",
    # intake
    "CutoutReport",
    "FaceReport",
    "IntakeReport",
    "DeliveryReport",
    "ReframeReport",
    "AudioReport",
    "ReferenceReport",
    # job
    "BudgetStatus",
    "JobRecord",
    "JobResult",
    "SpendEntry",
    "StageEvent",
]
