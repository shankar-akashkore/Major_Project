"""Provider interfaces.

Every external call in this project goes through one of these three ABCs.  The
reason is not architectural tidiness — it is that generation models are being
deprecated and repriced constantly, and a semester-long project that hard-codes
one vendor will break mid-semester.  Swapping ``kling-3`` for ``seedance-2.5``
should touch exactly one file.

Each provider declares its price model so the cost governor can estimate a
charge *before* the call is made.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from adschema import AspectRatio, AssetRef, ShotBrief, Tier

from .pricing import (
    IMAGE_PRICES,
    VIDEO_PRICES,
    estimate_image_cost,
    estimate_video_cost,
)


class ProviderError(RuntimeError):
    """A provider call failed in a way the pipeline may retry once."""


class ProviderUnavailable(ProviderError):
    """Provider is misconfigured or unreachable — do not retry."""


@dataclass
class ImageGenRequest:
    """One multi-reference composition call.

    ``references`` is ordered and the order is load-bearing: the prompt refers to
    the inputs positionally ("Image 1 is the model, Image 2 is the product"),
    which is what keeps identity from drifting.
    """

    brief: ShotBrief
    references: list[AssetRef]
    aspect_ratio: AspectRatio
    seed: int
    output_key: str
    reference_roles: list[str] = field(default_factory=list)
    #: Brand hex colours, for providers that accept a palette hint.
    palette: list[str] = field(default_factory=list)


@dataclass
class ImageGenResult:
    asset: AssetRef
    model: str
    tier: Tier
    cost_usd: float
    latency_ms: int
    seed: int
    cache_hit: bool = False
    raw: dict | None = None


@dataclass
class VideoGenRequest:
    """One image-to-video call: the frame becomes frame 1, motion does the rest."""

    brief: ShotBrief
    start_image: AssetRef
    duration_seconds: float
    aspect_ratio: AspectRatio
    seed: int
    output_key: str
    fps: int = 24


@dataclass
class VideoGenResult:
    asset: AssetRef
    model: str
    tier: Tier
    cost_usd: float
    latency_ms: int
    seed: int
    duration_seconds: float
    fps: int = 24
    was_chained: bool = False
    seam_consistency: float | None = None
    cache_hit: bool = False
    raw: dict | None = None


class ImageProvider(ABC):
    """Multi-reference image composition."""

    name: str = "abstract"
    model: str = "abstract"

    @property
    def tier(self) -> Tier:
        return IMAGE_PRICES[self.model].tier

    @property
    def max_reference_images(self) -> int:
        return IMAGE_PRICES[self.model].max_reference_images

    def estimate_cost(self, count: int = 1) -> float:
        return estimate_image_cost(self.model, count)

    @abstractmethod
    async def generate(self, request: ImageGenRequest) -> ImageGenResult: ...


class VideoProvider(ABC):
    """Image-to-video animation."""

    name: str = "abstract"
    model: str = "abstract"

    @property
    def tier(self) -> Tier:
        return VIDEO_PRICES[self.model].tier

    @property
    def native_max_seconds(self) -> float:
        return VIDEO_PRICES[self.model].native_max_seconds

    def supports_duration(self, seconds: float) -> bool:
        """Whether this model reaches ``seconds`` without a concatenation seam.

        The pipeline checks this before spending: a False here means the clip
        will be chained, and chained clips carry a measured seam-consistency
        score so the quality cost is visible rather than hidden.
        """
        return seconds <= self.native_max_seconds

    def estimate_cost(self, seconds: float, count: int = 1) -> float:
        return estimate_video_cost(self.model, seconds, count)

    @abstractmethod
    async def generate(self, request: VideoGenRequest) -> VideoGenResult: ...


class LLMProvider(ABC):
    """Structured brief expansion.

    Note what this is *not* used for: it never ranks candidates.  An LLM judge
    appears in the evaluation only as a baseline to beat, because "learning-based
    performance prediction" has to mean a trained model, not a prompt.
    """

    name: str = "abstract"
    model: str = "abstract"

    @abstractmethod
    async def complete_json(self, system: str, user: str, schema_hint: str) -> dict: ...
