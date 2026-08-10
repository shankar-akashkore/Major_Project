"""Provider price table and cost estimation.

Prices are USD and were current as of August 2026 (see the plan's sources).
They move, so treat this table as configuration rather than fact — the governor
reads it to *estimate* a charge before making a call, and every estimate is
reconciled against the provider's reported usage afterwards where available.

The table is also what makes the two-tier strategy legible: the research tier
is genuinely $0 because it runs open weights on Colab, and the premium tier is
priced per second, which is why an 8-10 s video is the dominant line item.
"""

from __future__ import annotations

from dataclasses import dataclass

from adschema import Tier


@dataclass(frozen=True)
class ImagePrice:
    model: str
    usd_per_image: float
    max_reference_images: int
    tier: Tier = Tier.PREMIUM
    note: str = ""


@dataclass(frozen=True)
class VideoPrice:
    model: str
    usd_per_second: float
    native_max_seconds: float
    tier: Tier = Tier.PREMIUM
    has_native_audio: bool = False
    note: str = ""

    @property
    def supports_project_window(self) -> bool:
        """Whether this model reaches 8-10 s without chaining two clips.

        This is the criterion the week-5 provider spike has to settle: models
        capping at 5 s force a concatenation seam, which costs temporal
        consistency and compounds identity drift.
        """
        from adschema import MIN_DURATION_S

        return self.native_max_seconds >= MIN_DURATION_S


IMAGE_PRICES: dict[str, ImagePrice] = {
    "mock": ImagePrice("mock", 0.0, 16, Tier.MOCK, "Deterministic placeholder, no network."),
    # Multi-reference composition — the capability this project actually needs.
    "gemini-flash-image": ImagePrice(
        "gemini-flash-image", 0.039, 14, Tier.PREMIUM, "Has a free tier via AI Studio; best fit."
    ),
    "seedream-4": ImagePrice("seedream-4", 0.030, 6, Tier.PREMIUM, "Universal Reference system."),
    "flux-2-pro": ImagePrice("flux-2-pro", 0.040, 8, Tier.PREMIUM, "Strong instruction editing."),
    "flux-kontext-dev": ImagePrice(
        "flux-kontext-dev", 0.0, 4, Tier.RESEARCH, "Open weights; runs on Colab free."
    ),
}

VIDEO_PRICES: dict[str, VideoPrice] = {
    "mock": VideoPrice("mock", 0.0, 60.0, Tier.MOCK, note="Synthetic clip, no network."),
    # Research tier — free, open weights on a Colab GPU. Lower resolution is
    # fine because ranking research does not need 1080p.
    "ltx-video": VideoPrice(
        "ltx-video", 0.0, 10.0, Tier.RESEARCH, note="Lowest VRAM; best Colab free fit."
    ),
    "wan-2.1-i2v": VideoPrice(
        "wan-2.1-i2v", 0.0, 5.0, Tier.RESEARCH, note="Needs chaining for 8-10 s."
    ),
    # Premium tier — paid, reserved for golden demo jobs.
    "kling-3": VideoPrice("kling-3", 0.10, 10.0, Tier.PREMIUM, note="10 s native; the default."),
    "seedance-2.5": VideoPrice("seedance-2.5", 0.12, 12.0, Tier.PREMIUM, note="Long-form capable."),
    "runway-gen-4.5": VideoPrice("runway-gen-4.5", 0.20, 10.0, Tier.PREMIUM),
    "veo-3.1-fast": VideoPrice("veo-3.1-fast", 0.15, 8.0, Tier.PREMIUM, has_native_audio=True),
    "veo-3.1": VideoPrice(
        "veo-3.1",
        0.40,
        8.0,
        Tier.PREMIUM,
        has_native_audio=True,
        note="Native audio, but 4x the budget of kling-3. Out of reach at $35.",
    ),
}

#: Flat per-call estimate for brief compilation. A structured 3-brief response
#: is a few thousand tokens; pricing it per call keeps the ledger simple.
LLM_PRICES: dict[str, float] = {
    "mock": 0.0,
    "claude-haiku": 0.004,
    "claude-sonnet": 0.012,
    "gemini-flash": 0.002,
}


def estimate_image_cost(model: str, count: int = 1) -> float:
    price = IMAGE_PRICES.get(model)
    if price is None:
        raise KeyError(f"unknown image model {model!r}; add it to IMAGE_PRICES")
    return round(price.usd_per_image * count, 6)


def estimate_video_cost(model: str, seconds: float, count: int = 1) -> float:
    price = VIDEO_PRICES.get(model)
    if price is None:
        raise KeyError(f"unknown video model {model!r}; add it to VIDEO_PRICES")
    billable = seconds
    if not price.supports_project_window and seconds > price.native_max_seconds:
        # Chaining bills each segment, so round up to whole clips.
        import math

        segments = math.ceil(seconds / price.native_max_seconds)
        billable = segments * price.native_max_seconds
    return round(price.usd_per_second * billable * count, 6)


def estimate_llm_cost(model: str, calls: int = 1) -> float:
    if model not in LLM_PRICES:
        raise KeyError(f"unknown llm model {model!r}; add it to LLM_PRICES")
    return round(LLM_PRICES[model] * calls, 6)


def estimate_job_cost(
    image_model: str,
    video_model: str,
    llm_model: str,
    candidate_count: int,
    duration_seconds: float,
) -> float:
    """Total estimated cost of one full job, used for the pre-flight check.

    The governor calls this *before* stage 1 so an unaffordable job is refused
    up front rather than failing halfway through with money already spent.
    """
    return round(
        estimate_image_cost(image_model, candidate_count)
        + estimate_video_cost(video_model, duration_seconds, candidate_count)
        + estimate_llm_cost(llm_model, 1),
        6,
    )
