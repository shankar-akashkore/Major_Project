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
    #: Durations the API will actually accept, when it takes an enum rather than a
    #: number. ``None`` means any duration up to ``native_max_seconds``.
    #:
    #: This distinction turned out to matter. The project commits to an 8-10 s
    #: window and defaults to 9 s, and 9 s is simply not requestable on the
    #: cheapest provider that reaches the window at all: Kling's image-to-video
    #: endpoint takes ``duration`` as the enum {"5", "10"}. A continuous request
    #: therefore has to snap to a real option, and the cost has to be estimated on
    #: what will be billed rather than on what was asked for.
    supported_durations: tuple[float, ...] | None = None
    #: Whether the API accepts a seed. Kling's image-to-video endpoint does not,
    #: so video-stage generation is not reproducible there — which the evaluation
    #: has to account for rather than assume away.
    honours_seed: bool = True
    note: str = ""

    @property
    def supports_project_window(self) -> bool:
        """Whether this model reaches 8-10 s without chaining two clips.

        This is the criterion the week-5 provider spike had to settle. It is
        settled: several models reach the window natively, so chaining is a
        fallback for cheaper tiers rather than the default path.
        """
        from adschema import MIN_DURATION_S

        return self.native_max_seconds >= MIN_DURATION_S

    def snap_duration(self, seconds: float) -> float:
        """The duration this provider will actually deliver for a request.

        Always rounds *up* to the next available option. Rounding down would
        deliver a clip shorter than the 8 s the project commits to, and quietly
        breaking a stated guarantee to save two cents is the wrong trade.
        """
        if self.supported_durations is None:
            return min(seconds, self.native_max_seconds)
        longer = [d for d in sorted(self.supported_durations) if d >= seconds]
        return longer[0] if longer else max(self.supported_durations)


IMAGE_PRICES: dict[str, ImagePrice] = {
    "mock": ImagePrice("mock", 0.0, 16, Tier.MOCK, "Deterministic placeholder, no network."),
    # Multi-reference composition — the capability this project actually needs.
    # `seedream-4.5-edit` is the one with a wired adapter; it takes up to 10
    # reference images in one call and honours a seed, which the image-stage
    # ablations depend on.
    "seedream-4.5-edit": ImagePrice(
        "seedream-4.5-edit",
        0.04,
        10,
        Tier.PREMIUM,
        "fal-ai/bytedance/seedream/v4.5/edit — up to 10 refs, seed honoured. The default.",
    ),
    "gemini-flash-image": ImagePrice(
        "gemini-flash-image", 0.039, 14, Tier.PREMIUM, "Has a free tier via AI Studio."
    ),
    "seedream-4": ImagePrice("seedream-4", 0.030, 10, Tier.PREMIUM, "Previous generation."),
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
    #
    # `kling-2.5-turbo-pro` is the one with a wired adapter, and it is the
    # cheapest model that reaches the project's duration window at all.
    "kling-2.5-turbo-pro": VideoPrice(
        "kling-2.5-turbo-pro",
        0.07,
        10.0,
        Tier.PREMIUM,
        supported_durations=(5.0, 10.0),
        honours_seed=False,
        note=(
            "fal-ai/kling-video/v2.5-turbo/pro/image-to-video. $0.35 for 5 s then "
            "$0.07/s, so a 10 s clip is $0.70. Duration is the enum {5, 10} and there "
            "is no seed parameter. The default."
        ),
    ),
    "kling-3-pro": VideoPrice(
        "kling-3-pro",
        0.112,
        15.0,
        Tier.PREMIUM,
        note="3-15 s continuous, so 9 s is requestable — 1.6x the price for that freedom.",
    ),
    "seedance-2.0": VideoPrice(
        "seedance-2.0",
        0.30,
        15.0,
        Tier.PREMIUM,
        note="720p. 4x kling-2.5-turbo-pro; the plan's $0.12/s estimate was too optimistic.",
    ),
    "veo-3.1-fast": VideoPrice(
        "veo-3.1-fast",
        0.10,
        8.0,
        Tier.PREMIUM,
        has_native_audio=True,
        supported_durations=(4.0, 6.0, 8.0),
        note="$0.10/s without audio, $0.15/s with. Reaches 8 s exactly — the window's floor.",
    ),
    "veo-3.1": VideoPrice(
        "veo-3.1",
        0.20,
        8.0,
        Tier.PREMIUM,
        has_native_audio=True,
        supported_durations=(4.0, 6.0, 8.0),
        note="$0.40/s with audio. Out of reach at $35.",
    ),
    "sora-2-pro": VideoPrice(
        "sora-2-pro",
        0.30,
        20.0,
        Tier.PREMIUM,
        supported_durations=(4.0, 8.0, 12.0, 16.0, 20.0),
        note="720p; $0.50/s at 1080p. Out of reach.",
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
    """Cost of the clip that will actually be delivered, not the one requested.

    Two adjustments, both of which raise the estimate. Under-estimating is the
    dangerous direction: the governor reserves against this number, so an estimate
    below the real charge is how a budget cap gets quietly exceeded.
    """
    price = VIDEO_PRICES.get(model)
    if price is None:
        raise KeyError(f"unknown video model {model!r}; add it to VIDEO_PRICES")

    if seconds > price.native_max_seconds:
        # Chaining bills each segment, so round up to whole clips.
        import math

        segments = math.ceil(seconds / price.native_max_seconds)
        billable = segments * price.native_max_seconds
    else:
        # A provider with a discrete duration enum bills the option it delivers.
        billable = price.snap_duration(seconds)
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
    video_count: int | None = None,
) -> float:
    """Total estimated cost of one full job, used for the pre-flight check.

    The governor calls this *before* stage 1 so an unaffordable job is refused
    up front rather than failing halfway through with money already spent.

    ``video_count`` defaults to ``candidate_count`` — the 1:1 cascade this project
    ran until the stages were decoupled. It is a separate argument because the two
    counts price *very* differently: at $0.04 an image against $0.70 a clip, the
    video term is 94% of a default job, so estimating it from the image count
    reserves more than three times what a 5-to-2 job will actually spend. A
    governor that over-reserves refuses jobs the budget could have afforded, which
    is a quieter failure than overspending and just as wrong.
    """
    animated = candidate_count if video_count is None else min(video_count, candidate_count)
    return round(
        estimate_image_cost(image_model, candidate_count)
        + estimate_video_cost(video_model, duration_seconds, animated)
        + estimate_llm_cost(llm_model, 1),
        6,
    )
