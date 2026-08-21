"""The job request — what the user actually submits.

Design notes on the deliberate departures from a naive input list:

* **Aspect ratio is derived**, not asked for.  The user picks a platform; the
  geometry and safe areas follow.  ``aspect_ratio_override`` exists for the
  advanced panel only.
* **Colour theme and background colour are merged** into one :class:`ThemeSpec`,
  whose palette is auto-extracted from the product image when the user does not
  supply one.
* **Camera angle is absent** from the request by default.  It is a design-space
  axis the sampler varies across candidates; ``locked_angle`` lets a user pin it
  when they really do want all three candidates shot the same way.
* **Consent is mandatory.**  Uploading someone's likeness to a generative model
  needs an affirmative attestation, and the gate refuses the job without one.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import (
    AspectRatio,
    BackgroundTreatment,
    CameraAngle,
    Mood,
    Platform,
    ProductScale,
    Tier,
    Vertical,
    default_scale_for,
)

HEX_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

#: The video duration window the project commits to.  Most image-to-video
#: models generate 5 s natively, so anything in this range demands either a
#: provider with native long generation or 5 s + 5 s chaining.  See
#: ``VideoProvider.supports_duration``.
MIN_DURATION_S = 8.0
MAX_DURATION_S = 10.0

#: Image candidates generated per job, and how many of them are animated.
#:
#: These were equal, and the pipeline was a 1:1 cascade — every image became a
#: video, so the image-stage ranking was a *prediction* that cost nothing to be
#: wrong about. They are separate now because the shape of the product changed:
#: explore widely where generation is cheap, spend narrowly where it is not.
#:
#: The asymmetry is the whole point. An image is $0.04 and a 10 s clip is $0.70 —
#: seventeen and a half times the price — so five images and two videos costs
#: $1.60 against the old $2.22 while showing the user *more* creative range, not
#: less. What it buys is paid for by the image-stage predictor, which stops being
#: a diagnostic and becomes the thing that decides where the money goes.
DEFAULT_CANDIDATE_COUNT = 5
DEFAULT_VIDEO_COUNT = 2


class AssetRef(BaseModel):
    """A pointer to a stored upload.  Never raw bytes — media lives in object
    storage and moves around as a key plus a signed URL."""

    key: str = Field(description="Storage key, e.g. 'uploads/<job>/human.jpg'")
    url: str | None = Field(default=None, description="Signed URL, short-lived")
    mime_type: str = "image/jpeg"
    width: int | None = None
    height: int | None = None
    sha256: str | None = Field(default=None, description="Content hash, for the cache key")


class ThemeSpec(BaseModel):
    """Brand look.  Replaces the separate 'color theme' and 'background color'
    inputs — they were two controls for one decision."""

    palette: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Brand hex colours. Left empty, intake extracts them from the product image.",
    )
    background: BackgroundTreatment = BackgroundTreatment.SOFT_GRADIENT
    background_color: str | None = Field(
        default=None,
        description="Only meaningful when background is seamless_color.",
    )
    palette_auto_extracted: bool = Field(
        default=False,
        description="Set by intake when the palette was derived rather than supplied.",
    )

    @field_validator("palette")
    @classmethod
    def _validate_palette(cls, v: list[str]) -> list[str]:
        for colour in v:
            if not HEX_COLOR.match(colour):
                raise ValueError(f"palette entry {colour!r} is not a hex colour like '#1a2b3c'")
        return v

    @field_validator("background_color")
    @classmethod
    def _validate_bg(cls, v: str | None) -> str | None:
        if v is not None and not HEX_COLOR.match(v):
            raise ValueError(f"background_color {v!r} is not a hex colour")
        return v

    @model_validator(mode="after")
    def _seamless_needs_colour(self) -> ThemeSpec:
        if self.background is BackgroundTreatment.SEAMLESS_COLOR and not self.background_color:
            # Fall back to the dominant brand colour rather than failing the job.
            self.background_color = self.palette[0] if self.palette else "#f2f2f2"
        return self


class AudienceSpec(BaseModel):
    """Who the ad is for.  Feeds the predictor as tabular context."""

    age_band: str = Field(default="18-34", pattern=r"^\d{2}-\d{2}$")
    region: str = Field(default="IN", min_length=2, max_length=8)
    language: str = Field(default="en", min_length=2, max_length=8)


class ConsentAttestation(BaseModel):
    """Affirmative rights confirmation for the human-model image.

    This is not decoration: the pipeline hard-refuses at intake without it.  It
    is also the concrete artefact the report's ethics section describes.
    """

    has_model_release: bool = Field(
        description="User confirms they hold rights to use this person's likeness."
    )
    not_a_public_figure: bool = Field(
        description="User confirms the image is not of a public figure or celebrity."
    )
    attested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attested_by: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.has_model_release and self.not_a_public_figure


class AdJobRequest(BaseModel):
    """Everything needed to run one ad job.

    Deliberately not "a 3-candidate ad job" any more. The candidate count and
    the video count are separate request fields because they are separate
    decisions — how widely to explore, and how much of that to pay to animate.
    """

    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # --- Assets ---
    human_model_image: AssetRef
    product_image: AssetRef
    logo_image: AssetRef | None = None

    # --- Copy ---
    product_name: str = Field(min_length=1, max_length=120)
    caption: str = Field(default="", max_length=300)
    cta_text: str = Field(default="", max_length=40, description="e.g. 'Shop now'")
    additional_prompt: str = Field(default="", max_length=1000)
    negative_constraints: str = Field(
        default="",
        max_length=500,
        description="Free text, e.g. 'no hands covering the label, no visible text on product'",
    )

    # --- Targeting ---
    vertical: Vertical = Vertical.OTHER
    product_scale: ProductScale | None = Field(
        default=None,
        description="How big the product is in real life. Left unset it is derived "
        "from the vertical; set it explicitly when the vertical is 'other' or when "
        "the product is unusual for its category. This is the single control that "
        "stops the generator rendering a phone the size of a person.",
    )
    platform: Platform = Platform.INSTAGRAM_REELS
    audience: AudienceSpec = Field(default_factory=AudienceSpec)

    # --- Look ---
    theme: ThemeSpec = Field(default_factory=ThemeSpec)
    mood: Mood = Mood.WARM_LIFESTYLE

    # --- Geometry & timing (derived where possible) ---
    aspect_ratio_override: AspectRatio | None = Field(
        default=None, description="Advanced panel only; normally derived from platform."
    )
    duration_seconds: float = Field(
        default=9.0,
        ge=MIN_DURATION_S,
        le=MAX_DURATION_S,
        description="Video length. Kept in the 8-10 s window the project commits to.",
    )

    # --- Candidate control ---
    candidate_count: int = Field(
        default=DEFAULT_CANDIDATE_COUNT,
        ge=1,
        le=6,
        description="Image candidates to generate. Cheap, so this is the exploration budget.",
    )
    video_count: int = Field(
        default=DEFAULT_VIDEO_COUNT,
        ge=1,
        le=6,
        description="How many of the candidates are animated, taken in image-stage "
        "predicted order. Set equal to candidate_count to animate everything, which "
        "is what an evaluation run needs — see AdJobRequest.animates_everything.",
    )
    locked_angle: CameraAngle | None = Field(
        default=None,
        description="Pin the camera angle across all candidates. Left None, the "
        "sampler varies angle — which is what makes the candidates genuinely different.",
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="Base seed. Candidate i uses seed + i, so runs are reproducible "
        "and ablations are comparable.",
    )

    # --- Governance ---
    consent: ConsentAttestation
    tier: Tier = Tier.MOCK

    @model_validator(mode="after")
    def _cannot_animate_more_than_was_generated(self) -> AdJobRequest:
        """``video_count`` is clamped, not rejected.

        Asking for more videos than there are candidates is incoherent rather than
        malicious — a leftover form value, a script that changed one number and not
        the other. Refusing the job would cost the user their uploads and tell them
        nothing they could not be told by quietly animating everything, which is
        exactly what the request describes at the limit.

        The gate below it is the one that matters and it is enforced elsewhere: a
        candidate that fails quality control is never promoted no matter how many
        videos were asked for.
        """
        if self.video_count > self.candidate_count:
            self.video_count = self.candidate_count
        return self

    @property
    def animates_everything(self) -> bool:
        """True when no candidate is cut, so the two stages rank the same set.

        The distinction is load-bearing for the research claim rather than for the
        product. A job that animates everything yields a *complete* paired
        observation — every image-stage rank has a video-stage rank to be compared
        against. A 5-to-2 job yields a truncated one, because the three candidates
        the predictor rejected have no video-stage outcome and never will.

        Both are legitimate; they answer different questions. Evaluation runs want
        this True. See docs/prediction-protocol.md.
        """
        return self.video_count >= self.candidate_count

    @property
    def aspect_ratio(self) -> AspectRatio:
        """Resolved geometry: the override if given, else the platform default."""
        return self.aspect_ratio_override or self.platform.aspect_ratio

    @property
    def effective_scale(self) -> ProductScale | None:
        """Resolved product size: the explicit value if given, else the vertical's.

        Still ``None`` for an unlabelled product in the ``other`` vertical, and
        deliberately so — that case is genuinely unknown, and the brief compiler
        answers it by constraining the *relationship* rather than inventing a
        measurement. Guessing "one hand" for a sofa would trade one silent scale
        error for another.
        """
        return self.product_scale or default_scale_for(self.vertical)

    def candidate_seed(self, index: int) -> int:
        return self.seed + index

    def cache_fingerprint(self) -> str:
        """Stable hash of everything that affects generation output.

        The cost governor keys its cache on this, so an identical resubmission
        costs nothing.  Deliberately excludes ``job_id`` and ``created_at``.
        """
        import hashlib
        import json

        payload = self.model_dump(
            mode="json",
            exclude={"job_id", "created_at", "consent", "tier"},
        )
        # Asset identity is the content hash, not the storage key.
        for field in ("human_model_image", "product_image", "logo_image"):
            asset = payload.get(field)
            if isinstance(asset, dict):
                payload[field] = asset.get("sha256") or asset.get("key")
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:32]
