"""Controlled vocabularies for the ad-generation pipeline.

Two kinds of enum live here and it is worth keeping the distinction in mind:

* **User-facing choices** (``Platform``, ``Vertical``, ``Mood``, ...) come from the
  request and are fixed for a whole job.
* **Design-space axes** (``CameraAngle``, ``Lighting``, ``BackgroundTreatment``,
  ``Composition``) are what the sampler *varies across candidates*.  They are
  deliberately not single request fields — that was the key change from the
  original design, where camera angle was a user input and every candidate
  therefore looked the same.
"""

from __future__ import annotations

from enum import Enum


class AspectRatio(str, Enum):
    """Output geometry.  Derived from :class:`Platform` rather than asked for."""

    VERTICAL_9_16 = "9:16"
    PORTRAIT_4_5 = "4:5"
    SQUARE_1_1 = "1:1"
    LANDSCAPE_16_9 = "16:9"

    @property
    def wh(self) -> tuple[int, int]:
        w, h = self.value.split(":")
        return int(w), int(h)

    @property
    def ratio(self) -> float:
        w, h = self.wh
        return w / h

    def pixel_size(self, long_edge: int = 1280) -> tuple[int, int]:
        """Pixel dimensions with the longer edge fixed to ``long_edge``."""
        w, h = self.wh
        if w >= h:
            return long_edge, round(long_edge * h / w)
        return round(long_edge * w / h), long_edge


class Platform(str, Enum):
    """Delivery target.  Drives aspect ratio and the safe-area overlay."""

    INSTAGRAM_REELS = "instagram_reels"
    INSTAGRAM_FEED = "instagram_feed"
    INSTAGRAM_STORY = "instagram_story"
    YOUTUBE_SHORTS = "youtube_shorts"
    YOUTUBE_INSTREAM = "youtube_instream"
    TIKTOK = "tiktok"
    FACEBOOK_FEED = "facebook_feed"

    @property
    def aspect_ratio(self) -> AspectRatio:
        return _PLATFORM_ASPECT[self]

    @property
    def safe_area(self) -> SafeArea:
        """Fraction of each edge that platform chrome may cover.

        The product and any CTA text must stay inside this box or the ranker
        penalises the candidate — a UI overlay eating the logo is a real and
        very common failure mode for vertical ad creative.
        """
        return _PLATFORM_SAFE_AREA[self]


class SafeArea(tuple):
    """``(top, bottom, left, right)`` as fractions of frame height/width."""

    __slots__ = ()

    def __new__(cls, top: float, bottom: float, left: float = 0.0, right: float = 0.0):
        return super().__new__(cls, (top, bottom, left, right))

    @property
    def top(self) -> float:
        return self[0]

    @property
    def bottom(self) -> float:
        return self[1]

    @property
    def left(self) -> float:
        return self[2]

    @property
    def right(self) -> float:
        return self[3]


_PLATFORM_ASPECT: dict[Platform, AspectRatio] = {
    Platform.INSTAGRAM_REELS: AspectRatio.VERTICAL_9_16,
    Platform.INSTAGRAM_FEED: AspectRatio.PORTRAIT_4_5,
    Platform.INSTAGRAM_STORY: AspectRatio.VERTICAL_9_16,
    Platform.YOUTUBE_SHORTS: AspectRatio.VERTICAL_9_16,
    Platform.YOUTUBE_INSTREAM: AspectRatio.LANDSCAPE_16_9,
    Platform.TIKTOK: AspectRatio.VERTICAL_9_16,
    Platform.FACEBOOK_FEED: AspectRatio.PORTRAIT_4_5,
}

_PLATFORM_SAFE_AREA: dict[Platform, SafeArea] = {
    Platform.INSTAGRAM_REELS: SafeArea(0.14, 0.20, 0.0, 0.14),
    Platform.INSTAGRAM_FEED: SafeArea(0.05, 0.05),
    Platform.INSTAGRAM_STORY: SafeArea(0.14, 0.20, 0.0, 0.14),
    Platform.YOUTUBE_SHORTS: SafeArea(0.10, 0.18, 0.0, 0.12),
    Platform.YOUTUBE_INSTREAM: SafeArea(0.05, 0.12),
    Platform.TIKTOK: SafeArea(0.12, 0.22, 0.0, 0.14),
    Platform.FACEBOOK_FEED: SafeArea(0.05, 0.05),
}


class Vertical(str, Enum):
    """Product category.  A real predictor feature — click behaviour differs
    enormously between, say, beauty and electronics, so the ranker conditions
    on it rather than treating all ads alike."""

    APPAREL = "apparel"
    BEAUTY = "beauty"
    FOOD_BEVERAGE = "food_beverage"
    ELECTRONICS = "electronics"
    JEWELLERY = "jewellery"
    FITNESS = "fitness"
    HOME = "home"
    FOOTWEAR = "footwear"
    OTHER = "other"


class PromptStyle(str, Enum):
    """How verbose the image prompt is.

    An ablation axis, not a preference.  A competing implementation reached
    visibly better composition with a 37-word prompt where ours ran to 343, which
    is a testable claim rather than a matter of taste: reference-edit models have
    a finite instruction-following budget, and a prompt that says twelve things
    may get fewer of them honoured than one that says four.

    ``COMPACT`` is not "the same prompt, shorter". It carries every design axis —
    dropping them would delete the multi-candidate diversity the project rests on
    — but states them as plain imperatives rather than as labelled blocks.
    """

    FULL = "full"
    COMPACT = "compact"


class ProductScale(str, Enum):
    """How large the product is in the real world.

    This exists because a reference-composition model has no idea, and nothing in
    the job told it.  The model is handed a full-frame photograph of a person and
    a full-frame cutout of a phone; absent any instruction it composites the two
    at comparable *apparent* size.  That is how the first live iPhone job returned
    a handset taller than the man holding it, and why the most physically
    plausible candidate of the five scored lowest.

    The levels are deliberately coarse.  A generator cannot act on "147 mm" with
    any more precision than it can act on "fits in one hand" — the hand relation
    is the part it can actually render, so that is what the levels name.
    """

    PALM = "palm"
    ONE_HAND = "one_hand"
    TWO_HANDS = "two_hands"
    WORN = "worn"
    FLOOR_STANDING = "floor_standing"


#: The scale each vertical implies.  ``None`` means the category genuinely does not
#: imply one — ``OTHER`` covers everything from lipstick to a sofa, and guessing
#: "one hand" for a sofa would replace one silent scale error with another.  The
#: brief compiler handles the unknown case by asserting the *relationship*
#: ("render at true size relative to the model") without asserting a number, which
#: is what actually stops the inflation.
_VERTICAL_SCALE: dict[Vertical, ProductScale | None] = {
    Vertical.APPAREL: ProductScale.WORN,
    Vertical.BEAUTY: ProductScale.PALM,
    Vertical.FOOD_BEVERAGE: ProductScale.ONE_HAND,
    Vertical.ELECTRONICS: ProductScale.ONE_HAND,
    Vertical.JEWELLERY: ProductScale.PALM,
    Vertical.FITNESS: ProductScale.TWO_HANDS,
    Vertical.HOME: ProductScale.TWO_HANDS,
    Vertical.FOOTWEAR: ProductScale.WORN,
    Vertical.OTHER: None,
}


def default_scale_for(vertical: Vertical) -> ProductScale | None:
    """The scale a vertical implies, or ``None`` when it implies nothing useful."""
    return _VERTICAL_SCALE[vertical]


class Mood(str, Enum):
    """Single user-facing energy control.  Drives lighting and motion intent."""

    CALM_PREMIUM = "calm_premium"
    WARM_LIFESTYLE = "warm_lifestyle"
    BOLD_CONFIDENT = "bold_confident"
    HIGH_ENERGY = "high_energy"


class BackgroundTreatment(str, Enum):
    STUDIO_WHITE = "studio_white"
    SEAMLESS_COLOR = "seamless_color"
    SOFT_GRADIENT = "soft_gradient"
    LIFESTYLE_SCENE = "lifestyle_scene"
    OUTDOOR_NATURAL = "outdoor_natural"


# --- Design-space axes: varied ACROSS candidates by the sampler -------------


class CameraAngle(str, Enum):
    EYE_LEVEL = "eye_level"
    LOW_ANGLE = "low_angle"
    HIGH_ANGLE = "high_angle"
    THREE_QUARTER = "three_quarter"
    PROFILE = "profile"
    CLOSE_UP_PRODUCT = "close_up_product"


class Lighting(str, Enum):
    SOFT_DIFFUSED = "soft_diffused"
    HARD_DIRECTIONAL = "hard_directional"
    RIM_BACKLIT = "rim_backlit"
    GOLDEN_HOUR = "golden_hour"
    HIGH_KEY = "high_key"


class Composition(str, Enum):
    CENTERED_HERO = "centered_hero"
    RULE_OF_THIRDS_LEFT = "rule_of_thirds_left"
    RULE_OF_THIRDS_RIGHT = "rule_of_thirds_right"
    NEGATIVE_SPACE_TOP = "negative_space_top"
    PRODUCT_FOREGROUND = "product_foreground"


class MotionIntent(str, Enum):
    """What the *subject* does during the clip.

    This axis used to name camera moves: ``slow_dolly_in``, ``slow_dolly_out``,
    ``orbit_left``, ``handheld_drift``, ``static_subtle`` — five of six levels
    describing where the lens goes and nothing about the person in front of it.

    Image-to-video models do what they are told, so that is what came back: a
    still photograph with a camera gliding over it.  The first live job produced
    three candidates and all three were zooms, which is not an advertisement.

    Every level now names a **human action first**.  Camera movement still varies
    across the levels, because visual variety is the point of the axis, but it
    only ever appears as support for something the model is doing.  There is
    deliberately no level that permits the subject to stand still: a level that
    can only produce a moving poster does not belong in the design space, and
    leaving one in means the sampler will eventually spend real money on it.
    """

    PRODUCT_REVEAL = "product_reveal"
    HERO_TURN = "hero_turn"
    IN_USE = "in_use"
    OFFER_TO_CAMERA = "offer_to_camera"
    PICK_UP = "pick_up"
    WALK_IN = "walk_in"


# --- Job lifecycle ----------------------------------------------------------


class Stage(str, Enum):
    """The eight pipeline stages, in order.  Emitted over SSE for progress."""

    INTAKE = "intake"
    BRIEF = "brief"
    IMAGE_GEN = "image_gen"
    QUALITY_GATE = "quality_gate"
    IMAGE_RANK = "image_rank"
    VIDEO_GEN = "video_gen"
    VIDEO_RANK = "video_rank"
    DELIVERY = "delivery"

    @property
    def index(self) -> int:
        return list(Stage).index(self)


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUSED_OVER_BUDGET = "refused_over_budget"


class GateVerdict(str, Enum):
    """Quality-gate outcome.  Deliberately separate from the learned score:
    the gate answers "is this usable at all", the predictor answers "how well
    will it perform".  Conflating the two is how these systems get muddled."""

    PASS = "pass"
    RETRY = "retry"
    REJECT = "reject"


class ProviderMode(str, Enum):
    """How generations are obtained.

    ``REPLAY`` is a third mode rather than a variant of ``MOCK`` because the two
    make opposite promises about the pixels.  Mock output is synthetic and says so;
    replay output is the real thing a paid provider returned, recorded to disk.
    Both cost nothing, which is why only ``LIVE`` can spend — see
    ``Settings.is_live``, which every budget check reads.
    """

    MOCK = "mock"
    LIVE = "live"
    REPLAY = "replay"


class Tier(str, Enum):
    """Which generation tier served a candidate.

    ``RESEARCH`` is free (open weights on Colab) and supplies training and
    validation volume; ``PREMIUM`` is paid and reserved for demo quality.
    Recorded per candidate so the report can compare the two.
    """

    RESEARCH = "research"
    PREMIUM = "premium"
    MOCK = "mock"
