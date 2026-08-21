"""The design-space sampler.

This is the component that makes "multi-candidate" mean something.  The obvious
approach — ask an LLM for three different creative ideas — produces candidates
whose diversity you cannot measure, cannot reproduce, and cannot ablate.  So
diversity is constructed geometrically instead.

The creative space is four categorical axes: camera angle, lighting,
composition, and motion intent.  A Latin-hypercube construction picks, for each
axis independently, ``n`` *distinct* levels and then pairs them up.  The
consequence is worth stating plainly: every pair of candidates then differs on
every axis, so the minimum pairwise Hamming distance is the maximum possible
(4).  There is nothing to optimise — the construction is already optimal
whenever ``n`` does not exceed the smallest axis's cardinality.

Two things modulate the draw without breaking that guarantee:

* **Mood** weights which levels are *preferred*.  A calm-premium brief should not
  usually get handheld camera shake.  Preferences are soft: if the preferred pool
  is too small to supply ``n`` distinct levels, the sampler widens to the full
  axis rather than duplicating a level.
* **A locked angle** collapses the angle axis to a constant.  That necessarily
  drops the achievable distance to 3, which the sampler reports rather than
  hides — the user asked for it, and the report should show what it cost.
"""

from __future__ import annotations

import random
from typing import TypeVar

from adschema import (
    BriefSet,
    CameraAngle,
    Composition,
    DesignPoint,
    Lighting,
    Mood,
    MotionIntent,
    Platform,
    Vertical,
)

#: Compositions that need a clear bottom edge, mapped to the largest bottom-chrome
#: fraction they tolerate.
#:
#: ``negative_space_top`` places the subject low in frame to leave room for text
#: above.  On Reels/Shorts/TikTok, where platform UI covers the bottom 18-22%,
#: that puts the subject directly under the chrome.  Measured safe-area saliency
#: share for this composition is 0.44-0.67 against 0.68-0.91 for every other
#: composition — it fails the gate structurally, not occasionally.
#:
#: Excluding it here rather than raising the gate threshold is the right layer:
#: the sampler should not spend money on a candidate whose framing is guaranteed
#: to be rejected. On 4:5 feed placements, where chrome is ~5%, it is fine.
_COMPOSITION_MAX_BOTTOM_CHROME: dict[Composition, float] = {
    Composition.NEGATIVE_SPACE_TOP: 0.10,
}

#: Levels each mood prefers.  Anything not listed is still reachable; these are
#: priors, not constraints.
_MOOD_LIGHTING: dict[Mood, list[Lighting]] = {
    Mood.CALM_PREMIUM: [Lighting.SOFT_DIFFUSED, Lighting.HIGH_KEY, Lighting.RIM_BACKLIT],
    Mood.WARM_LIFESTYLE: [Lighting.GOLDEN_HOUR, Lighting.SOFT_DIFFUSED, Lighting.HIGH_KEY],
    Mood.BOLD_CONFIDENT: [Lighting.HARD_DIRECTIONAL, Lighting.RIM_BACKLIT, Lighting.SOFT_DIFFUSED],
    Mood.HIGH_ENERGY: [Lighting.HARD_DIRECTIONAL, Lighting.GOLDEN_HOUR, Lighting.RIM_BACKLIT],
}

#: Motion preferences per mood.
#:
#: These pools used to be the quiet half of the zoom problem.  The sampler draws
#: the preferred levels *first*, and three of the four moods preferred three
#: camera moves — the default, ``warm_lifestyle``, preferred
#: ``[slow_dolly_in, product_present, static_subtle]``.  With three candidates
#: that is not a preference at all, it is a guarantee: two of every three clips
#: were camera-only and one was explicitly asked to hold still.
#:
#: Every pool now holds three performances.  Mood varies *which* performance and
#: how the camera supports it, which is what mood should have been varying.
_MOOD_MOTION: dict[Mood, list[MotionIntent]] = {
    Mood.CALM_PREMIUM: [
        MotionIntent.PRODUCT_REVEAL,
        MotionIntent.HERO_TURN,
        MotionIntent.OFFER_TO_CAMERA,
    ],
    Mood.WARM_LIFESTYLE: [
        MotionIntent.IN_USE,
        MotionIntent.PICK_UP,
        MotionIntent.HERO_TURN,
    ],
    Mood.BOLD_CONFIDENT: [
        MotionIntent.OFFER_TO_CAMERA,
        MotionIntent.HERO_TURN,
        MotionIntent.WALK_IN,
    ],
    Mood.HIGH_ENERGY: [
        MotionIntent.WALK_IN,
        MotionIntent.IN_USE,
        MotionIntent.PRODUCT_REVEAL,
    ],
}

#: Verticals where the product itself is the hero and deserves screen area.
_PRODUCT_LED_VERTICALS = {
    Vertical.BEAUTY,
    Vertical.JEWELLERY,
    Vertical.ELECTRONICS,
    Vertical.FOOD_BEVERAGE,
}

_PRODUCT_LED_ANGLES = [
    CameraAngle.CLOSE_UP_PRODUCT,
    CameraAngle.THREE_QUARTER,
    CameraAngle.EYE_LEVEL,
]
_PRODUCT_LED_COMPOSITIONS = [
    Composition.PRODUCT_FOREGROUND,
    Composition.CENTERED_HERO,
    Composition.RULE_OF_THIRDS_RIGHT,
]


T = TypeVar("T")


def _draw_levels(
    rng: random.Random,
    preferred: list[T],
    full: list[T],
    n: int,
) -> list[T]:
    """Draw ``n`` levels for one axis, distinct wherever the axis allows it.

    Preferred levels come first, in shuffled order.  If they run out we widen to
    the rest of the axis before ever repeating a level, because a repeat costs
    pairwise distance and distance is the whole point.
    """
    pool = list(dict.fromkeys([*preferred, *full]))  # preserve order, drop dupes
    head, tail = pool[: len(preferred)], pool[len(preferred) :]
    rng.shuffle(head)
    rng.shuffle(tail)
    ordered = [*head, *tail]

    if n <= len(ordered):
        return ordered[:n]

    # Axis is smaller than the candidate count: cycle, which is the least-bad
    # option and the reason min_pairwise_distance can legitimately drop below 4.
    return [ordered[i % len(ordered)] for i in range(n)]


def compositions_for(platform: Platform) -> list[Composition]:
    """Compositions that are viable on a given placement.

    See :data:`_COMPOSITION_MAX_BOTTOM_CHROME` for why this filter exists.
    """
    bottom = platform.safe_area.bottom
    return [c for c in Composition if bottom <= _COMPOSITION_MAX_BOTTOM_CHROME.get(c, 1.0)]


def sample_design_points(
    n: int,
    *,
    mood: Mood,
    vertical: Vertical,
    seed: int,
    platform: Platform = Platform.INSTAGRAM_REELS,
    locked_angle: CameraAngle | None = None,
) -> list[DesignPoint]:
    """Draw ``n`` well-separated points from the creative design space."""
    if n < 1:
        raise ValueError("n must be at least 1")
    rng = random.Random(seed)

    product_led = vertical in _PRODUCT_LED_VERTICALS
    viable_compositions = compositions_for(platform)

    if locked_angle is not None:
        angles = [locked_angle] * n
    else:
        angles = _draw_levels(
            rng,
            _PRODUCT_LED_ANGLES if product_led else [],
            list(CameraAngle),
            n,
        )

    lightings = _draw_levels(rng, _MOOD_LIGHTING[mood], list(Lighting), n)
    preferred_compositions = [
        c for c in (_PRODUCT_LED_COMPOSITIONS if product_led else []) if c in viable_compositions
    ]
    compositions = _draw_levels(rng, preferred_compositions, viable_compositions, n)
    motions = _draw_levels(rng, _MOOD_MOTION[mood], list(MotionIntent), n)

    return [
        DesignPoint(
            index=i,
            angle=angles[i],
            lighting=lightings[i],
            composition=compositions[i],
            motion=motions[i],
            seed=seed + i,
        )
        for i in range(n)
    ]


def max_achievable_distance(
    n: int,
    *,
    platform: Platform = Platform.INSTAGRAM_REELS,
    locked_angle: CameraAngle | None = None,
) -> int:
    """The best minimum pairwise distance the design space can offer for ``n``.

    Not a constant. An axis can only keep ``n`` candidates apart if it has ``n``
    distinct levels to give them; below that ``_draw_levels`` cycles, some pair
    collides on that axis, and the achievable minimum drops by one.

    This exists because the guarantee stopped being 4. Moving to five candidates
    put the composition axis under pressure on Reels, where ``negative_space_top``
    is excluded for placing the subject under the platform's bottom chrome — four
    viable levels for five candidates, so two candidates must share one and the
    best possible minimum is 3. On Feed, where all five compositions are viable,
    five candidates still separate completely.

    Asserting a computed bound rather than a remembered number is the point: the
    diversity guarantee is now a claim about the axes, and it stays true when a
    level is added, retired, or filtered out by a platform.
    """
    if n < 2:
        return 4
    cardinalities = (
        1 if locked_angle is not None else len(list(CameraAngle)),
        len(list(Lighting)),
        len(compositions_for(platform)),
        len(list(MotionIntent)),
    )
    return sum(1 for c in cardinalities if c >= n)


def min_pairwise_distance(points: list[DesignPoint]) -> int:
    """Smallest Hamming distance between any two points.

    4 means every candidate differs on every axis.  It is no longer *always* the
    best achievable: locking the camera angle costs one, and so does asking for
    more candidates than an axis has levels — see :func:`max_achievable_distance`,
    which is what this should be compared against.  0 would mean two identical
    candidates, which is a bug worth failing a test over.
    """
    if len(points) < 2:
        return 4
    return min(a.distance(b) for i, a in enumerate(points) for b in points[i + 1 :])


def describe_diversity(points: list[DesignPoint]) -> str:
    """Human-readable diversity summary, for logs and the report."""
    d = min_pairwise_distance(points)
    per_axis = []
    for axis_index, axis_name in enumerate(("angle", "lighting", "composition", "motion")):
        distinct = len({p.axes()[axis_index] for p in points})
        per_axis.append(f"{axis_name}={distinct}/{len(points)}")
    return f"min pairwise distance {d}/4 · distinct levels " + " ".join(per_axis)


def attach_diversity(brief_set: BriefSet) -> BriefSet:
    """Fill in the diversity metric on a completed brief set."""
    brief_set.min_pairwise_distance = min_pairwise_distance(
        [b.design_point for b in brief_set.briefs]
    )
    return brief_set
