"""Reframing a clip for a platform it was not generated for.

A job generates at one aspect ratio — the one its target platform derives.  Delivery
needs the others: the same creative as a 1:1 feed post and a 16:9 pre-roll.  The
naive answer is a centre crop, and it is wrong for the specific thing this project
cares about, because **the product is usually not in the centre**.  The design-space
sampler deliberately varies composition across candidates, and three of its five
settings put the subject off-centre; ``product_foreground`` and the rule-of-thirds
options are exactly the ones a centre crop mutilates.

So the crop window is placed by saliency.  Three decisions in here are worth
knowing about.

**Cropping is not always the right operation.**  A 9:16 source reframed to 16:9
keeps a horizontal band about 32% of the original height — most of the frame is
gone, and with it usually the model's face.  :func:`plan` measures how much salient
mass a crop would retain and recommends padding instead when that figure is poor.
The measurement is reported either way, so the loss is visible rather than implied.

**The window is static, and that is a measured choice rather than a shortcut.**  A
per-frame window tracks a moving subject better and jitters; a jittering frame is
worse to watch than a slightly mis-framed steady one.  :class:`CropPlan` carries both
numbers — what the static window retains, and what a perfect per-frame tracker would
have retained — so the gap is on the record.

On this project's clips that gap is **0.002**.  Measured across every mock motion
intent reframed to 9:16: ``static_subtle`` and ``handheld_drift`` both showed a gain
of 0.000, and ``orbit_left``, whose subject travels 12.5% of the short edge, showed
0.002.  8-10 s ad motion is a dolly or a drift, not a chase.  The metric is not
merely insensitive — driven with a subject crossing half the frame it reports gains
above 0.08 — so the small number is a fact about the content, not about the
measurement.

**The safe area is part of the objective, not a check afterwards.**  A window that
centres the product perfectly but leaves it under Instagram's caption bar has not
solved the problem.  Saliency inside the target platform's safe area is weighted
above saliency merely inside the frame.

A caveat that is easy to trip over.  This is only as good as the saliency map, and
:func:`adml.features.saliency_map` is a spectral residual — which behaves badly on
*synthetic* imagery with hard edges and no texture.  Measured on a single bright disc
over a perfectly flat field, the map comes out periodic with a period of about a
sixteenth of the frame, and its column-wise **minimum sits exactly on the disc**: the
FFT of a disc rings, and the residual amplifies the ringing.  On a photograph, or on
the mock renderer's gradient-and-grain frames, it behaves as intended — the same mock
frame with its subject anchored at 0.33 of the width put 79% of its saliency mass in
the correct half.  So a flat vector test fixture will produce a nonsense crop and it
is the fixture that is wrong.  The same limitation already applies to
:func:`adml.features.sharpness`, and for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import features as F

#: Saliency is computed at this resolution. The map is a spectral residual with no
#: learned parameters, so a small grid loses very little and costs an FFT per frame.
SALIENCY_GRID = 64

#: Weight applied to saliency mass that lands outside the platform's safe area.
#: Not zero: content under the caption bar is still visible, just compromised, and a
#: hard zero makes the objective indifferent between "slightly clipped" and "gone".
OUTSIDE_SAFE_AREA_WEIGHT = 0.35

#: Pixels of difference between the source and the largest fitting window below
#: which no reframe is worth doing. Two, because that is exactly the error even-
#: dimension rounding introduces.
NOOP_SLACK_PX = 2

#: Below this share of retained salience, cropping is doing more harm than
#: letterboxing. 0.70 is a judgement rather than a measurement — see the module note
#: in ``docs/delivery-protocol.md`` — but it separates the cases that matter: a 9:16
#: to 16:9 reframe of a standing figure retains far less, a 9:16 to 4:5 retains far
#: more.
MIN_RETAINED_SALIENCE = 0.70


@dataclass(frozen=True)
class Window:
    """An axis-aligned crop rectangle in source pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    def as_even(self) -> Window:
        """Round the rectangle to even dimensions.

        ``yuv420p`` subsamples chroma by two in each direction, so an odd width or
        height cannot be encoded. Done here rather than in the encoder so the
        window that gets measured is the window that gets rendered.
        """
        return Window(
            x=self.x - (self.x % 2),
            y=self.y - (self.y % 2),
            width=self.width - (self.width % 2),
            height=self.height - (self.height % 2),
        )


@dataclass(frozen=True)
class CropPlan:
    """How to reframe one clip, and what it costs."""

    mode: str  # "crop" | "pad" | "none"
    window: Window
    source_width: int
    source_height: int
    target_aspect: float
    #: Share of total saliency mass the chosen static window retains.
    retained_salience: float
    #: What a per-frame tracker would have retained. The gap between this and
    #: ``retained_salience`` is the price of holding the window still.
    tracked_salience: float
    #: How far the per-frame ideal centre travelled, as a fraction of the source's
    #: short edge. Large values mean a static window is genuinely compromising.
    centre_travel: float
    reason: str = ""

    @property
    def tracking_gain(self) -> float:
        """Salience a per-frame tracker would add. Reported, not acted on."""
        return self.tracked_salience - self.retained_salience

    @property
    def is_noop(self) -> bool:
        return self.mode == "none"

    def summary(self) -> str:
        if self.is_noop:
            return f"already {self.target_aspect:.3f}, no reframe needed"
        text = (
            f"{self.mode} to {self.window.width}x{self.window.height} "
            f"at ({self.window.x},{self.window.y}), "
            f"retains {self.retained_salience:.0%} of salience"
        )
        if self.tracking_gain > 0.02:
            text += f" (a per-frame tracker would retain {self.tracked_salience:.0%})"
        if self.centre_travel > 0.10:
            text += f", subject travelled {self.centre_travel:.0%} of the short edge"
        if self.reason:
            text += f" — {self.reason}"
        return text


def window_for_aspect(width: int, height: int, target_aspect: float) -> tuple[int, int]:
    """Largest ``target_aspect`` rectangle that fits inside ``width x height``."""
    if target_aspect <= 0:
        raise ValueError(f"target aspect must be positive, got {target_aspect}")
    source_aspect = width / height
    if source_aspect > target_aspect:
        # Source is wider than the target: full height, narrower width.
        return max(2, round(height * target_aspect)), height
    return width, max(2, round(width / target_aspect))


def _safe_area_weights(grid: int, safe: tuple[float, float, float, float] | None) -> np.ndarray:
    """Per-cell weights that prefer the platform's safe area.

    Saliency outside the safe box is not discarded, only discounted: content under
    a caption bar is compromised rather than absent, and zeroing it would make the
    objective indifferent between clipping the product and losing it entirely.
    """
    weights = np.full((grid, grid), OUTSIDE_SAFE_AREA_WEIGHT, dtype=float)
    if safe is None:
        return np.ones((grid, grid), dtype=float)
    top, bottom, left, right = safe
    y0 = int(round(grid * top))
    y1 = grid - int(round(grid * bottom))
    x0 = int(round(grid * left))
    x1 = grid - int(round(grid * right))
    if y1 > y0 and x1 > x0:
        weights[y0:y1, x0:x1] = 1.0
    return weights


def _best_offset(profile: np.ndarray, span: int) -> tuple[int, float]:
    """Sliding-window maximum over a 1-D mass profile.

    Returns ``(offset, mass_inside)``.  Computed from a cumulative sum, so the whole
    search is O(n) rather than O(n x span) — which matters because this runs once
    per frame per target ratio.
    """
    total = len(profile)
    if span >= total:
        return 0, float(profile.sum())
    cumulative = np.concatenate([[0.0], np.cumsum(profile)])
    sums = cumulative[span:] - cumulative[:-span]
    best = int(np.argmax(sums))
    return best, float(sums[best])


def _frame_window(
    saliency: np.ndarray,
    weights: np.ndarray,
    target_aspect: float,
    width: int,
    height: int,
) -> tuple[Window, float]:
    """The best window for one frame, and the weighted mass it captures."""
    grid = saliency.shape[0]
    weighted = saliency * weights
    win_w, win_h = window_for_aspect(width, height, target_aspect)

    # Reduce to marginals. The window is axis-aligned and one of its dimensions
    # always spans the full source (see window_for_aspect), so only one offset is
    # free and a 1-D search over the other axis is exact, not an approximation.
    if win_w < width:
        span = max(1, round(grid * win_w / width))
        offset, mass = _best_offset(weighted.sum(axis=0), span)
        x = int(round(offset * width / grid))
        x = min(max(0, x), width - win_w)
        return Window(x, 0, win_w, win_h), mass
    span = max(1, round(grid * win_h / height))
    offset, mass = _best_offset(weighted.sum(axis=1), span)
    y = int(round(offset * height / grid))
    y = min(max(0, y), height - win_h)
    return Window(0, y, win_w, win_h), mass


def plan(
    frames: list[np.ndarray],
    target_aspect: float,
    *,
    safe_area: tuple[float, float, float, float] | None = None,
    min_retained: float = MIN_RETAINED_SALIENCE,
) -> CropPlan:
    """Choose one window for the whole clip, and measure what it costs.

    ``frames`` should be the sampled frames from :func:`adml.video.decode` — the
    same subsample every other measurement uses, so the crop is planned against the
    clip the scorer saw.
    """
    if not frames:
        raise ValueError("cannot plan a crop with no frames")

    height, width = frames[0].shape[:2]
    # A no-op is decided in pixels, not in aspect ratios. An epsilon on the ratio
    # looks equivalent and is not: encoders pad odd dimensions to even, so a nominally
    # 9:16 clip is delivered at 136x240 = 0.5667 rather than 0.5625, and a tight ratio
    # epsilon then orders a two-pixel crop and a full re-encode for nothing.
    fit_w, fit_h = window_for_aspect(width, height, target_aspect)
    if width - fit_w <= NOOP_SLACK_PX and height - fit_h <= NOOP_SLACK_PX:
        return CropPlan(
            mode="none",
            window=Window(0, 0, width, height),
            source_width=width,
            source_height=height,
            target_aspect=target_aspect,
            retained_salience=1.0,
            tracked_salience=1.0,
            centre_travel=0.0,
            reason="source is already within a rounding error of the target ratio",
        )

    maps = [F.saliency_map(frame, size=SALIENCY_GRID) for frame in frames]
    weights = _safe_area_weights(SALIENCY_GRID, safe_area)

    per_frame: list[Window] = []
    tracked_mass = 0.0
    total_mass = 0.0
    for saliency in maps:
        window, mass = _frame_window(saliency, weights, target_aspect, width, height)
        per_frame.append(window)
        tracked_mass += mass
        total_mass += float((saliency * weights).sum())

    # The static window: place it at the median of the per-frame ideal offsets. The
    # median rather than the mean, so one frame where the subject leaves the shot
    # does not drag the window off the other forty-seven.
    win_w, win_h = window_for_aspect(width, height, target_aspect)
    if win_w < width:
        x = int(np.median([w.x for w in per_frame]))
        static = Window(min(max(0, x), width - win_w), 0, win_w, win_h)
        travel = float(np.ptp([w.x for w in per_frame])) / min(width, height)
    else:
        y = int(np.median([w.y for w in per_frame]))
        static = Window(0, min(max(0, y), height - win_h), win_w, win_h)
        travel = float(np.ptp([w.y for w in per_frame])) / min(width, height)

    static_mass = 0.0
    for saliency in maps:
        static_mass += _mass_inside(saliency * weights, static, width, height)

    retained = static_mass / total_mass if total_mass > 0 else 0.0
    tracked = tracked_mass / total_mass if total_mass > 0 else 0.0

    mode = "crop"
    reason = ""
    if retained < min_retained:
        mode = "pad"
        lost = 1.0 - retained
        reason = (
            f"a crop would discard {lost:.0%} of the salient content, so the frame is "
            f"letterboxed instead"
        )

    return CropPlan(
        mode=mode,
        window=static.as_even(),
        source_width=width,
        source_height=height,
        target_aspect=target_aspect,
        retained_salience=retained,
        tracked_salience=tracked,
        centre_travel=travel,
        reason=reason,
    )


def _mass_inside(weighted: np.ndarray, window: Window, width: int, height: int) -> float:
    grid = weighted.shape[0]
    x0 = int(round(window.x * grid / width))
    x1 = int(round(window.right * grid / width))
    y0 = int(round(window.y * grid / height))
    y1 = int(round(window.bottom * grid / height))
    return float(weighted[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)].sum())


def centre_plan(width: int, height: int, target_aspect: float) -> Window:
    """The centre crop, for comparison.

    Exists so the delivery report can quote what a naive reframe would have kept.
    "Smart crop retains 91% where a centre crop retains 62%" is a claim; "smart crop
    retains 91%" on its own is not.
    """
    win_w, win_h = window_for_aspect(width, height, target_aspect)
    return Window(
        x=(width - win_w) // 2, y=(height - win_h) // 2, width=win_w, height=win_h
    ).as_even()


def retained_by(
    frames: list[np.ndarray],
    window: Window,
    *,
    safe_area: tuple[float, float, float, float] | None = None,
) -> float:
    """Share of weighted salience a given window retains, for any window."""
    if not frames:
        return 0.0
    height, width = frames[0].shape[:2]
    weights = _safe_area_weights(SALIENCY_GRID, safe_area)
    inside = 0.0
    total = 0.0
    for frame in frames:
        weighted = F.saliency_map(frame, size=SALIENCY_GRID) * weights
        inside += _mass_inside(weighted, window, width, height)
        total += float(weighted.sum())
    return inside / total if total > 0 else 0.0
