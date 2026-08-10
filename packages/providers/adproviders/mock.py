"""Mock providers — free, deterministic, and the default everywhere.

These are not throwaway stubs.  They are the substrate the whole project is
developed against, so they earn a few properties on purpose:

* **Deterministic.**  Same seed, same bytes.  That makes the cache testable and
  the pipeline's output stable across runs.
* **Visually distinct per design point.**  Angle, lighting and composition
  actually change the rendered frame, so a broken sampler is visible at a glance
  instead of hiding behind three identical placeholders.
* **Genuinely animated.**  Mock video moves according to the brief's motion
  intent, at the requested duration.  Downstream motion features (optical flow,
  temporal variance) therefore have real signal to chew on, which means the
  scorers can be developed and tested long before any money is spent.

Video is written as an animated GIF rather than MP4 because ffmpeg is not
installed on this machine.  Frame timings are real, so duration assertions hold.
"""

from __future__ import annotations

import io
import math
import random
import time

from adschema import AspectRatio, AssetRef, CameraAngle, Composition, Lighting, MotionIntent, Tier
from PIL import Image, ImageDraw

from .base import (
    ImageGenRequest,
    ImageGenResult,
    ImageProvider,
    LLMProvider,
    VideoGenRequest,
    VideoGenResult,
    VideoProvider,
)
from .storage import Storage

#: Mock video renders at a low frame rate to keep GIFs small.  10 fps is chosen
#: deliberately: GIF stores frame delays in centiseconds, so only frame times
#: that are a whole multiple of 10 ms survive the encoder.  At 6 fps the 167 ms
#: delay silently became 160 ms and the reported duration drifted from the real
#: one — which would have made the "every video is 8-10 s" check meaningless.
MOCK_FPS = 10
MOCK_LONG_EDGE = 480

#: GIF frame-delay granularity, in milliseconds.
GIF_DELAY_QUANTUM_MS = 10

_FALLBACK_PALETTE = ["#2b3a55", "#ce7777", "#e8c4c4", "#f2e7d5"]


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def _lighting_gain(lighting: Lighting) -> tuple[float, float]:
    """``(brightness, contrast)`` multipliers per lighting style.

    Ranges are kept moderate on purpose.  An earlier version pushed brightness to
    1.22 and contrast to 0.70, which washed brand navy out into grey — the mock
    was claiming to render the brand palette while actually destroying it, and the
    quality gate was right to reject it.
    """
    return {
        Lighting.SOFT_DIFFUSED: (1.00, 0.90),
        Lighting.HARD_DIRECTIONAL: (0.95, 1.22),
        Lighting.RIM_BACKLIT: (0.90, 1.12),
        Lighting.GOLDEN_HOUR: (1.06, 0.96),
        Lighting.HIGH_KEY: (1.12, 0.82),
    }[lighting]


def _tone(colour: tuple[int, int, int], brightness: float, contrast: float) -> tuple[int, int, int]:
    """Apply a tone curve without shifting hue.

    Applying an affine curve per channel around mid-grey desaturates — it pulls
    every channel toward 0.5 independently, so a saturated colour loses its
    chroma. Scaling all three channels by a gain derived from *luminance* keeps
    the channel ratios, and therefore the hue, intact.
    """
    r, g, b = colour
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    target = ((lum - 0.5) * contrast + 0.5) * brightness
    gain = target / lum if lum > 1e-4 else target * 2.0
    return tuple(max(0, min(255, round(c * gain))) for c in (r, g, b))  # type: ignore[return-value]


def _composition_anchor(composition: Composition) -> tuple[float, float]:
    """Where the subject sits, as a fraction of frame width/height."""
    return {
        Composition.CENTERED_HERO: (0.50, 0.55),
        Composition.RULE_OF_THIRDS_LEFT: (0.33, 0.55),
        Composition.RULE_OF_THIRDS_RIGHT: (0.67, 0.55),
        Composition.NEGATIVE_SPACE_TOP: (0.50, 0.68),
        Composition.PRODUCT_FOREGROUND: (0.55, 0.48),
    }[composition]


def _angle_geometry(angle: CameraAngle) -> tuple[float, float, float]:
    """``(subject_scale, product_scale, horizon_shift)`` per camera angle."""
    return {
        CameraAngle.EYE_LEVEL: (1.00, 1.00, 0.00),
        CameraAngle.LOW_ANGLE: (1.12, 1.05, 0.12),
        CameraAngle.HIGH_ANGLE: (0.90, 0.95, -0.12),
        CameraAngle.THREE_QUARTER: (1.02, 1.02, 0.04),
        CameraAngle.PROFILE: (0.98, 0.92, 0.00),
        CameraAngle.CLOSE_UP_PRODUCT: (0.70, 1.70, 0.02),
    }[angle]


def _motion_offset(motion: MotionIntent, t: float) -> tuple[float, float, float]:
    """``(dx, dy, zoom)`` at normalised time ``t`` in [0, 1].

    Each intent produces a different optical-flow signature, which is exactly
    what the video-stage motion features need in order to be testable.
    """
    if motion is MotionIntent.SLOW_DOLLY_IN:
        return 0.0, 0.0, 1.0 + 0.18 * t
    if motion is MotionIntent.SLOW_DOLLY_OUT:
        return 0.0, 0.0, 1.18 - 0.18 * t
    if motion is MotionIntent.ORBIT_LEFT:
        return -0.10 * math.sin(math.pi * t), 0.0, 1.04
    if motion is MotionIntent.PRODUCT_PRESENT:
        return 0.03 * math.sin(2 * math.pi * t), -0.05 * t, 1.0 + 0.10 * t
    if motion is MotionIntent.HANDHELD_DRIFT:
        return 0.02 * math.sin(5 * math.pi * t), 0.015 * math.cos(4 * math.pi * t), 1.02
    return 0.004 * math.sin(2 * math.pi * t), 0.0, 1.005


def _render_frame(
    size: tuple[int, int],
    palette: list[str],
    angle: CameraAngle,
    lighting: Lighting,
    composition: Composition,
    label: str,
    t: float = 0.0,
    motion: MotionIntent = MotionIntent.STATIC_SUBTLE,
    rng_seed: int = 0,
) -> Image.Image:
    """Draw one synthetic ad frame.

    A crude stand-in for a real generation, but it varies along the same axes the
    sampler varies, which is what makes it useful rather than merely present.
    """
    w, h = size
    colours = [_hex_to_rgb(c) for c in (palette or _FALLBACK_PALETTE)]
    while len(colours) < 3:
        colours.append(colours[-1])
    brightness, contrast = _lighting_gain(lighting)
    dx, dy, zoom = _motion_offset(motion, t)

    img = Image.new("RGB", (w, h), colours[0])
    draw = ImageDraw.Draw(img)

    # Background gradient — direction depends on lighting so styles read apart.
    top, bottom = colours[0], colours[-1]
    if lighting is Lighting.RIM_BACKLIT:
        top, bottom = bottom, top
    for y in range(h):
        f = y / max(1, h - 1)
        draw.line([(0, y), (w, y)], fill=_tone(_mix(top, bottom, f), brightness, contrast))

    subj_scale, prod_scale, horizon = _angle_geometry(angle)
    ax, ay = _composition_anchor(composition)
    ax += dx
    ay += dy - horizon * 0.10
    cx, cy = ax * w, ay * h

    # "Model": head + torso, standing in for the human reference.
    torso_w = 0.30 * w * subj_scale * zoom
    torso_h = 0.46 * h * subj_scale * zoom
    skin = _mix(colours[1], (255, 235, 215), 0.45)
    draw.rounded_rectangle(
        [cx - torso_w / 2, cy - torso_h / 2, cx + torso_w / 2, cy + torso_h / 2],
        radius=int(torso_w * 0.28),
        fill=colours[1],
    )
    head_r = torso_w * 0.30
    draw.ellipse(
        [
            cx - head_r,
            cy - torso_h / 2 - head_r * 1.7,
            cx + head_r,
            cy - torso_h / 2 + head_r * 0.3,
        ],
        fill=skin,
    )

    # "Product": a distinct block held to one side, standing in for the product
    # reference. Its size tracks the camera angle so close-ups look like close-ups.
    pw = 0.13 * w * prod_scale * zoom
    ph = 0.22 * h * prod_scale * zoom
    px = cx + torso_w * 0.52
    py = cy + torso_h * 0.05
    accent = colours[2] if len(colours) > 2 else colours[1]
    draw.rounded_rectangle(
        [px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2],
        radius=int(pw * 0.18),
        fill=accent,
        outline=(255, 255, 255),
        width=max(1, round(w * 0.004)),
    )
    draw.rectangle(
        [px - pw * 0.34, py - ph * 0.10, px + pw * 0.34, py + ph * 0.10],
        fill=_mix(accent, (0, 0, 0), 0.35),
    )

    # Deterministic film grain, so identical seeds yield identical bytes.
    rng = random.Random(rng_seed)
    px_map = img.load()
    for _ in range(int(w * h * 0.004)):
        gx, gy = rng.randrange(w), rng.randrange(h)
        r, g, b = px_map[gx, gy]
        j = rng.randint(-12, 12)
        px_map[gx, gy] = (
            max(0, min(255, r + j)),
            max(0, min(255, g + j)),
            max(0, min(255, b + j)),
        )

    draw.text((round(w * 0.04), round(h * 0.03)), label, fill=(255, 255, 255))
    return img


class MockImageProvider(ImageProvider):
    """Renders a deterministic placeholder frame. Never touches the network."""

    name = "mock"
    model = "mock"

    def __init__(self, storage: Storage):
        self.storage = storage

    async def generate(self, request: ImageGenRequest) -> ImageGenResult:
        started = time.perf_counter()
        dp = request.brief.design_point
        size = request.aspect_ratio.pixel_size(MOCK_LONG_EDGE)
        label = f"{request.brief.slot_label}\nseed {request.seed}"
        frame = _render_frame(
            size=size,
            palette=request.palette,
            angle=dp.angle,
            lighting=dp.lighting,
            composition=dp.composition,
            label=label,
            rng_seed=request.seed,
        )
        buf = io.BytesIO()
        frame.save(buf, format="PNG", optimize=True)
        asset = self.storage.put_bytes(request.output_key, buf.getvalue(), "image/png")
        asset.width, asset.height = size
        return ImageGenResult(
            asset=asset,
            model=self.model,
            tier=Tier.MOCK,
            cost_usd=0.0,
            latency_ms=round((time.perf_counter() - started) * 1000),
            seed=request.seed,
        )


class MockVideoProvider(VideoProvider):
    """Animates the placeholder frame as a GIF with real frame timings.

    GIF rather than MP4 because ffmpeg is absent on this machine.  The duration
    encoded in the file is the requested duration, so the pipeline's "every
    delivered video is 8-10 s" check tests something real.
    """

    name = "mock"
    model = "mock"

    def __init__(self, storage: Storage):
        self.storage = storage

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        started = time.perf_counter()
        dp = request.brief.design_point
        size = request.aspect_ratio.pixel_size(MOCK_LONG_EDGE // 2)
        # Quantise the frame delay to what GIF can actually represent, then derive
        # the frame count from it, so the duration we report is the duration the
        # file really has.
        frame_ms = max(
            GIF_DELAY_QUANTUM_MS,
            round(1000 / MOCK_FPS / GIF_DELAY_QUANTUM_MS) * GIF_DELAY_QUANTUM_MS,
        )
        n_frames = max(2, round(request.duration_seconds * 1000 / frame_ms))

        frames = [
            _render_frame(
                size=size,
                palette=[],
                angle=dp.angle,
                lighting=dp.lighting,
                composition=dp.composition,
                label=f"{request.brief.slot_label}\n{dp.motion.value}",
                t=i / max(1, n_frames - 1),
                motion=dp.motion,
                rng_seed=request.seed + i,
            )
            for i in range(n_frames)
        ]

        buf = io.BytesIO()
        frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=frame_ms,
            loop=0,
            optimize=True,
        )
        key = request.output_key
        if key.endswith(".mp4"):
            key = key[: -len(".mp4")] + ".gif"
        asset = self.storage.put_bytes(key, buf.getvalue(), "image/gif")
        asset.width, asset.height = size

        return VideoGenResult(
            asset=asset,
            model=self.model,
            tier=Tier.MOCK,
            cost_usd=0.0,
            latency_ms=round((time.perf_counter() - started) * 1000),
            seed=request.seed,
            duration_seconds=round(n_frames * frame_ms / 1000, 3),
            fps=MOCK_FPS,
            was_chained=False,
        )


class MockLLMProvider(LLMProvider):
    """Returns nothing useful on purpose.

    Brief compilation has a deterministic template path that produces complete,
    valid briefs without an LLM (see ``adworker.briefs``).  The mock therefore
    signals "no LLM available" rather than fabricating text, so mock runs
    exercise the same template fallback that protects live runs when the LLM
    call fails.
    """

    name = "mock"
    model = "mock"

    async def complete_json(self, system: str, user: str, schema_hint: str) -> dict:
        return {}


def mock_reference_asset(
    storage: Storage,
    key: str,
    aspect: AspectRatio = AspectRatio.SQUARE_1_1,
    seed: int = 7,
) -> AssetRef:
    """Write a synthetic upload, so tests and demos need no real photographs."""
    size = aspect.pixel_size(512)
    frame = _render_frame(
        size=size,
        palette=_FALLBACK_PALETTE,
        angle=CameraAngle.EYE_LEVEL,
        lighting=Lighting.SOFT_DIFFUSED,
        composition=Composition.CENTERED_HERO,
        label="reference",
        rng_seed=seed,
    )
    buf = io.BytesIO()
    frame.save(buf, format="PNG", optimize=True)
    asset = storage.put_bytes(key, buf.getvalue(), "image/png")
    asset.width, asset.height = size
    return asset
