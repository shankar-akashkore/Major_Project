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

Video is written as **H.264 MP4 when ffmpeg is available**, and falls back to an
animated GIF when it is not.  That ordering matters more than it looks: the whole
video-scoring half of the pipeline decodes with ffmpeg, and while the mock only
ever produced GIFs, none of that path was exercised by a single test.  PIL cannot
open an MP4, so the first paid Kling clip would have failed in stage 7 after being
billed.  Free mock runs now travel the same decode path as paid ones.

Frame timings are real in both containers, so duration assertions hold either way.
"""

from __future__ import annotations

import io
import math
import random
import time

from adml import video as V
from adschema import AspectRatio, AssetRef, CameraAngle, Composition, Lighting, MotionIntent, Tier
from PIL import Image, ImageDraw, ImageFilter

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

    Worth being blunt about the limit of this stand-in: the renderer moves a flat
    frame, so it can only ever express *camera* motion.  Every intent now names a
    human action, and none of that is representable here.  These are flow
    signatures chosen to stay distinguishable from one another, not previews of
    what a real generation looks like — which is why the motion thresholds in the
    quality gate have to be recalibrated against paid clips rather than these.
    """
    if motion is MotionIntent.PRODUCT_REVEAL:
        # Something rising into frame while the camera closes in.
        return 0.0, -0.06 * t, 1.0 + 0.10 * t
    if motion is MotionIntent.HERO_TURN:
        # Lateral swing, camera locked off.
        return -0.08 * math.sin(math.pi * t), 0.0, 1.02
    if motion is MotionIntent.IN_USE:
        # Busy, small-amplitude hand-scale movement.
        return 0.03 * math.sin(3 * math.pi * t), 0.02 * math.cos(2 * math.pi * t), 1.0 + 0.06 * t
    if motion is MotionIntent.OFFER_TO_CAMERA:
        # The product grows in frame; the camera itself does not move.
        return 0.0, 0.0, 1.0 + 0.18 * t
    if motion is MotionIntent.PICK_UP:
        # Down to the surface, then up and in with the hands.
        return 0.02 * t, 0.07 * math.sin(math.pi * t) - 0.04 * t, 1.0 + 0.08 * t
    # WALK_IN: stride bob against a slow pull-back.
    return 0.02 * math.sin(4 * math.pi * t), 0.01 * math.cos(4 * math.pi * t), 1.14 - 0.12 * t


def _render_frame(
    size: tuple[int, int],
    palette: list[str],
    angle: CameraAngle,
    lighting: Lighting,
    composition: Composition,
    label: str,
    t: float = 0.0,
    motion: MotionIntent = MotionIntent.HERO_TURN,
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

    if label:
        draw.text((round(w * 0.04), round(h * 0.03)), label, fill=(255, 255, 255))
    return img


class MockImageProvider(ImageProvider):
    """Renders a deterministic placeholder frame. Never touches the network.

    ``annotate_frames`` burns the design point and the seed into the corner, which
    is what makes the dev harness readable — and must be turned **off** for images
    that a human will be asked to compare. An annotator who can read
    "close up product / rim backlit" off the frame is not judging the image, and
    real generations carry no such caption, so labels collected against captioned
    mock frames would not transfer.
    """

    name = "mock"
    model = "mock"

    def __init__(self, storage: Storage, *, annotate_frames: bool = True):
        self.storage = storage
        self.annotate_frames = annotate_frames

    async def generate(self, request: ImageGenRequest) -> ImageGenResult:
        started = time.perf_counter()
        dp = request.brief.design_point
        size = request.aspect_ratio.pixel_size(MOCK_LONG_EDGE)
        label = f"{request.brief.slot_label}\nseed {request.seed}" if self.annotate_frames else ""
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
    """Animates the placeholder frame with real frame timings.

    MP4 when ffmpeg is present, GIF otherwise.  The duration encoded in the file
    is the requested duration in both cases, so the pipeline's "every delivered
    video is 8-10 s" check tests something real — and because it now reads the
    duration back out of a container ffmpeg wrote, it tests the same code path a
    Kling clip will take.

    ``force_gif`` exists for the one test that has to cover the no-ffmpeg fallback
    on a machine where ffmpeg *is* installed.
    """

    name = "mock"
    model = "mock"

    def __init__(self, storage: Storage, *, force_gif: bool = False):
        self.storage = storage
        self.force_gif = force_gif

    @property
    def writes_mp4(self) -> bool:
        return V.FFMPEG is not None and not self.force_gif

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        started = time.perf_counter()
        dp = request.brief.design_point
        size = request.aspect_ratio.pixel_size(MOCK_LONG_EDGE // 2)

        if self.writes_mp4:
            # H.264 takes an exact frame rate, so the frame count follows directly
            # from the duration with no quantisation to work around.
            fps = MOCK_FPS
            n_frames = max(2, round(request.duration_seconds * fps))
            duration = n_frames / fps
        else:
            # GIF stores frame delays in centiseconds. Quantise the delay to what
            # the format can represent, then derive the frame count from *that*, so
            # the duration reported is the duration the file really has. At 6 fps
            # an unquantised 167 ms delay silently became 160 ms and the reported
            # duration drifted from the real one.
            frame_ms = max(
                GIF_DELAY_QUANTUM_MS,
                round(1000 / MOCK_FPS / GIF_DELAY_QUANTUM_MS) * GIF_DELAY_QUANTUM_MS,
            )
            fps = 1000 / frame_ms
            n_frames = max(2, round(request.duration_seconds * 1000 / frame_ms))
            duration = n_frames * frame_ms / 1000

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

        key = request.output_key
        if self.writes_mp4:
            import numpy as np

            data = V.encode_mp4([np.asarray(f) for f in frames], fps=fps)
            mime = "video/mp4"
            if not key.endswith(".mp4"):
                key = f"{key.rsplit('.', 1)[0]}.mp4"
        else:
            buf = io.BytesIO()
            frames[0].save(
                buf,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=round(1000 / fps),
                loop=0,
                optimize=True,
            )
            data = buf.getvalue()
            mime = "image/gif"
            if key.endswith(".mp4"):
                key = key[: -len(".mp4")] + ".gif"

        asset = self.storage.put_bytes(key, data, mime)
        asset.width, asset.height = size

        return VideoGenResult(
            asset=asset,
            model=self.model,
            tier=Tier.MOCK,
            cost_usd=0.0,
            latency_ms=round((time.perf_counter() - started) * 1000),
            seed=request.seed,
            duration_seconds=round(duration, 3),
            fps=round(fps),
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
    return _store_png(storage, key, frame)


def _store_png(storage: Storage, key: str, frame: Image.Image) -> AssetRef:
    buf = io.BytesIO()
    frame.save(buf, format="PNG", optimize=True)
    asset = storage.put_bytes(key, buf.getvalue(), "image/png")
    asset.width, asset.height = frame.size
    return asset


# --- Role-specific synthetic uploads ---------------------------------------
#
# ``mock_reference_asset`` renders the same abstract composition for every role,
# which is fine for the generation path and useless for intake: intake's job is to
# cut a product away from its backdrop and find a face, and an abstract gradient
# has neither.  These two produce uploads with the structure intake looks for, so
# the cutout and face-detection paths are exercised by the default demo job rather
# than only by whatever real photographs happen to be lying around.


def mock_product_asset(
    storage: Storage,
    key: str,
    aspect: AspectRatio = AspectRatio.SQUARE_1_1,
    seed: int = 22,
    backdrop: int = 245,
) -> AssetRef:
    """A product on a near-seamless studio sweep.

    Flat backdrop on purpose: it is the case a flood-fill cutout can handle, so the
    fallback path is reachable without ``rembg`` installed.
    """
    width, height = aspect.pixel_size(512)
    rng = random.Random(seed)
    img = Image.new("RGB", (width, height), (backdrop, backdrop, backdrop))
    draw = ImageDraw.Draw(img)

    body_w, body_h = round(width * 0.34), round(height * 0.55)
    left, top = (width - body_w) // 2, round(height * 0.22)
    navy = _hex_to_rgb(_FALLBACK_PALETTE[0])
    accent = _hex_to_rgb(_FALLBACK_PALETTE[1])
    cream = _hex_to_rgb(_FALLBACK_PALETTE[3])

    draw.rounded_rectangle(
        [left, top, left + body_w, top + body_h], radius=round(body_w * 0.16), fill=navy
    )
    label_pad = round(body_w * 0.13)
    draw.rectangle(
        [
            left + label_pad,
            top + round(body_h * 0.34),
            left + body_w - label_pad,
            top + round(body_h * 0.58),
        ],
        fill=cream,
    )
    cap_w = round(body_w * 0.42)
    draw.ellipse(
        [
            left + (body_w - cap_w) // 2,
            top - round(body_h * 0.06),
            left + (body_w + cap_w) // 2,
            top + round(body_h * 0.10),
        ],
        fill=accent,
    )
    # A faint contact shadow, so the backdrop is not perfectly synthetic-flat.
    draw.ellipse(
        [left - 8, top + body_h - 6, left + body_w + 8, top + body_h + 14],
        fill=(backdrop - 18, backdrop - 18, backdrop - 16),
    )
    for _ in range(round(width * height * 0.0004)):
        x, y = rng.randrange(width), rng.randrange(height)
        shade = backdrop + rng.randint(-3, 3)
        draw.point((x, y), fill=(shade, shade, shade))

    return _store_png(storage, key, img)


def mock_portrait_asset(
    storage: Storage,
    key: str,
    aspect: AspectRatio = AspectRatio.PORTRAIT_4_5,
    seed: int = 11,
) -> AssetRef:
    """A synthetic frontal portrait that a Haar cascade actually detects.

    Verified to produce a detection rather than assumed to: the proportions here
    (brow mass above the eyes, eye spacing, a mouth shadow below) are what the
    frontal-face cascade responds to, and the slight blur matters — a cascade
    trained on photographs does not fire on hard vector edges.

    Two limits worth being explicit about, because this fixture is what several
    tests stand on:

    * It only just clears the detector.  Grain above about 4 levels loses it, where
      a real photograph would be found comfortably.  Treat a detection here as
      evidence the code path works, not that the detector is good.
    * Its focus figure is close to meaningless.  Flat vector shading has almost no
      high-frequency content, so :func:`adml.features.sharpness` reads it as soft,
      and adding grain raises that number without adding any real detail.  That is
      a property of Laplacian-variance sharpness, and it is why intake reports focus
      as an advisory rather than blocking on it.
    """
    width, height = aspect.pixel_size(512)
    img = Image.new("RGB", (width, height), (150, 150, 152))
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, round(height * 0.42)
    fw, fh = round(width * 0.38), round(height * 0.30)
    skin = (215, 190, 170)
    hair = (90, 70, 60)

    draw.ellipse([cx - fw, cy - fh, cx + fw, cy + fh], fill=hair)
    draw.ellipse(
        [cx - round(fw * 0.86), cy - round(fh * 0.30), cx + round(fw * 0.86), cy + fh], fill=skin
    )
    eye_dx, eye_dy = round(fw * 0.42), round(fh * 0.06)
    eye_r = max(4, round(fw * 0.16))
    for sign in (-1, 1):
        draw.ellipse(
            [
                cx + sign * eye_dx - eye_r,
                cy + eye_dy - round(eye_r * 0.7),
                cx + sign * eye_dx + eye_r,
                cy + eye_dy + round(eye_r * 0.7),
            ],
            fill=(45, 40, 40),
        )
    draw.polygon(
        [
            (cx, cy + round(fh * 0.18)),
            (cx - round(fw * 0.12), cy + round(fh * 0.44)),
            (cx + round(fw * 0.12), cy + round(fh * 0.44)),
        ],
        fill=(192, 167, 150),
    )
    draw.ellipse(
        [
            cx - round(fw * 0.30),
            cy + round(fh * 0.56),
            cx + round(fw * 0.30),
            cy + round(fh * 0.76),
        ],
        fill=(140, 95, 90),
    )
    # Shoulders, so the frame reads as a portrait rather than a floating head.
    draw.ellipse(
        [cx - round(width * 0.46), cy + fh, cx + round(width * 0.46), height + round(height * 0.3)],
        fill=_hex_to_rgb(_FALLBACK_PALETTE[0]),
    )

    img = img.filter(ImageFilter.GaussianBlur(1.5))
    return _store_png(storage, key, _add_grain(img, seed=seed))


#: Grain amplitude, in 8-bit levels, uniform so σ ≈ amplitude/√3.
#:
#: 3 gives σ ≈ 1.7, which is about a clean low-ISO sensor. It is chosen for detector
#: margin, not for any metric it produces: measured on this fixture, the Haar
#: cascade still finds the face at amplitudes 1-4 and loses it at 6, so 3 sits
#: comfortably inside the working range. Worth knowing that a real photograph is
#: not nearly this fragile — see :func:`mock_portrait_asset`.
_GRAIN_AMPLITUDE = 3


def _add_grain(img: Image.Image, seed: int) -> Image.Image:
    """Add fine luminance noise, so a synthetic frame behaves like a photograph.

    Flat vector shading is not merely unrealistic, it is unrepresentative in ways
    that matter: contrast, colourfulness, saliency and sharpness all read a
    hand-drawn region very differently from a photographed one, and a fixture that
    stands in for an upload should not quietly flatter every feature that looks at
    it.  A real photograph carries sensor noise and surface texture everywhere.
    """
    rng = random.Random(seed)
    pixels = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            delta = rng.randint(-_GRAIN_AMPLITUDE, _GRAIN_AMPLITUDE)
            pixels[x, y] = (
                min(255, max(0, r + delta)),
                min(255, max(0, g + delta)),
                min(255, max(0, b + delta)),
            )
    return img
