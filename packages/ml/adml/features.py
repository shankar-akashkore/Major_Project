"""Pixel-level features, computed with numpy only.

Everything in this module runs on an 8 GB laptop with no GPU and no torch.  That
is a deliberate choice about build order: the embedding features (SigLIP, DINOv2,
ArcFace, LAION aesthetic) arrive later and are trained in Colab, but the geometric
and photometric features can be real from week one.  So the quality gate and the
ranker have genuine signal to work with long before any model weights exist —
which means they can be tested, and their failures can be distinguished from the
predictor's failures.

The saliency implementation is Hou & Zhang's spectral residual, which is a few
lines of FFT and no learned parameters.  It matters more than it looks: the plan
leans on attention as the highest-validity published proxy for ad performance, and
this is the cheapest defensible way to estimate where a viewer's eye goes.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

# --- Colour ----------------------------------------------------------------

_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
#: D65 white point.
_WHITE = np.array([0.95047, 1.00000, 1.08883])


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64) / 255.0
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255, ``(..., 3)``) → CIE L*a*b*.

    Lab is used rather than RGB distance because palette adherence is a
    perceptual question — two colours can be far apart in RGB and
    indistinguishable to a viewer, and the score should reflect the viewer.
    """
    linear = _srgb_to_linear(np.asarray(rgb))
    xyz = linear @ _SRGB_TO_XYZ.T / _WHITE
    eps = (6 / 29) ** 3
    f = np.where(xyz > eps, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def delta_e_76(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """CIE76 colour difference.  ~2.3 is the just-noticeable threshold."""
    return np.sqrt(np.sum((lab_a - lab_b) ** 2, axis=-1))


# --- Loading ---------------------------------------------------------------


def load_image(data: bytes) -> np.ndarray:
    """Decode to an RGB array."""
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"))


def load_frames(data: bytes, max_frames: int = 48) -> list[np.ndarray]:
    """Decode an animated file (GIF now, MP4 once ffmpeg lands) to RGB frames.

    Frames are subsampled evenly to ``max_frames`` so a 10 s clip costs the same
    to analyse as a 5 s one.  The first frame is always included, because the
    hook-strength feature depends on it specifically.
    """
    with Image.open(io.BytesIO(data)) as img:
        total = getattr(img, "n_frames", 1)
        if total <= max_frames:
            indices = list(range(total))
        else:
            step = total / max_frames
            indices = sorted({min(total - 1, int(i * step)) for i in range(max_frames)})
        frames = []
        for i in indices:
            img.seek(i)
            frames.append(np.asarray(img.convert("RGB")))
    return frames


def frame_durations_seconds(data: bytes) -> float:
    """Total playback duration read from the file itself.

    Used to verify a delivered clip really is 8-10 s, rather than trusting the
    provider's reported number.
    """
    with Image.open(io.BytesIO(data)) as img:
        total = getattr(img, "n_frames", 1)
        ms = 0
        for i in range(total):
            img.seek(i)
            ms += img.info.get("duration", 0)
    return ms / 1000.0


# --- Basic photometrics ----------------------------------------------------


def _gray(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., :3] @ np.array([0.299, 0.587, 0.114])


def mean_luminance(rgb: np.ndarray) -> float:
    return float(_gray(rgb).mean() / 255.0)


def rms_contrast(rgb: np.ndarray) -> float:
    g = _gray(rgb) / 255.0
    return float(g.std())


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Süsstrunk colourfulness, normalised to roughly 0-1.

    Correlates with how much an image "pops" in a feed, which is why it is a
    reasonable input to a performance predictor rather than just a curiosity.
    """
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = math.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(min(1.0, (std + 0.3 * mean) / 110.0))


# --- Box filter and saliency ----------------------------------------------


def _box_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Mean filter with a ``k x k`` window, via summed-area table."""
    if k <= 1:
        return x
    pad = k // 2
    padded = np.pad(x, pad, mode="edge")
    cs = padded.cumsum(axis=0).cumsum(axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)), mode="constant")
    h, w = x.shape
    total = cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]
    return total[:h, :w] / (k * k)


def _blur(x: np.ndarray, passes: int = 3, k: int = 5) -> np.ndarray:
    """Approximate Gaussian blur by repeated box filtering."""
    for _ in range(passes):
        x = _box_filter(x, k)
    return x


def saliency_map(rgb: np.ndarray, size: int = 64) -> np.ndarray:
    """Spectral-residual saliency, normalised to 0-1 at ``size x size``.

    No learned parameters, so it behaves identically on this laptop and in Colab,
    and it gives the ranker a defensible attention proxy from day one.
    """
    small = np.asarray(Image.fromarray(rgb).convert("L").resize((size, size), Image.BILINEAR))
    fft = np.fft.fft2(small.astype(np.float64))
    log_amp = np.log(np.abs(fft) + 1e-8)
    phase = np.angle(fft)
    residual = log_amp - _box_filter(log_amp, 3)
    recon = np.fft.ifft2(np.exp(residual + 1j * phase))
    sal = _blur(np.abs(recon) ** 2, passes=2, k=5)
    lo, hi = sal.min(), sal.max()
    return (sal - lo) / (hi - lo) if hi > lo else np.zeros_like(sal)


def focal_concentration(sal: np.ndarray, top_fraction: float = 0.10) -> float:
    """Share of total saliency mass held by the most salient ``top_fraction``.

    High values mean one clear focal point; low values mean attention is smeared
    across the frame, which for an ad usually means the viewer does not know what
    they are being sold.

    This is a *proxy* for product salience.  Attributing saliency to the product
    specifically needs the product mask, which arrives with the ``rembg`` intake
    work — until then :mod:`adml.scoring` maps this in and flags the score as a
    stub rather than claiming product-level attention.
    """
    flat = np.sort(sal.ravel())[::-1]
    if flat.sum() <= 0:
        return 0.0
    k = max(1, int(len(flat) * top_fraction))
    return float(flat[:k].sum() / flat.sum())


def region_saliency_share(sal: np.ndarray, top: float, bottom: float) -> float:
    """Fraction of saliency mass falling *outside* the platform-chrome bands.

    1.0 means nothing important sits where Instagram's UI will cover it.
    """
    h = sal.shape[0]
    t = int(round(h * top))
    b = int(round(h * (1.0 - bottom)))
    if b <= t:
        return 0.0
    total = sal.sum()
    if total <= 0:
        return 0.0
    return float(sal[t:b, :].sum() / total)


# --- Composition -----------------------------------------------------------


def thirds_alignment(sal: np.ndarray) -> float:
    """How close the saliency centroid sits to a rule-of-thirds power point.

    Scored by distance to the nearest of the four intersections, normalised so
    dead-centre is not penalised into the floor — centred hero framing is a
    legitimate choice, just a different one.
    """
    h, w = sal.shape
    total = sal.sum()
    if total <= 0:
        return 0.5
    ys, xs = np.mgrid[0:h, 0:w]
    cy = float((sal * ys).sum() / total) / h
    cx = float((sal * xs).sum() / total) / w
    points = [(1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3)]
    best = min(math.hypot(cx - px, cy - py) for px, py in points)
    # 0.24 is roughly the distance from a power point to the frame centre.
    return float(max(0.0, 1.0 - best / 0.24))


def subject_scale(sal: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of the frame occupied by clearly-salient content."""
    return float((sal >= threshold).mean())


# --- Palette ---------------------------------------------------------------


def dominant_colors(rgb: np.ndarray, k: int = 5) -> list[tuple[tuple[int, int, int], float]]:
    """Dominant colours as ``((r, g, b), weight)``, via PIL's adaptive palette.

    Cheaper and steadier than running k-means for every candidate, and the
    weights are what the palette score needs.
    """
    img = Image.fromarray(rgb).convert("RGB")
    img.thumbnail((128, 128), Image.BILINEAR)
    quant = img.quantize(colors=k, method=Image.Quantize.FASTOCTREE)
    palette = quant.getpalette() or []
    counts = quant.getcolors() or []
    total = sum(c for c, _ in counts) or 1
    out = []
    for count, idx in sorted(counts, reverse=True):
        r, g, b = palette[idx * 3 : idx * 3 + 3]
        out.append(((r, g, b), count / total))
    return out


def _hex_to_rgb_tuple(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _point_segment_distance(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from ``q`` to the segment ``ab``, all in Lab."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(q - a))
    t = float(np.clip((q - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(q - (a + t * ab)))


def brand_gamut_distance(colour_lab: np.ndarray, brand_lab: np.ndarray) -> float:
    """Distance from a colour to the brand's *permissible* colour set.

    Matching only against exact palette entries is the wrong question.  A soft
    gradient between two brand colours contains midpoints that belong to neither,
    and a photograph of a navy backdrop contains a hundred tints of navy — both
    are entirely on-brand, and a vertex-only metric punishes them.

    So the permissible set is taken to be what a brand guideline actually allows:

    * the palette colours themselves,
    * blends between any two of them (segments between vertices),
    * tints and shades of each (segments toward white and toward black).

    A hue absent from the brand entirely still scores badly, which is the
    discrimination the check exists to provide.
    """
    white = np.array([100.0, 0.0, 0.0])
    black = np.array([0.0, 0.0, 0.0])

    best = min(float(np.linalg.norm(colour_lab - p)) for p in brand_lab)
    n = len(brand_lab)
    for i in range(n):
        for j in range(i + 1, n):
            best = min(best, _point_segment_distance(colour_lab, brand_lab[i], brand_lab[j]))
        best = min(
            best,
            _point_segment_distance(colour_lab, brand_lab[i], white),
            _point_segment_distance(colour_lab, brand_lab[i], black),
        )
    return best


#: ΔE at which a colour is considered to have left the brand's world entirely.
#: Calibrated in ``scripts/calibrate_gate.py`` against matched vs deliberately
#: mismatched palettes rather than guessed.
PALETTE_DELTA_E_CEILING = 30.0


def palette_adherence(rgb: np.ndarray, palette_hex: list[str]) -> tuple[float, float]:
    """``(score_0_to_1, weighted_mean_delta_e)`` against the brand palette.

    Each dominant colour is matched to the nearest point of the brand's
    permissible set (see :func:`brand_gamut_distance`) and the differences are
    weighted by how much of the frame each covers — so an off-brand background
    costs far more than an off-brand accent.
    """
    if not palette_hex:
        # No palette supplied means no constraint, not a failure.
        return 1.0, 0.0

    brand_lab = rgb_to_lab(np.array([_hex_to_rgb_tuple(h) for h in palette_hex], dtype=np.float64))
    doms = dominant_colors(rgb, k=5)
    if not doms:
        return 0.0, PALETTE_DELTA_E_CEILING

    weights = np.array([w for _, w in doms])
    dom_lab = rgb_to_lab(np.array([c for c, _ in doms], dtype=np.float64))
    nearest = np.array([brand_gamut_distance(lab, brand_lab) for lab in dom_lab])
    mean_de = float((nearest * weights).sum() / weights.sum())
    score = max(0.0, 1.0 - mean_de / PALETTE_DELTA_E_CEILING)
    return float(score), mean_de


# --- Video -----------------------------------------------------------------


def motion_energy(frames: list[np.ndarray]) -> list[float]:
    """Per-transition mean absolute frame difference, normalised to 0-1."""
    if len(frames) < 2:
        return [0.0]
    grays = [_gray(f) / 255.0 for f in frames]
    return [float(np.abs(grays[i + 1] - grays[i]).mean()) for i in range(len(grays) - 1)]


def temporal_consistency(frames: list[np.ndarray]) -> float:
    """Mean frame-to-frame correlation.

    Low values mean the content is not holding together between frames — the
    identity-drift and flicker failure mode that makes generated video unusable.
    """
    if len(frames) < 2:
        return 1.0
    grays = [(_gray(f) / 255.0).ravel() for f in frames]
    cors = []
    for i in range(len(grays) - 1):
        a, b = grays[i], grays[i + 1]
        if a.shape != b.shape:
            return 0.0
        sa, sb = a.std(), b.std()
        cors.append(1.0 if sa < 1e-6 and sb < 1e-6 else float(np.corrcoef(a, b)[0, 1]))
    vals = [c for c in cors if not math.isnan(c)]
    return float(np.clip(np.mean(vals), 0.0, 1.0)) if vals else 0.0


def hook_strength(frames: list[np.ndarray], fps: float, window_s: float = 1.0) -> float:
    """How arresting the opening second is.

    Combines saliency concentration in the first frame with how much movement
    happens inside the first ``window_s``.  Short-form ads live or die in the
    first second, so this is scored separately rather than averaged away across
    the whole clip.
    """
    if not frames:
        return 0.0
    first_focus = focal_concentration(saliency_map(frames[0]))
    n = max(2, min(len(frames), int(round(fps * window_s))))
    early = motion_energy(frames[:n])
    early_motion = float(np.mean(early)) if early else 0.0
    # Movement helps, but only up to a point — a chaotic first second is not a hook.
    motion_term = min(1.0, early_motion / 0.06)
    return float(np.clip(0.6 * first_focus + 0.4 * motion_term, 0.0, 1.0))


def seam_consistency(frames: list[np.ndarray], seam_index: int) -> float:
    """Temporal consistency across a concatenation seam.

    Only meaningful for chained clips.  Reported rather than smoothed over,
    because if 8-10 s has to be reached by stitching two ~5 s generations, the
    quality cost of that decision belongs in the write-up.
    """
    if not (0 < seam_index < len(frames)):
        return 1.0
    return temporal_consistency(frames[seam_index - 1 : seam_index + 1])
