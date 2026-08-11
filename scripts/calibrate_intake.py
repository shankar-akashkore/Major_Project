"""Calibrate the intake thresholds against measured data, not intuition.

Run:  .venv/bin/python scripts/calibrate_intake.py

Three questions, each answered by measuring both classes rather than picking a
number that feels right:

1. **Sharpness** — where do usable uploads sit versus blurred or undersized ones?
   The awkward case is a product shot with a deliberately blurred background:
   shallow depth of field is *good* photography and must not be rejected.

2. **Border uniformity** — how uniform must a backdrop be before the flood-fill
   cutout can be trusted?  Because the fixtures are synthetic, the true product
   mask is known, so this reports the flood's actual IoU against ground truth
   rather than guessing from coverage alone.

3. **Flood tolerance** — how much ΔE latitude the flood needs to clear a real
   backdrop without eating into the product.

The same discipline as ``calibrate_gate.py``, which is where it earned its keep:
that run showed the threshold was fine and the metric was wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "ml"))

from adml import features as F  # noqa: E402

SIZE = (640, 640)
#: Product geometry, shared by every fixture so the ground-truth mask is exact.
BODY = (200, 150, 420, 500)
LABEL = (228, 250, 392, 330)
CAP = (255, 162, 365, 212)
NAVY = (43, 58, 85)
CREAM = (240, 236, 226)
GOLD = (198, 160, 92)


def _draw_product(img: Image.Image) -> None:
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(BODY, radius=28, fill=NAVY)
    d.rectangle(LABEL, fill=CREAM)
    d.ellipse(CAP, fill=GOLD)


def ground_truth_mask() -> np.ndarray:
    """Exact product mask, from the same geometry the fixtures are drawn with."""
    stencil = Image.new("L", SIZE, 0)
    _draw_product_stencil(stencil)
    return np.asarray(stencil) > 127


def _draw_product_stencil(img: Image.Image) -> None:
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(BODY, radius=28, fill=255)


def plain_sweep(shade: int = 245) -> np.ndarray:
    img = Image.new("RGB", SIZE, (shade, shade, shade))
    _draw_product(img)
    return np.asarray(img)


def gradient_sweep() -> np.ndarray:
    ramp = np.linspace(250, 190, SIZE[1])[:, None, None] * np.ones((1, SIZE[0], 3))
    img = Image.fromarray(ramp.astype(np.uint8), "RGB")
    _draw_product(img)
    return np.asarray(img)


def lifestyle(blur_radius: float) -> np.ndarray:
    """A room-like backdrop: distinct regions, optionally thrown out of focus.

    Deliberately *not* blurred noise — noise blurs into flat grey and would make
    a busy background look uniform, which is how a fixture can quietly validate
    nothing.  Large blocks of differing colour stay distinguishable at any blur.
    """
    img = Image.new("RGB", SIZE, (206, 190, 168))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 430, SIZE[0], SIZE[1]], fill=(122, 92, 66))  # floor
    d.rectangle([40, 120, 250, 430], fill=(88, 110, 96))  # plant
    d.rectangle([430, 60, 620, 300], fill=(230, 226, 214))  # window
    d.ellipse([470, 330, 600, 460], fill=(168, 84, 72))  # cushion
    if blur_radius:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    _draw_product(img)
    return np.asarray(img)


def blurred(rgb: np.ndarray, radius: float) -> np.ndarray:
    """Blur the *whole* frame — a camera-shake or missed-focus upload."""
    return np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius)))


def resampled(rgb: np.ndarray, edge: int) -> np.ndarray:
    """Simulate a small upload: downscale, then measure as-is."""
    return np.asarray(Image.fromarray(rgb).resize((edge, edge), Image.LANCZOS))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def main() -> None:
    truth = ground_truth_mask()

    print("=" * 78)
    print("1. SHARPNESS")
    print("=" * 78)
    print("\n  Usable uploads (must pass):")
    usable = {
        "plain sweep, sharp": plain_sweep(),
        "gradient sweep, sharp": gradient_sweep(),
        "lifestyle, sharp bg": lifestyle(0),
        "lifestyle, bokeh bg (r=10)": lifestyle(10),
        "lifestyle, bokeh bg (r=20)": lifestyle(20),
        "plain sweep @ 512px": resampled(plain_sweep(), 512),
    }
    for name, frame in usable.items():
        print(f"    {name:32} {F.sharpness(frame):.4f}")
    usable_min = min(F.sharpness(f) for f in usable.values())

    print("\n  Unusable uploads (must fail):")
    unusable = {
        "whole frame blur r=1": blurred(plain_sweep(), 1),
        "whole frame blur r=2": blurred(plain_sweep(), 2),
        "whole frame blur r=4": blurred(plain_sweep(), 4),
        "lifestyle, all blurred r=3": blurred(lifestyle(0), 3),
        "plain sweep @ 128px": resampled(plain_sweep(), 128),
        "plain sweep @ 192px": resampled(plain_sweep(), 192),
    }
    for name, frame in unusable.items():
        print(f"    {name:32} {F.sharpness(frame):.4f}")
    unusable_max = max(F.sharpness(f) for f in unusable.values())

    print(f"\n  usable min  = {usable_min:.4f}")
    print(f"  unusable max = {unusable_max:.4f}")
    print(
        "\n  READ THIS BEFORE ACTING ON THE NUMBERS ABOVE.\n"
        "  These fixtures are vector drawings, and hard vector edges carry far more\n"
        "  Laplacian energy than photographic detail does — which is why the usable\n"
        "  class sits at 0.96-1.00, a range real photographs do not occupy. A threshold\n"
        "  set from this separation looks safe and is not: a flat-shaded synthetic\n"
        "  portrait measures 0.11 and would be refused by it.\n"
        "  So `focus` is reported as an advisory and blocks nothing. Re-run this against\n"
        "  real uploads in week 6 before promoting it to a blocking check."
    )
    if usable_min > unusable_max:
        midpoint = (usable_min + unusable_max) / 2
        print(f"\n  synthetic classes separate at ~{midpoint:.3f} — not a usable threshold yet")
    else:
        print("  -> OVERLAP. A single global threshold cannot separate these.")
        print("     Borderline members of each class:")
        for name, frame in usable.items():
            if (v := F.sharpness(frame)) <= unusable_max:
                print(f"       usable   {name:30} {v:.4f}")
        for name, frame in unusable.items():
            if (v := F.sharpness(frame)) >= usable_min:
                print(f"       unusable {name:30} {v:.4f}")

    print()
    print("=" * 78)
    print("2. BORDER UNIFORMITY vs ACTUAL FLOOD QUALITY")
    print("=" * 78)
    print("\n  IoU is the flood's derived foreground against the known product mask.")
    print(f"  {'fixture':30} {'uniformity':>11} {'coverage':>9} {'IoU':>7}")
    backdrops = {
        "plain sweep (245)": plain_sweep(),
        "plain sweep (200)": plain_sweep(200),
        "gradient sweep": gradient_sweep(),
        "lifestyle, bokeh r=20": lifestyle(20),
        "lifestyle, bokeh r=10": lifestyle(10),
        "lifestyle, sharp": lifestyle(0),
    }
    rows = []
    for name, frame in backdrops.items():
        uniformity = F.border_uniformity(frame)
        bg = F.background_mask_by_flood(frame)
        quality = iou(~bg, truth)
        rows.append((name, uniformity, float(bg.mean()), quality))
        print(f"  {name:30} {uniformity:11.4f} {bg.mean():9.3f} {quality:7.3f}")

    good = [r for r in rows if r[3] >= 0.85]
    bad = [r for r in rows if r[3] < 0.85]
    if good and bad:
        lo = min(r[1] for r in good)
        hi = max(r[1] for r in bad)
        print(f"\n  trustworthy floods have uniformity >= {lo:.4f}")
        print(f"  untrustworthy floods have uniformity <= {hi:.4f}")
        if lo > hi:
            print(f"  -> suggest MIN_BORDER_UNIFORMITY = {(lo + hi) / 2:.3f}")
        else:
            print("  -> OVERLAP: uniformity alone does not predict flood quality.")
    elif not bad:
        print("\n  every fixture flooded well; need a harder backdrop to find the limit.")
    else:
        print("\n  no fixture flooded well; the flood is not usable as written.")

    print()
    print("=" * 78)
    print("3. FLOOD TOLERANCE (on the backdrops that are floodable at all)")
    print("=" * 78)
    print(f"\n  {'tolerance ΔE':>13} {'plain IoU':>10} {'gradient IoU':>13}")
    for tol in (4, 8, 12, 16, 24, 32):
        p = iou(~F.background_mask_by_flood(plain_sweep(), tolerance_de=tol), truth)
        g = iou(~F.background_mask_by_flood(gradient_sweep(), tolerance_de=tol), truth)
        print(f"  {tol:13} {p:10.3f} {g:13.3f}")

    print()
    print("=" * 78)
    print("4. PALETTE: whole frame vs masked cutout")
    print("=" * 78)
    print("\n  The product's real colours are navy #2b3a55, cream #f0ece2, gold #c6a05c.")
    for name, frame in (("plain sweep", plain_sweep()), ("lifestyle bokeh", lifestyle(10))):
        whole = F.extract_palette(frame)
        masked = F.extract_palette(frame, mask=truth)
        print(f"\n  {name}")
        print(f"    whole frame : {[c for c, _ in whole]}")
        print(f"    masked      : {[c for c, _ in masked]}")


if __name__ == "__main__":
    main()
