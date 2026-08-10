"""Calibrate quality-gate thresholds from measurements rather than intuition.

The thresholds in ``adworker.gate`` decide whether a candidate is thrown away, so
picking them by feel is not good enough — and tuning them upward until the demo
passes would be fitting the test to the answer.

Method: render the mock frame set twice, once with the brand palette it is
supposed to honour and once with a deliberately unrelated palette.  Those two
distributions are the positive and negative classes.  A usable threshold sits
between them; if they overlap, the metric itself is not discriminating and the
threshold is not the thing to change.

Run:  .venv/bin/python scripts/calibrate_gate.py
"""

from __future__ import annotations

import numpy as np
from adml import features as F
from adproviders.mock import _render_frame
from adschema import CameraAngle, Composition, Lighting

#: A cool navy / rose / cream brand.
BRAND = ["#2b3a55", "#ce7777", "#f2e7d5"]
#: Deliberately unrelated: saturated greens and yellows share no hue with BRAND.
OFF_BRAND = ["#1f7a1f", "#8fd41f", "#f5f50a"]


def _render_grid(palette_used: list[str]) -> list[np.ndarray]:
    """One frame per lighting x composition x angle combination."""
    frames = []
    for lighting in Lighting:
        for composition in Composition:
            for angle in CameraAngle:
                frames.append(
                    np.asarray(
                        _render_frame(
                            (270, 480),
                            palette_used,
                            angle,
                            lighting,
                            composition,
                            "cal",
                            rng_seed=17,
                        )
                    )
                )
    return frames


def _summarise(name: str, values: list[float]) -> dict[str, float]:
    arr = np.array(values)
    stats = {
        "n": len(arr),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }
    print(
        f"  {name:22s} n={stats['n']:3d}  min={stats['min']:.3f}  p05={stats['p05']:.3f}  "
        f"median={stats['median']:.3f}  p95={stats['p95']:.3f}  max={stats['max']:.3f}"
    )
    return stats


def main() -> None:
    print("Rendering calibration grid (lighting x composition x angle)...")
    on_frames = _render_grid(BRAND)
    off_frames = _render_grid(OFF_BRAND)

    print("\n=== palette_adherence ===")
    print("  positive class: frames rendered FROM the brand palette")
    print("  negative class: frames rendered from an unrelated palette\n")
    on = [F.palette_adherence(f, BRAND)[0] for f in on_frames]
    off = [F.palette_adherence(f, BRAND)[0] for f in off_frames]
    on_stats = _summarise("on-brand", on)
    off_stats = _summarise("off-brand", off)

    gap = on_stats["min"] - off_stats["max"]
    if gap > 0:
        threshold = round((on_stats["min"] + off_stats["max"]) / 2, 3)
        print(
            f"\n  ✅ classes separate cleanly (gap {gap:.3f}). "
            f"Suggested THRESH_PALETTE = {threshold}"
        )
    else:
        # Midpoint between the 5th percentile of the positives and the 95th of the
        # negatives — accepts almost every genuine frame while still rejecting
        # almost every off-brand one.
        threshold = round((on_stats["p05"] + off_stats["p95"]) / 2, 3)
        fn = sum(1 for v in on if v < threshold) / len(on)
        fp = sum(1 for v in off if v >= threshold) / len(off)
        print(
            f"\n  ⚠️  classes overlap by {-gap:.3f}. "
            f"Suggested THRESH_PALETTE = {threshold} "
            f"(rejects {fn:.0%} of on-brand, admits {fp:.0%} of off-brand)"
        )

    print("\n=== other gate metrics on the on-brand grid ===")
    print("  (these have no negative class here; the numbers set a sane floor)\n")
    sal = [F.saliency_map(f) for f in on_frames]
    _summarise("focal_clarity", [F.focal_concentration(s) for s in sal])
    _summarise("safe_area(.14/.20)", [F.region_saliency_share(s, 0.14, 0.20) for s in sal])
    _summarise("exposure", [F.mean_luminance(f) for f in on_frames])
    _summarise("contrast", [F.rms_contrast(f) for f in on_frames])
    _summarise("colorfulness", [F.colorfulness(f) for f in on_frames])
    _summarise("thirds_alignment", [F.thirds_alignment(s) for s in sal])

    print(
        "\nNote: these are calibrated against the *mock* renderer. Re-run against a "
        "fixture set of real generations in week 6 before trusting them with money."
    )


if __name__ == "__main__":
    main()
