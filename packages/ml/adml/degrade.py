"""Making catch-trial decoys: copies of real candidates that are obviously worse.

A catch trial only works if the right answer is not a matter of taste.  An
annotator who picks the blurred, washed-out copy of an image over the original
was not comparing them, and that is the one inference about annotator behaviour
that can be drawn without assuming a ground truth for quality.

The degradation is therefore deliberately crude and large: heavy blur *and* a
collapse of colour *and* a contrast crush, well past anything a generator would
produce on a bad day.  A subtle decoy would test the annotator's eyesight, which
is not the point and would flag careful people as careless.

:func:`degrade` measures the result, so the corpus builder can refuse a decoy
that did not come out visibly worse rather than shipping a broken catch trial.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .features import load_image, sharpness

#: Gaussian blur radius as a fraction of the image's short edge. 2.5% of a
#: 1024 px edge is ~26 px — unmistakable.
BLUR_FRACTION = 0.025

#: Remaining colour saturation and contrast. Both well below anything a
#: generation model produces.
SATURATION = 0.25
CONTRAST = 0.55

#: A decoy must lose at least this much measured sharpness, or it is not
#: obviously worse and the catch trial would be unfair rather than diagnostic.
MIN_SHARPNESS_DROP = 0.30


@dataclass
class Degradation:
    """The decoy's bytes plus evidence that the degradation actually landed."""

    data: bytes
    label: str
    sharpness_before: float
    sharpness_after: float

    @property
    def drop(self) -> float:
        return self.sharpness_before - self.sharpness_after

    @property
    def is_obvious(self) -> bool:
        return self.drop >= MIN_SHARPNESS_DROP


def degrade(data: bytes) -> Degradation:
    """Blur, desaturate and flatten an image into an unmistakably worse copy."""
    before = sharpness(load_image(data))

    image = Image.open(io.BytesIO(data)).convert("RGB")
    radius = max(2.0, BLUR_FRACTION * min(image.size))
    out = image.filter(ImageFilter.GaussianBlur(radius=radius))
    out = ImageEnhance.Color(out).enhance(SATURATION)
    out = ImageEnhance.Contrast(out).enhance(CONTRAST)

    buffer = io.BytesIO()
    out.save(buffer, format="PNG")
    payload = buffer.getvalue()

    after = sharpness(np.asarray(out, dtype=np.uint8))
    return Degradation(
        data=payload,
        label=f"blur-{radius:.0f}px-sat{SATURATION:g}-con{CONTRAST:g}",
        sharpness_before=before,
        sharpness_after=after,
    )
