"""Feature extraction and the performance predictor.

Split by hardware requirement on purpose:

* :mod:`adml.features` is numpy-only and runs anywhere, including this 8 GB
  laptop. It is real today.
* The embedding extractors and the trained head (SigLIP, DINOv2, ArcFace, LAION
  aesthetic, the Bradley-Terry calibration) need torch and a GPU, so they run in
  Colab and land later. They will import from here, not replace it.
"""

from . import features

__all__ = ["features"]
