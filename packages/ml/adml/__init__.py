"""Feature extraction, label collection design, and the performance predictor.

Split by hardware requirement on purpose:

* :mod:`adml.features` is numpy-only and runs anywhere, including this 8 GB
  laptop. It is real today.
* :mod:`adml.pairs` designs *which* comparisons to collect, and :mod:`adml.ranking`
  turns the collected judgements into rankings and the metrics the report rests
  on. Both are numpy-only for the same reason — the evaluation has to be
  reproducible in the same environment as the pipeline, not only in Colab.
* :mod:`adml.degrade` builds the deliberately-worse copies used as catch trials.
* The embedding extractors and the trained head (SigLIP, DINOv2, ArcFace, LAION
  aesthetic) need torch and a GPU, so they run in Colab and land later. They will
  import from here, not replace it — in particular they consume
  :func:`adml.ranking.fit_bradley_terry`'s output as their calibration target.
"""

from . import degrade, features, pairs, ranking

__all__ = ["degrade", "features", "pairs", "ranking"]
