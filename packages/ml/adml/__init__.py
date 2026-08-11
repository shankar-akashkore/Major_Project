"""Feature extraction, label collection design, and the performance predictor.

Split by hardware requirement on purpose:

* :mod:`adml.features` is numpy-only and runs anywhere, including this 8 GB
  laptop. It is real today.
* :mod:`adml.pairs` designs *which* comparisons to collect, and :mod:`adml.ranking`
  turns the collected judgements into rankings and the metrics the report rests
  on. Both are numpy-only for the same reason — the evaluation has to be
  reproducible in the same environment as the pipeline, not only in Colab.
* :mod:`adml.degrade` builds the deliberately-worse copies used as catch trials.
* :mod:`adml.split` divides data by *generation set*, which is the difference
  between an honest held-out number and a leaked one, and
  :mod:`adml.featureset` assembles grouped feature matrices sized to the labels
  that actually exist.
* :mod:`adml.predictor` is the trainable head and :mod:`adml.evaluate` is the
  harness that reports it against baselines and against the annotator noise
  ceiling. Both numpy, so the model in the ablation table is the same object the
  API serves — there is no second implementation to disagree with the first.
* :mod:`adml.embeddings` is the boundary to the parts that need torch. The frozen
  encoders (SigLIP, DINOv2, ArcFace, LAION aesthetic) run in Colab and cross into
  this package as an npz file. Nothing here imports torch.
* :mod:`adml.video` is the boundary to ffmpeg, and the only place a clip becomes
  frames. It exists because everything video previously decoded through PIL, which
  cannot open an MP4 — so the video half of the pipeline would have failed on the
  first real generation, after paying for it.
* :mod:`adml.serving` persists a fitted transform and head together and scores one
  candidate set with them. Without it the trained model could not reach the
  product: the evaluation and the pipeline were scoring with different things.
"""

from . import (
    degrade,
    embeddings,
    evaluate,
    features,
    featureset,
    pairs,
    predictor,
    ranking,
    serving,
    split,
    video,
)

__all__ = [
    "degrade",
    "embeddings",
    "evaluate",
    "features",
    "featureset",
    "pairs",
    "predictor",
    "ranking",
    "serving",
    "split",
    "video",
]
