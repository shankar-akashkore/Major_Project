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
* :mod:`adml.crop` and :mod:`adml.audio` are the delivery half: placing a reframe
  window by saliency rather than by centring it, and mixing a licensed music bed to
  a stated loudness. Both report what they cost — a reframe discards content, and a
  bed without recorded provenance is refused outright.
* :mod:`adml.stages` harvests the two-stage rank agreement out of finished jobs. It
  identifies a generation set by the content of its candidates, because a golden
  replay writes a new job record over identical bytes and counting those as
  separate observations would manufacture the project's headline number.
* :mod:`adml.figures` draws the report's figures as SVG from the standard library
  alone. There is no matplotlib on this machine and buying one for four plots would
  cost more disk than the parts of the project that have no alternative. Every
  interval is drawn and every figure carries its own provenance line, so a plot
  pasted into a slide arrives with its caveats attached.
* :mod:`adml.report` turns the evaluation objects into tables in markdown and LaTeX.
  It exists to remove the one unrecorded step in any report — a number copied from a
  terminal into a document — which is where a figure measured on simulated labels
  becomes a figure presented as measured. A cell is a number or a stated absence;
  there are no blanks.
"""

from . import (
    audio,
    crop,
    degrade,
    embeddings,
    evaluate,
    features,
    featureset,
    figures,
    pairs,
    predictor,
    ranking,
    report,
    serving,
    split,
    stages,
    video,
)

__all__ = [
    "audio",
    "crop",
    "degrade",
    "embeddings",
    "evaluate",
    "features",
    "featureset",
    "figures",
    "pairs",
    "predictor",
    "ranking",
    "report",
    "serving",
    "split",
    "stages",
    "video",
]
