"""The generation pipeline: sampling, brief compilation, gating, scoring, ranking."""

from .briefs import compile_briefs
from .gate import evaluate_image, stricter_prompt
from .intake import preprocess
from .pipeline import ConsentRefused, NoViableCandidates, Pipeline, UnusableReference
from .sampler import (
    describe_diversity,
    min_pairwise_distance,
    sample_design_points,
)
from .scoring import explain, rank_videos, score_image, score_video

__all__ = [
    "Pipeline",
    "ConsentRefused",
    "NoViableCandidates",
    "UnusableReference",
    "preprocess",
    "compile_briefs",
    "sample_design_points",
    "min_pairwise_distance",
    "describe_diversity",
    "evaluate_image",
    "stricter_prompt",
    "score_image",
    "score_video",
    "rank_videos",
    "explain",
]
