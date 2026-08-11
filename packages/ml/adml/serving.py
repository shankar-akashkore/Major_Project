"""Persisting a fitted model, and scoring one candidate set with it.

Until this module existed the trained head could not be used by the product at
all.  Training happened in ``scripts/train_predictor.py``, the fitted objects lived
for the length of that process, and :mod:`adworker.scoring` went on computing a
hand-weighted blend stamped ``heuristic-0``.  The evaluation and the pipeline were
scoring with two different things, and only one of them was in the report.

**The fitted transform travels with the weights.**  A :class:`FeaturePipeline`
holds PCA components and standardisation statistics fitted on training rows; those
are as much part of the model as ``w1`` is.  Saving weights alone and re-fitting
the transform at serving time would standardise against whatever candidates
happened to be in front of it — three frames from one job — so a feature's value
would depend on its peers rather than on itself.  Both are saved together, in one
file, or neither is.

**A model is refused unless it fits what it is asked to score.**  The saved
feature names are compared against the ones computed at serving time, and a
mismatch raises.  This is the failure that would otherwise be silent and total:
add a feature column, and every weight after the insertion point applies to the
wrong feature while the model keeps returning plausible numbers.

**The scale is a gauge, so ranking is the only claim.**  The pairwise objective
has no output bias, so a constant added to every score cancels in every comparison
and is not identifiable.  Raw scores are therefore not comparable between models,
between folds, or against any absolute standard.  :func:`rank_within_set` returns
an *order*, and :class:`ServedScores` exposes a 0-1 value only as a within-set
relative position, labelled as such — never as a calibrated probability of
anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import EmbeddingSpec
from .featureset import BlockPCA, FeatureGroup, FeaturePipeline, FeatureTable, Standardiser
from .predictor import PairwiseRanker, TrainConfig

#: Bumped when the on-disk layout changes, so an old file fails with a version
#: message rather than a confusing shape error.
MODEL_FORMAT_VERSION = 2

#: Default location, under the repo's fixtures. Small — a linear head over ~40
#: features is a few kilobytes, so this is committable if wanted.
DEFAULT_MODEL_PATH = Path("fixtures/models/ranker.npz")


class ModelMismatch(RuntimeError):
    """The saved model does not match the features it is being asked to score."""


@dataclass(frozen=True)
class ModelCard:
    """Provenance that has to survive to the point of use.

    Every field here answers a question someone reading a ranked result will ask,
    and each is a question the numbers alone cannot answer.

    ``is_real_features`` is the one that matters most.  It is False when any column
    came from :func:`adml.embeddings.hash_embeddings` — the deterministic stand-in
    used before a GPU has run.  A model fitted on stand-in embeddings produces
    perfectly plausible rankings, and shipping one unlabelled is how a placeholder
    reaches a reported result.  It propagates all the way to
    ``ScoreBreakdown.is_stub``.
    """

    model_version: str
    trained_at: str
    n_train_observations: int
    n_features: int
    n_parameters: int
    #: Held-out pairwise accuracy at training time, and the ceiling it was measured
    #: against. Carried so a served ranking can say how good it actually is.
    holdout_accuracy: float = float("nan")
    ceiling_accuracy: float = float("nan")
    #: Lower bound of the 95% bootstrap interval on ``holdout_accuracy``. Carried
    #: because the point estimate alone cannot answer "is this better than a coin
    #: flip" — 0.53 over 100 comparisons and 0.53 over 3,000 are different claims.
    holdout_ci_low: float = float("nan")
    is_real_features: bool = True
    embedding_sources: dict[str, str] | None = None
    notes: str = ""

    @property
    def beats_chance(self) -> bool:
        """Whether held-out accuracy resolved above 0.5 at 95%.

        The interval's lower bound, not the point estimate. This project's whole
        statistical discipline is that an unresolved difference is not a result, and
        a model is not exempt from its own standard.
        """
        return bool(np.isfinite(self.holdout_ci_low) and self.holdout_ci_low > 0.5)

    @property
    def is_stub(self) -> bool:
        """True while anything about this model is a placeholder.

        Three ways to be a stub, and the third is the one worth stating.

        A model trained on stand-in embeddings is a stub however well it scores.  A
        model whose held-out accuracy was never measured is a stub because nothing
        establishes it beats guessing.  And **a model measured at chance is a stub
        too** — it has been evaluated and found not to work, which is a stronger
        reason to label it than never having been evaluated at all.  Without that
        last clause the first model this project ever saved would have shipped as a
        validated predictor while sitting at 0.496.
        """
        if not self.is_real_features:
            return True
        if not np.isfinite(self.holdout_accuracy):
            return True
        return not self.beats_chance

    @property
    def ceiling_fraction(self) -> float:
        """Held-out accuracy as a share of what a perfect predictor could reach.

        The honest way to read a 0.68: against a 0.72 annotator-noise ceiling it is
        most of what is reachable, not a mediocre absolute score.

        **Chance-corrected**, so 0.68 against a 0.72 ceiling is 82%, not 94%.  Both
        figures are measured from the 0.5 floor of a coin flip, because the raw ratio
        ``0.68 / 0.72 = 0.94`` awards a model at chance 69% of the ceiling — which
        flatters a predictor that has learned nothing.  ``docs/prediction-protocol.md``
        quotes the raw ratio in its scale study ("94-95% of oracle"); the two are
        different quantities and the difference is not a discrepancy.

        NaN unless both were measured, or if the ceiling is itself at chance.
        """
        if not (np.isfinite(self.holdout_accuracy) and np.isfinite(self.ceiling_accuracy)):
            return float("nan")
        if self.ceiling_accuracy <= 0.5:
            return float("nan")
        # Both measured against the 0.5 floor of a coin flip.
        return (self.holdout_accuracy - 0.5) / (self.ceiling_accuracy - 0.5)

    def summary(self) -> str:
        parts = [f"{self.model_version}", f"{self.n_features} features"]
        if np.isfinite(self.holdout_accuracy):
            text = f"held-out {self.holdout_accuracy:.3f}"
            if np.isfinite(self.holdout_ci_low):
                text += f" (95% CI low {self.holdout_ci_low:.3f})"
            if np.isfinite(self.ceiling_accuracy):
                text += f" of {self.ceiling_accuracy:.3f} ceiling"
                fraction = self.ceiling_fraction
                if np.isfinite(fraction) and fraction > 0.0:
                    text += f" ({fraction:.0%} of achievable)"
            parts.append(text)
            if not self.beats_chance:
                # Stated whether or not a ceiling was measured. A negative share of
                # achievable is arithmetically correct and reads like a rounding
                # artifact, so say the actual thing instead.
                parts.append("DOES NOT BEAT CHANCE")
        else:
            parts.append("held-out accuracy not measured")
        if not self.is_real_features:
            parts.append("STAND-IN FEATURES")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "trained_at": self.trained_at,
            "n_train_observations": self.n_train_observations,
            "n_features": self.n_features,
            "n_parameters": self.n_parameters,
            "holdout_accuracy": self.holdout_accuracy,
            "ceiling_accuracy": self.ceiling_accuracy,
            "holdout_ci_low": self.holdout_ci_low,
            "is_real_features": self.is_real_features,
            "embedding_sources": self.embedding_sources or {},
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelCard:
        return cls(
            model_version=str(raw.get("model_version", "unknown")),
            trained_at=str(raw.get("trained_at", "")),
            n_train_observations=int(raw.get("n_train_observations", 0)),
            n_features=int(raw.get("n_features", 0)),
            n_parameters=int(raw.get("n_parameters", 0)),
            holdout_accuracy=float(raw.get("holdout_accuracy", float("nan"))),
            ceiling_accuracy=float(raw.get("ceiling_accuracy", float("nan"))),
            holdout_ci_low=float(raw.get("holdout_ci_low", float("nan"))),
            is_real_features=bool(raw.get("is_real_features", True)),
            embedding_sources=dict(raw.get("embedding_sources") or {}),
            notes=str(raw.get("notes", "")),
        )


@dataclass
class ServedModel:
    """A fitted transform plus a fitted head, ready to score."""

    pipeline: FeaturePipeline
    ranker: PairwiseRanker
    card: ModelCard

    @property
    def feature_names(self) -> list[str]:
        return self.pipeline.kept_names()

    def check_table(self, table: FeatureTable) -> None:
        """Refuse a table whose columns do not match what this model was fitted on.

        Compared as an *ordered* list, not as a set. A reordering is exactly as
        wrong as a missing column and considerably harder to notice: every weight
        still finds a number to multiply, and the output stays in range.
        """
        expected = self.pipeline.names
        if list(table.names) != list(expected):
            missing = [n for n in expected if n not in table.names]
            extra = [n for n in table.names if n not in expected]
            detail = []
            if missing:
                detail.append(f"missing {missing[:6]}")
            if extra:
                detail.append(f"unexpected {extra[:6]}")
            if not detail:
                detail.append("same columns in a different order")
            raise ModelMismatch(
                f"model {self.card.model_version} was fitted on {len(expected)} feature "
                f"columns and is being asked to score {len(table.names)}: "
                + "; ".join(detail)
                + ". Retrain with scripts/train_predictor.py --save, or score with the "
                "heuristic baseline instead of a model that no longer matches."
            )

    def score_items(self, table: FeatureTable, item_ids: list[str]) -> dict[str, float]:
        self.check_table(table)
        x = self.pipeline.transform(table, item_ids)
        return self.ranker.score_items(x, item_ids)


@dataclass(frozen=True)
class ServedScores:
    """Scores for one candidate set, with the relative position made explicit."""

    raw: dict[str, float]
    card: ModelCard

    @property
    def order(self) -> list[str]:
        """Item ids, best first. Ties broken by id so the order is deterministic."""
        return sorted(self.raw, key=lambda i: (-self.raw[i], i))

    def relative(self) -> dict[str, float]:
        """Scores mapped to 0-1 *within this set*.

        Explicitly a within-set position, not a calibrated score. The pairwise
        objective fixes no origin and no unit, so 0.0 means "worst of these three"
        and nothing more; the same candidate in a different set would map
        differently. A single candidate maps to 0.5 rather than to either
        extreme — one item carries no ranking information at all.
        """
        if not self.raw:
            return {}
        values = np.array(list(self.raw.values()), dtype=float)
        lo, hi = float(values.min()), float(values.max())
        if hi - lo < 1e-12:
            return dict.fromkeys(self.raw, 0.5)
        return {k: (v - lo) / (hi - lo) for k, v in self.raw.items()}


# --- Persistence -------------------------------------------------------------


def _pca_keys(name: str) -> tuple[str, str, str]:
    return f"pca.{name}.mean", f"pca.{name}.components", f"pca.{name}.explained"


def save_model(model: ServedModel, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
    """Write transform, weights and card to a single npz.

    One file rather than three, because a transform and the weights fitted on top
    of it are only meaningful together — and two files can be copied
    independently, which is how a model comes to be served with somebody else's
    standardiser.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pipeline, ranker = model.pipeline, model.ranker
    arrays: dict[str, np.ndarray] = {
        "std.mean": pipeline.standardiser.mean,
        "std.std": pipeline.standardiser.std,
        "std.keep": pipeline.standardiser.keep,
        "std.is_indicator": pipeline.standardiser.is_indicator,
        "head.w1": ranker.w1,
    }
    if ranker.b1 is not None:
        arrays["head.b1"] = ranker.b1
    if ranker.w2 is not None:
        arrays["head.w2"] = ranker.w2
    for block, pca in pipeline.block_pcas.items():
        mean_key, comp_key, exp_key = _pca_keys(block)
        arrays[mean_key] = pca.mean
        arrays[comp_key] = pca.components
        arrays[exp_key] = pca.explained

    meta = {
        "version": MODEL_FORMAT_VERSION,
        "card": model.card.to_dict(),
        "pipeline": {
            "embedding_dim": pipeline.embedding_dim,
            "names": list(pipeline.names),
            "groups": [g.value for g in pipeline.groups],
            "is_real": bool(pipeline.is_real),
            "dropped_names": list(pipeline.standardiser.dropped_names),
            "blocks": sorted(pipeline.block_pcas),
        },
        "head": {
            "names": list(ranker.names),
            "hidden": ranker.config.hidden,
            "l2": ranker.config.l2,
            "lr": ranker.config.lr,
            "max_steps": ranker.config.max_steps,
            "patience": ranker.config.patience,
            "seed": ranker.config.seed,
        },
    }
    # JSON inside the npz as a uint8 array, matching how `adml.embeddings` carries
    # its spec: npz stores arrays, and a str_ array's dtype depends on the platform.
    arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)

    np.savez_compressed(path, **arrays)
    return path


def load_model(path: Path | str = DEFAULT_MODEL_PATH) -> ServedModel:
    """Read back a model saved by :func:`save_model`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no trained ranker at {path}. Train one with "
            "`.venv/bin/python scripts/train_predictor.py --save`, or let the pipeline "
            "fall back to the heuristic baseline."
        )

    with np.load(path, allow_pickle=False) as data:
        if "__meta__" not in data:
            raise ModelMismatch(
                f"{path} has no __meta__ entry, so its feature layout is unknown. "
                "It was not written by adml.serving.save_model."
            )
        meta = json.loads(bytes(data["__meta__"]).decode("utf-8"))
        version = meta.get("version")
        if version != MODEL_FORMAT_VERSION:
            raise ModelMismatch(
                f"{path} is model format version {version!r}, this code reads "
                f"{MODEL_FORMAT_VERSION}. Retrain to regenerate it."
            )

        spec = meta["pipeline"]
        standardiser = Standardiser(
            mean=data["std.mean"],
            std=data["std.std"],
            keep=data["std.keep"].astype(bool),
            is_indicator=data["std.is_indicator"].astype(bool),
            dropped_names=list(spec.get("dropped_names", [])),
        )
        pcas: dict[str, BlockPCA] = {}
        for block in spec.get("blocks", []):
            mean_key, comp_key, exp_key = _pca_keys(block)
            pcas[block] = BlockPCA(
                mean=data[mean_key], components=data[comp_key], explained=data[exp_key]
            )
        pipeline = FeaturePipeline(
            embedding_dim=int(spec["embedding_dim"]),
            standardiser=standardiser,
            block_pcas=pcas,
            names=list(spec["names"]),
            groups=[FeatureGroup(g) for g in spec["groups"]],
            is_real=bool(spec["is_real"]),
        )

        head = meta["head"]
        ranker = PairwiseRanker(
            config=TrainConfig(
                hidden=int(head.get("hidden", 0)),
                l2=float(head["l2"]),
                lr=float(head.get("lr", 0.05)),
                max_steps=int(head.get("max_steps", 0)),
                patience=int(head.get("patience", 0)),
                seed=int(head.get("seed", 0)),
            ),
            w1=data["head.w1"],
            b1=data["head.b1"] if "head.b1" in data else None,
            w2=data["head.w2"] if "head.w2" in data else None,
            names=list(head.get("names", [])),
        )
        card = ModelCard.from_dict(meta["card"])

    return ServedModel(pipeline=pipeline, ranker=ranker, card=card)


def build_card(
    *,
    pipeline: FeaturePipeline,
    ranker: PairwiseRanker,
    n_train_observations: int,
    holdout_accuracy: float = float("nan"),
    ceiling_accuracy: float = float("nan"),
    holdout_ci_low: float = float("nan"),
    embedding_specs: dict[str, EmbeddingSpec] | None = None,
    notes: str = "",
) -> ModelCard:
    """Assemble the provenance card from the objects that were actually fitted.

    ``is_real_features`` is taken from the pipeline rather than passed in, so a
    stand-in cannot be relabelled as real by a caller that forgot.
    """
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    kind = "linear" if ranker.is_linear else f"mlp{ranker.config.hidden}"
    # Taken from the pipeline and the specs, never from a caller's argument: a
    # stand-in must not be relabelled real by code that forgot to pass the flag.
    real = pipeline.is_real and all(spec.is_real for spec in (embedding_specs or {}).values())
    return ModelCard(
        model_version=f"pairwise-{kind}-{stamp[:10].replace('-', '')}",
        trained_at=stamp,
        n_train_observations=n_train_observations,
        n_features=pipeline.n_features,
        n_parameters=ranker.n_parameters,
        holdout_accuracy=holdout_accuracy,
        ceiling_accuracy=ceiling_accuracy,
        holdout_ci_low=holdout_ci_low,
        is_real_features=real,
        embedding_sources={name: spec.source for name, spec in (embedding_specs or {}).items()},
        notes=notes,
    )


# --- Scoring one candidate set ----------------------------------------------


def rank_within_set(model: ServedModel, table: FeatureTable, item_ids: list[str]) -> ServedScores:
    """Score and order one job's candidates.

    The only operation the product needs from a trained ranker, and the only one
    the pairwise objective supports without further assumptions.
    """
    if not item_ids:
        return ServedScores(raw={}, card=model.card)
    return ServedScores(raw=model.score_items(table, item_ids), card=model.card)


def try_load(path: Path | str = DEFAULT_MODEL_PATH) -> ServedModel | None:
    """Load a model if one is there, otherwise ``None``.

    The pipeline's entry point: no trained model is the *normal* state early on,
    and it should fall back to the documented heuristic rather than fail a job. A
    model that exists but is broken still raises — that is a misconfiguration, not
    an absence, and silently serving the baseline instead would hide it.
    """
    path = Path(path)
    if not path.is_file():
        return None
    return load_model(path)
