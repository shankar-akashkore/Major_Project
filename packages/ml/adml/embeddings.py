"""Frozen-encoder embeddings, kept on the far side of a file boundary.

SigLIP, DINOv2, ArcFace and the LAION aesthetic head all need torch, and torch
does not belong on an 8 GB laptop with 19 GB of disk free.  Rather than make the
whole project depend on an environment it cannot have, the split is a file:

    Colab (GPU, torch)                     laptop / API (numpy only)
    --------------------------             --------------------------
    load corpus images         ->
    run the frozen encoders    ->
    write embeddings.npz       ->  load_npz()  ->  featureset  ->  predictor

Nothing in this module imports torch.  The Colab notebook writes the npz; this
module reads it.  That has a benefit beyond disk space: the *trainable* head is
numpy on both sides, so the model reported in the evaluation is byte-for-byte the
model the API serves.  There is no train/serve reimplementation gap to be wrong
about.

Two failure modes this module exists to prevent.

**Silent zero-fill.**  An item with no embedding must not be given a zero vector.
After standardisation a zero vector *is the mean* — indistinguishable from a
perfectly average image — so a missing encoder run would show up as a mediocre
candidate rather than as missing data.  :meth:`EmbeddingSet.matrix` returns
coverage and the caller drops uncovered items explicitly.

**A stand-in reaching a reported result.**  :func:`hash_embeddings` produces
deterministic pseudo-embeddings so the training and evaluation path can be
exercised end to end with no downloads.  They carry no visual information
whatsoever.  ``EmbeddingSpec.is_real`` is False for them and propagates through
the feature matrix into the evaluation output, so a table built on stand-ins says
so on its face instead of looking like a result.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Spec entry inside the npz.  Double-underscored so it cannot collide with an
#: item id.
_SPEC_KEY = "__spec__"

#: Model ids the Colab notebook is expected to produce.  Recorded here so the
#: reader can say what is missing rather than only what is present.
EXPECTED = {
    "siglip": "google/siglip-base-patch16-224",
    "dinov2": "facebook/dinov2-base",
    "aesthetic": "laion/aesthetic-predictor-v2-linear",
    "arcface": "insightface/buffalo_l",
}


@dataclass(frozen=True)
class EmbeddingSpec:
    """What produced a block of vectors, and whether it can be believed."""

    name: str
    dim: int
    source: str
    #: False for anything not computed by a real encoder.  Propagates into the
    #: evaluation so a stand-in cannot masquerade as a measurement.
    is_real: bool = True
    #: True when vectors are already unit-norm, so callers do not normalise twice.
    l2_normalised: bool = False

    def to_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "dim": self.dim,
                "source": self.source,
                "is_real": self.is_real,
                "l2_normalised": self.l2_normalised,
            }
        )

    @staticmethod
    def from_json(raw: str) -> EmbeddingSpec:
        data = json.loads(raw)
        return EmbeddingSpec(
            name=data["name"],
            dim=int(data["dim"]),
            source=data["source"],
            is_real=bool(data.get("is_real", True)),
            l2_normalised=bool(data.get("l2_normalised", False)),
        )


@dataclass
class Coverage:
    """Which items an embedding block actually covers."""

    present: list[str]
    missing: list[str]

    @property
    def fraction(self) -> float:
        total = len(self.present) + len(self.missing)
        return len(self.present) / total if total else float("nan")

    def summary(self) -> str:
        if not self.missing:
            return f"{len(self.present)} items, complete"
        shown = ", ".join(self.missing[:3])
        more = f" (+{len(self.missing) - 3} more)" if len(self.missing) > 3 else ""
        return (
            f"{len(self.present)}/{len(self.present) + len(self.missing)} items "
            f"({self.fraction:.0%}) — missing: {shown}{more}"
        )


class EmbeddingSet:
    """One named block of vectors, keyed by item id."""

    def __init__(self, spec: EmbeddingSpec, vectors: dict[str, np.ndarray]) -> None:
        for item, vec in vectors.items():
            if vec.shape != (spec.dim,):
                raise ValueError(
                    f"{spec.name}: {item} has shape {vec.shape}, expected ({spec.dim},)"
                )
        self.spec = spec
        self.vectors = vectors

    def __len__(self) -> int:
        return len(self.vectors)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self.vectors

    def coverage(self, item_ids: Iterable[str]) -> Coverage:
        wanted = list(item_ids)
        return Coverage(
            present=[i for i in wanted if i in self.vectors],
            missing=[i for i in wanted if i not in self.vectors],
        )

    def matrix(self, item_ids: Sequence[str]) -> tuple[np.ndarray, Coverage]:
        """Stack the covered items, and say which were left out.

        Deliberately does *not* accept a fill value.  See the module docstring:
        there is no number that honestly stands for "this encoder never ran".
        """
        cov = self.coverage(item_ids)
        if not cov.present:
            return np.zeros((0, self.spec.dim), dtype=np.float64), cov
        rows = np.stack([self.vectors[i] for i in cov.present]).astype(np.float64)
        return rows, cov

    def l2_normalised(self) -> EmbeddingSet:
        """Unit-norm copy.

        Cosine geometry is what these encoders are trained for, and it also stops
        one block dominating a concatenation purely through vector magnitude —
        DINOv2 activations are numerically much larger than a scalar aesthetic
        score, and a linear head has no way to know that is not signal.
        """
        if self.spec.l2_normalised:
            return self
        out = {}
        for item, vec in self.vectors.items():
            norm = float(np.linalg.norm(vec))
            out[item] = vec / norm if norm > 1e-12 else vec
        spec = EmbeddingSpec(
            name=self.spec.name,
            dim=self.spec.dim,
            source=self.spec.source,
            is_real=self.spec.is_real,
            l2_normalised=True,
        )
        return EmbeddingSet(spec, out)

    def save_npz(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            item: vec.astype(np.float32) for item, vec in self.vectors.items()
        }
        # float32 on disk: these are 768-dim blocks over hundreds of items and the
        # encoders themselves emit float32, so float64 storage would double the
        # file for digits the model never produced.
        payload[_SPEC_KEY] = np.frombuffer(self.spec.to_json().encode(), dtype=np.uint8)
        np.savez_compressed(target, **payload)
        return target

    @staticmethod
    def load_npz(path: str | Path) -> EmbeddingSet:
        with np.load(Path(path), allow_pickle=False) as data:
            if _SPEC_KEY not in data:
                raise ValueError(
                    f"{path} has no {_SPEC_KEY} entry — it was not written by "
                    "EmbeddingSet.save_npz, so its provenance is unknown"
                )
            spec = EmbeddingSpec.from_json(bytes(data[_SPEC_KEY]).decode())
            vectors = {
                key: np.asarray(data[key], dtype=np.float64)
                for key in data.files
                if key != _SPEC_KEY
            }
        return EmbeddingSet(spec, vectors)


def load_directory(directory: str | Path) -> dict[str, EmbeddingSet]:
    """Load every ``*.npz`` in a directory, keyed by block name.

    The Colab notebook writes one file per encoder so a single failed encoder does
    not invalidate the others, and so a re-run of one encoder does not have to
    recompute the rest.
    """
    out: dict[str, EmbeddingSet] = {}
    for path in sorted(Path(directory).glob("*.npz")):
        block = EmbeddingSet.load_npz(path)
        out[block.spec.name] = block
    return out


def missing_blocks(loaded: dict[str, EmbeddingSet]) -> list[str]:
    """Expected encoder blocks that are absent.

    Reported rather than raised: the numpy feature groups are real without any of
    these, so the pipeline runs and the ablation table simply shows the embedding
    rows as unavailable.
    """
    return [name for name in sorted(EXPECTED) if name not in loaded]


def hash_embeddings(
    item_ids: Iterable[str], *, dim: int = 32, name: str = "siglip"
) -> EmbeddingSet:
    """Deterministic stand-in vectors carrying no visual information.

    For exercising the training and evaluation path with zero downloads.  Derived
    from the item id alone, so they are stable across runs and reproducible — and
    completely uninformative, which is the point: a model trained on these should
    score at chance, and if it does not, the evaluation is leaking.  That makes
    this a *test* as much as a placeholder.

    ``is_real=False`` travels with the vectors into every downstream result.
    """
    vectors: dict[str, np.ndarray] = {}
    for item in item_ids:
        raw = b""
        counter = 0
        while len(raw) < dim * 8:
            raw += hashlib.sha256(f"{name}:{item}:{counter}".encode()).digest()
            counter += 1
        ints = np.frombuffer(raw[: dim * 8], dtype=np.uint64)
        # Uniform in [-1, 1); no attempt at a Gaussian, since the vectors are not
        # pretending to approximate an encoder's output distribution either.
        vectors[item] = (ints / np.float64(2**64)) * 2.0 - 1.0
    spec = EmbeddingSpec(name=name, dim=dim, source="hash-stand-in", is_real=False)
    return EmbeddingSet(spec, vectors)
