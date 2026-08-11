"""Feature vectors: named, grouped, and sized to the labels that exist.

Three jobs, each of which is a place this kind of code usually goes wrong.

**Naming and grouping.**  Every column carries a name and a :class:`FeatureGroup`.
The ablation table is the report's main evidence about *which* signals matter, and
it is only trustworthy if a group can be removed exactly — dropping "the
composition features" has to mean dropping precisely those columns, not whichever
ones a stale index list happens to point at.

**Fitting only on training data.**  Standardisation means and PCA components are
statistics of the data, so fitting them on the whole corpus leaks test-set
information into training.  The leak is small compared with a bad train/test split
but it is free to avoid: :class:`FeaturePipeline` fits on the training rows of one
fold and transforms everything with those parameters.

**The parameter budget.**  This is the constraint that decides the model, and it
is easy to ignore until the numbers come out too good.  SigLIP and DINOv2 are 768
dimensions each; the corpus will carry on the order of a thousand pairwise
training observations.  A linear head on the raw concatenation has more parameters
than it has labels and will fit the training comparisons perfectly while learning
nothing.  :func:`parameter_budget` states what the label count supports, and the
pipeline reduces each embedding block to fit inside it.

A note on degenerate columns.  The corpus is generated on a single platform
deliberately (mixing aspect ratios would make a comparison partly about framing
shape), so ``platform`` is *constant* across the corpus.  A constant column has
zero variance: standardising it divides by zero, and a model fitted on it is
fitting noise into an intercept.  :class:`Standardiser` drops such columns and
names them, because "this feature was constant" belongs in the write-up — it is
the honest reason a whole feature group can show no contribution.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from adschema import CameraAngle, Composition, Lighting, MotionIntent, Platform, Vertical
from adschema.annotation import CorpusItem

from . import features as F
from .embeddings import EmbeddingSet


class FeatureGroup(str, Enum):
    """Ablation units.  One row of the ablation table per group."""

    #: Exposure, contrast, saturation — cheap, real, and computed from pixels.
    PHOTOMETRIC = "photometric"
    #: Framing: where the salient mass sits and how big it is.
    COMPOSITION = "composition"
    #: Attention proxy: saliency concentration and safe-area compliance.
    SALIENCE = "salience"
    #: Brand-palette adherence.
    PALETTE = "palette"
    #: The design coordinate the sampler chose, plus vertical and platform.
    #: Ablating this group answers "does the creative decision matter at all,
    #: independent of how the render turned out?"
    CONTEXT = "context"
    #: Reduced frozen-encoder blocks (SigLIP, DINOv2).  Needs Colab.
    EMBEDDING = "embedding"
    #: LAION aesthetic score.  Needs Colab.
    AESTHETIC = "aesthetic"
    #: Product-DINO and face-ArcFace similarity against the references.
    IDENTITY = "identity"
    #: Video-only: motion energy, temporal consistency, hook strength.
    MOTION = "motion"


#: Column name -> group.  Kept as one table rather than scattered through the
#: builders so that a column can never exist without a group, which is what makes
#: :meth:`FeatureTable.without` exact.
GROUP_OF: dict[str, FeatureGroup] = {}


def _register(group: FeatureGroup, *names: str) -> None:
    for name in names:
        GROUP_OF[name] = group


_register(FeatureGroup.PHOTOMETRIC, "luminance", "contrast", "colorfulness", "sharpness")
_register(FeatureGroup.COMPOSITION, "thirds_alignment", "subject_scale", "border_uniformity")
_register(FeatureGroup.SALIENCE, "focal_concentration", "safe_area_share")
_register(FeatureGroup.PALETTE, "palette_adherence", "palette_delta_e")
_register(FeatureGroup.MOTION, "motion_energy_mean", "motion_energy_std", "motion_energy_peak")
_register(FeatureGroup.MOTION, "motion_trend", "temporal_consistency", "hook_strength")
_register(FeatureGroup.MOTION, "focal_persistence", "has_cut")

#: Categorical axes expanded to one-hot columns.  Every level gets a column
#: including the first: with the intercept-free pairwise loss below there is no
#: reference-level trap, and a dropped level would silently become "the baseline"
#: in a way the ablation table could not show.
_CATEGORICAL: dict[str, type[Enum]] = {
    "angle": CameraAngle,
    "lighting": Lighting,
    "composition_axis": Composition,
    "motion_axis": MotionIntent,
    "vertical": Vertical,
    "platform": Platform,
}
for _axis, _enum in _CATEGORICAL.items():
    _register(FeatureGroup.CONTEXT, *[f"{_axis}={m.value}" for m in _enum])


# --- Per-item feature construction ------------------------------------------


def image_features(
    item: CorpusItem, data: bytes, palette_hex: Sequence[str] = ()
) -> dict[str, float]:
    """The numpy-computable features for one image, as ``name -> value``.

    Everything here runs on the laptop.  The embedding and aesthetic groups are
    added separately from a Colab npz, so this function's output is complete and
    real on its own, and a run with no embeddings available is a smaller
    experiment rather than a broken one.
    """
    rgb = F.load_image(data)
    sal = F.saliency_map(rgb)
    safe = item.platform.safe_area
    palette_score, delta_e = F.palette_adherence(rgb, list(palette_hex))

    return {
        "luminance": F.mean_luminance(rgb),
        "contrast": F.rms_contrast(rgb),
        "colorfulness": F.colorfulness(rgb),
        "sharpness": F.sharpness(rgb),
        "thirds_alignment": F.thirds_alignment(sal),
        "subject_scale": F.subject_scale(sal),
        "border_uniformity": F.border_uniformity(rgb),
        "focal_concentration": F.focal_concentration(sal),
        "safe_area_share": F.region_saliency_share(sal, safe.top, safe.bottom),
        "palette_adherence": palette_score,
        "palette_delta_e": delta_e,
    }


def video_features(
    item: CorpusItem, data: bytes, palette_hex: Sequence[str] = ()
) -> dict[str, float]:
    """Motion features, plus the still features of the middle frame.

    The middle frame rather than the first: the first frame is the generated still
    that the image stage already scored, so measuring it again would make the two
    stages correlated by construction and flatter the headline agreement number.

    Two things are deliberately *not* returned as features, even though
    :class:`adml.video.ClipFeatures` measures both.

    ``seam_consistency`` is NaN for any clip without a cut, which is every natively
    generated one.  A column that is missing for most rows cannot be learned from,
    and filling it with 1.0 would assert a perfect seam where there is no seam at
    all.  It stays a diagnostic.

    ``duration_seconds`` would be a **tier label in disguise**.  The premium
    provider delivers 10 s natively and the research tier reaches the window by
    chaining two ~5 s clips, so in this corpus duration separates the tiers almost
    perfectly.  A ranker given that column could score "premium" rather than
    "good", and the tier comparison the report wants to make would be circular.
    """
    from . import video as V

    clip, motion = V.measure(data)
    if clip.n_sampled < 2:
        # A still is not a video. Returning partial features would put a row with
        # no temporal information into the motion ablation.
        return {}

    mid = clip.frames[len(clip.frames) // 2]
    sal = F.saliency_map(mid)
    safe = item.platform.safe_area
    palette_score, delta_e = F.palette_adherence(mid, list(palette_hex))

    return {
        "luminance": F.mean_luminance(mid),
        "contrast": F.rms_contrast(mid),
        "colorfulness": F.colorfulness(mid),
        "sharpness": F.sharpness(mid),
        "thirds_alignment": F.thirds_alignment(sal),
        "subject_scale": F.subject_scale(sal),
        "border_uniformity": F.border_uniformity(mid),
        "focal_concentration": F.focal_concentration(sal),
        "safe_area_share": F.region_saliency_share(sal, safe.top, safe.bottom),
        # Measured on the middle frame, by the same call :func:`image_features`
        # makes. An earlier version hardcoded 1.0/0.0 here, which happened to agree
        # with the no-palette default but would have kept disagreeing once a
        # palette was threaded through — the two stages would then have been
        # scoring adherence against different things while reporting one column.
        "palette_adherence": palette_score,
        "palette_delta_e": delta_e,
        "motion_energy_mean": motion.motion_energy_mean,
        "motion_energy_std": motion.motion_energy_std,
        "motion_energy_peak": motion.motion_energy_peak,
        "motion_trend": motion.motion_trend,
        "temporal_consistency": motion.temporal_consistency,
        "hook_strength": motion.hook_strength,
        "focal_persistence": motion.focal_persistence,
        "has_cut": float(motion.has_seam),
    }


def context_features(item: CorpusItem) -> dict[str, float]:
    """One-hot design coordinate and provenance."""
    values = {
        "angle": item.angle.value if item.angle else None,
        "lighting": item.lighting.value if item.lighting else None,
        "composition_axis": item.composition.value if item.composition else None,
        "motion_axis": item.motion.value if item.motion else None,
        "vertical": item.vertical.value,
        "platform": item.platform.value,
    }
    out: dict[str, float] = {}
    for axis, enum in _CATEGORICAL.items():
        chosen = values[axis]
        for member in enum:
            out[f"{axis}={member.value}"] = 1.0 if member.value == chosen else 0.0
    return out


# --- The table ---------------------------------------------------------------


@dataclass
class FeatureTable:
    """A dense matrix with a name and a group for every column."""

    item_ids: list[str]
    names: list[str]
    matrix: np.ndarray
    #: Group per column, parallel to ``names``.  Embedding columns get theirs from
    #: the block that produced them rather than from :data:`GROUP_OF`.
    groups: list[FeatureGroup]
    #: False as soon as any column came from a stand-in rather than a real
    #: encoder.  Propagates into the evaluation output.
    is_real: bool = True

    def __post_init__(self) -> None:
        n, d = self.matrix.shape
        if n != len(self.item_ids):
            raise ValueError(f"{n} rows for {len(self.item_ids)} items")
        if d != len(self.names) or d != len(self.groups):
            raise ValueError(f"{d} columns for {len(self.names)} names / {len(self.groups)} groups")

    @property
    def index(self) -> dict[str, int]:
        return {item: i for i, item in enumerate(self.item_ids)}

    def present_groups(self) -> list[FeatureGroup]:
        seen: list[FeatureGroup] = []
        for group in self.groups:
            if group not in seen:
                seen.append(group)
        return seen

    def group_sizes(self) -> dict[FeatureGroup, int]:
        sizes: dict[FeatureGroup, int] = {}
        for group in self.groups:
            sizes[group] = sizes.get(group, 0) + 1
        return sizes

    def _mask(self, keep: Iterable[FeatureGroup]) -> np.ndarray:
        wanted = set(keep)
        return np.array([g in wanted for g in self.groups], dtype=bool)

    def select(self, keep: Iterable[FeatureGroup]) -> FeatureTable:
        mask = self._mask(keep)
        return FeatureTable(
            item_ids=list(self.item_ids),
            names=[n for n, m in zip(self.names, mask, strict=True) if m],
            matrix=self.matrix[:, mask],
            groups=[g for g, m in zip(self.groups, mask, strict=True) if m],
            is_real=self.is_real,
        )

    def without(self, drop: FeatureGroup) -> FeatureTable:
        """The ablation operation: everything except one group."""
        return self.select([g for g in self.present_groups() if g is not drop])

    def rows(self, item_ids: Sequence[str]) -> np.ndarray:
        idx = self.index
        missing = [i for i in item_ids if i not in idx]
        if missing:
            raise KeyError(f"{len(missing)} items not in the table, e.g. {missing[:3]}")
        return self.matrix[[idx[i] for i in item_ids], :]

    def summary(self) -> str:
        sizes = ", ".join(f"{g.value} {n}" for g, n in self.group_sizes().items())
        flag = "" if self.is_real else "  [CONTAINS STAND-IN FEATURES — not a result]"
        return f"{len(self.item_ids)} items x {self.matrix.shape[1]} columns ({sizes}){flag}"


def build_table(
    items: Sequence[CorpusItem],
    measured: dict[str, dict[str, float]],
    *,
    embeddings: dict[str, EmbeddingSet] | None = None,
    include_context: bool = True,
) -> FeatureTable:
    """Assemble the table from measured features plus optional embedding blocks.

    ``measured`` maps item id to the output of :func:`image_features` or
    :func:`video_features`.  Items missing from it are dropped: a feature row of
    zeros would read as a perfectly average image (see
    :mod:`adml.embeddings`), and there is no honest fill value.

    Items missing from *any* supplied embedding block are dropped too, for the
    same reason.  The count that survives is what the model actually trains on, so
    the caller should look at it.
    """
    embeddings = embeddings or {}
    usable = [
        it
        for it in items
        if not it.is_decoy
        and it.item_id in measured
        and all(it.item_id in block for block in embeddings.values())
    ]
    if not usable:
        return FeatureTable([], [], np.zeros((0, 0)), [])

    scalar_names = sorted({k for it in usable for k in measured[it.item_id]})
    names = list(scalar_names)
    groups = [GROUP_OF[n] for n in scalar_names]
    columns = [
        np.array([measured[it.item_id].get(n, 0.0) for it in usable], dtype=np.float64)
        for n in scalar_names
    ]

    if include_context:
        ctx = [context_features(it) for it in usable]
        ctx_names = sorted(ctx[0])
        names += ctx_names
        groups += [GROUP_OF[n] for n in ctx_names]
        columns += [np.array([row[n] for row in ctx], dtype=np.float64) for n in ctx_names]

    is_real = True
    ids = [it.item_id for it in usable]
    for block_name in sorted(embeddings):
        block = embeddings[block_name].l2_normalised()
        vectors, _ = block.matrix(ids)
        group = (
            FeatureGroup.AESTHETIC
            if block_name == "aesthetic"
            else FeatureGroup.IDENTITY
            if block_name in ("arcface", "product_dino")
            else FeatureGroup.EMBEDDING
        )
        for j in range(block.spec.dim):
            names.append(f"{block_name}[{j}]")
            groups.append(group)
            columns.append(vectors[:, j])
        is_real = is_real and block.spec.is_real

    return FeatureTable(
        item_ids=ids,
        names=names,
        matrix=np.column_stack(columns),
        groups=groups,
        is_real=is_real,
    )


# --- Fitted transforms ------------------------------------------------------


#: Standard deviation below which a column carries no usable variation.  Not
#: exactly zero: a one-hot that is set for a single item out of three hundred has
#: nonzero variance and is still unfittable.
MIN_COLUMN_STD = 1e-6


@dataclass
class Standardiser:
    """Scaling fitted on training rows only, dropping degenerate columns.

    **Indicator columns are left at 0/1 rather than z-scored**, and that is not a
    cosmetic choice.  Dividing an indicator by its own standard deviation scales it
    by how *rare* the level is: measured on this corpus, a one-hot set for 3 of 87
    items reaches ``z = +5.3`` while one set for 44 of 87 reaches ``z = +1.0``.  The
    rare level therefore arrives with five times the leverage of a continuous
    feature and absorbs five times as much of the L2 penalty, purely because few
    items have it.  Rarity is not importance, and a design axis the sampler rarely
    picked should not dominate the fit for that reason.

    Centring is skipped for the same columns and would not matter anyway: the
    pairwise loss sees only score *differences*, in which any per-column constant
    cancels exactly.
    """

    mean: np.ndarray
    std: np.ndarray
    keep: np.ndarray
    is_indicator: np.ndarray
    dropped_names: list[str] = field(default_factory=list)

    @staticmethod
    def fit(matrix: np.ndarray, names: Sequence[str]) -> Standardiser:
        std = matrix.std(axis=0)
        keep = std > MIN_COLUMN_STD
        indicator = np.all((matrix == 0.0) | (matrix == 1.0), axis=0)
        return Standardiser(
            mean=np.where(indicator, 0.0, matrix.mean(axis=0)),
            std=np.where(keep & ~indicator, std, 1.0),
            keep=keep,
            is_indicator=indicator,
            dropped_names=[n for n, k in zip(names, keep, strict=True) if not k],
        )

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return ((matrix - self.mean) / self.std)[:, self.keep]


@dataclass
class BlockPCA:
    """Dimension reduction for one embedding block, fitted on training rows.

    Unsupervised, so it can be fitted without labels — but still fitted inside the
    fold, because the *directions of variation of the test images* are test-set
    information.
    """

    mean: np.ndarray
    components: np.ndarray  # (k, d)
    explained: np.ndarray

    @staticmethod
    def fit(matrix: np.ndarray, k: int) -> BlockPCA:
        k = max(1, min(k, *matrix.shape))
        mean = matrix.mean(axis=0)
        centred = matrix - mean
        # SVD rather than an eigendecomposition of the covariance: d is 768 and n
        # is a few hundred, so the covariance matrix would be larger than the data
        # and rank-deficient by construction.
        _, singular, vt = np.linalg.svd(centred, full_matrices=False)
        total = float((singular**2).sum())
        return BlockPCA(
            mean=mean,
            components=vt[:k],
            explained=(singular[:k] ** 2) / total if total > 0 else np.zeros(k),
        )

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.mean) @ self.components.T


#: Training observations wanted per free parameter.  10 is the usual floor quoted
#: for logistic regression; 15 is used here because the labels are noisy human
#: preferences rather than clean outcomes, so each observation carries less
#: information than the rule of thumb assumes.
OBSERVATIONS_PER_PARAMETER = 15


def parameter_budget(
    n_observations: int, *, per_parameter: int = OBSERVATIONS_PER_PARAMETER
) -> int:
    """How many free parameters this many pairwise observations supports.

    The number that should decide the model's size, rather than the number of
    dimensions the encoders happen to emit.  With ~1,200 training comparisons the
    budget is ~80 parameters — which a 768-dimensional block blows through forty
    times over, and is why the embedding blocks are reduced rather than used raw.
    """
    return max(1, n_observations // max(1, per_parameter))


@dataclass
class FeaturePipeline:
    """Fitted feature transform: per-block PCA, then standardisation.

    Fit on one fold's training rows, then applied unchanged to that fold's test
    rows.  Holding the fitted object explicitly (rather than transforming in
    place) is what makes it possible to assert in a test that no test row
    influenced any fitted parameter.
    """

    embedding_dim: int
    standardiser: Standardiser
    block_pcas: dict[str, BlockPCA]
    names: list[str]
    groups: list[FeatureGroup]
    is_real: bool

    @property
    def n_features(self) -> int:
        return int(self.standardiser.keep.sum())

    @staticmethod
    def fit(
        table: FeatureTable,
        train_items: Sequence[str],
        *,
        embedding_dim: int = 8,
    ) -> FeaturePipeline:
        rows = table.rows(train_items)
        blocks = _embedding_blocks(table)

        pcas: dict[str, BlockPCA] = {}
        names: list[str] = []
        groups: list[FeatureGroup] = []
        pieces: list[np.ndarray] = []

        scalar = np.array([g is not FeatureGroup.EMBEDDING for g in table.groups], dtype=bool)
        if scalar.any():
            pieces.append(rows[:, scalar])
            names += [n for n, m in zip(table.names, scalar, strict=True) if m]
            groups += [g for g, m in zip(table.groups, scalar, strict=True) if m]

        for block_name, mask in blocks.items():
            pca = BlockPCA.fit(rows[:, mask], embedding_dim)
            pcas[block_name] = pca
            pieces.append(pca.transform(rows[:, mask]))
            k = pca.components.shape[0]
            names += [f"{block_name}.pc{j}" for j in range(k)]
            groups += [FeatureGroup.EMBEDDING] * k

        reduced = np.column_stack(pieces) if pieces else np.zeros((len(train_items), 0))
        std = Standardiser.fit(reduced, names)
        return FeaturePipeline(
            embedding_dim=embedding_dim,
            standardiser=std,
            block_pcas=pcas,
            names=names,
            groups=groups,
            is_real=table.is_real,
        )

    def transform(self, table: FeatureTable, item_ids: Sequence[str]) -> np.ndarray:
        rows = table.rows(item_ids)
        blocks = _embedding_blocks(table)
        pieces: list[np.ndarray] = []
        scalar = np.array([g is not FeatureGroup.EMBEDDING for g in table.groups], dtype=bool)
        if scalar.any():
            pieces.append(rows[:, scalar])
        for block_name, mask in blocks.items():
            pieces.append(self.block_pcas[block_name].transform(rows[:, mask]))
        reduced = np.column_stack(pieces) if pieces else np.zeros((len(item_ids), 0))
        return self.standardiser.transform(reduced)

    def kept_names(self) -> list[str]:
        return [n for n, k in zip(self.names, self.standardiser.keep, strict=True) if k]

    def kept_groups(self) -> list[FeatureGroup]:
        return [g for g, k in zip(self.groups, self.standardiser.keep, strict=True) if k]


def _embedding_blocks(table: FeatureTable) -> dict[str, np.ndarray]:
    """Column masks for each embedding block, keyed by block name.

    Blocks are recovered from the ``name[j]`` column naming rather than tracked
    separately, so a table round-tripped through :meth:`FeatureTable.select` still
    knows where its blocks are.
    """
    blocks: dict[str, list[int]] = {}
    for j, (name, group) in enumerate(zip(table.names, table.groups, strict=True)):
        if group is not FeatureGroup.EMBEDDING or "[" not in name:
            continue
        blocks.setdefault(name.split("[", 1)[0], []).append(j)
    return {k: np.isin(np.arange(len(table.names)), v) for k, v in sorted(blocks.items())}
