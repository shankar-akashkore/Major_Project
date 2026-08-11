"""The trainable head: Bradley-Terry loss on features, in numpy.

Numpy rather than torch, for a reason beyond the laptop's disk.  The evaluation
runs here and the API serves here, so the model in the ablation table is the same
object the product scores with — there is no second implementation to disagree with
the first.  The frozen encoders stay in Colab and cross the boundary as an npz
(:mod:`adml.embeddings`); the part with learned parameters never leaves this file.

**Why train on comparisons directly — and the measurement that qualifies it.**  The
obvious alternative is two steps: fit Bradley-Terry to get one strength per item,
then regress features onto those strengths.  That is :class:`StrengthRegressor`
below.  The theoretical objection to it is real — a BT strength estimated from
three comparisons and one estimated from twelve arrive as equally confident
numbers, and squared error then works hardest to match whichever happened to be
most extreme, usually the least-observed item.

The objection did not show up in practice, and that is worth recording rather than
hiding.  Over simulated corpora of 25 to 200 sets the two approaches were within a
point of each other, and the difference never resolved except once in the wrong
direction (``docs/prediction-protocol.md``).  The likely reason is the previous
phase: :func:`adml.pairs.design_pairs` gives *every* item the same number of
comparisons by construction, so the variance imbalance the objection depends on
has largely been designed out.

The pairwise objective is still the primary method, for reasons that survive the
measurement: it needs no intermediate fit, it handles ties as observations rather
than as an interpolation problem, and it does not depend on degree balance — which
holds for the annotation corpus by design but will *not* hold for judgements that
accumulate unevenly on live product jobs.  What it is not, on this corpus, is
demonstrably more accurate than the simpler thing.

**Model size is set by the labels, not by the encoders.**  The default is linear.
With ~1,200 training comparisons the parameter budget is ~80
(:func:`adml.featureset.parameter_budget`), and a single 8-unit hidden layer over
30 inputs already needs 250.  ``hidden`` exists so the report can show what
happens when that budget is exceeded — which is worth one row of a table and is
not the headline model.

**Ties are kept.**  A tie enters as two half-weight observations, one in each
direction, matching :func:`adml.ranking.fit_bradley_terry`.  Discarding ties would
throw away the observations where the candidates are genuinely close, which is
precisely where a ranker's behaviour matters.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .ranking import fit_bradley_terry
from .split import Observation

#: Adam's usual constants.  Not tuned: with a few thousand full-batch steps on a
#: convex (linear) objective the optimiser is not where the difficulty lives.
_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8


def _log1p_exp(x: np.ndarray) -> np.ndarray:
    """``log(1 + exp(x))`` without overflowing for large ``x``."""
    return np.logaddexp(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function, evaluated on whichever side does not overflow."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# --- Observation encoding ---------------------------------------------------


@dataclass
class EncodedPairs:
    """Comparisons as row indices into a feature matrix, plus weights."""

    winner: np.ndarray  # int, (n,)
    loser: np.ndarray  # int, (n,)
    weight: np.ndarray  # float, (n,)

    def __len__(self) -> int:
        return len(self.winner)

    @property
    def total_weight(self) -> float:
        return float(self.weight.sum())


def encode(observations: Sequence[Observation], row_of: dict[str, int]) -> EncodedPairs:
    """Map observations onto feature-matrix rows.

    Observations naming an item with no feature row are skipped.  That happens
    legitimately — an item whose embedding block is missing was dropped from the
    table — so it is a count to report, not an error to raise.
    """
    winners: list[int] = []
    losers: list[int] = []
    weights: list[float] = []
    for obs in observations:
        if obs.winner not in row_of or obs.loser not in row_of:
            continue
        a, b = row_of[obs.winner], row_of[obs.loser]
        if obs.is_tie:
            winners += [a, b]
            losers += [b, a]
            weights += [0.5, 0.5]
        else:
            winners.append(a)
            losers.append(b)
            weights.append(1.0)
    return EncodedPairs(
        winner=np.array(winners, dtype=np.intp),
        loser=np.array(losers, dtype=np.intp),
        weight=np.array(weights, dtype=np.float64),
    )


# --- Configuration and history ----------------------------------------------


@dataclass
class TrainConfig:
    """Everything that changes the fitted parameters, in one recordable object."""

    hidden: int = 0
    l2: float = 1e-2
    lr: float = 0.05
    max_steps: int = 3000
    #: Steps without validation improvement before stopping.  Generous, because
    #: the validation set is small enough to be noisy step to step.
    patience: int = 200
    seed: int = 0

    def describe(self) -> str:
        shape = "linear" if self.hidden <= 0 else f"mlp({self.hidden})"
        return f"{shape} l2={self.l2:g} lr={self.lr:g}"


@dataclass
class TrainHistory:
    steps: int
    best_step: int
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_val_loss: float = float("nan")
    stopped_early: bool = False

    def summary(self) -> str:
        stop = "early" if self.stopped_early else "max steps"
        val = "n/a" if math.isnan(self.best_val_loss) else f"{self.best_val_loss:.4f}"
        return (
            f"{self.steps} steps ({stop}), best at {self.best_step}, "
            f"train {self.train_loss[-1]:.4f} val {val}"
        )


# --- The model --------------------------------------------------------------


class PairwiseRanker:
    """A scorer fitted by maximising the Bradley-Terry likelihood of comparisons."""

    def __init__(
        self,
        *,
        config: TrainConfig,
        w1: np.ndarray,
        b1: np.ndarray | None,
        w2: np.ndarray | None,
        names: Sequence[str] = (),
        history: TrainHistory | None = None,
    ) -> None:
        self.config = config
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.names = list(names)
        self.history = history

    # -- shape bookkeeping --

    @property
    def is_linear(self) -> bool:
        return self.w2 is None

    @property
    def n_parameters(self) -> int:
        total = self.w1.size
        if self.b1 is not None:
            total += self.b1.size
        if self.w2 is not None:
            total += self.w2.size
        return int(total)

    # -- forward --

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if self.is_linear:
            return x @ self.w1, None
        hidden = np.tanh(x @ self.w1.T + self.b1)
        return hidden @ self.w2, hidden

    def scores(self, x: np.ndarray) -> np.ndarray:
        """Latent quality on an arbitrary scale.

        Only differences are meaningful: there is no output bias, because a
        constant added to every score cancels in every comparison and so is not
        identifiable from pairwise data.  Any absolute interpretation of these
        numbers would be reading structure into a gauge freedom.
        """
        return self._forward(x)[0]

    def score_items(self, x: np.ndarray, item_ids: Sequence[str]) -> dict[str, float]:
        values = self.scores(x)
        return {item: float(v) for item, v in zip(item_ids, values, strict=True)}

    def coefficients(self) -> dict[str, float]:
        """Per-feature weights, for the linear model only.

        Reported in the write-up as *directions*, not as effect sizes: the inputs
        are z-scored, so the magnitudes are comparable to each other but carry no
        units, and correlated features share credit arbitrarily between them.
        """
        if not self.is_linear:
            raise ValueError("coefficients are only interpretable for the linear model")
        if len(self.names) != len(self.w1):
            return {f"f{i}": float(v) for i, v in enumerate(self.w1)}
        return dict(zip(self.names, (float(v) for v in self.w1), strict=True))

    # -- loss --

    def _loss_and_grad_scores(
        self, scores: np.ndarray, pairs: EncodedPairs
    ) -> tuple[float, np.ndarray]:
        margin = scores[pairs.winner] - scores[pairs.loser]
        weight_sum = pairs.total_weight or 1.0
        loss = float((pairs.weight * _log1p_exp(-margin)).sum() / weight_sum)

        # d/d(margin) of -log sigmoid(margin) is -sigmoid(-margin).
        per_pair = -pairs.weight * _sigmoid(-margin) / weight_sum
        grad = np.zeros_like(scores)
        np.add.at(grad, pairs.winner, per_pair)
        np.add.at(grad, pairs.loser, -per_pair)
        return loss, grad

    def loss(self, x: np.ndarray, pairs: EncodedPairs) -> float:
        """Mean weighted negative log-likelihood, without the penalty term."""
        if len(pairs) == 0:
            return float("nan")
        return self._loss_and_grad_scores(self.scores(x), pairs)[0]

    def accuracy(self, x: np.ndarray, pairs: EncodedPairs) -> float:
        """Weighted fraction of comparisons ordered correctly; ties count half."""
        if len(pairs) == 0:
            return float("nan")
        margin = self.scores(x)[pairs.winner] - self.scores(x)[pairs.loser]
        hits = np.where(margin > 0, 1.0, np.where(margin == 0, 0.5, 0.0))
        return float((pairs.weight * hits).sum() / (pairs.total_weight or 1.0))

    # -- fitting --

    @staticmethod
    def fit(
        x: np.ndarray,
        pairs: EncodedPairs,
        *,
        config: TrainConfig | None = None,
        val_x: np.ndarray | None = None,
        val_pairs: EncodedPairs | None = None,
        names: Sequence[str] = (),
    ) -> PairwiseRanker:
        """Full-batch Adam on the pairwise likelihood.

        Full batch because the whole problem is a few thousand rows of a few dozen
        columns — mini-batching would add a source of run-to-run variation for no
        speed at all, and reproducibility is worth more here than throughput.

        ``val_pairs`` drives early stopping.  It must come from *different
        generation sets* than ``pairs`` (see :func:`adml.split.holdout`); a
        validation split that shares items with training will keep improving long
        after generalisation has stopped, and the stopping rule then does nothing.
        """
        config = config or TrainConfig()
        if len(pairs) == 0:
            raise ValueError("no comparisons to fit")
        d = x.shape[1]
        rng = np.random.default_rng(config.seed)

        if config.hidden <= 0:
            # Zero init is safe and deterministic for the linear model: the
            # objective is convex, so there is no symmetry to break.
            params = [np.zeros(d)]
        else:
            h = config.hidden
            params = [
                rng.normal(0.0, 1.0 / math.sqrt(max(d, 1)), size=(h, d)),
                np.zeros(h),
                rng.normal(0.0, 1.0 / math.sqrt(h), size=h),
            ]

        moments = [np.zeros_like(p) for p in params]
        velocities = [np.zeros_like(p) for p in params]
        best = [p.copy() for p in params]
        best_val = float("inf")
        best_step = 0
        history = TrainHistory(steps=0, best_step=0)

        def model_of(values: list[np.ndarray]) -> PairwiseRanker:
            if config.hidden <= 0:
                return PairwiseRanker(config=config, w1=values[0], b1=None, w2=None, names=names)
            return PairwiseRanker(
                config=config, w1=values[0], b1=values[1], w2=values[2], names=names
            )

        for step in range(1, config.max_steps + 1):
            model = model_of(params)
            scores, hidden = model._forward(x)
            loss, grad_scores = model._loss_and_grad_scores(scores, pairs)

            if config.hidden <= 0:
                grads = [x.T @ grad_scores]
            else:
                grad_w2 = hidden.T @ grad_scores
                grad_pre = (grad_scores[:, None] * params[2][None, :]) * (1.0 - hidden**2)
                grads = [grad_pre.T @ x, grad_pre.sum(axis=0), grad_w2]

            penalty = 0.0
            for i, p in enumerate(params):
                grads[i] = grads[i] + config.l2 * p
                penalty += 0.5 * config.l2 * float((p**2).sum())

            for i, (p, g) in enumerate(zip(params, grads, strict=True)):
                moments[i] = _BETA1 * moments[i] + (1 - _BETA1) * g
                velocities[i] = _BETA2 * velocities[i] + (1 - _BETA2) * g * g
                m_hat = moments[i] / (1 - _BETA1**step)
                v_hat = velocities[i] / (1 - _BETA2**step)
                params[i] = p - config.lr * m_hat / (np.sqrt(v_hat) + _EPS)

            history.steps = step
            history.train_loss.append(loss + penalty)

            if val_pairs is not None and val_x is not None and len(val_pairs) > 0:
                val = model_of(params).loss(val_x, val_pairs)
                history.val_loss.append(val)
                if val < best_val - 1e-9:
                    best_val, best_step = val, step
                    best = [p.copy() for p in params]
                elif step - best_step >= config.patience:
                    history.stopped_early = True
                    break
            else:
                best = [p.copy() for p in params]
                best_step = step

        history.best_step = best_step
        history.best_val_loss = best_val if best_val < float("inf") else float("nan")
        model = model_of(best)
        model.history = history
        return model


#: L2 strengths swept by :func:`fit_with_selection`.  Log-spaced and wide, because
#: with a few hundred effective observations the useful value can be anywhere.
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


@dataclass
class Selection:
    """Which hyper-parameter won on the inner validation split, and by how much."""

    best: TrainConfig
    scores: list[tuple[float, float]]  # (l2, val_loss)

    def summary(self) -> str:
        table = ", ".join(f"{l2:g}:{loss:.4f}" for l2, loss in self.scores)
        return f"chose l2={self.best.l2:g}  [{table}]"


def fit_with_selection(
    x: np.ndarray,
    train_pairs: EncodedPairs,
    inner_folds: Sequence[tuple[EncodedPairs, EncodedPairs]],
    *,
    base: TrainConfig | None = None,
    grid: Sequence[float] = L2_GRID,
    names: Sequence[str] = (),
) -> tuple[PairwiseRanker, Selection]:
    """Choose the penalty by inner cross-validation, then refit on everything.

    ``inner_folds`` is a list of ``(train, validation)`` comparison splits drawn
    from the outer fold's *training* sets only.  Choosing the penalty on the outer
    test fold — even just choosing, not fitting — is a test-set peek, and it is the
    most common way an honest-looking cross-validation still reports an optimistic
    number.

    **Several inner folds rather than one, because one is not enough here.**  A
    single 20% inner split of 23 training sets holds out about five sets, which
    yields on the order of twenty comparisons.  Selecting a hyper-parameter on
    twenty noisy binary observations is worse than not selecting at all: measured on
    this corpus it picked a different penalty on nearly every outer fold, varying
    across four orders of magnitude.  Averaging the validation loss over several
    inner folds uses every training set for validation exactly once and makes the
    choice stable.

    The returned model is refit on the whole training fold at the chosen penalty.
    The inner splits were spent choosing it; once chosen, their sets are ordinary
    training data.  That refit has no validation set and so runs to ``max_steps``,
    which is correct for the linear model — the objective is convex and the penalty
    is doing the regularising — and is one more reason the linear model is the
    headline rather than ``hidden > 0``, where L2 alone has to carry it.
    """
    base = base or TrainConfig()
    results: list[tuple[float, float]] = []
    best_l2 = base.l2
    best_loss = float("inf")

    for l2 in grid:
        config = _with_l2(base, l2)
        losses: list[float] = []
        for inner_train, inner_val in inner_folds:
            if len(inner_train) == 0 or len(inner_val) == 0:
                continue
            model = PairwiseRanker.fit(
                x, inner_train, config=config, val_x=x, val_pairs=inner_val, names=names
            )
            losses.append(model.loss(x, inner_val))
        if not losses:
            continue
        mean_loss = float(np.mean(losses))
        results.append((l2, mean_loss))
        if mean_loss < best_loss:
            best_loss, best_l2 = mean_loss, l2

    final = PairwiseRanker.fit(x, train_pairs, config=_with_l2(base, best_l2), names=names)
    return final, Selection(best=final.config, scores=results)


def _with_l2(base: TrainConfig, l2: float) -> TrainConfig:
    return TrainConfig(
        hidden=base.hidden,
        l2=l2,
        lr=base.lr,
        max_steps=base.max_steps,
        patience=base.patience,
        seed=base.seed,
    )


# --- Baselines ---------------------------------------------------------------
#
# The trained model has to beat all of these, and a couple of them are stronger
# than they look.  Reporting only "beats random" would be no result at all.


def random_scores(item_ids: Sequence[str], *, seed: int = 0) -> dict[str, float]:
    """The floor: 50% pairwise accuracy in expectation."""
    rng = np.random.default_rng(seed)
    return {
        item: float(v) for item, v in zip(item_ids, rng.normal(size=len(item_ids)), strict=True)
    }


def single_feature_scores(
    table_names: Sequence[str],
    x: np.ndarray,
    item_ids: Sequence[str],
    feature: str,
    *,
    sign: float = 1.0,
) -> dict[str, float]:
    """Rank by one standardised feature.

    The honest version of an "aesthetic-only" or "CLIPScore-only" baseline: no
    fitting, one column, and a sign chosen in advance from what the feature means
    rather than from which direction happens to score better on the test fold.
    """
    if feature not in table_names:
        raise KeyError(f"{feature!r} is not a column; have {list(table_names)[:6]}...")
    column = list(table_names).index(feature)
    return {item: float(sign * x[i, column]) for i, item in enumerate(item_ids)}


class StrengthRegressor:
    """Ridge regression of features onto Bradley-Terry strengths.

    The natural-looking two-step alternative, and — measured — a *strong* baseline
    rather than a straw man: it matched the pairwise head on every simulated corpus
    size tried.  Kept and reported for exactly that reason.  See the module
    docstring for why the expected weakness did not materialise.
    """

    def __init__(self, weights: np.ndarray, intercept: float, names: Sequence[str] = ()) -> None:
        self.weights = weights
        self.intercept = intercept
        self.names = list(names)

    @staticmethod
    def fit(
        x: np.ndarray,
        item_ids: Sequence[str],
        observations: Sequence[Observation],
        *,
        l2: float = 1.0,
        names: Sequence[str] = (),
    ) -> StrengthRegressor:
        fit = fit_bradley_terry(
            [(o.winner, o.loser) for o in observations if not o.is_tie],
            ties=[(o.winner, o.loser) for o in observations if o.is_tie],
            items=item_ids,
        )
        y = np.array([fit.strengths.get(i, 0.0) for i in item_ids], dtype=np.float64)
        centre = float(y.mean())
        d = x.shape[1]
        gram = x.T @ x + l2 * np.eye(d)
        weights = np.linalg.solve(gram, x.T @ (y - centre))
        return StrengthRegressor(weights=weights, intercept=centre, names=names)

    def score_items(self, x: np.ndarray, item_ids: Sequence[str]) -> dict[str, float]:
        values = x @ self.weights + self.intercept
        return {item: float(v) for item, v in zip(item_ids, values, strict=True)}
