"""Turning pairwise judgements into rankings, and measuring whether to believe them.

Two halves:

**Bradley-Terry** (:func:`fit_bradley_terry`) recovers one latent strength per
item from "A beat B" observations, via Hunter's minorization-maximization
iteration.  MM is used rather than gradient descent because it needs no learning
rate, no autograd and no GPU — it converges monotonically in a few dozen
iterations on a laptop, which is the whole point at this scale.

The unregularised MLE has a real failure mode: an item that wins every comparison
it appears in has *unbounded* strength, and the iteration walks off toward
infinity while looking like it is still making progress.  ``prior_strength``
fixes this by giving every item a small number of games against a phantom
opponent of fixed strength, half won and half lost.  Estimates stay finite, and
items with little data are pulled toward the middle rather than to an extreme —
which is also the honest thing to report for an item nobody looked at twice.

**Metrics** (everything below the fit) are the evaluation spine the report rests
on: rank correlation for the headline image-stage/video-stage agreement, pairwise
accuracy for beating the baselines, inter-annotator agreement for arguing the
labels mean something, and bootstrap intervals because a point estimate over a
few hundred comparisons is not a result.

All numpy, no scipy: this has to import in the same environment as the pipeline.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

#: Phantom-opponent games per item.  1.0 is a weak prior — enough to keep an
#: undefeated item finite, small enough that ten real comparisons dominate it.
DEFAULT_PRIOR_STRENGTH = 1.0

Comparison = tuple[str, str]  # (winner_id, loser_id)


# --- Bradley-Terry -----------------------------------------------------------


@dataclass
class BradleyTerryFit:
    """Fitted strengths on a log scale, centred so the mean strength is 0."""

    strengths: dict[str, float]
    iterations: int
    converged: bool
    n_comparisons: float
    prior_strength: float
    is_connected: bool

    @property
    def n_items(self) -> int:
        return len(self.strengths)

    def ranking(self) -> list[str]:
        """Item ids, strongest first.  Ties broken by id for determinism."""
        return sorted(self.strengths, key=lambda i: (-self.strengths[i], i))

    def rank_of(self) -> dict[str, int]:
        """1-based rank per item."""
        return {item: i + 1 for i, item in enumerate(self.ranking())}

    def probability(self, a: str, b: str) -> float:
        """Bradley-Terry probability that ``a`` beats ``b``."""
        return 1.0 / (1.0 + math.exp(-(self.strengths[a] - self.strengths[b])))

    def normalised(self) -> dict[str, float]:
        """Strengths squashed to 0-1, for display only.

        Deliberately not used as a model target: the squashing is monotone but
        arbitrary, and only the ordering and the differences carry meaning.
        """
        if not self.strengths:
            return {}
        values = np.array(list(self.strengths.values()), dtype=float)
        lo, hi = float(values.min()), float(values.max())
        if hi - lo < 1e-12:
            return dict.fromkeys(self.strengths, 0.5)
        return {k: (v - lo) / (hi - lo) for k, v in self.strengths.items()}


def fit_bradley_terry(
    comparisons: Iterable[Comparison],
    *,
    ties: Iterable[Comparison] = (),
    items: Iterable[str] | None = None,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> BradleyTerryFit:
    """Fit item strengths from pairwise wins.

    ``ties`` are counted as half a win in each direction, which keeps the
    information rather than discarding the observation.  A forced choice between
    two genuinely equivalent candidates is noise; a recorded tie is data.

    ``items`` lets callers include items with no comparisons.  Those come back at
    strength 0 — determined entirely by the prior, which is the correct answer to
    "how good is the thing nobody compared?" and is flagged by ``is_connected``.

    Connectivity is reported, not enforced.  With a prior every item is linked
    through the phantom opponent, so the fit is always computable; but strengths
    in different components of the *real* comparison graph are separated only by
    the prior, so comparing them across components is meaningless.
    """
    wins: dict[str, float] = {}
    counts: dict[tuple[str, str], float] = {}
    ids: dict[str, int] = {}

    def register(item: str) -> None:
        if item not in ids:
            ids[item] = len(ids)
            wins[item] = 0.0

    for item in items or ():
        register(item)

    total = 0.0
    for winner, loser in comparisons:
        if winner == loser:
            raise ValueError(f"a comparison needs two different items, got {winner!r} twice")
        register(winner)
        register(loser)
        wins[winner] += 1.0
        key = (winner, loser) if winner <= loser else (loser, winner)
        counts[key] = counts.get(key, 0.0) + 1.0
        total += 1.0

    for a, b in ties:
        if a == b:
            raise ValueError(f"a tie needs two different items, got {a!r} twice")
        register(a)
        register(b)
        wins[a] += 0.5
        wins[b] += 0.5
        key = (a, b) if a <= b else (b, a)
        counts[key] = counts.get(key, 0.0) + 1.0
        total += 1.0

    order = list(ids)
    n = len(order)
    if n == 0:
        return BradleyTerryFit({}, 0, True, 0.0, prior_strength, True)

    index = {item: i for i, item in enumerate(order)}
    win_vec = np.array([wins[i] for i in order], dtype=float) + prior_strength / 2.0

    # Symmetric game-count matrix. Dense is fine: n is in the hundreds, and a
    # 300x300 float array is 700 kB.
    games = np.zeros((n, n), dtype=float)
    for (a, b), c in counts.items():
        i, j = index[a], index[b]
        games[i, j] += c
        games[j, i] += c

    p = np.ones(n, dtype=float)
    converged = False
    iterations = 0
    for step in range(1, max_iter + 1):
        iterations = step
        # denominator_i = sum_j n_ij / (p_i + p_j)  +  prior / (p_i + 1)
        denom = (games / (p[:, None] + p[None, :])).sum(axis=1)
        denom += prior_strength / (p + 1.0)
        new = win_vec / np.maximum(denom, 1e-300)
        new /= np.exp(np.log(new).mean())  # fix the scale: geometric mean 1
        shift = float(np.abs(np.log(new) - np.log(p)).max())
        p = new
        if shift < tol:
            converged = True
            break

    theta = np.log(p)
    theta -= theta.mean()

    # Connectivity of the observed graph, ignoring the phantom opponent.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in counts:
        ra, rb = find(index[a]), find(index[b])
        if ra != rb:
            parent[rb] = ra
    connected = len({find(i) for i in range(n)}) <= 1

    return BradleyTerryFit(
        strengths={item: float(theta[index[item]]) for item in order},
        iterations=iterations,
        converged=converged,
        n_comparisons=total,
        prior_strength=prior_strength,
        is_connected=connected,
    )


# --- Rank correlation --------------------------------------------------------


def _rankdata(values: Sequence[float]) -> np.ndarray:
    """Ranks with ties averaged, matching the usual Spearman convention."""
    arr = np.asarray(values, dtype=float)
    order = arr.argsort(kind="stable")
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    # Average ranks within each tied group.
    sorted_vals = arr[order]
    start = 0
    for i in range(1, len(arr) + 1):
        if i == len(arr) or sorted_vals[i] != sorted_vals[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation.

    NaN when either input is constant: a vector with no rank information has no
    correlation, and returning 0.0 would read as "measured, and independent".
    """
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    if len(x) < 2:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b, which corrects for ties in either variable.

    tau-b rather than tau-a because rankings over 3-candidate sets tie constantly
    and tau-a silently understates agreement when they do.
    """
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = discordant = tie_x = tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue  # tied in both: excluded from every term
            if dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


# --- Predictive accuracy -----------------------------------------------------


def pairwise_accuracy(scores: dict[str, float], comparisons: Iterable[Comparison]) -> float:
    """Fraction of held-out comparisons the scores order correctly.

    Equal scores count as half, which is what a coin flip earns. Comparisons
    naming an item the scorer has never seen are skipped rather than counted as
    failures — the model was not asked about them.
    """
    hits = 0.0
    n = 0
    for winner, loser in comparisons:
        if winner not in scores or loser not in scores:
            continue
        n += 1
        diff = scores[winner] - scores[loser]
        hits += 1.0 if diff > 0 else (0.5 if diff == 0 else 0.0)
    return hits / n if n else float("nan")


def top1_retention(sets: Iterable[tuple[Sequence[str], Sequence[str]]]) -> float:
    """How often the predicted best is also the true best.

    This is the project's counterfactual: had only the top-ranked image been
    promoted to video, how often would the human-preferred candidate have
    survived? Sets whose orders are empty are skipped.
    """
    hits = 0
    n = 0
    for predicted, actual in sets:
        if not predicted or not actual:
            continue
        n += 1
        hits += int(predicted[0] == actual[0])
    return hits / n if n else float("nan")


def ndcg_at_k(predicted: Sequence[str], relevance: dict[str, float], k: int = 1) -> float:
    """Normalised discounted cumulative gain over a small candidate set.

    With ``k=1`` over 3 candidates this is a graded version of top-1 accuracy:
    picking the second-best is penalised less than picking the worst, which is
    the right shape for a 3-candidate ranking task.
    """
    if not predicted:
        return float("nan")

    def dcg(items: Sequence[str]) -> float:
        return sum(
            relevance.get(item, 0.0) / math.log2(rank + 1)
            for rank, item in enumerate(items[:k], start=1)
        )

    ideal = sorted(predicted, key=lambda i: -relevance.get(i, 0.0))
    best = dcg(ideal)
    return dcg(predicted) / best if best > 0 else float("nan")


# --- Inter-annotator agreement ----------------------------------------------


def percent_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    """Raw agreement between two aligned label sequences."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if not a:
        return float("nan")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Chance-corrected agreement between exactly two annotators.

    Worth reporting alongside raw agreement: on a two-alternative task, 50%
    agreement is what coin-flipping produces, so raw agreement alone always
    looks better than it is.
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    observed = percent_agreement(a, b)
    expected = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    if abs(1.0 - expected) < 1e-12:
        return float("nan")  # everyone always says the same thing; kappa undefined
    return (observed - expected) / (1.0 - expected)


def krippendorff_alpha_nominal(units: Iterable[Sequence[str]]) -> float:
    """Krippendorff's alpha for nominal labels, tolerating missing ratings.

    The right tool for this project's actual shape: more than two annotators, and
    each one judges a different, overlapping subset of pairs. Cohen's kappa needs
    two raters over the same items and cannot express that.

    ``units`` is one sequence of labels per compared pair. Units with fewer than
    two labels carry no agreement information and are ignored.
    """
    coincidence: dict[tuple[str, str], float] = {}
    for ratings in units:
        m = len(ratings)
        if m < 2:
            continue
        weight = 1.0 / (m - 1)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                key = (ratings[i], ratings[j])
                coincidence[key] = coincidence.get(key, 0.0) + weight

    if not coincidence:
        return float("nan")

    marginal: dict[str, float] = {}
    for (c, _other), v in coincidence.items():
        marginal[c] = marginal.get(c, 0.0) + v
    total = sum(marginal.values())
    if total <= 1:
        return float("nan")

    observed = sum(v for (c, k), v in coincidence.items() if c != k)
    expected = sum(marginal[c] * marginal[k] for c in marginal for k in marginal if c != k)
    if expected <= 0:
        return float("nan")
    return 1.0 - (total - 1.0) * observed / expected


# --- Reliability and intervals ----------------------------------------------


@dataclass
class SplitHalf:
    """Do two independent halves of the annotator pool agree on the ranking?

    This is the strongest single check that the labels carry signal rather than
    noise. It needs no ground truth: if half the annotators' fitted strengths do
    not correlate with the other half's, no model trained on them will generalise,
    and that is worth discovering in week 8 rather than week 15.
    """

    rho: float
    n_shared_items: int
    n_left: int
    n_right: int
    left_annotators: int
    right_annotators: int


def split_half_reliability(
    observations: Iterable[tuple[str, str, str]],
    *,
    seed: int = 0,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> SplitHalf:
    """Split annotators in two, fit each half, correlate the strengths.

    ``observations`` are ``(annotator_id, winner_id, loser_id)``. The split is on
    *annotators*, not on comparisons: splitting comparisons would put the same
    person's judgements on both sides and measure the fitter's stability rather
    than the annotators' agreement.
    """
    rows = list(observations)
    annotators = sorted({a for a, _, _ in rows})
    if len(annotators) < 2:
        return SplitHalf(float("nan"), 0, len(rows), 0, len(annotators), 0)

    rng = random.Random(seed)
    shuffled = annotators[:]
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    left_set, right_set = set(shuffled[:half]), set(shuffled[half:])

    left = [(w, x) for a, w, x in rows if a in left_set]
    right = [(w, x) for a, w, x in rows if a in right_set]
    if not left or not right:
        return SplitHalf(float("nan"), 0, len(left), len(right), len(left_set), len(right_set))

    fit_left = fit_bradley_terry(left, prior_strength=prior_strength)
    fit_right = fit_bradley_terry(right, prior_strength=prior_strength)
    shared = sorted(set(fit_left.strengths) & set(fit_right.strengths))
    rho = (
        spearman_rho(
            [fit_left.strengths[i] for i in shared], [fit_right.strengths[i] for i in shared]
        )
        if len(shared) >= 2
        else float("nan")
    )
    return SplitHalf(
        rho=rho,
        n_shared_items=len(shared),
        n_left=len(left),
        n_right=len(right),
        left_annotators=len(left_set),
        right_annotators=len(right_set),
    )


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values``.

    Used for every headline number. A pairwise accuracy of 0.71 over 300
    comparisons and one over 3,000 are different claims, and only the interval
    says which is which.
    """
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(resamples, len(arr)))].mean(axis=1)
    lo = float(np.percentile(means, 100 * (1 - confidence) / 2))
    hi = float(np.percentile(means, 100 * (1 + confidence) / 2))
    return (lo, hi)
