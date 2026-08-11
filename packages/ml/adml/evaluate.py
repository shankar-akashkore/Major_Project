"""The evaluation harness: what the numbers mean, and what they cannot mean.

Four decisions here do more for the credibility of the results than any modelling
choice, and each exists because the naive version is misleading.

**1. Results are pooled across folds, not averaged as per-fold means.**  An honest
set-wise split leaves roughly a hundred testable comparisons per fold
(:mod:`adml.split`), and a proportion measured on a hundred observations carries a
+-9 point interval — wide enough to swallow the entire difference between a good
model and a poor one.  Folds are disjoint by construction, so every held-out
comparison is predicted exactly once and the pooled set is a valid sample.

**2. Models are compared with a paired test on the same comparisons, never with
two independent intervals.**  Two models that share features agree on most
comparisons; their accuracies are strongly correlated, so overlapping marginal
intervals say almost nothing about whether one is better.  What matters is the
*difference* on the comparisons where they disagree, which is
:func:`paired_difference` and :func:`mcnemar_exact`.

**3. Every accuracy is reported against a ceiling, not against 1.0.**  Labels are
single human judgements, and humans disagree with *themselves* on repeats.  If
test-retest agreement is 0.60, a predictor that knew each annotator's true
preference perfectly would still only match a single judgement about 72% of the
time.  Against that ceiling, 68% is near-perfect; against 100% it reads as
mediocre.  Reporting accuracy without the ceiling is the difference between those
two sentences.  See :func:`ceiling_from_agreement` for the derivation and its
assumptions.

**4. Intransitive triples are counted, not resolved.**  Three candidates judged
once per pair can produce a cycle: A beats B, B beats C, C beats A.  There is no
"true best" in that set, and quietly breaking the tie by score or by id would
manufacture a ground truth to be graded against.  Such sets are excluded from
top-1 retention and their number is reported, because the rate is itself a
measurement of how hard the task is.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from adschema.annotation import PairKind

from .ranking import bootstrap_ci, kendall_tau, spearman_rho
from .split import Observation

# --- The noise ceiling ------------------------------------------------------


def ceiling_from_agreement(agreement: float) -> float:
    """Highest pairwise accuracy attainable against single noisy judgements.

    Model a judgement as expressing the underlying preference with probability
    ``q`` and flipping otherwise.  Two independent judgements of the same pair then
    agree with probability ``q^2 + (1-q)^2``, so an observed agreement ``a`` implies

        q = (1 + sqrt(2a - 1)) / 2

    and a predictor that knew the underlying preference exactly would match a
    single judgement with probability ``q`` — never 1.0.

    Two assumptions worth stating in the write-up, because both can be wrong:

    * *Errors are independent across showings.*  A repeat shown to the same person
      is partly remembered, which inflates agreement and therefore inflates the
      ceiling.  That direction is conservative — it makes the model look further
      from ceiling than it is — so it is the safe way to be wrong.
    * *There is one underlying preference.*  A consistent idiosyncratic taste is
      not noise, and where annotators genuinely differ this treats real
      disagreement as error.  The inter-annotator ceiling is therefore lower than
      the test-retest one, and both are reported.

    Agreement at or below chance carries no information about ``q`` and returns
    NaN rather than a number.
    """
    if not 0.5 < agreement <= 1.0:
        return float("nan")
    return (1.0 + math.sqrt(2.0 * agreement - 1.0)) / 2.0


@dataclass
class Ceiling:
    """The ceilings a reported accuracy has to be read against."""

    test_retest_agreement: float
    test_retest_ceiling: float
    inter_annotator_agreement: float
    inter_annotator_ceiling: float
    n_repeats: int
    n_multi_judged_pairs: int

    def summary(self) -> str:
        def fmt(value: float) -> str:
            return "n/a" if math.isnan(value) else f"{value:.3f}"

        return (
            f"test-retest    agreement {fmt(self.test_retest_agreement)} "
            f"(n={self.n_repeats})  -> ceiling {fmt(self.test_retest_ceiling)}\n"
            f"inter-annotator agreement {fmt(self.inter_annotator_agreement)} "
            f"(n={self.n_multi_judged_pairs}) -> ceiling {fmt(self.inter_annotator_ceiling)}"
        )


def measure_ceiling(
    repeat_agreements: Sequence[float],
    multi_judged: Iterable[Sequence[str]],
) -> Ceiling:
    """Derive both ceilings from what the annotation tool already recorded.

    ``repeat_agreements`` is one 0/1 (or fractional) agreement per repeated
    showing — the test-retest observations the annotation store collects at
    ``REPEAT_RATE``.  ``multi_judged`` is one sequence of chosen-item labels per
    pair judged by more than one annotator.

    Both come free from the existing collection design, which is the reason the
    repeat and coverage policies were built into ``next_pair`` rather than added
    afterwards: without them there is no ceiling, and without a ceiling the
    accuracy numbers cannot be interpreted at all.
    """
    repeats = [float(v) for v in repeat_agreements]
    retest = float(np.mean(repeats)) if repeats else float("nan")

    pairwise: list[float] = []
    n_units = 0
    for labels in multi_judged:
        values = list(labels)
        if len(values) < 2:
            continue
        n_units += 1
        agree = sum(
            1.0
            for i in range(len(values))
            for j in range(i + 1, len(values))
            if values[i] == values[j]
        )
        total = len(values) * (len(values) - 1) / 2
        pairwise.append(agree / total)
    inter = float(np.mean(pairwise)) if pairwise else float("nan")

    return Ceiling(
        test_retest_agreement=retest,
        test_retest_ceiling=ceiling_from_agreement(retest) if repeats else float("nan"),
        inter_annotator_agreement=inter,
        inter_annotator_ceiling=ceiling_from_agreement(inter) if pairwise else float("nan"),
        n_repeats=len(repeats),
        n_multi_judged_pairs=n_units,
    )


# --- Per-observation credit -------------------------------------------------


def credit(scores: dict[str, float], observations: Sequence[Observation]) -> np.ndarray:
    """1.0 for a correctly ordered comparison, 0.5 for a tie, 0.0 for wrong.

    NaN where the scorer has no opinion because an item is absent from it — kept
    as NaN rather than dropped so that two models evaluated on the same
    observation list stay index-aligned, which is what the paired tests need.
    """
    out = np.full(len(observations), np.nan)
    for i, obs in enumerate(observations):
        if obs.winner not in scores or obs.loser not in scores:
            continue
        if obs.is_tie:
            # A tie has no correct direction. Scored as 0.5 regardless, so that a
            # model is neither rewarded nor punished for its choice, and the
            # observation still counts toward the denominator it belongs in.
            out[i] = 0.5
            continue
        diff = scores[obs.winner] - scores[obs.loser]
        out[i] = 1.0 if diff > 0 else (0.5 if diff == 0 else 0.0)
    return out


@dataclass
class Evaluation:
    """One scorer's pooled held-out performance.

    **Scores from different folds are not on a common scale.**  Each fold fits its
    own model, and a Bradley-Terry score is only defined up to an additive constant
    (see :meth:`adml.predictor.PairwiseRanker.scores`), so two items scored by
    different folds cannot be compared with each other.  Everything computed here
    stays inside a fold, which the set-wise split guarantees: both items of a
    held-out comparison always land in the same fold, and a generation set is never
    divided.  ``item_scores`` is pooled for inspection and for within-set metrics
    only — ranking the whole corpus by it would be meaningless.
    """

    name: str
    observations: list[Observation]
    credit: np.ndarray
    item_scores: dict[str, float] = field(default_factory=dict)
    #: False when any input feature came from a stand-in rather than a real
    #: encoder.  Printed alongside every number it touches.
    is_real: bool = True

    @property
    def n(self) -> int:
        return int(np.isfinite(self.credit).sum())

    @property
    def accuracy(self) -> float:
        valid = self.credit[np.isfinite(self.credit)]
        return float(valid.mean()) if len(valid) else float("nan")

    def ci(self, *, seed: int = 0) -> tuple[float, float]:
        return bootstrap_ci(list(self.credit[np.isfinite(self.credit)]), seed=seed)

    def accuracy_by_kind(self) -> dict[str, tuple[float, int]]:
        """Accuracy split by comparison kind.

        Within-set accuracy is the deployment number — rank three candidates for
        one job — and cross-set accuracy is a different, easier question about
        general image quality.  Pooling them into one figure hides which one the
        model is actually good at.
        """
        out: dict[str, tuple[float, int]] = {}
        for kind in PairKind:
            mask = np.array(
                [
                    o.kind is kind and np.isfinite(c)
                    for o, c in zip(self.observations, self.credit, strict=True)
                ]
            )
            if mask.any():
                out[kind.value] = (float(self.credit[mask].mean()), int(mask.sum()))
        return out

    def summary(self, ceiling: float | None = None) -> str:
        lo, hi = self.ci()
        line = f"{self.name:<22} {self.accuracy:.3f} [{lo:.3f}, {hi:.3f}]  n={self.n}"
        if ceiling is not None and not math.isnan(ceiling) and ceiling > 0:
            line += f"  ({self.accuracy / ceiling:.0%} of ceiling {ceiling:.3f})"
        if not self.is_real:
            line += "  [STAND-IN FEATURES]"
        return line


# --- Paired comparison ------------------------------------------------------


@dataclass
class Difference:
    """Whether one scorer really beats another on the same held-out comparisons."""

    name_a: str
    name_b: str
    delta: float
    ci: tuple[float, float]
    n_compared: int
    a_only_correct: int
    b_only_correct: int
    p_value: float

    @property
    def resolved(self) -> bool:
        """True when the interval for the difference excludes zero."""
        lo, hi = self.ci
        return not (math.isnan(lo) or math.isnan(hi)) and (lo > 0 or hi < 0)

    def summary(self) -> str:
        lo, hi = self.ci
        verdict = "resolved" if self.resolved else "UNRESOLVED at this sample size"
        return (
            f"{self.name_a} - {self.name_b}: {self.delta:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
            f"discordant {self.a_only_correct}/{self.b_only_correct}  "
            f"p={self.p_value:.4f}  {verdict}"
        )


def paired_difference(
    a: Evaluation,
    b: Evaluation,
    *,
    resamples: int = 4000,
    seed: int = 0,
) -> Difference:
    """Bootstrap the *difference* over the comparisons both scorers answered.

    Resampling the difference rather than each accuracy separately is what makes a
    few hundred observations enough to resolve anything: the two models agree on
    most comparisons, and those contribute exactly zero variance to the difference
    while contributing plenty to each marginal interval.
    """
    if len(a.observations) != len(b.observations):
        raise ValueError("paired comparison needs the same observation list on both sides")
    mask = np.isfinite(a.credit) & np.isfinite(b.credit)
    diff = a.credit[mask] - b.credit[mask]
    n = int(mask.sum())

    if n < 2:
        return Difference(a.name, b.name, float("nan"), (float("nan"), float("nan")), n, 0, 0, 1.0)

    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
    ci = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

    a_only = int((a.credit[mask] > b.credit[mask]).sum())
    b_only = int((b.credit[mask] > a.credit[mask]).sum())
    return Difference(
        name_a=a.name,
        name_b=b.name,
        delta=float(diff.mean()),
        ci=ci,
        n_compared=n,
        a_only_correct=a_only,
        b_only_correct=b_only,
        p_value=mcnemar_exact(a_only, b_only),
    )


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar test on the discordant comparisons.

    Exact rather than the chi-squared approximation because the discordant count
    is often under thirty here, where the approximation is not trustworthy.  Only
    comparisons the two models answer differently carry information about which is
    better; the ones they both get right are the same evidence twice.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


# --- Set-level metrics ------------------------------------------------------


@dataclass
class SetOutcome:
    """The human verdict inside one generation set, if there is one."""

    set_id: str
    order: list[str]
    wins: dict[str, float]
    is_intransitive: bool


def within_set_outcomes(
    observations: Iterable[Observation],
    set_of: dict[str, str],
) -> list[SetOutcome]:
    """Human ordering per set from its within-set comparisons.

    Ordered by win count.  A set whose items all have the same win count is a
    cycle — with three items judged once per pair, A>B>C>A gives everyone exactly
    one win — and is marked rather than ordered.  See the module docstring.
    """
    by_set: dict[str, dict[str, float]] = {}
    for obs in observations:
        if obs.kind is not PairKind.WITHIN_SET:
            continue
        set_id = set_of.get(obs.winner)
        if set_id is None or set_id != set_of.get(obs.loser):
            continue
        wins = by_set.setdefault(set_id, {})
        wins.setdefault(obs.winner, 0.0)
        wins.setdefault(obs.loser, 0.0)
        if obs.is_tie:
            wins[obs.winner] += 0.5
            wins[obs.loser] += 0.5
        else:
            wins[obs.winner] += 1.0

    out: list[SetOutcome] = []
    for set_id, wins in sorted(by_set.items()):
        values = sorted(wins.values())
        intransitive = len(wins) >= 3 and max(values) - min(values) < 1e-9
        out.append(
            SetOutcome(
                set_id=set_id,
                order=sorted(wins, key=lambda i: (-wins[i], i)),
                wins=wins,
                is_intransitive=intransitive,
            )
        )
    return out


@dataclass
class SetMetrics:
    """Deployment-shaped metrics: pick the best of three, per job."""

    top1_retention: float
    top1_ci: tuple[float, float]
    ndcg_at_1: float
    mean_spearman: float
    n_sets: int
    n_intransitive: int

    def summary(self) -> str:
        lo, hi = self.top1_ci
        return (
            f"top-1 retention {self.top1_retention:.3f} [{lo:.3f}, {hi:.3f}]  "
            f"NDCG@1 {self.ndcg_at_1:.3f}  mean rho {self.mean_spearman:+.3f}  "
            f"over {self.n_sets} sets ({self.n_intransitive} intransitive, excluded)"
        )


def set_metrics(
    scores: dict[str, float],
    outcomes: Sequence[SetOutcome],
    *,
    seed: int = 0,
) -> SetMetrics:
    """Top-1 retention and rank agreement, per set, over determinate sets only.

    Top-1 retention is the project's counterfactual: had only the best-predicted
    image been promoted to video, how often would the human-preferred candidate
    have survived?  With three candidates, chance is 1/3.
    """
    hits: list[float] = []
    gains: list[float] = []
    rhos: list[float] = []
    intransitive = 0

    for outcome in outcomes:
        if outcome.is_intransitive:
            intransitive += 1
            continue
        items = [i for i in outcome.order if i in scores]
        if len(items) < 2:
            continue
        predicted = sorted(items, key=lambda i: (-scores[i], i))
        hits.append(1.0 if predicted[0] == outcome.order[0] else 0.0)

        # Graded credit: choosing the second-best costs less than choosing the
        # worst, which is the right shape for a three-candidate decision.
        relevance = {item: float(len(items) - rank) for rank, item in enumerate(outcome.order)}
        best = max(relevance.values())
        gains.append(relevance[predicted[0]] / best if best > 0 else float("nan"))

        rho = spearman_rho(
            [float(len(items) - outcome.order.index(i)) for i in items],
            [scores[i] for i in items],
        )
        if not math.isnan(rho):
            rhos.append(rho)

    return SetMetrics(
        top1_retention=float(np.mean(hits)) if hits else float("nan"),
        top1_ci=bootstrap_ci(hits, seed=seed) if len(hits) > 1 else (float("nan"), float("nan")),
        ndcg_at_1=float(np.nanmean(gains)) if gains else float("nan"),
        mean_spearman=float(np.mean(rhos)) if rhos else float("nan"),
        n_sets=len(hits),
        n_intransitive=intransitive,
    )


# --- Stage agreement: the headline claim ------------------------------------


@dataclass
class StageAgreement:
    """Does the image-stage ranking survive to the video stage?

    The number the project stands on.  Measured per set and pooled, because a
    single correlation over all items would be dominated by cross-set variation in
    image quality rather than by the within-job question that is actually asked.
    """

    mean_spearman: float
    spearman_ci: tuple[float, float]
    mean_kendall: float
    top1_retention: float
    top1_ci: tuple[float, float]
    n_sets: int

    def summary(self) -> str:
        lo, hi = self.spearman_ci
        t_lo, t_hi = self.top1_ci
        return (
            f"image->video rank agreement: rho {self.mean_spearman:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
            f"tau {self.mean_kendall:+.3f}  "
            f"top-1 retention {self.top1_retention:.3f} [{t_lo:.3f}, {t_hi:.3f}]  "
            f"over {self.n_sets} sets"
        )


def stage_agreement(
    image_order: dict[str, Sequence[str]],
    video_order: dict[str, Sequence[str]],
    *,
    seed: int = 0,
) -> StageAgreement:
    """Correlate the two stages' orderings set by set.

    ``image_order`` and ``video_order`` map set id to items best-first.  Sets
    present in only one mapping are skipped: a job whose videos were never ranked
    contributes no agreement observation, and treating it as agreement or as
    disagreement would both be inventions.
    """
    rhos: list[float] = []
    taus: list[float] = []
    hits: list[float] = []

    for set_id, predicted in sorted(image_order.items()):
        actual = video_order.get(set_id)
        if actual is None:
            continue
        shared = [i for i in predicted if i in set(actual)]
        if len(shared) < 2:
            continue
        pred_rank = {item: len(predicted) - i for i, item in enumerate(predicted)}
        true_rank = {item: len(actual) - i for i, item in enumerate(actual)}
        rho = spearman_rho([pred_rank[i] for i in shared], [true_rank[i] for i in shared])
        tau = kendall_tau([pred_rank[i] for i in shared], [true_rank[i] for i in shared])
        if not math.isnan(rho):
            rhos.append(rho)
        if not math.isnan(tau):
            taus.append(tau)
        hits.append(1.0 if predicted[0] == actual[0] else 0.0)

    return StageAgreement(
        mean_spearman=float(np.mean(rhos)) if rhos else float("nan"),
        spearman_ci=bootstrap_ci(rhos, seed=seed)
        if len(rhos) > 1
        else (float("nan"), float("nan")),
        mean_kendall=float(np.mean(taus)) if taus else float("nan"),
        top1_retention=float(np.mean(hits)) if hits else float("nan"),
        top1_ci=bootstrap_ci(hits, seed=seed) if len(hits) > 1 else (float("nan"), float("nan")),
        n_sets=len(hits),
    )


# --- Ablation ---------------------------------------------------------------


@dataclass
class AblationRow:
    """One line of the ablation table."""

    label: str
    n_features: int
    accuracy: float
    ci: tuple[float, float]
    delta_vs_full: Difference | None = None

    def summary(self) -> str:
        lo, hi = self.ci
        line = f"{self.label:<26} {self.n_features:>4}  {self.accuracy:.3f} [{lo:.3f}, {hi:.3f}]"
        if self.delta_vs_full is not None:
            d = self.delta_vs_full
            marker = "*" if d.resolved else " "
            line += f"  {d.delta:+.3f}{marker}"
        return line


def ablate(
    full: Evaluation,
    variants: Iterable[tuple[str, int, Evaluation]],
    *,
    n_features: int = 0,
    seed: int = 0,
) -> list[AblationRow]:
    """Build the ablation table, each row paired against the full model.

    Every row is a paired comparison against the same full model on the same
    held-out comparisons, so "removing this group costs 3 points" is a claim with
    an interval attached rather than a difference of two noisy point estimates.
    An unresolved row is marked as such: at this corpus size many groups genuinely
    cannot be distinguished, and saying so is the result.
    """
    rows = [
        AblationRow(
            label="full", n_features=n_features, accuracy=full.accuracy, ci=full.ci(seed=seed)
        )
    ]
    for label, n_features, variant in variants:
        rows.append(
            AblationRow(
                label=label,
                n_features=n_features,
                accuracy=variant.accuracy,
                ci=variant.ci(seed=seed),
                delta_vs_full=paired_difference(variant, full, seed=seed),
            )
        )
    return rows


ScoreFn = Callable[[Sequence[str]], dict[str, float]]
