"""The human-annotation contract: corpus items, comparison pairs, judgements.

This is where the project's labels come from, and the labels are the part of the
research claim that cannot be recovered later — a predictor trained on noisy
preferences will report a confident number that means nothing.  So the design
here is built around three ideas that are easy to skip and expensive to add
afterwards.

**1. Record the raw observation, not the interpretation.**  A judgement stores
*which side the annotator clicked*, together with which item was rendered on
that side.  The winning item is derived.  Storing only the winner would be
smaller and would silently destroy the ability to measure position bias — and
left-side preference is a well-known artefact of two-alternative forced choice.

**2. Comparisons are grouped, and the grouping matters.**  A ``set_id`` marks
items generated from the same references, so a *within-set* comparison holds the
product and the model constant and isolates the design point.  That is exactly
the deployment task (rank 3 candidates for one job).  *Cross-set* comparisons are
what make all the items mutually comparable at all — see :mod:`adml.pairs`.

**3. Annotator quality is measured, not assumed.**  Volunteers clicking through
a task for a friend are the single biggest threat to label quality, so the schema
carries the machinery to catch it: deliberate repeats for test-retest agreement,
catch trials against a visibly degraded decoy, response latency, and a side-bias
test.  :class:`AnnotatorQuality` computes the flags; nothing is silently dropped.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from .enums import CameraAngle, Composition, Lighting, MotionIntent, Platform, Tier, Vertical
from .request import AssetRef

# --- Quality-control thresholds ---------------------------------------------
#
# PROVISIONAL.  These are starting points from the 2AFC literature's usual
# ranges, not measurements of this task, and the intake thresholds already taught
# the cost of trusting a number set against unrepresentative data.  Re-derive all
# four from the first real annotation session before excluding anyone's work:
# `scripts/annotation_report.py` prints the observed distribution for exactly this.

#: A repeated pair the annotator answers differently more often than this is not
#: expressing a stable preference.  Human aesthetic 2AFC test-retest typically
#: lands around 0.70-0.85, so agreement below 0.60 is worse than the task's floor.
MIN_REPEAT_CONSISTENCY = 0.60

#: Catch trials pair a real candidate against a heavily degraded copy of itself.
#: Missing these is not a matter of taste.  Reported for reference — the *flag*
#: uses the binomial evidence below rather than this raw fraction.
MIN_CATCH_ACCURACY = 0.80

#: Catch accuracy is flagged when the score is this consistent with pure guessing,
#: as a one-sided binomial tail, rather than when a raw fraction falls short.
#:
#: A simulated session showed why the raw fraction is wrong: with 6 catch trials,
#: 5/6 passes an 80% threshold and 4/6 fails it, so an attentive annotator who
#: mis-clicks twice is excluded while a careless one who guesses lucky is kept.
#: Six trials genuinely cannot separate 83% from 95% accuracy, and the fix is
#: more trials plus a sample-size-aware test — not a prettier threshold.
MAX_CATCH_GUESS_PROBABILITY = 0.10

#: Catch trials needed before the flag is allowed to fire.
#:
#: 10, chosen from the false-positive rate rather than by feel. Take an attentive
#: annotator who is right on 95% of catch trials.  Over 6 trials the flag fires
#: unless they are perfect, so it wrongly flags them 27% of the time — useless with
#: six volunteers.  Over 10 trials it fires only at 7 or fewer correct, which
#: happens 1.1% of the time.  Fewer trials than this cannot separate carelessness
#: from a couple of mis-clicks, and no choice of threshold changes that.
#:
#: Note the ceiling this implies elsewhere: an annotator sees at most one catch
#: trial per decoy, so the corpus needs at least this many decoys for the screen to
#: work at all (`scripts/build_corpus.py --decoys`).
MIN_CATCH_TRIALS = 10

#: Below this, a "decision" is not plausibly a comparison of two images.
MIN_PLAUSIBLE_LATENCY_MS = 600

#: Side bias is tested as a z-score against the 50/50 null rather than as a flat
#: fraction: 0.65 means nothing over 20 judgements and a great deal over 500.
MAX_SIDE_BIAS_Z = 3.0

#: Statistics computed on fewer trials than this are reported but never used to
#: exclude anyone — the small-sample noise is larger than the effect.
MIN_TRIALS_TO_JUDGE_AN_ANNOTATOR = 12


class ItemKind(str, Enum):
    """Which pipeline stage produced the item being compared.

    Image and video items are never compared against each other; the two stages
    are ranked separately and the *agreement between those rankings* is the
    project's headline measurement.
    """

    IMAGE = "image"
    VIDEO = "video"


class PairKind(str, Enum):
    """Why a pair exists.  Recorded because it changes how the pair is used."""

    #: Same references, different design point.  Holds product and model
    #: constant, so the comparison isolates the creative decision.
    WITHIN_SET = "within_set"

    #: Different sets.  Individually less informative, collectively essential:
    #: without them the comparison graph is a pile of disconnected triangles and
    #: Bradley-Terry strengths cannot be compared across sets.
    CROSS_SET = "cross_set"

    #: Added specifically to join two components of the comparison graph.  Kept
    #: distinct from CROSS_SET so the report can say how much of the graph's
    #: connectivity was engineered rather than sampled.
    BRIDGE = "bridge"

    #: A real candidate against a visibly degraded copy.  Screening only —
    #: **never** fed to the Bradley-Terry fit, where a decoy's strength would
    #: distort the scale it shares with real items.
    CATCH = "catch"


class Choice(str, Enum):
    """The raw click.  Sides, not items — see the module docstring."""

    LEFT = "left"
    RIGHT = "right"
    TIE = "tie"


class CorpusItem(BaseModel):
    """One comparable artefact, with enough provenance to stratify and ablate.

    Fields are flat rather than nested so the store can index them: the pair
    sampler filters on ``set_id`` and ``kind``, and the ablation tables group by
    the four design axes.
    """

    item_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    set_id: str = Field(
        description="Items sharing this were generated from the same references. "
        "Usually a job id; for bulk corpus generation, the generated set's id."
    )
    kind: ItemKind = ItemKind.IMAGE
    asset: AssetRef

    # --- Provenance ---
    tier: Tier = Tier.MOCK
    provider: str = ""
    model: str = ""
    vertical: Vertical = Vertical.OTHER
    platform: Platform = Platform.INSTAGRAM_REELS
    seed: int = 0

    # --- Design-space coordinate, flattened for grouping ---
    angle: CameraAngle | None = None
    lighting: Lighting | None = None
    composition: Composition | None = None
    motion: MotionIntent | None = None

    # --- Decoys ---
    degraded_from: str | None = Field(
        default=None,
        description="Set on catch-trial decoys: the item_id this was degraded from. "
        "A decoy is never a legitimate training item.",
    )
    degradation: str = Field(default="", description="How it was degraded, e.g. 'blur-12'.")

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_decoy(self) -> bool:
        return self.degraded_from is not None

    def axes(self) -> tuple[str, str, str, str] | None:
        """The design coordinate, or None when the item carries no design point."""
        if self.angle is None or self.lighting is None:
            return None
        assert self.composition is not None and self.motion is not None
        return (
            self.angle.value,
            self.lighting.value,
            self.composition.value,
            self.motion.value,
        )


class ComparisonPair(BaseModel):
    """Two items to be shown side by side.

    The pair is *unordered*: which item appears on the left is decided per
    showing, by the server, so that the same pair shown twice to the same person
    is a genuine test of preference rather than of motor memory.
    """

    pair_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    item_a: str
    item_b: str
    kind: PairKind = PairKind.CROSS_SET
    expected_winner: str | None = Field(
        default=None,
        description="Only for CATCH pairs: the non-degraded item. Answering otherwise "
        "is a screening failure, not a preference.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _check_distinct(self) -> ComparisonPair:
        if self.item_a == self.item_b:
            raise ValueError("a comparison pair needs two different items")
        if self.kind is PairKind.CATCH and self.expected_winner not in (self.item_a, self.item_b):
            raise ValueError("a catch pair must name which of its two items should win")
        return self

    @property
    def key(self) -> tuple[str, str]:
        """Order-independent identity, for de-duplicating a design."""
        return (
            (self.item_a, self.item_b) if self.item_a <= self.item_b else (self.item_b, self.item_a)
        )

    def other(self, item_id: str) -> str:
        if item_id == self.item_a:
            return self.item_b
        if item_id == self.item_b:
            return self.item_a
        raise KeyError(f"{item_id!r} is not in pair {self.pair_id!r}")


class AnnotatorProfile(BaseModel):
    """Who is judging.

    ``agreed_to_research_use`` is mandatory and enrolment is refused without it.
    Judgements from classmates are human-subject data going into a report, and
    the affirmative opt-in is both the right thing and the ethics section's
    evidence — the same stance the pipeline takes on model releases.
    """

    annotator_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    label: str = Field(default="", max_length=64, description="A display name or nickname.")
    cohort: str = Field(
        default="",
        max_length=32,
        description="Free-text grouping, e.g. 'classmate' or 'design-student'. Lets the "
        "report check whether agreement is higher within a cohort than across.",
    )
    agreed_to_research_use: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def may_annotate(self) -> bool:
        return self.agreed_to_research_use


class Judgement(BaseModel):
    """One recorded comparison.

    ``shown_left`` plus ``choice`` is the raw observation; :attr:`winner` is the
    interpretation.  Keeping both is what makes position bias measurable.
    """

    judgement_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    pair_id: str
    annotator_id: str
    shown_left: str = Field(description="item_id that was rendered on the left this showing.")
    choice: Choice
    latency_ms: int = Field(
        default=0,
        ge=0,
        description="Measured from when both images finished rendering to the keypress — "
        "not from the request, or network jitter would swamp the signal.",
    )
    showing: int = Field(
        default=0,
        ge=0,
        description="0 for the first time this annotator saw this pair, 1 for the "
        "deliberate repeat, and so on. Test-retest agreement is computed across these.",
    )
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_repeat(self) -> bool:
        return self.showing > 0

    def winner(self, pair: ComparisonPair) -> str | None:
        """The item chosen, or None for a tie."""
        if self.choice is Choice.TIE:
            return None
        if self.choice is Choice.LEFT:
            return self.shown_left
        return pair.other(self.shown_left)

    def loser(self, pair: ComparisonPair) -> str | None:
        won = self.winner(pair)
        return None if won is None else pair.other(won)


class AnnotatorQuality(BaseModel):
    """Measured reliability for one annotator.

    Nothing here excludes anyone by itself.  The report prints the numbers, the
    flags say which thresholds were crossed, and dropping an annotator's work is
    a decision made deliberately and written down — not a side effect.
    """

    annotator_id: str
    label: str = ""
    cohort: str = ""
    n_judgements: int = 0
    n_pairs: int = Field(default=0, description="Distinct pairs seen, excluding repeats.")
    tie_rate: float = 0.0
    median_latency_ms: int = 0
    fast_click_rate: float = Field(
        default=0.0, description=f"Fraction under {MIN_PLAUSIBLE_LATENCY_MS} ms."
    )

    chose_left: int = 0
    n_sided: int = Field(default=0, description="Non-tie judgements — the side-bias denominator.")

    n_repeats: int = 0
    repeat_agreements: int = 0

    n_catch: int = 0
    catch_correct: int = 0

    @property
    def side_bias(self) -> float | None:
        """Fraction of non-tie judgements that chose the left image."""
        return self.chose_left / self.n_sided if self.n_sided else None

    @property
    def side_bias_z(self) -> float | None:
        """Deviation from the 50/50 null, in standard errors.

        Scales with sample size, which a raw fraction does not: an annotator who
        picked left 13 times out of 20 is unremarkable, and one who did it 325
        times out of 500 is not looking at the images.
        """
        if self.n_sided < MIN_TRIALS_TO_JUDGE_AN_ANNOTATOR:
            return None
        n = self.n_sided
        return (self.chose_left - n / 2) / math.sqrt(n / 4)

    @property
    def repeat_consistency(self) -> float | None:
        return self.repeat_agreements / self.n_repeats if self.n_repeats else None

    @property
    def catch_accuracy(self) -> float | None:
        return self.catch_correct / self.n_catch if self.n_catch else None

    @property
    def catch_guess_probability(self) -> float | None:
        """One-sided binomial tail: P(this many or more correct | pure guessing).

        Small means the score is hard to explain by guessing, so the annotator was
        attending.  Large means the catch trials carry no evidence that they were.
        Exact rather than normal-approximated, because the counts here are single
        digits and the approximation is poor exactly there.
        """
        if not self.n_catch:
            return None
        n, k = self.n_catch, self.catch_correct
        return sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)

    @property
    def flags(self) -> list[str]:
        """Thresholds crossed, each with the number that crossed it."""
        out: list[str] = []
        if self.n_judgements < MIN_TRIALS_TO_JUDGE_AN_ANNOTATOR:
            out.append(f"too few judgements to assess ({self.n_judgements})")
            return out

        guess = self.catch_guess_probability
        if guess is not None and self.n_catch >= MIN_CATCH_TRIALS:
            if guess > MAX_CATCH_GUESS_PROBABILITY:
                out.append(
                    f"catch trials no better than guessing "
                    f"({self.catch_correct}/{self.n_catch}, p={guess:.2f})"
                )

        repeat = self.repeat_consistency
        if repeat is not None and self.n_repeats >= 5 and repeat < MIN_REPEAT_CONSISTENCY:
            out.append(f"inconsistent on repeats ({repeat:.0%} of {self.n_repeats})")

        z = self.side_bias_z
        if z is not None and abs(z) > MAX_SIDE_BIAS_Z:
            side = "left" if z > 0 else "right"
            out.append(f"prefers the {side} side (z={z:+.1f})")

        if self.fast_click_rate > 0.30:
            out.append(
                f"{self.fast_click_rate:.0%} of responses under {MIN_PLAUSIBLE_LATENCY_MS} ms"
            )
        return out

    @property
    def is_trustworthy(self) -> bool:
        """No threshold crossed *and* enough trials to have been able to cross one."""
        return not self.flags


class CorpusStats(BaseModel):
    """Where the label collection has got to.  Rendered in the annotation UI."""

    n_items: int = 0
    n_decoys: int = 0
    n_sets: int = 0
    n_pairs: int = 0
    n_judgements: int = 0
    n_annotators: int = 0
    pairs_judged: int = Field(default=0, description="Pairs with at least one judgement.")

    @property
    def coverage(self) -> float:
        return self.pairs_judged / self.n_pairs if self.n_pairs else 0.0
