"""Harvesting the two-stage rank agreement out of jobs that have actually run.

This is the project's headline measurement, and the thing most easily faked by
accident.  Three distinctions decide whether the number means anything.

**1. A set is identified by its content, not by its job id.**  Every golden replay
writes a fresh :class:`~adschema.JobRecord` with a new job id, a new storage
prefix, and *bit-identical* candidates.  Harvesting per job id would therefore let
``replay_golden.py --all``, run twenty times before the viva, report twenty sets in
perfect agreement with a confidence interval of zero width — a fabricated headline
produced entirely by honest-looking code.  :func:`set_key` hashes the candidates'
own content digests, so a replayed job collapses onto the job it replays.

**2. Model-to-model agreement is not the claim.**  The claim is that the
image-stage prediction anticipates how the *finished videos* are preferred, and
"preferred" means by a person.  Both stages currently being scored by the same
codebase, over overlapping feature groups, guarantees some agreement by
construction and measures internal consistency rather than predictive validity.
:class:`Harvest` keeps the two apart: :attr:`Harvest.model_vs_model` is
computable today and is reported as a diagnostic, and the human comparison is
reported as absent until video-stage judgements exist.  Quoting the first as
though it were the second is the single most likely way this project would
overclaim.

**3. Identical inputs must produce an identical ranking.**  Two jobs sharing a set
key ran the same ranker over the same bytes, so any disagreement between their
orderings is nondeterminism in the scorer, not a second observation.  That is a
defect worth surfacing loudly rather than averaging away, so conflicts are
collected in :attr:`Harvest.conflicts`.

Stub scores propagate: a set whose candidates were ranked by the heuristic
stand-in is counted, marked, and excluded from anything quoted as a result, in the
same way :class:`adml.featureset.FeatureTable` carries ``is_real``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from adschema import JobRecord, JobResult, JobState

from .evaluate import StageAgreement, stage_agreement

#: Joined into the content digest so a set key cannot collide with a bare SHA-256
#: appearing anywhere else.
_KEY_PREFIX = "stageset-v1"

#: Fewest sets an agreement may be quoted from.  One set gives rho = +-1 exactly and
#: no interval at all, and "rho +1.000" reads as a strong result whatever the sample
#: size printed beside it.  Below this the number is withheld rather than qualified.
MIN_QUOTABLE_SETS = 2


def set_key(result: JobResult) -> tuple[str, bool]:
    """Content identity of a generation set, and whether it is trustworthy.

    The digest covers the image candidates' content hashes, sorted by candidate
    index so the key is stable, plus the candidate count.  Storage keys are
    deliberately excluded: they carry the job id, which is exactly the thing that
    differs between a job and its replay.

    Returns ``(key, is_content_based)``.  When any candidate is missing a
    ``sha256`` — an older record, or a provider that did not report one — there is
    no content to key on and the job id is used instead, with ``False`` so the
    caller can say that deduplication was not possible rather than assuming it
    worked.
    """
    digests = [(c.index, c.asset.sha256) for c in result.images]
    if not digests or any(d is None for _, d in digests):
        return result.job_id, False
    joined = "|".join(f"{i}:{d}" for i, d in sorted(digests))
    payload = f"{_KEY_PREFIX}|{len(digests)}|{joined}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32], True


@dataclass
class StageObservation:
    """One generation set's paired image-stage and video-stage ordering.

    ``image_order`` and ``video_order`` are candidate indices as strings, best
    first, because :func:`adml.evaluate.stage_agreement` keys on item ids and a
    candidate index is this set's item id.
    """

    key: str
    job_ids: list[str]
    image_order: list[str]
    video_order: list[str]
    #: True when every score behind either ordering came from the stub scorer.
    is_stub: bool
    #: False when the key fell back to the job id, so replays were not collapsed.
    keyed_by_content: bool = True
    tier: str = "mock"

    @property
    def n_replays(self) -> int:
        """How many job records produced this one observation."""
        return len(self.job_ids)


@dataclass
class Conflict:
    """Two jobs over identical candidates that ranked them differently."""

    key: str
    job_a: str
    job_b: str
    order_a: list[str]
    order_b: list[str]
    stage: str

    def summary(self) -> str:
        return (
            f"{self.stage} ordering is not deterministic: {self.job_a} gave "
            f"{self.order_a} and {self.job_b} gave {self.order_b} over the same bytes"
        )


@dataclass
class Harvest:
    """What the jobs on disk can and cannot support.

    ``model_vs_model`` deliberately does not have a "headline" alias.  The name is
    the caveat, and a caller that wants to print it has to print what it is.
    """

    observations: list[StageObservation]
    conflicts: list[Conflict] = field(default_factory=list)
    #: Completed jobs that carried no usable pair of orderings, by reason.
    skipped: dict[str, int] = field(default_factory=dict)
    n_records: int = 0
    n_completed: int = 0

    @property
    def n_sets(self) -> int:
        return len(self.observations)

    @property
    def n_real_sets(self) -> int:
        """Sets whose orderings came from a trained scorer rather than the stub."""
        return sum(1 for o in self.observations if not o.is_stub)

    @property
    def n_collapsed(self) -> int:
        """Job records that were replays of a set already counted."""
        return sum(o.n_replays - 1 for o in self.observations)

    @property
    def all_stub(self) -> bool:
        return bool(self.observations) and self.n_real_sets == 0

    @property
    def dedup_incomplete(self) -> bool:
        return any(not o.keyed_by_content for o in self.observations)

    def model_vs_model(self, *, real_only: bool = False) -> StageAgreement:
        """Agreement between the two stages' *predicted* orders.

        A diagnostic, not the claim — see the module docstring.  ``real_only``
        restricts it to sets scored by a trained model, which is the only version
        fit to appear in a results table.
        """
        chosen = [o for o in self.observations if not (real_only and o.is_stub)]
        image = {o.key: o.image_order for o in chosen}
        video = {o.key: o.video_order for o in chosen}
        return stage_agreement(image, video)

    def quotable(self, *, real_only: bool = False) -> bool:
        """Whether there are enough sets for an agreement to be printed as a number.

        Separate from whether it is a *result* — a stub-scored agreement over ten
        sets is quotable and still not a result. Both gates have to pass before a
        number belongs in the report.
        """
        n = self.n_real_sets if real_only else self.n_sets
        return n >= MIN_QUOTABLE_SETS

    def rank_shifts(self) -> dict[int, int]:
        """How far candidates moved between stages, counted over all sets.

        A distribution of zeros means the image stage told us everything, which is
        a result; large shifts mean animation changes the ordering, which is also a
        result. It is the flat, uninformative middle that would say nothing.
        """
        out: dict[int, int] = {}
        for obs in self.observations:
            video_rank = {item: i for i, item in enumerate(obs.video_order)}
            for image_rank, item in enumerate(obs.image_order):
                if item not in video_rank:
                    continue
                shift = image_rank - video_rank[item]
                out[shift] = out.get(shift, 0) + 1
        return dict(sorted(out.items()))

    def summary(self) -> str:
        lines = [
            f"{self.n_records} job record(s), {self.n_completed} completed, "
            f"{self.n_sets} distinct generation set(s)"
        ]
        if self.n_collapsed:
            lines.append(
                f"{self.n_collapsed} record(s) were replays of a set already counted "
                "and contribute no extra observation"
            )
        if self.dedup_incomplete:
            lines.append(
                "some sets have no content digests, so replays of them could not be "
                "collapsed — treat n as an upper bound"
            )
        for reason, count in sorted(self.skipped.items()):
            lines.append(f"{count} completed job(s) skipped: {reason}")
        if self.all_stub:
            lines.append("every set was ranked by the stub scorer, so none of this is a result")
        elif self.n_real_sets < self.n_sets:
            lines.append(
                f"{self.n_sets - self.n_real_sets} of {self.n_sets} set(s) are stub-scored"
            )
        for conflict in self.conflicts:
            lines.append("CONFLICT " + conflict.summary())
        return "\n".join(lines)


def _orders(result: JobResult) -> tuple[list[str], list[str], str | None]:
    """The two orderings for one job, or a reason there are none."""
    image = [str(i) for i in result.image_stage_order]
    video = [str(i) for i in result.video_stage_order]
    if len(image) < 2:
        return [], [], "fewer than two image candidates carried a score"
    if len(video) < 2:
        return [], [], "fewer than two candidates reached the final ranking"
    # An ordering over items the other stage never saw cannot be correlated. This
    # happens when a slot fails its gate: the image stage ranks three, the video
    # stage ranks two, and the shared subset is what gets compared.
    shared = set(image) & set(video)
    if len(shared) < 2:
        return [], [], "the two stages have fewer than two candidates in common"
    return image, video, None


def _is_stub(result: JobResult) -> bool:
    """True unless some score behind either ordering came from a trained model.

    Pessimistic on purpose: a set counts as real only on positive evidence that a
    non-stub score exists, so a missing flag reads as stub rather than as real.
    """
    flags = [c.score.is_stub for c in result.images if c.score is not None]
    flags += [r.video.score.is_stub for r in result.ranking if r.video.score is not None]
    return all(flags) if flags else True


def harvest(records: Iterable[JobRecord]) -> Harvest:
    """Collect one paired observation per distinct generation set.

    Records are processed oldest first so that the job kept for a set is the one
    that produced it rather than the most recent replay of it, which makes the
    reported job id the useful one.
    """
    ordered = sorted(
        records,
        key=lambda r: (r.finished_at is None, r.finished_at or r.request.created_at),
    )

    by_key: dict[str, StageObservation] = {}
    conflicts: list[Conflict] = []
    skipped: dict[str, int] = {}
    n_completed = 0
    n_records = 0

    for record in ordered:
        n_records += 1
        if record.state is not JobState.COMPLETED or record.result is None:
            continue
        n_completed += 1
        result = record.result

        image, video, reason = _orders(result)
        if reason is not None:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        key, by_content = set_key(result)
        tier = result.images[0].tier.value if result.images else "mock"
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = StageObservation(
                key=key,
                job_ids=[record.job_id],
                image_order=image,
                video_order=video,
                is_stub=_is_stub(result),
                keyed_by_content=by_content,
                tier=tier,
            )
            continue

        # Same bytes, already counted. The only thing left to check is whether the
        # ranking reproduced; if it did not, the scorer is nondeterministic.
        for stage, was, now in (
            ("image-stage", existing.image_order, image),
            ("video-stage", existing.video_order, video),
        ):
            if was != now:
                conflicts.append(
                    Conflict(
                        key=key,
                        job_a=existing.job_ids[0],
                        job_b=record.job_id,
                        order_a=was,
                        order_b=now,
                        stage=stage,
                    )
                )
        existing.job_ids.append(record.job_id)

    return Harvest(
        observations=[by_key[k] for k in sorted(by_key)],
        conflicts=conflicts,
        skipped=skipped,
        n_records=n_records,
        n_completed=n_completed,
    )


# --- The claim that is still missing ----------------------------------------


@dataclass
class HumanStageAgreement:
    """Image-stage prediction against video-stage *human* preference.

    Separate from :class:`Harvest` because it needs data the project does not have
    yet: pairwise judgements over generated videos.  The type exists so the report
    has somewhere to say "pending" with a reason attached, rather than leaving the
    headline row blank and letting a reader assume it was measured and omitted.
    """

    agreement: StageAgreement | None
    n_video_judgements: int
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.agreement is not None

    def summary(self) -> str:
        if self.agreement is None:
            return f"PENDING — {self.reason}"
        return self.agreement.summary()


def human_stage_agreement(
    image_order: dict[str, Sequence[str]],
    human_video_order: dict[str, Sequence[str]],
    *,
    n_video_judgements: int,
) -> HumanStageAgreement:
    """The headline claim, when there are video judgements to compute it from.

    Requires at least two sets: a correlation over one set has no interval and
    would be quoted as though it did.
    """
    usable = {k: v for k, v in human_video_order.items() if k in image_order}
    if n_video_judgements == 0:
        return HumanStageAgreement(
            agreement=None,
            n_video_judgements=0,
            reason=(
                "no video-stage pairwise judgements exist. The video corpus is empty "
                "and no annotator has ranked a generated clip, so the image->video "
                "claim cannot be measured against human preference at all"
            ),
        )
    if len(usable) < 2:
        return HumanStageAgreement(
            agreement=None,
            n_video_judgements=n_video_judgements,
            reason=(
                f"{n_video_judgements} video judgement(s) cover {len(usable)} set(s) "
                "with a predicted ordering; two are needed before the agreement has "
                "an interval"
            ),
        )
    return HumanStageAgreement(
        agreement=stage_agreement(
            {k: image_order[k] for k in usable}, {k: list(v) for k, v in usable.items()}
        ),
        n_video_judgements=n_video_judgements,
    )
