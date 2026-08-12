"""Measure the project's headline claim against the jobs that have run.

    # what the finished jobs support
    .venv/bin/python scripts/stage_agreement.py

    # per-set detail, and every job that was collapsed or skipped
    .venv/bin/python scripts/stage_agreement.py --verbose

    # machine-readable, for scripts/report.py
    .venv/bin/python scripts/stage_agreement.py --json docs/results/stage-agreement.json

The claim is that ranking three candidates *as images* anticipates how they rank
once animated — which is what makes early prediction worth anything, because the
video stage is where the money goes.

Two things this script exists to stop.

**Counting replays as observations.**  Every golden replay writes a fresh job
record over bit-identical candidates.  Keyed by job id, running the week-14 replay
suite a few times before the viva would report a dozen sets in perfect agreement
with a zero-width interval.  Sets are keyed by candidate content instead, so a
replay collapses onto the job it replays and is reported as collapsed.

**Passing model-to-model agreement off as the claim.**  Both stages are scored by
this codebase over overlapping feature groups, so they agree partly by
construction.  The real claim needs human preference over the finished videos.
This script prints the diagnostic under a name that says what it is, and prints the
claim itself as PENDING with the reason, until video judgements exist.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("schema", "providers", "ml"):
    sys.path.insert(0, str(ROOT / "packages" / package))
sys.path.insert(0, str(ROOT / "services" / "api"))

import adproviders as P  # noqa: E402
from adapi.annotation_store import AnnotationStore  # noqa: E402
from adapi.store import JobStore  # noqa: E402
from adml import stages as ST  # noqa: E402
from adschema import ItemKind  # noqa: E402


async def _video_judgement_count(url: str) -> int:
    """How many judgements exist over video items.

    Counted rather than assumed: the annotation tool can already collect them, the
    contract has ``ItemKind.VIDEO``, and the only thing missing is a video corpus.
    So this is a number that will start moving on its own once the Colab notebook
    has run, and the report should notice without being edited.
    """
    store = AnnotationStore.from_url(url)
    await store.create_all()
    try:
        video_items = {i.item_id for i in await store.list_items(kind=ItemKind.VIDEO)}
        if not video_items:
            return 0
        pairs = {p.pair_id: p for p in await store.list_pairs()}
        return sum(
            1
            for j in await store.list_judgements()
            if (pair := pairs.get(j.pair_id)) is not None
            and (pair.item_a in video_items or pair.item_b in video_items)
        )
    finally:
        await store.engine.dispose()


def _report(harvest: ST.Harvest, human: ST.HumanStageAgreement, *, verbose: bool) -> None:
    print("=" * 78)
    print("IMAGE -> VIDEO RANK AGREEMENT")
    print("=" * 78)

    print("\n1. WHAT THE JOBS SUPPORT")
    print("-" * 78)
    print("  " + harvest.summary().replace("\n", "\n  "))

    if not harvest.observations:
        print(
            "\n  Nothing to measure. Run a job end to end, or replay the golden set:\n"
            "    .venv/bin/python scripts/replay_golden.py --all"
        )
        return

    print("\n2. THE CLAIM: image-stage prediction vs video-stage human preference")
    print("-" * 78)
    print("  " + human.summary())
    if not human.available:
        print(
            "\n  This is the number the project stands on and it has no value yet.\n"
            "  It needs generated videos in the annotation corpus and annotators who\n"
            "  have ranked them:\n"
            "    1. run notebooks/colab_video.ipynb on a GPU to fill the clip corpus\n"
            "    2. .venv/bin/python scripts/build_corpus.py --kind video --commit\n"
            "    3. collect judgements at /annotate/ui"
        )

    print("\n3. DIAGNOSTIC: the two stages' own orderings, against each other")
    print("-" * 78)
    if harvest.quotable():
        print("  " + harvest.model_vs_model().summary())
    else:
        # Withheld rather than qualified. Over one set rho is exactly +-1 with no
        # interval, and "rho +1.000  over 1 sets" is read as the first half of that
        # sentence by everyone who sees it on a slide.
        print(
            f"  withheld: {harvest.n_sets} distinct set(s), and an agreement needs "
            f"{ST.MIN_QUOTABLE_SETS} before it has an interval.\n"
            f"  Over one set the correlation is +1 or -1 by arithmetic, whatever the "
            f"ranker did."
        )
    print(
        "\n  Not the claim above. Both orderings come from this codebase over\n"
        "  overlapping features, so agreement here is partly built in — it measures\n"
        "  whether the video stage reorders anything, not whether either stage is right."
    )
    if harvest.all_stub:
        print(
            "\n  !! Every set was ranked by the stub scorer. These orderings are\n"
            "     heuristic output, so the agreement above is a property of the\n"
            "     heuristic and is NOT A RESULT. Train and save a model first:\n"
            "       .venv/bin/python scripts/train_predictor.py --save"
        )
    elif harvest.n_real_sets < harvest.n_sets:
        real = harvest.model_vs_model(real_only=True)
        print(f"\n  over the {harvest.n_real_sets} trained-scorer set(s) only: {real.summary()}")

    shifts = harvest.rank_shifts()
    if shifts:
        print("\n  rank shift between stages (image rank - video rank):")
        total = sum(shifts.values())
        for shift, count in shifts.items():
            bar = "#" * max(1, round(30 * count / total))
            print(f"    {shift:+d}  {count:>4}  {bar}")
        if set(shifts) == {0}:
            print(
                "    Every candidate held its place. With this few sets that is as\n"
                "    consistent with a video stage that adds nothing as with one that\n"
                "    agrees — it does not distinguish them."
            )

    if verbose:
        print("\n4. PER SET")
        print("-" * 78)
        for obs in harvest.observations:
            mark = "stub" if obs.is_stub else "real"
            print(
                f"  {obs.key[:12]}  {mark}  tier={obs.tier}  "
                f"image {obs.image_order} -> video {obs.video_order}"
            )
            print(f"      {obs.n_replays} job(s): {', '.join(j[:12] for j in obs.job_ids)}")
            if not obs.keyed_by_content:
                print("      keyed by job id — no content digests, replays not collapsed")

    if harvest.conflicts:
        print(
            "\n  The conflicts above are a defect, not data: identical candidates\n"
            "  ranked differently means the scorer is not deterministic."
        )


def _payload(harvest: ST.Harvest, human: ST.HumanStageAgreement) -> dict:
    """What scripts/report.py consumes. Carries the caveats, not just the numbers."""
    diagnostic = harvest.model_vs_model()

    def agreement_dict(agreement) -> dict:
        return {
            "mean_spearman": agreement.mean_spearman,
            "spearman_ci": list(agreement.spearman_ci),
            "mean_kendall": agreement.mean_kendall,
            "top1_retention": agreement.top1_retention,
            "top1_ci": list(agreement.top1_ci),
            "n_sets": agreement.n_sets,
        }

    return {
        "n_records": harvest.n_records,
        "n_completed": harvest.n_completed,
        "n_sets": harvest.n_sets,
        "n_real_sets": harvest.n_real_sets,
        "n_collapsed_replays": harvest.n_collapsed,
        "all_stub": harvest.all_stub,
        "dedup_incomplete": harvest.dedup_incomplete,
        "skipped": harvest.skipped,
        "conflicts": [c.summary() for c in harvest.conflicts],
        "rank_shifts": {str(k): v for k, v in harvest.rank_shifts().items()},
        "model_vs_model": agreement_dict(diagnostic),
        # Two independent gates, both of which the report has to pass before it
        # prints a number: enough sets for an interval, and a trained scorer behind
        # the orderings. A stub agreement over fifty sets is quotable and still not
        # a result; a trained agreement over one set is a result and still not
        # quotable.
        "model_vs_model_quotable": harvest.quotable(),
        "model_vs_model_is_a_result": not harvest.all_stub,
        "min_quotable_sets": ST.MIN_QUOTABLE_SETS,
        "claim": {
            "available": human.available,
            "reason": human.reason,
            "n_video_judgements": human.n_video_judgements,
            **(agreement_dict(human.agreement) if human.agreement else {}),
        },
        "sets": [
            {
                "key": o.key,
                "job_ids": o.job_ids,
                "image_order": o.image_order,
                "video_order": o.video_order,
                "is_stub": o.is_stub,
                "tier": o.tier,
                "keyed_by_content": o.keyed_by_content,
            }
            for o in harvest.observations
        ],
    }


async def _run(args: argparse.Namespace) -> int:
    settings = P.get_settings()
    store = JobStore.from_url(settings.database_url)
    await store.create_all()
    records = await store.list_recent(limit=args.limit)
    await store.engine.dispose()

    harvest = ST.harvest(records)
    n_video = await _video_judgement_count(settings.database_url)
    human = ST.human_stage_agreement(
        {o.key: o.image_order for o in harvest.observations},
        {},  # No video judgements are mapped onto generation sets yet.
        n_video_judgements=n_video,
    )

    _report(harvest, human, verbose=args.verbose)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_payload(harvest, human), indent=2) + "\n")
        print(f"\nwrote {path}")

    print()
    # A conflict is the only thing here that indicates a broken build. Missing data
    # is the expected state and must not fail a script the report runs.
    if harvest.conflicts:
        print("FAIL: the ranker gave different orders for identical inputs.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000, help="how many records to read")
    parser.add_argument("--verbose", action="store_true", help="per-set detail")
    parser.add_argument("--json", default=None, help="also write the payload here")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
