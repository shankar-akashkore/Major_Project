"""Are the collected labels good enough to train on?

    .venv/bin/python scripts/annotation_report.py

Six sections, in the order the questions actually need answering:

1. **Coverage** — how much of the design has been judged, and is the *observed*
   graph connected?  A connected design says nothing if half the pairs are still
   unjudged: Bradley-Terry is fit on what was collected, not on what was planned.
2. **Per-annotator reliability** — test-retest consistency, catch accuracy, side
   bias, response latency.
3. **Observed distributions for the thresholds** — the numbers actually seen, so
   the provisional constants in ``adschema.annotation`` can be replaced with
   measurements. The intake thresholds were set once from unrepresentative
   fixtures and had to be walked back; this section exists so that does not repeat.
4. **Inter-annotator agreement** — Krippendorff's alpha over pairs judged more
   than once, and cohort-wise agreement.
5. **Split-half reliability** — fit Bradley-Terry on half the annotators, fit on
   the other half, correlate. This is the single strongest check that the labels
   carry signal, and it needs no ground truth.
6. **The fit itself** — convergence, and the strongest and weakest items so the
   ranking can be eyeballed against the images.

Nothing here excludes anybody.  It prints who is flagged and why; dropping an
annotator's work is a decision to make deliberately and record in the write-up.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("schema", "providers", "ml"):
    sys.path.insert(0, str(ROOT / "packages" / package))
sys.path.insert(0, str(ROOT / "services" / "api"))

import adproviders as P  # noqa: E402
from adapi.annotation_store import AnnotationStore  # noqa: E402
from adml import pairs as PAIRS  # noqa: E402
from adml import ranking as R  # noqa: E402
from adschema import (  # noqa: E402
    MIN_CATCH_TRIALS,
    MIN_PLAUSIBLE_LATENCY_MS,
    Choice,
    ItemKind,
    PairKind,
)

RULE = "─" * 78


def _fmt(value: float | None, spec: str = ".3f", blank: str = "n/a") -> str:
    if value is None:
        return blank
    if value != value:  # NaN
        return blank
    return format(value, spec)


async def _run(args: argparse.Namespace) -> int:
    settings = P.Settings()
    store = AnnotationStore.from_url(settings.database_url)
    await store.create_all()

    stats = await store.stats()
    items = await store.list_items(kind=ItemKind.IMAGE)
    all_pairs = await store.list_pairs()
    judgements = await store.list_judgements()

    if not judgements:
        print("No judgements yet. Start the API and share /api/annotate/ui .")
        print(f"Corpus is ready: {stats.n_items} items, {stats.n_pairs} pairs.")
        return 0

    pairs_by_id = {p.pair_id: p for p in all_pairs}
    item_ids = [i.item_id for i in items]

    # --- 1. Coverage and connectivity -------------------------------------
    print(RULE)
    print("1. COVERAGE")
    print(RULE)
    judged_ids = {j.pair_id for j in judgements}
    observed = [pairs_by_id[pid] for pid in judged_ids if pid in pairs_by_id]
    designed = [p for p in all_pairs if p.kind is not PairKind.CATCH]

    # Coverage is measured over the *design*, excluding catch pairs. Including them
    # would flatter the number: catch trials are served on a schedule and are always
    # judged, so they would push coverage up without a single extra real comparison.
    judged_design = sum(1 for p in observed if p.kind is not PairKind.CATCH)
    coverage = judged_design / len(designed) if designed else 0.0
    print(f"items                 : {stats.n_items} in {stats.n_sets} sets")
    print(f"pairs designed        : {len(designed)} (+{stats.n_decoys} decoys as catch trials)")
    print(f"pairs judged at least once: {judged_design} ({coverage:.0%} of the design)")
    print(f"judgements            : {stats.n_judgements} from {stats.n_annotators} annotators")

    design_components = PAIRS.connected_components(designed, include=item_ids)
    observed_components = PAIRS.connected_components(observed, include=item_ids)
    print(f"designed graph        : {len(design_components)} component(s)")
    print(f"OBSERVED graph        : {len(observed_components)} component(s)", end="")
    if len(observed_components) > 1:
        sizes = ", ".join(str(len(c)) for c in observed_components[:6])
        print(f"  <-- sizes {sizes}{' …' if len(observed_components) > 6 else ''}")
        print(
            "    Strengths are NOT comparable across components. Keep collecting, or\n"
            "    restrict the analysis to the largest component and say so."
        )
    else:
        print("  ✓ every item's strength is comparable with every other's")

    per_item: dict[str, int] = defaultdict(int)
    for pair in observed:
        n = sum(1 for j in judgements if j.pair_id == pair.pair_id)
        per_item[pair.item_a] += n
        per_item[pair.item_b] += n
    covered = [per_item.get(i, 0) for i in item_ids]
    if covered:
        print(
            f"judgements per item   : min {min(covered)} / median "
            f"{statistics.median(covered):.0f} / max {max(covered)}"
        )
        zero = sum(1 for c in covered if c == 0)
        if zero:
            print(f"    {zero} item(s) have not been compared at all")

    # --- 2. Per-annotator reliability -------------------------------------
    print()
    print(RULE)
    print("2. ANNOTATOR RELIABILITY")
    print(RULE)
    quality = await store.annotator_quality()
    print(
        f"{'annotator':<16}{'n':>5}{'retest':>8}{'catch':>10}{'p(guess)':>10}"
        f"{'left%':>7}{'z':>7}{'median ms':>11}"
    )
    for q in quality:
        if q.n_judgements == 0:
            continue
        print(
            f"{(q.label or q.annotator_id)[:15]:<16}"
            f"{q.n_judgements:>5}"
            f"{_fmt(q.repeat_consistency, '.0%', '  —'):>8}"
            f"{(f'{q.catch_correct}/{q.n_catch}' if q.n_catch else '  —'):>10}"
            f"{_fmt(q.catch_guess_probability, '.2f', '  —'):>10}"
            f"{_fmt(q.side_bias, '.0%', '  —'):>7}"
            f"{_fmt(q.side_bias_z, '+.1f', ' —'):>7}"
            f"{q.median_latency_ms:>11}"
        )
    print(f"  MIN_CATCH_TRIALS is {MIN_CATCH_TRIALS}; p(guess) is only acted on at or above that.")
    flagged = [q for q in quality if q.flags and q.n_judgements > 0]
    if flagged:
        print("\nflagged:")
        for q in flagged:
            print(f"  {q.label or q.annotator_id}: {'; '.join(q.flags)}")
    else:
        print("\nno annotator crossed a threshold")

    # --- 3. Observed distributions ----------------------------------------
    print()
    print(RULE)
    print("3. WHAT THE THRESHOLDS SHOULD BE (observed, not assumed)")
    print(RULE)
    latencies = sorted(j.latency_ms for j in judgements if j.latency_ms > 0)
    if latencies:

        def pct(p: float) -> int:
            return latencies[min(len(latencies) - 1, int(p * len(latencies)))]

        print(
            f"latency ms            : p5 {pct(0.05)} / p25 {pct(0.25)} / median {pct(0.5)} "
            f"/ p75 {pct(0.75)} / p95 {pct(0.95)}"
        )
        under = sum(1 for latency in latencies if latency < MIN_PLAUSIBLE_LATENCY_MS)
        print(
            f"    MIN_PLAUSIBLE_LATENCY_MS is {MIN_PLAUSIBLE_LATENCY_MS}; "
            f"{under}/{len(latencies)} ({under / len(latencies):.0%}) fall under it"
        )
        print(
            f"    seconds per comparison, observed median: {pct(0.5) / 1000:.1f}s "
            f"(adml.pairs assumes {PAIRS.ASSUMED_SECONDS_PER_PAIR:.1f}s for planning)"
        )
    retests = [q.repeat_consistency for q in quality if q.repeat_consistency is not None]
    if retests:
        print(
            f"retest consistency    : min {min(retests):.0%} / median "
            f"{statistics.median(retests):.0%} / max {max(retests):.0%} "
            f"over {len(retests)} annotator(s)"
        )
        print("    Set MIN_REPEAT_CONSISTENCY below the honest floor of this spread,")
        print("    not at a round number that happens to look strict.")
    ties = sum(1 for j in judgements if j.choice is Choice.TIE)
    print(f"tie rate              : {ties}/{len(judgements)} ({ties / len(judgements):.0%})")

    # --- 4. Inter-annotator agreement -------------------------------------
    print()
    print(RULE)
    print("4. INTER-ANNOTATOR AGREEMENT")
    print(RULE)
    # One unit per pair, one label per annotator: the winning item id, or "tie".
    # Only the first showing counts — a repeat is the same person, and including it
    # would inflate agreement with an annotator's agreement with themselves.
    units: dict[str, list[str]] = defaultdict(list)
    for j in judgements:
        pair = pairs_by_id.get(j.pair_id)
        if pair is None or pair.kind is PairKind.CATCH or j.showing != 0:
            continue
        units[j.pair_id].append(j.winner(pair) or "tie")
    multi = [labels for labels in units.values() if len(labels) >= 2]
    print(f"pairs judged by 2+ annotators: {len(multi)}")
    if len(multi) < 20:
        print(
            "    Too few for a stable agreement figure. Agreement needs deliberate\n"
            "    overlap: run build_corpus with --judgements-per-pair 2, or keep\n"
            "    collecting until the coverage-first policy starts doubling up."
        )
    if multi:
        alpha = R.krippendorff_alpha_nominal(multi)
        raw = statistics.mean(
            sum(1 for x in labels if x == labels[0]) / len(labels) for labels in multi
        )
        print(f"Krippendorff alpha    : {_fmt(alpha)}")
        print(f"raw agreement         : {raw:.3f}")
        print(
            "    On a two-alternative task, 0.50 raw agreement is what coin-flipping\n"
            "    produces, which is why alpha is the number to quote."
        )

    # --- 5. Split-half reliability ----------------------------------------
    print()
    print(RULE)
    print("5. SPLIT-HALF RELIABILITY")
    print(RULE)
    wins, tie_obs = await store.observations()
    split = R.split_half_reliability(wins, seed=args.seed)
    print(f"annotators split      : {split.left_annotators} vs {split.right_annotators}")
    print(f"comparisons           : {split.n_left} vs {split.n_right}")
    print(f"items in both halves  : {split.n_shared_items}")
    print(f"Spearman rho          : {_fmt(split.rho)}")
    if split.n_shared_items < 10:
        print("    Not enough shared items yet for this to mean anything.")
    elif split.rho == split.rho and split.rho < 0.3:
        print(
            "    Two halves of the pool disagree about the ranking. Investigate before\n"
            "    training: either the task is ambiguous, or some annotators are noise."
        )

    # --- 6. The fit -------------------------------------------------------
    print()
    print(RULE)
    print("6. BRADLEY-TERRY FIT")
    print(RULE)
    excluded = {q.annotator_id for q in quality if q.flags and args.exclude_flagged}
    if excluded:
        print(f"excluding {len(excluded)} flagged annotator(s) (--exclude-flagged)")
        wins, tie_obs = await store.observations(exclude_annotators=excluded)
    fit = R.fit_bradley_terry(
        [(w, x) for _, w, x in wins],
        ties=[(a, b) for _, a, b in tie_obs],
        items=item_ids,
    )
    print(f"items                 : {fit.n_items}")
    print(f"comparisons           : {fit.n_comparisons:.0f} ({len(tie_obs)} ties, half-weighted)")
    print(f"converged             : {fit.converged} in {fit.iterations} iterations")
    print(f"graph connected       : {fit.is_connected}")
    accuracy = R.pairwise_accuracy(fit.strengths, [(w, x) for _, w, x in wins])
    print(f"in-sample pairwise acc: {_fmt(accuracy)}  (fit on this data; not a result)")

    order = fit.ranking()
    labels = {i.item_id: i for i in items}
    print("\nstrongest:")
    for item_id in order[: args.show]:
        item = labels.get(item_id)
        axes = "/".join(item.axes() or ()) if item else ""
        print(f"  {fit.strengths[item_id]:+.3f}  {item_id}  {axes}")
    print("weakest:")
    for item_id in order[-args.show :]:
        item = labels.get(item_id)
        axes = "/".join(item.axes() or ()) if item else ""
        print(f"  {fit.strengths[item_id]:+.3f}  {item_id}  {axes}")

    await store.engine.dispose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", type=int, default=5, help="items to list at each end")
    parser.add_argument("--seed", type=int, default=0, help="seed for the annotator split")
    parser.add_argument(
        "--exclude-flagged",
        action="store_true",
        help="refit without annotators that crossed a quality threshold",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
