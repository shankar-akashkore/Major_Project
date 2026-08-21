"""Generate the evaluation write-up: tables, figures, and what is still missing.

    # everything the project can currently support
    .venv/bin/python scripts/report.py

    # with the deterministic stand-in embeddings, to exercise the whole path
    .venv/bin/python scripts/report.py --hash-embeddings

    # just tell me whether the numbers are quotable yet
    .venv/bin/python scripts/report.py --check

Writes ``docs/results/``: one markdown document, the same tables as LaTeX floats,
and the figures as SVG.  Nothing in there should ever be edited — it is regenerated,
and an edit would be silently reverted on the next run.

Why generate a report at all rather than write one.  Every number in a write-up
normally makes one unrecorded journey, out of a terminal and into a document, and
that journey is where a figure measured on simulated labels becomes a figure
presented as measured.  It is also where numbers go stale: the model is retrained
and the table is not.  Here the tables and the figures come from the same objects
the evaluation harness produced, in one command, so they cannot disagree with each
other or with the code.

What this deliberately does not write is the argument.  Interpretation, related
work and the discussion are the author's; this produces the evidence they refer to,
and the list of things that have to happen before the evidence means anything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("schema", "providers", "ml"):
    sys.path.insert(0, str(ROOT / "packages" / package))
sys.path.insert(0, str(ROOT / "services" / "api"))

import adproviders as P  # noqa: E402
from adml import figures as FIG  # noqa: E402
from adml import report as RP  # noqa: E402
from adml import stages as ST  # noqa: E402
from adschema import DEFAULT_CANDIDATE_COUNT, DEFAULT_VIDEO_COUNT  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "results"
COMMAND = "scripts/report.py"

#: Corpus sizes and the pooled 95% half-width each one buys, from the simulation in
#: section 6 of docs/prediction-protocol.md. Recorded here rather than recomputed so
#: the figure that justifies the $12 generation run does not need a training run to
#: exist first.
SAMPLE_SIZE_CURVE = [(12, 0.104), (24, 0.074), (48, 0.052), (100, 0.040), (200, 0.028)]
#: Below this half-width the model differences that matter become resolvable.
RESOLVABLE_WIDTH = 0.045
#: What the corpus needs to reach. Measured, not chosen: see the same section.
TARGET_SETS = 100


def _git_revision() -> str:
    """Which commit produced these numbers.

    Worth the subprocess: a table without a revision cannot be reproduced, and the
    first question about a surprising number is what the code was doing at the time.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# --- Section 1: the headline claim -------------------------------------------


async def _stage_section(report: RP.Report, out: Path) -> ST.Harvest:
    from adapi.store import JobStore

    settings = P.get_settings()
    store = JobStore.from_url(settings.database_url)
    await store.create_all()
    records = await store.list_recent(limit=1000)
    await store.engine.dispose()

    harvest = ST.harvest(records)
    human = ST.human_stage_agreement(
        {o.key: o.image_order for o in harvest.observations}, {}, n_video_judgements=0
    )

    section = report.section(
        "The headline claim: does the image-stage ranking predict the video-stage one?",
        "The project's claim is that ranking three candidates as *images* anticipates how "
        "they rank once animated, which is what makes early prediction worth anything: "
        "the video stage is where the money goes. The claim has two halves and only one "
        "of them is computable today.",
    )

    stamp = RP.Stamp(
        n=harvest.n_sets,
        unit="generation sets",
        n_sets=harvest.n_sets,
        labels_are_real=False,
        scorer="stub heuristic" if harvest.all_stub else "trained head",
        note=f"{harvest.n_collapsed} replayed record(s) collapsed by content",
    )
    table = RP.Table(
        key="stage-agreement",
        caption="Image-stage to video-stage rank agreement",
        columns=[
            RP.Column("comparison", "Comparison"),
            RP.Column("rho", "Mean Spearman ρ [95% CI]", "right"),
            RP.Column("tau", "Mean Kendall τ", "right"),
            RP.Column("top1", "Top-1 retention [95% CI]", "right"),
            RP.Column("sets", "Sets", "right"),
        ],
        stamp=stamp,
    )

    # The claim itself, which needs human preference over the finished videos.
    table.pending_row("image-stage prediction vs video-stage human preference", human.reason)

    # The diagnostic, which does not.
    if harvest.observations and harvest.quotable():
        agreement = harvest.model_vs_model()
        table.add(
            comparison="image-stage prediction vs video-stage prediction (diagnostic)",
            rho=RP.Value(agreement.mean_spearman, agreement.spearman_ci, signed=True),
            tau=RP.Value(agreement.mean_kendall, signed=True),
            top1=RP.Value(agreement.top1_retention, agreement.top1_ci),
            sets=agreement.n_sets,
        )
    else:
        table.pending_row(
            "image-stage prediction vs video-stage prediction (diagnostic)",
            f"{harvest.n_sets} distinct generation set(s) on disk; "
            f"{ST.MIN_QUOTABLE_SETS} are needed before a correlation has an interval. "
            "Run more jobs end to end",
        )
    section.tables.append(table)

    shifts = harvest.rank_shifts()
    pairs = []
    for obs in harvest.observations:
        video_rank = {item: i for i, item in enumerate(obs.video_order)}
        for image_rank, item in enumerate(obs.image_order):
            if item in video_rank:
                pairs.append((image_rank, video_rank[item]))

    svg = FIG.stage_scatter(
        pairs,
        provenance=FIG.Provenance(
            n=harvest.n_sets,
            unit="generation sets",
            labels_are_real=False,
            note="stub-scored" if harvest.all_stub else "",
        ),
        rho=harvest.model_vs_model().mean_spearman if harvest.quotable() else None,
    )
    path = _write_figure(out, "stage-agreement.svg", svg)
    section.figures.append((f"figures/{path.name}", "Image-stage rank against video-stage rank"))

    if shifts:
        section.body += (
            "\n\nRank movement between the stages, over "
            f"{sum(shifts.values())} candidate placements: "
            + ", ".join(f"{shift:+d} → {count}" for shift, count in shifts.items())
            + "."
        )
    return harvest


# --- Section 2: the predictor -------------------------------------------------


def _predictor_section(report: RP.Report, out: Path, *, trained: dict | None) -> None:
    section = report.section(
        "The predictor against its baselines",
        "Every accuracy is pooled over set-wise cross-validation folds, compared against "
        "the baselines with a paired test on the same held-out comparisons, and read "
        "against the annotator noise ceiling rather than against 1.0. The reasoning for "
        "all three is in `docs/prediction-protocol.md`.",
    )

    if trained is None:
        stamp = RP.Stamp(n=0, unit="comparisons", labels_are_real=False)
        table = RP.Table(
            key="models",
            caption="Held-out pairwise accuracy against baselines",
            columns=[
                RP.Column("model", "Model"),
                RP.Column("accuracy", "Accuracy [95% CI]", "right"),
                RP.Column("of_ceiling", "% of ceiling", "right"),
                RP.Column("n", "n", "right"),
                RP.Column("delta", "Δ vs trained [95% CI]", "right"),
            ],
            stamp=stamp,
        )
        reason = (
            "no human pairwise judgements exist. The annotation tool is built and "
            "verified against a simulated session, and no annotator has used it, so "
            "there is nothing to train or evaluate on. This is the project's critical "
            "path: collect judgements at /annotate/ui"
        )
        for name in (
            "pairwise (trained)",
            "BT-target ridge",
            "salience-only",
            "aesthetic-only",
            "LLM-as-judge",
            "random",
        ):
            table.pending_row(name, reason)
        section.tables.append(table)

        ablation = RP.Table(
            key="ablation",
            caption="Feature-group ablation, each row paired against the full model",
            columns=[
                RP.Column("variant", "Variant"),
                RP.Column("cols", "Columns", "right"),
                RP.Column("accuracy", "Accuracy [95% CI]", "right"),
                RP.Column("delta", "Δ vs full [95% CI]", "right"),
                RP.Column("verdict", "Verdict"),
            ],
            stamp=stamp,
        )
        ablation.pending_row("every feature group", reason)
        ablation.pending_row(
            "embedding, aesthetic, identity groups",
            "additionally needs notebooks/colab_embeddings.ipynb run on a GPU; no GPU has "
            "executed it, so these rows are pending rather than zero",
        )
        section.tables.append(ablation)

        section.body += (
            "\n\nThe rows above are named rather than omitted. A reader who cannot see that "
            "LLM-as-judge is one of the planned baselines cannot tell whether it was tried "
            "and lost or never run."
        )
        return

    section.tables.append(trained["models"])
    section.tables.append(trained["ablation"])
    section.figures.append(("figures/models.svg", "Held-out pairwise accuracy"))
    section.figures.append(("figures/ablation.svg", "Cost of removing each feature group"))


# --- Section 3: what the corpus needs ----------------------------------------


async def _total_spent(url: str) -> float:
    """What the ledger says has been spent, read rather than asserted.

    This used to be the literal string ``"$0.0000"`` with the note "no API call has
    been made", which was true for as long as it was true and became a false claim
    in the project's own write-up the moment the first contract smoke test ran. A
    figure that cannot change is not a measurement.
    """
    ledger = P.SqlLedger.from_url(url)
    try:
        return await ledger.total_spent()
    finally:
        await ledger.engine.dispose()


async def _corpus_sets(url: str) -> int:
    """How many generation sets the annotation corpus actually holds.

    Read rather than assumed: the whole point of the curve below is to say where the
    corpus is against where it needs to be, and a hardcoded "now" would keep saying
    24 after the generation run had moved it.
    """
    from adapi.annotation_store import AnnotationStore

    store = AnnotationStore.from_url(url)
    await store.create_all()
    try:
        items = await store.list_items(include_decoys=False)
        return len({i.set_id for i in items})
    finally:
        await store.engine.dispose()


def _corpus_section(report: RP.Report, out: Path, *, n_sets_now: int) -> None:
    section = report.section(
        "Why the corpus has to grow",
        "The interval on a pairwise accuracy is set by how many held-out comparisons "
        "survive an honest set-wise split, and at the current corpus size it is wider "
        "than every difference between models — so nothing about the predictor is "
        "resolvable yet. This is the measurement behind treating the generation run as "
        "necessary rather than optional.",
    )

    stamp = RP.Stamp(
        n=len(SAMPLE_SIZE_CURVE),
        unit="simulated corpus sizes",
        labels_are_real=False,
        note=(
            f"simulation from docs/prediction-protocol.md §6, not an observation; "
            f"the corpus holds {n_sets_now} set(s) today"
        ),
    )
    table = RP.Table(
        key="sample-size",
        caption="Pooled 95% half-width against corpus size",
        columns=[
            RP.Column("sets", "Generation sets", "right"),
            RP.Column("width", "95% half-width", "right"),
            RP.Column("verdict", "Resolvable?"),
        ],
        stamp=stamp,
    )
    for sets, width in SAMPLE_SIZE_CURVE:
        table.add(
            sets=sets,
            width=RP.Value(width),
            verdict="yes" if width <= RESOLVABLE_WIDTH else "no",
        )
    section.tables.append(table)

    marks = [(TARGET_SETS, "needed")]
    if any(n == n_sets_now for n, _ in SAMPLE_SIZE_CURVE):
        marks.insert(0, (n_sets_now, "now"))
    svg = FIG.sample_size_curve(
        SAMPLE_SIZE_CURVE,
        provenance=FIG.Provenance(
            n=len(SAMPLE_SIZE_CURVE), unit="simulated corpus sizes", labels_are_real=False
        ),
        marks=marks,
        target=RESOLVABLE_WIDTH,
    )
    path = _write_figure(out, "sample-size.svg", svg)
    section.figures.append((f"figures/{path.name}", "Interval width against corpus size"))


# --- Section 4: the product side ---------------------------------------------


def _delivery_section(report: RP.Report, harvest: ST.Harvest, spent_usd: float) -> None:
    section = report.section(
        "The working product",
        # Counts come from the schema rather than from this sentence. They were
        # written out as "three" and "three" here, and stayed that way through the
        # change to five candidates and two videos — a report describing a pipeline
        # the project no longer runs.
        "The pipeline runs end to end: intake and product cutout, a sampled design "
        f"space expanded into shot briefs, {DEFAULT_CANDIDATE_COUNT} image candidates, "
        "a hard quality gate, the image-stage ranking, which promotes the top "
        f"{DEFAULT_VIDEO_COUNT} to 8–10 s video, the video-stage ranking with "
        "explanations, and delivery — saliency-aware reframes per platform, mockup "
        "previews, an ffmpeg audio mix carrying a licence or a synthesised bed that "
        "needs none, report cards and a download bundle.",
    )

    stamp = RP.Stamp(
        n=harvest.n_completed,
        unit="completed job records",
        n_sets=harvest.n_sets,
        labels_are_real=False,
        note=(
            f"{harvest.n_premium_sets} premium set(s) among them"
            if harvest.n_premium_sets
            else "mock and replay tiers only; no premium generation has been paid for"
        ),
    )
    table = RP.Table(
        key="delivery",
        caption="What has actually run",
        columns=[
            RP.Column("item", "Item"),
            RP.Column("value", "Value", "right"),
            RP.Column("note", "Note"),
        ],
        stamp=stamp,
    )
    table.add(item="Job records on disk", value=harvest.n_records, note="all tiers")
    table.add(item="Completed jobs", value=harvest.n_completed, note="reached delivery")
    table.add(
        item="Distinct generation sets",
        value=harvest.n_sets,
        note=f"{harvest.n_collapsed} replay(s) collapsed by candidate content",
    )
    table.add(
        item="Total spend",
        value=f"${spent_usd:.4f}",
        note=(
            f"{harvest.n_premium_sets} premium job(s) plus the contract smoke "
            "tests; mock mode remains the default"
            if harvest.n_premium_sets
            else "contract smoke tests only; mock mode remains the default"
            if spent_usd > 0
            else "no API call has been made; mock mode is the default"
        ),
    )
    # Three states, not two. Keying this off `spent_usd > 0` could not tell a
    # $0.39 pair of smoke tests from a $2.22 job that ran the entire pipeline, so
    # it went on claiming no full job existed after one had been paid for. The
    # tier is recorded per candidate; the count is read rather than inferred.
    if harvest.n_premium_sets:
        table.add(
            item="Premium-tier jobs",
            value=harvest.n_premium_sets,
            note="paid generation on the live API; the golden set is still frozen "
            "from synthetic references and should be re-frozen from one of these",
        )
    else:
        table.pending_row(
            "Premium-tier jobs",
            "the two fal contract smoke tests have been run and both adapters are confirmed "
            "against the live API (see docs/provider-spike.md), but no full job has been "
            "generated, so the golden set is still frozen from synthetic references"
            if spent_usd > 0
            else "the two fal contract smoke tests ($0.39) have not been run and no key has "
            "been supplied, so no paid generation exists and the golden set is frozen "
            "from synthetic references",
        )
    table.pending_row(
        "Research-tier clips",
        "notebooks/colab_video.ipynb has not been run on a GPU, so the free clip corpus "
        "is empty and the tier comparison has no data",
    )
    section.tables.append(table)


# --- Writing -----------------------------------------------------------------


def _write_figure(out: Path, name: str, svg: str) -> Path:
    directory = out / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(svg)
    return path


async def _run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    revision = _git_revision()
    report = RP.Report(
        title="Results: multi-candidate ad generation with learned ranking",
        command=f"{COMMAND} (at {revision})",
    )

    settings = P.get_settings()
    harvest = await _stage_section(report, out)
    _predictor_section(report, out, trained=None)
    _corpus_section(report, out, n_sets_now=await _corpus_sets(settings.database_url))
    _delivery_section(report, harvest, await _total_spent(settings.database_url))

    if args.check:
        print(f"tables: {len(report.tables)}   quotable as a result: {report.is_a_result}")
        for reason in report.outstanding:
            print(f"  PENDING {reason}")
        return 0 if report.is_a_result else 1

    out.mkdir(parents=True, exist_ok=True)
    markdown = out / "README.md"
    markdown.write_text(report.to_markdown())
    latex = out / "tables.tex"
    latex.write_text(report.to_latex_tables())
    payload = out / "results.json"
    payload.write_text(
        json.dumps(
            {
                "generated_at": report.generated_at.isoformat(),
                "revision": revision,
                "is_a_result": report.is_a_result,
                "outstanding": report.outstanding,
                "tables": [t.key for t in report.tables],
            },
            indent=2,
        )
        + "\n"
    )

    print(f"wrote {markdown}")
    print(f"      {latex}")
    print(f"      {payload}")
    print(f"      {out / 'figures'}/*.svg")
    print()
    print(f"{len(report.tables)} table(s), {len(report.outstanding)} outstanding item(s)")
    if not report.is_a_result:
        print(
            "\nThis is not a result yet. Every table says so above its own header, and\n"
            "the document opens with what has to happen first. The machinery is what is\n"
            "finished: when the labels land, the same command produces the real tables."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the artefact")
    parser.add_argument(
        "--check",
        action="store_true",
        help="print whether the numbers are quotable and exit non-zero if not",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
