"""Replay the golden demo set offline, and report anything that changed.

    # what has been frozen
    .venv/bin/python scripts/replay_golden.py --list

    # check the media is all there, without running anything
    .venv/bin/python scripts/replay_golden.py --verify --all

    # replay one bundle and diff it against the outcome that was frozen
    .venv/bin/python scripts/replay_golden.py aurora-reels

    # every bundle: the regression suite for the whole pipeline
    .venv/bin/python scripts/replay_golden.py --all

    # having read the drift and decided it is correct, adopt the new outcome
    .venv/bin/python scripts/replay_golden.py aurora-reels --accept

A replay runs the real pipeline — intake, cutout, palette, quality gate, both
ranking stages, video decoding, reframing, previews, the bundle — against frozen
generations.  So this is two things at once: the demo that gets shown, and a
regression test over almost every stage the project has.  It exits non-zero when a
replay disagrees with what was frozen, which is the point: the alternative is
finding out during the viva.

``--accept`` exists because some drift is correct.  When the ranker is retrained
the scores *should* move, and re-freezing to record that would mean paying for the
generations again — the media is the expensive part and it has not changed.
``--accept`` rewrites only the expectation block, leaves every asset alone, and
costs nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("schema", "providers", "ml"):
    sys.path.insert(0, str(ROOT / "packages" / package))
sys.path.insert(0, str(ROOT / "services" / "api"))
sys.path.insert(0, str(ROOT / "services" / "worker"))

import adproviders as P  # noqa: E402
from adschema import StageEvent  # noqa: E402
from adworker import Pipeline  # noqa: E402


def _bundles(settings, args) -> list[P.GoldenBundle]:
    if args.all:
        return P.list_bundles(settings.golden_root)
    return [P.GoldenBundle.load(P.golden_bundle_path(settings, slug)) for slug in args.slugs]


async def _replay_one(bundle: P.GoldenBundle, settings, storage, ledger, *, verbose: bool):
    session = P.GoldenSession(bundle, storage)
    job_id = uuid.uuid4().hex[:16]
    request = bundle.materialise_request(job_id, storage)

    before = await ledger.total_spent()
    pipeline = Pipeline(
        storage=storage,
        # Cache off: the bundle is the cache. With it on, every fingerprint hits
        # (the frozen uploads hash to what they hashed to at freeze time) and the
        # replay reads nothing at all — see CostGovernor.use_cache.
        governor=P.CostGovernor(ledger, settings, use_cache=False),
        image_provider=session.image_provider(),
        video_provider=session.video_provider(),
        llm_provider=session.llm_provider(),
        on_progress=_printer() if verbose else None,
    )
    record = await pipeline.run(request)
    spent = await ledger.total_spent() - before

    drift = P.compare_replay(bundle, record, spent_usd=spent, notes=session.notes + session.audit())
    return drift, record


def _printer():
    async def on_progress(event: StageEvent) -> None:
        if event.state in {"failed", "warning"} or event.progress >= 1.0:
            mark = {"failed": "✗", "warning": "!"}.get(event.state, "·")
            print(f"    {mark} {event.stage.value:<12} {event.message}")

    return on_progress


def _report(drift: P.GoldenDrift) -> None:
    print(f"  {drift.summary()}")
    for label, lines in (
        ("SPEND", drift.spend),
        ("RESULT", drift.ordering),
        ("SERVING", drift.serving),
        ("SCORES", drift.numeric),
    ):
        for line in lines:
            print(f"    {label:<8} {line}")


async def _run(args: argparse.Namespace) -> int:
    settings = P.get_settings()
    storage = P.get_storage(settings.storage_backend, settings.storage_root)

    if args.list:
        bundles = P.list_bundles(settings.golden_root)
        if not bundles:
            print(
                "no golden bundles yet. Freeze one with:\n"
                "  .venv/bin/python scripts/freeze_golden.py --slug <name>"
            )
            return 0
        print(f"{len(bundles)} golden bundle(s) under {settings.golden_root}/golden:")
        for bundle in bundles:
            print(f"  {bundle.summary()}")
            problems = bundle.verify(storage)
            for problem in problems:
                print(f"    PROBLEM {problem}")
        return 0

    if not args.all and not args.slugs:
        print("name a bundle, or pass --all. `--list` shows what exists.")
        return 2

    bundles = _bundles(settings, args)
    if not bundles:
        print("no bundles matched.")
        return 2

    # Verification is separate from replay because it answers a different question:
    # "is the media intact" is worth being able to ask on a machine that has just
    # unpacked a zip, without waiting for eight pipelines to run.
    failed = 0
    if args.verify:
        for bundle in bundles:
            problems = bundle.verify(storage)
            print(f"  {bundle.slug}: {'ok' if not problems else 'INCOMPLETE'}")
            for problem in problems:
                print(f"    {problem}")
            failed += bool(problems)
        return 1 if failed else 0

    ledger = P.SqlLedger.from_url(settings.database_url)
    await ledger.create_all()
    total_before = await ledger.total_spent()

    drifted: list[tuple[P.GoldenBundle, P.GoldenDrift, object]] = []
    for bundle in bundles:
        print(f"\n{bundle.slug}  ({bundle.title or 'untitled'}, frozen {bundle.created_at})")
        problems = bundle.verify(storage)
        if problems:
            print("  cannot replay — the bundle's media is incomplete:")
            for problem in problems:
                print(f"    {problem}")
            failed += 1
            continue
        drift, record = await _replay_one(bundle, settings, storage, ledger, verbose=args.verbose)
        _report(drift)
        if not drift.ok:
            drifted.append((bundle, drift, record))

    spent = await ledger.total_spent() - total_before
    await ledger.engine.dispose()

    print()
    print(f"replayed {len(bundles) - failed}/{len(bundles)} bundle(s), spent ${spent:.4f}")

    if args.accept and drifted:
        # Rewrites the expectation block only. Every asset is left alone, so
        # adopting a new outcome costs nothing and does not change a single pixel of
        # what the demo shows.
        print()
        for bundle, drift, record in drifted:
            if drift.spend:
                print(f"  refusing to accept {bundle.slug}: a replay that spent money is a bug")
                continue
            bundle.expectation = P.GoldenExpectation.from_record(record)  # type: ignore[arg-type]
            path = bundle.save(settings.golden_root)
            print(f"  accepted {bundle.slug}: expectation rewritten in {path}")
        print("  the frozen media was not touched. Commit the manifest.")
        return 1 if failed else 0

    if failed:
        return 1
    if drifted:
        print(
            "\nThe replays above disagree with what was frozen. Read the lines, then "
            "either fix the regression or, if the new outcome is the correct one, "
            "adopt it with --accept."
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slugs", nargs="*", help="bundles to replay")
    parser.add_argument("--all", action="store_true", help="replay every frozen bundle")
    parser.add_argument("--list", action="store_true", help="show what has been frozen")
    parser.add_argument(
        "--verify", action="store_true", help="hash-check the media, replay nothing"
    )
    parser.add_argument(
        "--accept",
        action="store_true",
        help="adopt the replayed outcome as the new expectation, without re-freezing media",
    )
    parser.add_argument("--verbose", action="store_true", help="print each stage as it runs")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
