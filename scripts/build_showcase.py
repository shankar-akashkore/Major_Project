"""Build the landing page's showcase media from a job that already ran.

    .venv/bin/python scripts/build_showcase.py --job 4dbb3eb9041e43bf

The landing page argues that the system composes five genuinely different frames,
measures them, and promotes two before spending anything on video.  Every one of
those claims is checkable against a run that actually happened, and the runs are
sitting in ``fixtures/`` — so the page shows them rather than describing them.

**The job is named on the command line, never discovered.**  This media carries a
real person's likeness and a real brand's product, and deciding which run is fit
to publish is a judgement a person makes.  A script that went looking for "the
best job" would be making it silently, and would eventually publish a watermarked
stock comp or somebody's trademarked hardware because it scored well.

Output is gitignored.  A checkout without it renders the page's abstract
treatment instead — see ``apps/web/app/home/showcase.ts`` — so nothing here is
required for the page to work, and nothing generated is ever committed.

Costs nothing and spends nothing: every byte read has already been generated and
already been paid for.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "schema"))
sys.path.insert(0, str(ROOT / "packages" / "ml"))

from adml import video as V  # noqa: E402

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"

DEFAULT_DB = ROOT / "adgen.db"
DEFAULT_STORAGE = ROOT / "fixtures"
DEFAULT_OUT = ROOT / "apps" / "web" / "public" / "showcase"

#: Rendered at 540 px against a 1920 px source. The stills appear at roughly a
#: fifth of the viewport width, so this is still comfortably above 2x on a phone,
#: and five of them at this size cost less than half a megabyte between them.
STILL_WIDTH = 540
#: Wider than the stills because a clip is the thing being watched, and narrower
#: than the source because nothing on this page is ever shown full-bleed.
CLIP_WIDTH = 480
#: Enough to read the motion, short enough to loop without becoming the page. The
#: full ad is a click away in the console; this is a moving thumbnail.
CLIP_SECONDS = 4.0
#: 30, against the 20 the pipeline delivers at. A looping four-second thumbnail is
#: not the deliverable, and the difference on disk is roughly four to one.
CLIP_CRF = 30


class ShowcaseError(RuntimeError):
    """Something the operator has to fix, reported without a traceback."""


# --- Reading the run ---------------------------------------------------------


def load_job(job_id: str, db: Path) -> dict[str, Any]:
    """The stored record for one job, or a message naming what went wrong."""
    if not db.exists():
        raise ShowcaseError(f"no database at {db}")

    con = sqlite3.connect(db)
    try:
        row = con.execute("select document from jobs where job_id = ?", (job_id,)).fetchone()
    finally:
        con.close()

    if row is None:
        raise ShowcaseError(f"no job {job_id!r} in {db.name}")

    document: dict[str, Any] = json.loads(row[0])
    if document.get("state") != "completed":
        raise ShowcaseError(
            f"job {job_id} is {document.get('state')}, not completed — "
            "a showcase built from a partial run would be showing an argument that did not finish"
        )
    return document


def settled_cost(job_id: str, db: Path) -> dict[str, float]:
    """What the run actually cost, split by operation, from the ledger.

    Read from ``spend_entries`` rather than from the job document because the
    document's ``total_cost_usd`` is one number and the page's meter needs the
    split — the whole point of the meter is that the image money and the video
    money are spent in different acts, with the free stages in between.
    """
    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            "select operation, sum(cost_usd) from spend_entries "
            "where job_id = ? and state = 'settled' group by operation",
            (job_id,),
        ).fetchall()
    finally:
        con.close()

    by_operation = {str(op): float(total or 0.0) for op, total in rows}
    return {
        "images": round(by_operation.get("image", 0.0), 4),
        "videos": round(by_operation.get("video", 0.0), 4),
    }


# --- Transcoding -------------------------------------------------------------


def still_webp(data: bytes, width: int) -> bytes:
    """One generated frame, downscaled to `width` and re-encoded as WebP."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    height = round(image.height * (width / image.width))
    image = image.resize((width, height), Image.LANCZOS)

    out = io.BytesIO()
    # method=6 is the slowest and smallest setting. This runs once, by hand, and
    # the result is served to every visitor — the trade is not close.
    image.save(out, format="WEBP", quality=82, method=6)
    return out.getvalue()


def clip_mp4(data: bytes, width: int, seconds: float) -> bytes:
    """A short, silent, loop-friendly cut of a delivered clip.

    Audio is dropped rather than kept quiet.  The page autoplays these, which
    browsers only permit muted, so the audio track would be bytes nobody could
    ever hear — and it is most of the file.
    """
    if V.FFMPEG is None:
        raise ShowcaseError("ffmpeg is not installed, so the clips cannot be transcoded")

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "in.mp4"
        target = Path(tmp) / "out.mp4"
        source.write_bytes(data)

        proc = subprocess.run(
            [
                V.FFMPEG,
                "-v", "error", "-y",
                "-i", str(source),
                "-t", f"{seconds:.3f}",
                # -2 keeps the height even, which H.264 with yuv420p requires.
                "-vf", f"scale={width}:-2",
                "-an",
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", str(CLIP_CRF),
                "-pix_fmt", "yuv420p",
                # Moves the index to the front so playback can start before the
                # whole file has arrived. It is a landing page; nobody waits.
                "-movflags", "+faststart",
                str(target),
            ],
            capture_output=True,
            timeout=V.SUBPROCESS_TIMEOUT_S,
        )  # fmt: skip
        if proc.returncode != 0 or not target.exists():
            raise ShowcaseError(f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()}")
        return target.read_bytes()


def poster_webp(clip: bytes, width: int) -> bytes:
    """The clip's first frame, for the `poster` attribute.

    Without one the video element is a black rectangle until it has buffered, and
    on this page a black rectangle on a black background is indistinguishable from
    nothing having loaded at all.
    """
    frames = V.decode(clip, max_frames=2).frames
    image = Image.fromarray(frames[0])
    height = round(image.height * (width / image.width))
    out = io.BytesIO()
    image.resize((width, height), Image.LANCZOS).save(out, format="WEBP", quality=82, method=6)
    return out.getvalue()


# --- Building ----------------------------------------------------------------


def _asset(storage: Path, key: str) -> bytes:
    path = storage / key
    if not path.exists():
        raise ShowcaseError(
            f"{key} is missing from {storage}. The record says it was generated, so "
            "the media has been cleaned up — re-run the job or pick another one."
        )
    return path.read_bytes()


def _checks(gate: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The gate's measurements, minus the ones whose model has not landed.

    A pending check always passes and therefore verifies nothing, so putting it on
    a landing page beside five real measurements would be borrowing credibility
    from a component that does not exist yet.
    """
    return [
        {
            "name": check["name"],
            "value": round(float(check.get("value") or 0.0), 4),
            "threshold": round(float(check.get("threshold") or 0.0), 4),
            "passed": bool(check.get("passed")),
        }
        for check in (gate or {}).get("checks", [])
        if check.get("implemented")
    ]


def build(
    job_id: str,
    *,
    db: Path = DEFAULT_DB,
    storage: Path = DEFAULT_STORAGE,
    out: Path = DEFAULT_OUT,
    still_width: int = STILL_WIDTH,
    clip_width: int = CLIP_WIDTH,
    clip_seconds: float = CLIP_SECONDS,
) -> dict[str, Any]:
    """Write the showcase directory and return the manifest that describes it."""
    document = load_job(job_id, db)
    result = document.get("result") or {}
    images = result.get("images") or []
    if not images:
        raise ShowcaseError(f"job {job_id} produced no images, so there is nothing to show")

    # Which raw video came from which still. The pipeline delivers the winner only,
    # so this is how a promoted candidate finds its own clip rather than the top one.
    clip_for = {
        int(video["source_image_index"]): video
        for video in (result.get("videos") or [])
        if video.get("source_image_index") is not None
    }
    rank_of = {
        int(entry["video"]["source_image_index"]): entry
        for entry in (result.get("ranking") or [])
        if (entry.get("video") or {}).get("source_image_index") is not None
    }

    # Built beside the target and swapped in at the end. A rebuild that fails
    # halfway — a missing clip, ffmpeg gone — would otherwise have already deleted a
    # showcase that was working, and the operator finds out by looking at the page.
    staging = out.parent / f".{out.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        return _write(
            staging,
            out,
            document,
            images,
            clip_for,
            rank_of,
            job_id,
            db,
            storage,
            still_width,
            clip_width,
            clip_seconds,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _write(
    staging: Path,
    out: Path,
    document: dict[str, Any],
    images: list[dict[str, Any]],
    clip_for: dict[int, Any],
    rank_of: dict[int, Any],
    job_id: str,
    db: Path,
    storage: Path,
    still_width: int,
    clip_width: int,
    clip_seconds: float,
) -> dict[str, Any]:
    """The body of :func:`build`, split out so its staging directory is cleaned up
    on any failure by a single ``except`` rather than by a ``finally`` that would
    also have to know whether the swap had already happened."""
    candidates: list[dict[str, Any]] = []
    for image in images:
        slot = int(image["index"])
        still = still_webp(_asset(storage, image["asset"]["key"]), still_width)
        (staging / f"still-{slot}.webp").write_bytes(still)

        score = image.get("score") or {}
        entry: dict[str, Any] = {
            "slot": slot,
            "still": f"/showcase/still-{slot}.webp",
            # The design point in the words the brief builder used for it, which is
            # already a plain-language sentence — there is no second phrasing here
            # that could disagree with what the job was actually asked for.
            "concept": (image.get("brief") or {}).get("concept") or "",
            "gate": {
                "verdict": (image.get("gate") or {}).get("verdict") or "",
                "checks": _checks(image.get("gate")),
            },
            "score": None if score.get("overall") is None else round(float(score["overall"]), 4),
            "promoted": bool(image.get("promoted")),
            "clip": None,
            "poster": None,
            "rank": None,
            "image_stage_rank": None,
            "explanation": "",
        }

        video = clip_for.get(slot)
        if video is not None:
            clip = clip_mp4(_asset(storage, video["asset"]["key"]), clip_width, clip_seconds)
            (staging / f"clip-{slot}.mp4").write_bytes(clip)
            (staging / f"poster-{slot}.webp").write_bytes(poster_webp(clip, clip_width))
            entry["clip"] = f"/showcase/clip-{slot}.mp4"
            entry["poster"] = f"/showcase/poster-{slot}.webp"

        ranked = rank_of.get(slot)
        if ranked is not None:
            entry["rank"] = ranked.get("rank")
            entry["image_stage_rank"] = ranked.get("image_stage_rank")
            entry["explanation"] = ranked.get("explanation") or ""

        candidates.append(entry)

    request = document.get("request") or {}
    cost = settled_cost(job_id, db)
    manifest = {
        "version": MANIFEST_VERSION,
        "job_id": job_id,
        "product_name": request.get("product_name") or "",
        "generated_at": (document.get("finished_at") or "")[:10],
        "cost": cost,
        # The ledger, not the document's `total_cost_usd`. That field counts the
        # candidates the run kept, so a job that retried a failed image reports
        # less than was actually charged. The page prints this total directly
        # beside the split and calls it "settled", so the two disagreeing would
        # be an arithmetic error visible on the first screen.
        "total_cost_usd": round(cost["images"] + cost["videos"], 4),
        # Every job on disk was ranked by the heuristic baseline, not a trained
        # model. The page has to say so wherever it shows a score, and this is where
        # it finds out — see `scoreDisplay` in the web app, which already does this
        # job for the console.
        "scores_are_stub": any((image.get("score") or {}).get("is_stub") for image in images),
        "candidates": candidates,
    }
    (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    if out.exists():
        shutil.rmtree(out)
    staging.rename(out)
    return manifest


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--job", required=True, help="Job id to publish, e.g. 4dbb3eb9041e43bf")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--still-width", type=int, default=STILL_WIDTH)
    parser.add_argument("--clip-width", type=int, default=CLIP_WIDTH)
    parser.add_argument("--clip-seconds", type=float, default=CLIP_SECONDS)
    args = parser.parse_args(argv)

    try:
        manifest = build(
            args.job,
            db=args.db,
            storage=args.storage,
            out=args.out,
            still_width=args.still_width,
            clip_width=args.clip_width,
            clip_seconds=args.clip_seconds,
        )
    except ShowcaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    written = sorted(args.out.glob("*"))
    total = sum(path.stat().st_size for path in written)
    print(f"{manifest['product_name']} — job {manifest['job_id']}, {manifest['generated_at']}")
    print(
        f"  {len(manifest['candidates'])} candidates, {len(written)} files, {total / 1024:.0f} kB"
    )
    print(f"  written to {args.out}")
    if manifest["scores_are_stub"]:
        print("  scores are placeholders and the page will mark them as such")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
