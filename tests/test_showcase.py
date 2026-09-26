"""The landing page's showcase builder.

What is asserted here is not that files appear. It is that the manifest and the
directory cannot disagree — the page reads the manifest and requests whatever it
names, so a manifest promising a clip that was never written is a broken video
element on the first screen a stranger sees, and nothing else in the project would
notice.

The honesty rules get the same treatment. A pending gate check always passes, so
one leaking into the showcase would put a measurement on a marketing page that
never ran.
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
from adml import video as V
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_showcase as B  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(V.FFMPEG is None, reason="ffmpeg is not installed")

JOB = "showcase00000001"


# --- A run on disk -----------------------------------------------------------


def _still(seed: int, size: tuple[int, int] = (1080, 1920)) -> bytes:
    """A frame with real structure, so the resize has something to lose."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    out = io.BytesIO()
    Image.fromarray(base).resize(size, Image.LANCZOS).save(out, format="PNG")
    return out.getvalue()


def _clip(seconds: float = 9.0, fps: int = 12) -> bytes:
    frames = [
        np.full((240, 136, 3), (i * 7) % 256, dtype=np.uint8) for i in range(int(seconds * fps))
    ]
    return V.encode_mp4(frames, fps=fps)


def _document(*, images: int = 3, promoted: tuple[int, ...] = (1,)) -> dict:
    """A job record shaped like the ones the pipeline writes."""
    return {
        "state": "completed",
        "finished_at": "2026-08-21T15:45:00+00:00",
        "request": {"product_name": "Aurora Serum"},
        "result": {
            "job_id": JOB,
            "total_cost_usd": 1.6,
            "images": [
                {
                    "index": i,
                    "asset": {"key": f"generations/{JOB}/img_{i}.png"},
                    "brief": {"concept": f"Design point {i}."},
                    "gate": {
                        "verdict": "pass",
                        "checks": [
                            {
                                "name": "palette_adherence",
                                "value": 0.7838,
                                "threshold": 0.456,
                                "passed": True,
                                "implemented": True,
                            },
                            {
                                "name": "face_identity",
                                "value": 0.0,
                                "threshold": 0.0,
                                "passed": True,
                                "implemented": False,
                            },
                        ],
                    },
                    "score": {"overall": 0.5 + i / 10, "is_stub": True},
                    "promoted": i in promoted,
                }
                for i in range(images)
            ],
            "videos": [
                {
                    "index": n,
                    "source_image_index": slot,
                    "asset": {"key": f"generations/{JOB}/vid_{slot}.mp4"},
                }
                for n, slot in enumerate(promoted)
            ],
            "ranking": [
                {
                    "rank": n + 1,
                    "image_stage_rank": n + 1,
                    "explanation": f"Ranked #{n + 1} — holds together frame to frame.",
                    "video": {"source_image_index": slot},
                }
                for n, slot in enumerate(promoted)
            ],
        },
    }


@pytest.fixture
def run(tmp_path):
    """A completed job: a database row, its stills, and its clips."""
    storage = tmp_path / "fixtures"
    (storage / "generations" / JOB).mkdir(parents=True)
    document = _document()

    for image in document["result"]["images"]:
        (storage / image["asset"]["key"]).write_bytes(_still(image["index"]))
    if V.FFMPEG is not None:
        clip = _clip()
        for video in document["result"]["videos"]:
            (storage / video["asset"]["key"]).write_bytes(clip)

    db = tmp_path / "adgen.db"
    con = sqlite3.connect(db)
    con.execute("create table jobs (job_id text, state text, document text)")
    con.execute("insert into jobs values (?, ?, ?)", (JOB, "completed", json.dumps(document)))
    con.execute(
        "create table spend_entries (job_id text, operation text, cost_usd real, state text)"
    )
    con.executemany(
        "insert into spend_entries values (?, ?, ?, 'settled')",
        [(JOB, "image", 0.04)] * 5 + [(JOB, "video", 0.7)] * 2,
    )
    con.commit()
    con.close()

    return {"db": db, "storage": storage, "out": tmp_path / "showcase", "document": document}


def _build(run, **kwargs):
    return B.build(JOB, db=run["db"], storage=run["storage"], out=run["out"], **kwargs)


# --- The manifest describes the directory ------------------------------------


@needs_ffmpeg
def test_the_manifest_and_the_directory_cannot_disagree(run):
    """Both directions, and the second one is the one that matters.

    A manifest naming a file that was not written is a broken image on the landing
    page. A file written that the manifest does not name is dead weight served to
    every visitor, and nothing would ever remove it.
    """
    manifest = _build(run)

    named = {
        Path(url).name
        for c in manifest["candidates"]
        for url in (c["still"], c["clip"], c["poster"])
        if url
    }
    written = {p.name for p in run["out"].iterdir()} - {B.MANIFEST_NAME}

    assert named == written, (
        f"named but missing: {named - written}; written but unnamed: {written - named}"
    )
    for name in named:
        assert (run["out"] / name).stat().st_size > 0


@needs_ffmpeg
def test_only_the_promoted_candidates_carry_a_clip(run):
    """The page's whole argument is that three of these were never animated.

    A showcase that quietly gave every candidate a clip would be illustrating the
    opposite of what the system does, using the system's own output.
    """
    manifest = _build(run)
    with_clips = [c["slot"] for c in manifest["candidates"] if c["clip"]]
    promoted = [c["slot"] for c in manifest["candidates"] if c["promoted"]]

    assert with_clips == promoted == [1]


@needs_ffmpeg
def test_stills_land_at_the_declared_width(run):
    """Read off the file, not off the argument that requested it."""
    manifest = _build(run, still_width=360)

    for candidate in manifest["candidates"]:
        image = Image.open(run["out"] / Path(candidate["still"]).name)
        assert image.width == 360
        # 1080x1920 in, so the aspect ratio has to survive the resize.
        assert image.height == 640


@needs_ffmpeg
def test_a_clip_is_cut_shorter_than_the_ad_it_came_from(run):
    """It is a moving thumbnail, not the deliverable."""
    manifest = _build(run, clip_seconds=3.0)
    clip = (run["out"] / Path(manifest["candidates"][1]["clip"]).name).read_bytes()
    probe = V.probe(clip)

    assert probe.duration_seconds == pytest.approx(3.0, abs=0.3)
    assert probe.duration_seconds < 9.0
    assert probe.width == B.CLIP_WIDTH


# --- The honesty rules -------------------------------------------------------


def test_a_pending_check_never_reaches_the_page(run):
    """A pending check always passes, so it verifies nothing.

    Putting `face_identity: passed` beside five real measurements would borrow
    credibility from a component that has not been built.
    """
    manifest = _build(run)
    names = {c["name"] for candidate in manifest["candidates"] for c in candidate["gate"]["checks"]}

    assert "palette_adherence" in names
    assert "face_identity" not in names


def test_placeholder_scores_are_flagged_for_the_page(run):
    """Every job on disk was ranked by the heuristic baseline, not a trained model."""
    assert _build(run)["scores_are_stub"] is True


def test_the_cost_is_split_the_way_the_meter_spends_it(run):
    """Images and video are spent in different acts, so one total is not enough."""
    manifest = _build(run)

    assert manifest["cost"] == {"images": 0.2, "videos": 1.4}


def test_the_total_is_the_ledger_rather_than_the_job_s_own_count(run):
    """A retried image is charged twice and counted once.

    The job document's ``total_cost_usd`` sums the candidates the run kept, so any
    job that retried a failed generation reports less than the provider charged.
    The page prints this total one line under the image/video split and calls it
    "settled", so taking the document's number would put two sums that do not
    agree within a few characters of a sentence promising every number was
    measured rather than chosen.
    """
    con = sqlite3.connect(run["db"])
    con.execute("insert into spend_entries values (?, 'image', 0.04, 'settled')", (JOB,))
    con.commit()
    con.close()

    manifest = _build(run)
    split = manifest["cost"]

    assert split == {"images": 0.24, "videos": 1.4}, "the retry was not charged"
    assert manifest["total_cost_usd"] == pytest.approx(split["images"] + split["videos"])
    assert manifest["total_cost_usd"] != run["document"]["result"]["total_cost_usd"]


# --- Refusals ----------------------------------------------------------------


def test_an_unknown_job_is_named_rather_than_guessed(run):
    with pytest.raises(B.ShowcaseError, match="no job"):
        B.build("nope", db=run["db"], storage=run["storage"], out=run["out"])


def test_a_job_that_produced_nothing_is_refused_rather_than_published(run, tmp_path):
    """Not an empty manifest. The page treats a present manifest as "there is a run
    to show", so an empty one is a section of captions with no frames under them."""
    document = _document()
    document["result"]["images"] = []
    con = sqlite3.connect(run["db"])
    con.execute("update jobs set document = ? where job_id = ?", (json.dumps(document), JOB))
    con.commit()
    con.close()

    with pytest.raises(B.ShowcaseError, match="no images"):
        _build(run)
    assert not run["out"].exists(), "a refused build must not leave a half-written directory"


def test_a_run_that_never_finished_is_refused(run):
    document = _document()
    document["state"] = "failed"
    con = sqlite3.connect(run["db"])
    con.execute("update jobs set document = ? where job_id = ?", (json.dumps(document), JOB))
    con.commit()
    con.close()

    with pytest.raises(B.ShowcaseError, match="not completed"):
        _build(run)


def test_media_cleaned_up_since_the_run_is_named(run):
    """The record outlives the files — `fixtures/generations` is gitignored and gets
    swept. The message has to say which file, because the operator's next move is to
    pick a different job."""
    (run["storage"] / "generations" / JOB / "img_1.png").unlink()

    with pytest.raises(B.ShowcaseError, match="img_1.png is missing"):
        _build(run)
    assert not run["out"].exists()


@needs_ffmpeg
def test_a_failed_rebuild_leaves_the_working_showcase_alone(run):
    """The build stages beside the target and swaps at the end.

    Deleting first would mean a rebuild that fails on a missing clip takes down a
    showcase that was serving perfectly well, and the operator learns about it by
    looking at the landing page.
    """
    before = _build(run)
    (run["storage"] / "generations" / JOB / "img_2.png").unlink()

    with pytest.raises(B.ShowcaseError):
        _build(run)

    kept = json.loads((run["out"] / B.MANIFEST_NAME).read_text())
    assert kept == before
    assert not list(run["out"].parent.glob(".*.partial")), "the staging directory was left behind"
