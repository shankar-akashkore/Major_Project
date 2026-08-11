"""Video container I/O, cut detection and clip measurement.

The first test in this file is the one that matters most: it pins the defect this
module was built to remove.  Every video measurement in the project used to decode
through PIL, the mock provider wrote GIFs, and so nothing ever established that an
MP4 could be read at all — which is the format every real provider returns.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
from adml import features as F
from adml import video as V

needs_ffmpeg = pytest.mark.skipif(V.FFMPEG is None, reason="ffmpeg is not installed")


# --- Fixtures ----------------------------------------------------------------


def _lavfi_mp4(tmp_path, spec: str, name: str) -> bytes:
    out = tmp_path / f"{name}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            spec,
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out.read_bytes()


@pytest.fixture
def moving_mp4(tmp_path) -> bytes:
    """9 s of continuously changing content — no cuts anywhere."""
    return _lavfi_mp4(tmp_path, "testsrc=size=270x480:rate=24:duration=9", "moving")


@pytest.fixture
def chained_mp4(tmp_path) -> bytes:
    """Two 4.5 s solid-colour clips concatenated: exactly the 5+5 chaining seam.

    Solid colours on purpose. They are the case that broke the first cut detector,
    and a flat end card behind a logo is ordinary ad creative, not a contrivance.
    """
    first = tmp_path / "p1.mp4"
    second = tmp_path / "p2.mp4"
    for path, colour in ((first, "navy"), (second, "orange")):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={colour}:size=270x480:rate=24:duration=4.5",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
    listing = tmp_path / "list.txt"
    listing.write_text(f"file '{first}'\nfile '{second}'\n")
    out = tmp_path / "chained.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out.read_bytes()


# --- The defect --------------------------------------------------------------


@needs_ffmpeg
def test_an_mp4_decodes_where_pil_could_not(moving_mp4):
    """The regression guard for the bug this module exists to fix.

    ``adml.features.load_frames`` is PIL and raises ``UnidentifiedImageError`` on an
    MP4. Mock video was GIF, so the failure was invisible until a paid clip arrived
    — at which point stage 7 would have died on already-billed output.
    """
    from PIL import UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError):
        F.load_frames(moving_mp4)

    clip, motion = V.measure(moving_mp4)
    assert clip.n_sampled >= 2
    assert motion.motion_energy_mean > 0.0


def test_a_missing_ffmpeg_raises_rather_than_returning_no_frames(monkeypatch, moving_mp4):
    """An absent decoder must fail, not produce a zero-filled measurement.

    Zeros would flow into the feature table as a perfectly average clip, and the
    ranker would score it without complaint.
    """
    monkeypatch.setattr(V, "FFMPEG", None)
    monkeypatch.setattr(V, "FFPROBE", None)
    with pytest.raises(V.ClipDecodeError, match="brew install ffmpeg"):
        V.decode(moving_mp4)


# --- Container sniffing ------------------------------------------------------


@needs_ffmpeg
def test_containers_are_recognised_from_bytes_not_extensions(moving_mp4, storage):
    import adproviders as P
    from adschema import AspectRatio

    asset = P.mock_reference_asset(storage, "x/still.png", AspectRatio.SQUARE_1_1)
    assert V.container_of(moving_mp4) == "mp4"
    assert V.container_of(storage.get_bytes(asset.key)) == "still"
    assert V.container_of(b"GIF89a" + b"\x00" * 16) == "gif"
    assert V.container_of(b"not a media file at all") == "unknown"


def test_an_unrecognised_container_names_what_it_saw():
    with pytest.raises(V.ClipDecodeError, match="unrecognised container"):
        V.probe(b"\x00\x01\x02\x03 definitely not media")


# --- Probing -----------------------------------------------------------------


@needs_ffmpeg
def test_probe_reads_geometry_and_timing_from_the_file(moving_mp4):
    probe = V.probe(moving_mp4)
    assert probe.container == "mp4"
    assert probe.width == 270 and probe.height == 480
    assert abs(probe.duration_seconds - 9.0) < 0.05
    assert abs(probe.fps - 24.0) < 0.1
    assert probe.duration_measured


@needs_ffmpeg
def test_frames_are_sampled_across_the_whole_clip(moving_mp4):
    """Not taken from the front: a statistic over them must describe the clip."""
    clip = V.decode(moving_mp4, max_frames=12)
    assert clip.n_sampled <= 12
    # The sampled rate is well below the file's, which is what makes the frames
    # span the duration rather than the first half-second.
    assert clip.sample_fps < clip.probe.fps
    assert abs(clip.n_sampled / clip.sample_fps - clip.duration_seconds) < 0.5


@needs_ffmpeg
def test_frames_are_downscaled_to_fit_the_memory_budget(moving_mp4):
    clip = V.decode(moving_mp4, long_edge=64)
    assert max(clip.frames[0].shape[:2]) <= 64
    assert clip.frames[0].shape[-1] == 3


def test_two_frames_are_the_minimum_for_a_motion_measurement(moving_mp4):
    with pytest.raises(ValueError, match="at least 2"):
        V.decode(moving_mp4, max_frames=1)


# --- The flat-frame correlation bug -----------------------------------------


def test_two_different_flat_frames_are_not_perfectly_similar():
    """The bug: 'both frames are constant' was treated as 'the frames are identical'.

    Pearson correlation is undefined for a constant vector, and the natural guard
    returns 1.0. Navy and orange are both constant and nothing alike — measured on a
    real concatenation, a navy-to-orange hard cut scored 1.000 consistency while the
    frames differed by 0.62 in mean intensity.
    """
    navy = np.full((16, 16, 3), (20, 20, 90), dtype=np.uint8)
    orange = np.full((16, 16, 3), (240, 160, 40), dtype=np.uint8)

    assert F.frame_similarity(navy, navy) == pytest.approx(1.0)
    assert F.frame_similarity(navy, orange) < 0.6
    # ...and the sequence statistic inherits the fix.
    assert F.temporal_consistency([navy, orange]) < 0.6


def test_a_single_frame_has_no_temporal_consistency():
    """NaN, not 1.0. A still has no temporal behaviour to score."""
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert np.isnan(F.temporal_consistency([frame]))


def test_exposure_change_is_not_a_loss_of_consistency():
    """Correlation is invariant to overall brightness, deliberately.

    A clip that brightens across its run has not fallen apart, and a similarity
    measure that says otherwise would penalise every golden-hour ramp.
    """
    rng = np.random.default_rng(0)
    base = rng.integers(40, 200, size=(24, 24, 3), dtype=np.uint8)
    brighter = np.clip(base.astype(int) + 30, 0, 255).astype(np.uint8)
    assert F.frame_similarity(base, brighter) > 0.95


# --- Cut and seam detection --------------------------------------------------


@needs_ffmpeg
def test_a_concatenation_seam_is_found_from_the_pixels(chained_mp4):
    """The seam is detected without being told it exists.

    This is what makes chained clips measurable at all — including one a provider
    stitched internally, which no ``was_chained`` flag would reveal.
    """
    clip, motion = V.measure(chained_mp4)
    cuts = V.detect_cuts(clip.frames)

    assert motion.cut_count == 1, cuts
    assert motion.has_seam
    # And it is at the midpoint, where 4.5 s + 4.5 s says it should be.
    midpoint = (clip.n_sampled - 1) / 2
    assert abs(cuts[0] - midpoint) <= 2, (cuts, midpoint)
    # The seam's cost is a real number, not the 1.0 the correlation version gave.
    assert 0.0 < motion.seam_consistency < 0.6


@needs_ffmpeg
def test_continuous_content_has_no_cuts(moving_mp4):
    clip, motion = V.measure(moving_mp4)
    assert motion.cut_count == 0
    assert not motion.has_seam
    # NaN, not 0.0: there is no seam, which differs from a totally broken one.
    assert np.isnan(motion.seam_consistency)


def test_film_grain_in_a_still_clip_is_not_a_cut():
    """Why the absolute floor exists alongside the robust z-score.

    A near-frozen clip has a tiny spread, so grain sits many robust deviations above
    its own median. Measured on a ``static_subtle`` mock clip, the largest transition
    was 30 deviations out at an absolute difference of 0.0021.
    """
    rng = np.random.default_rng(1)
    base = rng.integers(90, 160, size=(32, 32, 3), dtype=np.uint8)
    frames = [
        np.clip(base.astype(int) + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
        for _ in range(20)
    ]
    assert V.detect_cuts(frames) == []


def test_a_uniformly_flickering_clip_has_no_distinguishable_cut():
    """Why the robust z-score exists alongside the absolute floor.

    If every transition is enormous, none of them is a cut — the clip is simply
    broken, and reporting 19 cuts would be describing the wrong problem.
    """
    rng = np.random.default_rng(2)
    frames = [rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8) for _ in range(20)]
    cuts = V.detect_cuts(frames)
    assert len(cuts) < len(frames) // 2, cuts


# --- Duration verification ---------------------------------------------------


@needs_ffmpeg
def test_duration_is_checked_against_the_file_not_the_claim(moving_mp4):
    honest = V.verify_duration(moving_mp4, 9.0, min_seconds=8.0, max_seconds=10.0)
    assert honest.in_window
    assert honest.matches_claim
    assert abs(honest.drift_seconds) < 0.05

    # A provider that bills 10 s and delivers 9 s is caught, and the discrepancy
    # appears in the summary rather than being rounded away.
    lying = V.verify_duration(moving_mp4, 10.0, min_seconds=8.0, max_seconds=10.0)
    assert lying.in_window  # the file itself is fine
    assert not lying.matches_claim
    assert "provider reported" in lying.summary()


@needs_ffmpeg
def test_a_short_delivery_falls_outside_the_window(tmp_path):
    short = _lavfi_mp4(tmp_path, "testsrc=size=64x64:rate=24:duration=5", "short")
    check = V.verify_duration(short, 5.0, min_seconds=8.0, max_seconds=10.0)
    assert not check.in_window
    assert "OUTSIDE" in check.summary()


# --- Clip features -----------------------------------------------------------


@needs_ffmpeg
def test_motion_trend_keeps_its_sign(tmp_path):
    """A build and a fade are different creative choices, not the same magnitude."""
    building = _lavfi_mp4(
        tmp_path, "testsrc=size=64x64:rate=24:duration=8,fade=t=in:st=0:d=7", "build"
    )
    _, motion = V.measure(building)
    assert motion.motion_trend > 0.0


def test_a_still_yields_zero_motion_and_no_seam(storage):
    """A one-frame clip is legal input and must not fabricate temporal numbers."""
    import adproviders as P
    from adschema import AspectRatio

    asset = P.mock_reference_asset(storage, "x/one.png", AspectRatio.SQUARE_1_1)
    clip, motion = V.measure(storage.get_bytes(asset.key))

    assert clip.n_sampled == 1
    assert motion.motion_energy_mean == 0.0
    assert np.isnan(motion.temporal_consistency)
    assert np.isnan(motion.seam_consistency)
    assert motion.cut_count == 0


# --- Encoding ----------------------------------------------------------------


@needs_ffmpeg
def test_encoding_is_deterministic_so_the_cache_stays_correct():
    """Same frames in, same bytes out.

    The cost governor's cache is keyed on content hashes, and the pipeline asserts
    that an identical request does not pay twice. Non-deterministic encoding would
    silently break that.
    """
    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8) for _ in range(10)]
    assert V.encode_mp4(frames, fps=10) == V.encode_mp4(frames, fps=10)


@needs_ffmpeg
def test_odd_dimensions_are_padded_to_even():
    """yuv420p subsamples chroma by two and cannot represent an odd edge.

    Not an exotic guard: the mock provider's 9:16 frame at a 240 px long edge is
    135 px wide, so this is the common path.
    """
    frames = [np.zeros((240, 135, 3), dtype=np.uint8) for _ in range(4)]
    probe = V.probe(V.encode_mp4(frames, fps=10))
    assert (probe.width, probe.height) == (136, 240)


@needs_ffmpeg
def test_an_encoded_clip_round_trips_its_duration():
    frames = [np.full((32, 32, 3), i * 20, dtype=np.uint8) for i in range(12)]
    probe = V.probe(V.encode_mp4(frames, fps=12))
    assert abs(probe.duration_seconds - 1.0) < 0.05


def test_encoding_refuses_an_empty_frame_list():
    with pytest.raises(ValueError, match="empty frame list"):
        V.encode_mp4([], fps=10)
