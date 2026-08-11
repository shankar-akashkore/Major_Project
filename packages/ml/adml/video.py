"""Video containers — the one place a clip becomes frames.

Until this module existed, every video measurement in the project ran through
:func:`adml.features.load_frames`, which is PIL, and PIL cannot open an MP4.  Mock
video is written as a GIF, so nothing ever noticed: the mock provider even
rewrites ``.mp4`` output keys to ``.gif``.  The first real Kling clip would have
raised ``UnidentifiedImageError`` in stage 7 — *after* being generated and billed.
That is the specific failure this module exists to remove, and it is why frame
decoding is a named boundary rather than a helper tucked into the feature code.

Three properties are deliberate.

**One entry point, several containers.**  :func:`decode` sniffs magic bytes and
dispatches: GIF through PIL (no ffmpeg needed, so the mock path stays dependency
free), MP4/WebM through ffmpeg, a still image as a one-frame clip.  Callers never
branch on format.

**Measured, not claimed.**  :class:`ClipProbe` reads duration, frame rate and
dimensions out of the file.  The provider's reported duration is a separate
number, and the two disagreeing is a finding rather than an inconvenience — a
provider that bills for 10 s and delivers 8.4 s is something the write-up should
say out loud.  :func:`verify_duration` is what the pipeline asserts against.

**No silent zero.**  If ffmpeg is missing and an MP4 arrives, this raises.  It
does not return an empty frame list.  An empty list would flow into
:class:`ClipFeatures` as a set of zeros, which the standardiser would happily
accept and the ranker would happily score — a fabricated number reaching a
reported result, which is the same failure mode :mod:`adml.embeddings` refuses
zero-fill to prevent.

Frames are sampled down hard (48 frames at a 256 px long edge by default).  A 10 s
1080p clip decoded in full is about 1.5 GB of raw RGB, which does not fit in the
headroom of an 8 GB laptop; the sampled version is about 5 MB and every feature
here is a statistic over time, not a pixel-exact reconstruction.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass

import numpy as np

from . import features as F

#: Resolved once. ``None`` means the binary is absent, which is a legal state —
#: the GIF path and every image feature work without it.
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

#: Frames sampled per clip, and the long edge they are scaled to. Both are
#: memory decisions; see the module docstring.
DEFAULT_MAX_FRAMES = 48
DEFAULT_LONG_EDGE = 256

#: Wall-clock ceiling on any ffmpeg invocation. A decode of a 10 s clip takes
#: well under a second; anything near this bound means ffmpeg is stuck on a
#: malformed file and the job should fail rather than hang.
SUBPROCESS_TIMEOUT_S = 120.0


class ClipDecodeError(RuntimeError):
    """A clip could not be turned into frames.

    Always raised rather than returning an empty result: see the module docstring
    on why a zero-filled measurement is worse than a failed one.
    """


def _require_ffmpeg(container: str) -> str:
    if FFMPEG is None:
        raise ClipDecodeError(
            f"decoding a {container} clip needs ffmpeg, which is not on PATH. "
            "Install it with `brew install ffmpeg`. Mock video is written as GIF and "
            "decodes without it, so this only blocks real generations."
        )
    return FFMPEG


# --- Container sniffing ------------------------------------------------------

#: Containers this module can decode, and how to recognise them. Sniffed from
#: bytes rather than trusted from the storage key's extension: the mock provider
#: rewrites extensions, providers report their own content types, and a key is
#: request-derived data.
_ISOBMFF_BRAND_OFFSET = 4


def container_of(data: bytes) -> str:
    """``"gif"``, ``"mp4"``, ``"webm"``, ``"still"`` or ``"unknown"``."""
    if len(data) < 12:
        return "unknown"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[_ISOBMFF_BRAND_OFFSET : _ISOBMFF_BRAND_OFFSET + 4] == b"ftyp":
        # MP4, M4V and MOV all share the ISO base media container.
        return "mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"  # Matroska; WebM is the subset providers emit
    if data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff" or data[:4] == b"RIFF":
        return "still"
    return "unknown"


# --- Probing -----------------------------------------------------------------


@dataclass(frozen=True)
class ClipProbe:
    """What the file itself says it contains."""

    container: str
    duration_seconds: float
    fps: float
    width: int
    height: int
    n_frames: int
    codec: str = ""
    has_audio: bool = False
    #: False when the duration had to be derived from a frame count rather than
    #: read from the container. Fragmented MP4s and some GIF encoders omit it, and
    #: a derived figure should not be quoted as if it were read off the file.
    duration_measured: bool = True

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    def summary(self) -> str:
        measured = "" if self.duration_measured else " (derived)"
        audio = ", audio" if self.has_audio else ""
        return (
            f"{self.container} {self.width}x{self.height} "
            f"{self.duration_seconds:.2f}s{measured} @ {self.fps:.2f}fps, "
            f"{self.n_frames} frames{audio}"
        )


def _ffprobe_json(path: str) -> dict:
    if FFPROBE is None:
        raise ClipDecodeError(
            "probing a video needs ffprobe, which is not on PATH. Install it with "
            "`brew install ffmpeg` (ffprobe ships alongside ffmpeg)."
        )
    proc = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ],
        capture_output=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise ClipDecodeError(f"ffprobe failed: {proc.stderr.decode('utf-8', 'replace').strip()}")
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:  # pragma: no cover - ffprobe emitting non-JSON
        raise ClipDecodeError(f"ffprobe returned non-JSON: {proc.stdout[:200]!r}") from exc


def _parse_rate(value: str | None) -> float:
    """ffprobe reports frame rates as the rational string ``"30000/1001"``."""
    if not value:
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            numerator, denominator = float(num), float(den)
        except ValueError:
            return 0.0
        return numerator / denominator if denominator else 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _write_temp(data: bytes, suffix: str) -> str:
    """Spill bytes to a temp file.

    ffmpeg and ffprobe both need to seek, and an MP4 written by a generation API
    routinely has its ``moov`` atom at the end of the file — piping such a file on
    stdin fails, sometimes silently truncating instead of erroring.
    """
    handle, path = tempfile.mkstemp(suffix=f".{suffix}")
    with os.fdopen(handle, "wb") as fh:
        fh.write(data)
    return path


def probe(data: bytes) -> ClipProbe:
    """Read a clip's real geometry and timing out of its bytes."""
    container = container_of(data)

    if container == "gif":
        return _probe_animated_image(data)
    if container == "still":
        rgb = F.load_image(data)
        height, width = rgb.shape[:2]
        return ClipProbe("still", 0.0, 0.0, width, height, 1, codec="still")
    if container == "unknown":
        raise ClipDecodeError(
            f"unrecognised container; first bytes were {data[:12]!r}. "
            "Supported: GIF, MP4/MOV, WebM, and single still images."
        )

    _require_ffmpeg(container)
    path = _write_temp(data, container)
    try:
        info = _ffprobe_json(path)
    finally:
        os.unlink(path)

    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ClipDecodeError("the file has no video stream")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    # avg_frame_rate over r_frame_rate: r_frame_rate is the container's nominal
    # timebase and reads 1000+ on some variable-rate files, while avg_frame_rate
    # is frames divided by duration, which is what a sampling rate needs.
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))

    duration = 0.0
    measured = True
    for source in (video.get("duration"), (info.get("format") or {}).get("duration")):
        try:
            duration = float(source)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            break

    n_frames = 0
    try:
        n_frames = int(video.get("nb_frames"))
    except (TypeError, ValueError):
        n_frames = 0

    if duration <= 0 and n_frames and fps > 0:
        duration = n_frames / fps
        measured = False
    if n_frames <= 0 and duration > 0 and fps > 0:
        n_frames = max(1, round(duration * fps))
    if duration <= 0:
        raise ClipDecodeError(
            "could not determine the clip's duration from the container; "
            f"ffprobe reported duration={video.get('duration')!r} "
            f"nb_frames={video.get('nb_frames')!r} fps={fps}"
        )

    return ClipProbe(
        container=container,
        duration_seconds=duration,
        fps=fps,
        width=width,
        height=height,
        n_frames=n_frames,
        codec=str(video.get("codec_name") or ""),
        has_audio=has_audio,
        duration_measured=measured,
    )


def _probe_animated_image(data: bytes) -> ClipProbe:
    """GIF timing, read frame by frame.

    A GIF's frames can each carry their own delay, so the duration is a sum
    rather than ``n_frames / fps``. The reported fps is therefore an average, and
    for a variable-delay GIF it describes no individual frame.
    """
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        width, height = img.size
        total = getattr(img, "n_frames", 1)
        ms = 0
        for index in range(total):
            img.seek(index)
            ms += img.info.get("duration", 0)

    duration = ms / 1000.0
    measured = duration > 0
    if not measured:
        # A GIF with no delay metadata plays at the viewer's discretion. 10 fps
        # matches what the mock provider writes; flagged as derived either way.
        duration = total / 10.0
    return ClipProbe(
        container="gif",
        duration_seconds=duration,
        fps=(total / duration) if duration > 0 else 0.0,
        width=width,
        height=height,
        n_frames=total,
        codec="gif",
        duration_measured=measured,
    )


# --- Decoding ----------------------------------------------------------------


@dataclass
class Clip:
    """Sampled frames plus the probe they came from."""

    frames: list[np.ndarray]
    probe: ClipProbe
    #: The rate frames were *sampled* at, which is not ``probe.fps``. Motion
    #: energy is a per-transition quantity, so any feature that converts between
    #: frames and seconds has to use this one.
    sample_fps: float

    @property
    def duration_seconds(self) -> float:
        return self.probe.duration_seconds

    @property
    def n_sampled(self) -> int:
        return len(self.frames)

    def __len__(self) -> int:
        return len(self.frames)


def _scaled_size(probe: ClipProbe, long_edge: int) -> tuple[int, int]:
    if probe.width <= 0 or probe.height <= 0:
        raise ClipDecodeError(f"clip reports a degenerate size: {probe.width}x{probe.height}")
    longest = max(probe.width, probe.height)
    if longest <= long_edge:
        return probe.width, probe.height
    scale = long_edge / longest
    return max(2, round(probe.width * scale)), max(2, round(probe.height * scale))


def decode(
    data: bytes,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    long_edge: int = DEFAULT_LONG_EDGE,
) -> Clip:
    """Decode any supported container to a bounded, downscaled frame list.

    Frames are sampled at a constant rate spanning the whole clip rather than
    taken from the front, so a statistic over them describes the clip and not its
    opening. The first frame is always included: :func:`adml.features.hook_strength`
    is about the opening specifically.
    """
    if max_frames < 2:
        raise ValueError(f"max_frames must be at least 2 to measure motion, got {max_frames}")

    info = probe(data)

    if info.container == "still":
        rgb = F.load_image(data)
        return Clip(frames=[rgb], probe=info, sample_fps=0.0)

    if info.container == "gif":
        frames = F.load_frames(data, max_frames=max_frames)
        if not frames:
            raise ClipDecodeError("the GIF decoded to zero frames")
        # The sampled rate, not the file's: load_frames subsamples evenly, so
        # consecutive sampled frames are further apart in time than 1/probe.fps.
        sample_fps = len(frames) / info.duration_seconds if info.duration_seconds > 0 else 0.0
        return Clip(frames=frames, probe=info, sample_fps=sample_fps)

    ffmpeg = _require_ffmpeg(info.container)
    width, height = _scaled_size(info, long_edge)
    # Sample at whichever is lower: the clip's own rate, or the rate that fills
    # the frame budget. Asking for more than the source has makes ffmpeg
    # duplicate frames, which reads as zero motion and is worse than fewer frames.
    target_fps = max_frames / info.duration_seconds
    if info.fps > 0:
        target_fps = min(target_fps, info.fps)
    target_fps = max(target_fps, 1e-3)

    path = _write_temp(data, info.container)
    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                path,
                "-vf",
                f"fps={target_fps:.6f},scale={width}:{height}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            capture_output=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    finally:
        os.unlink(path)

    if proc.returncode != 0:
        raise ClipDecodeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace').strip()}")

    stride = width * height * 3
    # Count frames from the byte length rather than predicting them: the fps
    # filter's rounding depends on the source timestamps, so "duration x rate"
    # is off by one either way often enough to matter.
    count = len(proc.stdout) // stride
    if count < 1:
        raise ClipDecodeError(
            f"ffmpeg produced {len(proc.stdout)} bytes, under one {width}x{height} frame"
        )
    buffer = np.frombuffer(proc.stdout[: count * stride], dtype=np.uint8)
    frames = list(buffer.reshape(count, height, width, 3))

    sample_fps = count / info.duration_seconds if info.duration_seconds > 0 else target_fps
    return Clip(frames=frames, probe=info, sample_fps=sample_fps)


# --- Duration verification ---------------------------------------------------


@dataclass(frozen=True)
class DurationCheck:
    """Does the delivered file match what was promised and what was billed?"""

    measured_seconds: float
    claimed_seconds: float
    min_seconds: float
    max_seconds: float
    duration_measured: bool

    @property
    def in_window(self) -> bool:
        return self.min_seconds <= self.measured_seconds <= self.max_seconds

    @property
    def drift_seconds(self) -> float:
        return self.measured_seconds - self.claimed_seconds

    @property
    def matches_claim(self) -> bool:
        """Within a quarter second of the provider's reported duration.

        Loose on purpose: encoders round frame counts, and a 0.04 s difference on
        a 10 s clip is one frame, not a discrepancy worth reporting.
        """
        return abs(self.drift_seconds) <= 0.25

    def summary(self) -> str:
        window = f"{self.min_seconds:.0f}-{self.max_seconds:.0f}s"
        verdict = "in" if self.in_window else "OUTSIDE"
        text = f"{self.measured_seconds:.2f}s measured, {verdict} the {window} window"
        if not self.matches_claim:
            text += f" — provider reported {self.claimed_seconds:.2f}s ({self.drift_seconds:+.2f}s)"
        if not self.duration_measured:
            text += " (duration derived, not read from the container)"
        return text


def verify_duration(
    data: bytes,
    claimed_seconds: float,
    *,
    min_seconds: float,
    max_seconds: float,
) -> DurationCheck:
    """Check a delivered clip against the project's duration commitment.

    The project promises 8-10 s output, and the only trustworthy source for what
    a file contains is the file. Providers snap durations to their own enums, and
    encoders drop trailing frames; both are invisible to a check that reads the
    provider's response instead of the bytes.
    """
    info = probe(data)
    return DurationCheck(
        measured_seconds=info.duration_seconds,
        claimed_seconds=claimed_seconds,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        duration_measured=info.duration_measured,
    )


# --- Cut and seam detection --------------------------------------------------

#: Mean absolute frame difference (grayscale, 0-1) below which a transition is
#: never a cut, however anomalous it is for the clip.
#:
#: **Provisional, and derived from synthetic clips only.**  Measured across every
#: mock motion intent plus a high-motion ``testsrc`` pattern, the largest
#: per-transition difference produced by *continuous* content was 0.0093; a
#: navy-to-orange concatenation seam measured 0.6201, 67x higher.  0.15 sits an
#: order of magnitude above the former and well below the latter.  What has not
#: been tested is a real generation with fast camera movement, or a cut between
#: two genuinely similar shots — both narrow that gap.  Re-derive this against
#: real clips before quoting a cut count as a finding.
CUT_ABSOLUTE_FLOOR = 0.15
#: ...and a cut must also be this many robust deviations above the clip's own
#: typical transition. The conjunction is what makes the test work: the absolute
#: floor alone would flag every transition of a uniformly flickering clip, and the
#: relative test alone flags film grain — a ``static_subtle`` mock clip put its
#: largest transition 30 robust deviations above its own median at an absolute
#: difference of 0.0021, which is grain, not a cut.
CUT_SIGMA = 5.0


def detect_cuts(frames: list[np.ndarray]) -> list[int]:
    """Indices ``i`` where a hard cut sits between frame ``i`` and ``i + 1``.

    This doubles as the *seam* detector, which is the reason it is here.  Reaching
    8-10 s on a provider that caps at 5 s means concatenating two generations, and
    the seam is a hard cut in a clip that should contain none.  Detecting it from
    the pixels means the seam's cost can be measured on any clip — including one a
    provider chained internally without saying so, which no ``was_chained`` flag
    would reveal.

    Built on motion energy rather than frame correlation.  Correlation was the
    first attempt and it failed on exactly the case this function exists for: a
    concatenation of two flat-coloured segments has no spatial structure to
    correlate, so every transition including the seam scored 1.000.  Absolute
    difference has no such blind spot — the seam was the series maximum by a
    factor of 67.
    """
    if len(frames) < 3:
        return []

    series = np.asarray(F.motion_energy(frames), dtype=float)
    median = float(np.median(series))
    # Median absolute deviation, scaled to a standard deviation for Gaussian
    # data. Robust by construction: the cuts themselves are the outliers, and a
    # plain standard deviation would be inflated by the very thing being detected.
    mad = float(np.median(np.abs(series - median))) * 1.4826
    threshold = max(CUT_ABSOLUTE_FLOOR, median + CUT_SIGMA * max(mad, 1e-9))
    return [i for i, value in enumerate(series) if value >= threshold]


def seam_consistency_at(frames: list[np.ndarray], index: int) -> float:
    """Frame similarity across the transition after ``index``."""
    if not 0 <= index < len(frames) - 1:
        return float("nan")
    return F.frame_similarity(frames[index], frames[index + 1])


# --- Clip features -----------------------------------------------------------


@dataclass(frozen=True)
class ClipFeatures:
    """Everything the project measures about a clip's *motion*.

    Scoped to the temporal axis on purpose.  A clip's still-frame qualities —
    exposure, framing, palette — are measured by the image feature builder on the
    middle frame, so this dataclass and that one do not overlap, and the motion
    ablation row therefore isolates what movement contributes.
    """

    duration_seconds: float
    measured_fps: float
    temporal_consistency: float
    motion_energy_mean: float
    motion_energy_std: float
    motion_energy_peak: float
    #: Least-squares slope of motion energy against normalised time. Positive
    #: means the clip builds, negative means it fades. Kept signed: a build and a
    #: fade are different creative choices, and an absolute value would call them
    #: the same thing.
    motion_trend: float
    hook_strength: float
    cut_count: int
    #: Correlation across the worst cut. NaN when no cut was found — which is the
    #: expected state for a native long generation, and must stay distinguishable
    #: from a measured 0.0 (a clip whose seam is a total discontinuity).
    seam_consistency: float
    focal_persistence: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @property
    def has_seam(self) -> bool:
        return self.cut_count > 0


#: Focal-concentration level at which a frame is treated as still having a clear
#: subject. Provisional: it comes from the mock renderer's range, and needs
#: re-deriving against real generations before it is quoted as product screen time.
FOCAL_PRESENT_THRESHOLD = 0.12
#: Frames sampled for the per-frame saliency pass. Saliency is an FFT per frame
#: and by far the most expensive thing here, so it runs on a subsample.
FOCAL_SAMPLE_FRAMES = 8


def clip_features(clip: Clip) -> ClipFeatures:
    """Measure a decoded clip.

    A single frame is a legal input and yields zero motion with NaN seam — a still
    has no temporal properties, and reporting 1.0 consistency for it would claim a
    measurement that was never taken.
    """
    frames = clip.frames
    if not frames:
        raise ClipDecodeError("cannot measure a clip with no frames")

    if len(frames) < 2:
        return ClipFeatures(
            duration_seconds=clip.duration_seconds,
            measured_fps=clip.probe.fps,
            temporal_consistency=float("nan"),
            motion_energy_mean=0.0,
            motion_energy_std=0.0,
            motion_energy_peak=0.0,
            motion_trend=0.0,
            hook_strength=F.hook_strength(frames, fps=max(clip.sample_fps, 1.0)),
            cut_count=0,
            seam_consistency=float("nan"),
            focal_persistence=float(
                F.focal_concentration(F.saliency_map(frames[0])) >= FOCAL_PRESENT_THRESHOLD
            ),
        )

    energies = np.asarray(F.motion_energy(frames), dtype=float)
    # Normalised time, so the slope is "energy change across the whole clip" and
    # is comparable between a 24-frame sample and a 48-frame one.
    times = np.linspace(0.0, 1.0, len(energies))
    trend = (
        float(np.polyfit(times, energies, 1)[0]) if len(energies) >= 2 and times.std() > 0 else 0.0
    )

    cuts = detect_cuts(frames)
    seam = min((seam_consistency_at(frames, i) for i in cuts), default=float("nan"))

    step = max(1, len(frames) // FOCAL_SAMPLE_FRAMES)
    focal = [F.focal_concentration(F.saliency_map(f)) for f in frames[::step]]
    persistence = sum(1 for v in focal if v >= FOCAL_PRESENT_THRESHOLD) / max(1, len(focal))

    return ClipFeatures(
        duration_seconds=clip.duration_seconds,
        measured_fps=clip.probe.fps,
        temporal_consistency=F.temporal_consistency(frames),
        motion_energy_mean=float(energies.mean()),
        motion_energy_std=float(energies.std()),
        motion_energy_peak=float(energies.max()),
        motion_trend=trend,
        # The sampled rate: hook_strength converts a time window into a frame
        # count, and passing the file's fps would ask for more frames than were
        # sampled, silently widening the window past the first second.
        hook_strength=F.hook_strength(frames, fps=max(clip.sample_fps, 1.0)),
        cut_count=len(cuts),
        seam_consistency=seam,
        focal_persistence=persistence,
    )


def measure(
    data: bytes,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    long_edge: int = DEFAULT_LONG_EDGE,
) -> tuple[Clip, ClipFeatures]:
    """Decode and measure in one call — the usual entry point."""
    clip = decode(data, max_frames=max_frames, long_edge=long_edge)
    return clip, clip_features(clip)


# --- Encoding ----------------------------------------------------------------

#: Constant rate factor for H.264. 20 is visually near-lossless for synthetic
#: content and keeps a 9 s mock clip well under a megabyte.
H264_CRF = 20


def _run_filtergraph(data: bytes, container: str, filters: str, *, crf: int) -> bytes:
    """Re-encode a clip through one ffmpeg filtergraph.

    Shared by every delivery transform, so they all inherit the same determinism
    settings and the same failure reporting. Audio is passed through when present
    (``-c:a copy`` would break if the filter changes duration, so it is re-encoded).
    """
    ffmpeg = _require_ffmpeg(container)
    source = _write_temp(data, container)
    target = _write_temp(b"", "mp4")
    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-fflags",
                "+bitexact",
                "-i",
                source,
                "-vf",
                filters,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-threads",
                "1",
                "-map_metadata",
                "-1",
                "-fflags",
                "+bitexact",
                "-flags:v",
                "+bitexact",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                target,
            ],
            capture_output=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
        if proc.returncode != 0:
            raise ClipDecodeError(
                f"ffmpeg filter {filters!r} failed: "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}"
            )
        with open(target, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(source)
        os.unlink(target)


def render_crop(
    data: bytes,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    out_width: int,
    out_height: int,
    crf: int = H264_CRF,
) -> bytes:
    """Crop to a window, then scale to the delivery size."""
    info = probe(data)
    filters = f"crop={width}:{height}:{x}:{y},scale={out_width}:{out_height}"
    return _run_filtergraph(data, info.container, filters, crf=crf)


def render_padded(
    data: bytes,
    *,
    out_width: int,
    out_height: int,
    blur: int = 24,
    crf: int = H264_CRF,
) -> bytes:
    """Fit the whole frame inside the target and fill the margins.

    The fill is a blurred, over-scaled copy of the frame rather than a flat colour.
    That is what social platforms and every editing tool do for this case, and the
    reason is worth stating: a flat bar reads as a mistake, a blurred extension reads
    as a deliberate frame. It also keeps the palette on-brand for free, since the
    fill is made of the ad's own pixels.

    Used when :func:`adml.crop.plan` decides a crop would discard too much — a 9:16
    clip reframed to 16:9 keeps about a third of its height, which usually means
    losing the model's face.
    """
    info = probe(data)
    filters = (
        f"split[bg][fg];"
        # increase_scale then crop guarantees the background covers the target even
        # when the aspect change is extreme.
        f"[bg]scale={out_width}:{out_height}:force_original_aspect_ratio=increase,"
        f"crop={out_width}:{out_height},boxblur={blur}:2[blurred];"
        f"[fg]scale={out_width}:{out_height}:force_original_aspect_ratio=decrease[fitted];"
        f"[blurred][fitted]overlay=(W-w)/2:(H-h)/2"
    )
    return _run_filtergraph(data, info.container, filters, crf=crf)


def encode_mp4(frames: list[np.ndarray], fps: float, *, crf: int = H264_CRF) -> bytes:
    """Encode RGB frames to an H.264 MP4.

    Deterministic on purpose — same frames in, same bytes out — because the cost
    governor's cache is keyed on content hashes and the pipeline's tests assert
    that identical requests do not pay twice.  Three settings buy that:
    ``-threads 1`` (x264's slice threading makes the bitstream depend on how work
    was scheduled), ``-map_metadata -1`` (strips the encoding timestamp), and
    ``-fflags +bitexact`` (drops the encoder version string from the container).

    Dimensions are rounded up to even numbers.  ``yuv420p`` subsamples chroma by
    two in each direction and simply cannot represent an odd edge — and the mock
    provider's default 9:16 frame at a 240 px long edge is 135 px wide, so this is
    the common case rather than a guard against exotic input.
    """
    if not frames:
        raise ValueError("cannot encode an empty frame list")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    ffmpeg = _require_ffmpeg("mp4")

    height, width = frames[0].shape[:2]
    even_w, even_h = width + (width % 2), height + (height % 2)
    payload = b"".join(np.ascontiguousarray(f, dtype=np.uint8).tobytes() for f in frames)

    # A seekable output: MP4's moov atom is written after the media, so ffmpeg
    # rewinds to the front on close. Piping to stdout cannot do that without
    # fragmenting the file, and a fragmented MP4 is not what a provider hands back.
    path = _write_temp(b"", "mp4")
    try:
        command = [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-fflags",
            "+bitexact",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-map_metadata",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
        ]
        if (even_w, even_h) != (width, height):
            command += ["-vf", f"pad={even_w}:{even_h}"]
        command += ["-movflags", "+faststart", "-f", "mp4", path]

        proc = subprocess.run(
            command, input=payload, capture_output=True, timeout=SUBPROCESS_TIMEOUT_S
        )
        if proc.returncode != 0:
            raise ClipDecodeError(
                f"ffmpeg encode failed: {proc.stderr.decode('utf-8', 'replace').strip()}"
            )
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)
