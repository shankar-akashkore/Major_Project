"""Audio for the delivered ad: a music bed, an optional voiceover, mixed locally.

**Why this is not generated.**  Veo 3.1 produces native audio and costs $0.40 per
second — one 9 s clip would be $3.60, a tenth of this project's entire budget, for a
soundtrack. A royalty-free bed mixed with ffmpeg is free, gives full control over the
mix, and is what the plan committed to.

Two things here are deliberately strict.

**A bed without a recorded licence is refused.**  :class:`AudioBed` requires a
source, a licence name and a URL, and :func:`mix` will not accept anything else.
This is not paperwork: an ad is a commercial artefact, a submitted project is a
published one, and "I found the mp3 in a folder" is how a music rights claim
happens. The licence travels into the delivery manifest so every rendered video can
say where its audio came from.

**Loudness is normalised to a stated target, not to whatever the file happened to
be.**  Social platforms normalise playback to about -14 LUFS; a bed mastered louder
gets turned down on upload, which changes the mix balance the ad was approved with.
``loudnorm`` measures and corrects in one pass, so what is delivered is what plays.

Nothing in this module ships audio content.  :func:`tone_bed` synthesises a plainly
artificial tone so the mixing path can be developed and tested without a music file
present, and it is labelled as a test signal rather than as music.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from .video import (
    SUBPROCESS_TIMEOUT_S,
    ClipDecodeError,
    _require_ffmpeg,
    _write_temp,
    probe,
)

#: Integrated loudness target, in LUFS. -14 is what Spotify, YouTube and Instagram
#: normalise to; mastering above it just gets attenuated on upload.
TARGET_LUFS = -14.0
#: True peak ceiling, in dBTP. -1.0 leaves headroom for lossy-codec overshoot.
TARGET_TRUE_PEAK = -1.0

#: How far the bed is pulled down under a voiceover, in dB. 12 is the usual
#: broadcast starting point for speech over music.
VOICE_DUCK_DB = 12.0
#: Fade lengths, in seconds. A hard cut at the end of a 9 s clip reads as a file
#: that got truncated.
FADE_IN_S = 0.4
FADE_OUT_S = 0.8


class AudioError(RuntimeError):
    """The mix could not be produced."""


class LicenceMissing(AudioError):
    """A bed was supplied without provenance, and was refused.

    Separate from :class:`AudioError` so a caller can distinguish "the mix failed"
    from "you may not use this music", which are different problems with different
    fixes.
    """


@dataclass(frozen=True)
class AudioBed:
    """A music bed, with the provenance required to ship it.

    ``licence`` is free text on purpose — "CC0", "CC-BY 4.0", "Pixabay Content
    Licence" and "purchased single-use, invoice #1234" are all valid and none of
    them fits an enum. What is *not* optional is that something is written there.
    """

    data: bytes
    title: str
    source: str
    licence: str
    url: str = ""
    attribution_required: bool = False
    #: True for a synthesised test signal. Keeps a tone from being described as
    #: music in a delivery manifest.
    is_test_signal: bool = False

    def __post_init__(self) -> None:
        missing = [
            name for name in ("title", "source", "licence") if not str(getattr(self, name)).strip()
        ]
        if missing:
            raise LicenceMissing(
                f"an audio bed needs {', '.join(missing)} recorded before it can be "
                "mixed into a deliverable. An ad is a commercial artefact and a "
                "submitted project is a published one; audio with unknown provenance "
                "is a rights claim waiting to happen."
            )
        if not self.data:
            raise AudioError(f"audio bed {self.title!r} carries no data")

    @property
    def credit(self) -> str:
        """The attribution line, for the delivery manifest and the report card."""
        if self.is_test_signal:
            return f"{self.title} (synthesised test signal, not music)"
        text = f"{self.title} — {self.source} ({self.licence})"
        return f"{text} {self.url}".strip() if self.url else text

    def describe(self) -> dict[str, object]:
        return {
            "title": self.title,
            "source": self.source,
            "licence": self.licence,
            "url": self.url,
            "attribution_required": self.attribution_required,
            "is_test_signal": self.is_test_signal,
            "credit": self.credit,
        }


def tone_bed(seconds: float = 10.0, *, hz: float = 220.0) -> AudioBed:
    """A synthesised sine bed, for exercising the mix without a music file.

    Explicitly labelled a test signal.  The point of having it is that the whole
    ffmpeg path — mixing, ducking, loudness normalisation, fades — can be tested and
    demonstrated on a machine with no audio assets at all, and cannot be mistaken in
    a manifest for something a viewer would want to hear.
    """
    ffmpeg = _require_ffmpeg("audio")
    path = _write_temp(b"", "m4a")
    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={hz}:duration={seconds:.3f}",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-map_metadata",
                "-1",
                "-fflags",
                "+bitexact",
                "-f",
                "mp4",
                path,
            ],
            capture_output=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
        if proc.returncode != 0:
            raise AudioError(
                f"could not synthesise a tone: {proc.stderr.decode('utf-8', 'replace').strip()}"
            )
        with open(path, "rb") as fh:
            data = fh.read()
    finally:
        os.unlink(path)

    return AudioBed(
        data=data,
        title=f"{hz:.0f} Hz sine",
        source="synthesised by adml.audio.tone_bed",
        licence="not applicable — generated signal",
        is_test_signal=True,
    )


@dataclass(frozen=True)
class MixReport:
    """What the mix actually did."""

    duration_seconds: float
    has_voice: bool
    target_lufs: float
    bed: dict[str, object]
    filtergraph: str

    def summary(self) -> str:
        voice = "bed + voiceover, bed ducked" if self.has_voice else "bed only"
        return (
            f"{voice}, normalised to {self.target_lufs:.0f} LUFS, "
            f"{self.duration_seconds:.2f}s — {self.bed['credit']}"
        )


def _fade_graph(label: str, duration: float) -> str:
    """Fade in at the start and out at the end of ``duration`` seconds."""
    fade_out_at = max(0.0, duration - FADE_OUT_S)
    return f"[{label}]afade=t=in:st=0:d={FADE_IN_S},afade=t=out:st={fade_out_at:.3f}:d={FADE_OUT_S}"


def mix(
    video: bytes,
    bed: AudioBed,
    *,
    voice: bytes | None = None,
    duck_db: float = VOICE_DUCK_DB,
    target_lufs: float = TARGET_LUFS,
) -> tuple[bytes, MixReport]:
    """Attach audio to a clip, trimmed and faded to the clip's own length.

    The bed is looped if it is shorter than the clip and cut if longer, then faded at
    both ends.  A voiceover, when present, plays at full level over a ducked bed.

    Ducking is done with ``sidechaincompress`` keyed off the voice, not with a fixed
    volume cut.  The difference is audible and it is the whole point: a fixed cut
    leaves the music quiet through every pause, while a side-chained compressor lifts
    it back between phrases, which is what a mix is supposed to sound like.
    """
    info = probe(video)
    duration = info.duration_seconds
    if duration <= 0:
        raise AudioError("cannot mix audio onto a clip of unknown duration")

    ffmpeg = _require_ffmpeg("audio")
    video_path = _write_temp(video, info.container)
    bed_path = _write_temp(bed.data, "m4a")
    voice_path = _write_temp(voice, "m4a") if voice else None
    out_path = _write_temp(b"", "mp4")

    inputs = ["-i", video_path, "-stream_loop", "-1", "-i", bed_path]
    if voice_path:
        inputs += ["-i", voice_path]

    if voice_path:
        graph = (
            # The bed is the compressor's input and the voice its side-chain key, so
            # the music dips only while there is speech.
            f"[1:a]atrim=0:{duration:.3f},asetpts=N/SR/TB[bedraw];"
            f"[2:a]atrim=0:{duration:.3f},asetpts=N/SR/TB,asplit[voice][key];"
            f"[bedraw][key]sidechaincompress="
            f"threshold=0.05:ratio={max(1.5, duck_db / 4):.2f}:attack=20:release=400[ducked];"
            f"[ducked][voice]amix=inputs=2:duration=first:dropout_transition=0[mixed];"
            f"{_fade_graph('mixed', duration)},"
            f"loudnorm=I={target_lufs}:TP={TARGET_TRUE_PEAK}:LRA=11[out]"
        )
    else:
        graph = (
            f"[1:a]atrim=0:{duration:.3f},asetpts=N/SR/TB[bedraw];"
            f"{_fade_graph('bedraw', duration)},"
            f"loudnorm=I={target_lufs}:TP={TARGET_TRUE_PEAK}:LRA=11[out]"
        )

    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-fflags",
                "+bitexact",
                *inputs,
                "-filter_complex",
                graph,
                "-map",
                "0:v:0",
                "-map",
                "[out]",
                # The video is copied, not re-encoded: the mix must not silently cost
                # a generation another lossy pass.
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-shortest",
                "-threads",
                "1",
                "-map_metadata",
                "-1",
                "-fflags",
                "+bitexact",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                out_path,
            ],
            capture_output=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
        if proc.returncode != 0:
            raise AudioError(
                f"the audio mix failed: {proc.stderr.decode('utf-8', 'replace').strip()}"
            )
        with open(out_path, "rb") as fh:
            data = fh.read()
    finally:
        for path in (video_path, bed_path, out_path):
            os.unlink(path)
        if voice_path:
            os.unlink(voice_path)

    result = probe(data)
    if not result.has_audio:
        raise AudioError(
            "ffmpeg reported success but the output has no audio stream, so the mix "
            "silently produced a silent video"
        )
    return data, MixReport(
        duration_seconds=result.duration_seconds,
        has_voice=voice is not None,
        target_lufs=target_lufs,
        bed=bed.describe(),
        filtergraph=graph,
    )


def measure_loudness(data: bytes) -> float:
    """Integrated loudness of a file's audio, in LUFS.

    Used to check that normalisation did what it claimed. ``loudnorm`` runs in one
    pass by default, which corrects toward the target rather than landing exactly on
    it, so the delivered figure is worth measuring rather than assuming.
    """
    info = probe(data)
    if not info.has_audio:
        raise AudioError("the file has no audio stream to measure")

    ffmpeg = _require_ffmpeg("audio")
    path = _write_temp(data, info.container)
    try:
        proc = subprocess.run(
            [ffmpeg, "-v", "info", "-i", path, "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
            capture_output=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    finally:
        os.unlink(path)

    text = proc.stderr.decode("utf-8", "replace")
    marker = text.rfind("I:")
    if marker < 0:
        raise ClipDecodeError(f"could not read integrated loudness from ffmpeg: {text[-300:]}")
    for token in text[marker + 2 :].split():
        try:
            return float(token)
        except ValueError:
            continue
    raise ClipDecodeError("ffmpeg reported an integrated loudness that did not parse")


def voice_duration_estimate(text: str, *, words_per_minute: float = 150.0) -> float:
    """Rough spoken length of a caption, in seconds.

    Used before any TTS exists, to warn when a caption cannot fit the clip.  150 wpm
    is a measured average for advertising voiceover — conversational speech runs
    faster, but read copy over music is deliberately slower.

    An estimate, and labelled as one: it does not know about the words. The point is
    only to catch the case where a 40-word caption is aimed at a 9 s slot, which no
    delivery of the mix can rescue.
    """
    words = len([w for w in text.split() if w.strip()])
    return words / max(1.0, words_per_minute) * 60.0


def caption_fits(text: str, duration_seconds: float, *, headroom: float = 0.85) -> bool:
    """Whether a caption plausibly reads inside the clip, leaving breathing room."""
    return voice_duration_estimate(text) <= duration_seconds * headroom
