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

import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
    #: True when this project synthesised the audio rather than licensing it.
    #:
    #: Distinct from ``is_test_signal``: a generated bed *is* the soundtrack, and
    #: describing it as a test tone would understate what ships. What it is not is
    #: licensed music, and a manifest that leaves a reader to assume otherwise is
    #: the same provenance failure as an unrecorded licence, pointing the other way.
    generated: bool = False

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
        if self.generated:
            return f"{self.title} (synthesised for this ad — no third-party rights)"
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
            "generated": self.generated,
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


# --- A bed when there is no music file -------------------------------------
#
# The delivered clips were silent, and the reason was not a bug: nothing ever
# supplied a bed, because this project ships no music and `AudioBed` refuses
# audio without a recorded licence. Both of those are correct and neither is an
# answer to "why does my advertisement have no sound".
#
# So the bed is synthesised. Not a sine — `tone_bed` already exists for testing
# the mixing path and nobody would ship it — but an actual chord progression,
# scored to the mood the user picked. It is modest music. What it is is free,
# unencumbered, deterministic, and *there*, which beats silence and beats a
# rights claim. A real licensed track still wins: drop one in the library
# (`load_library`) and it takes precedence.

#: MIDI note 69 is A4 = 440 Hz, which is the only tuning fact needed here.
_A4_MIDI = 69
_A4_HZ = 440.0

#: Triad shapes as semitone offsets from the root.
_MAJOR = (0, 4, 7)
_MINOR = (0, 3, 7)
_SUS4 = (0, 5, 7)


def _hz(midi: int) -> float:
    return _A4_HZ * (2.0 ** ((midi - _A4_MIDI) / 12.0))


@dataclass(frozen=True)
class _Score:
    """How one mood sounds.

    The parameters are the ones that actually separate an ad bed from a drone:
    which chords, how often they change, how bright the result is, and how much
    low end sits under it.
    """

    #: (root MIDI note, triad shape) per bar. Length sets the harmonic rhythm —
    #: eight short bars reads as urgency, four long ones as composure.
    progression: tuple[tuple[int, tuple[int, ...]], ...]
    #: Top of the spectrum, in Hz. A bed is meant to sit under a voice and a
    #: product; anything with air at 12 kHz competes with the ad instead.
    lowpass_hz: int
    pad_gain: float
    bass_gain: float
    #: Attack and release as a fraction of one bar. Long swells for calm, short
    #: ones for energy — this is most of what makes a progression feel fast.
    swell: float


#: A3 is MIDI 57. Roots are written as MIDI numbers so a mood can be transposed
#: by changing one number rather than four frequencies.
_MOOD_SCORES: dict[str, _Score] = {
    "calm_premium": _Score(
        # Am - F - G - Em: no leading tone resolution, so it never arrives and
        # never demands attention. Which is what a premium bed is for.
        progression=((57, _MINOR), (53, _MAJOR), (55, _MAJOR), (52, _MINOR)),
        lowpass_hz=5000,
        pad_gain=0.20,
        bass_gain=0.24,
        swell=0.34,
    ),
    "warm_lifestyle": _Score(
        # C - F - Am - G: the most familiar progression in popular music, which
        # is the point. Warmth here means recognisable, not novel.
        progression=((48, _MAJOR), (53, _MAJOR), (57, _MINOR), (55, _MAJOR)),
        lowpass_hz=6500,
        pad_gain=0.22,
        bass_gain=0.26,
        swell=0.28,
    ),
    "bold_confident": _Score(
        # Dm - Bb - F - C, rooted low and open. Suspensions rather than triads on
        # the outer chords: a sus4 is unresolved without being sad.
        progression=((50, _MINOR), (46, _SUS4), (53, _MAJOR), (48, _SUS4)),
        lowpass_hz=8000,
        pad_gain=0.24,
        bass_gain=0.32,
        swell=0.18,
    ),
    "high_energy": _Score(
        # Eight bars instead of four, so the chords change twice as often over the
        # same clip. Nothing else about the sound says "fast" as clearly.
        progression=(
            (57, _MINOR),
            (53, _MAJOR),
            (48, _MAJOR),
            (55, _MAJOR),
            (57, _MINOR),
            (53, _MAJOR),
            (55, _MAJOR),
            (55, _SUS4),
        ),
        lowpass_hz=9000,
        pad_gain=0.24,
        bass_gain=0.30,
        swell=0.12,
    ),
}

#: Used when a mood has no score of its own. Warm rather than calm: an unmapped
#: mood is more likely to be an ordinary product ad than a luxury one.
_DEFAULT_SCORE = _MOOD_SCORES["warm_lifestyle"]


def generated_bed(mood: str = "warm_lifestyle", seconds: float = 10.0) -> AudioBed:
    """Synthesise a chord-progression bed for ``mood``.

    Built from one oscillator per note rather than from a frequency that steps
    through the progression: stepping a running oscillator's frequency is a phase
    discontinuity, which is a click, and a click every two and a half seconds is
    more noticeable than no music at all. Each note gets its own source, its own
    attack and release, and a place in the timeline.

    Deterministic — the same mood and duration produce identical bytes — so a
    golden replay is not disturbed by the soundtrack.
    """
    if seconds <= 0:
        raise AudioError("a bed needs a positive duration")

    score = _MOOD_SCORES.get(str(mood), _DEFAULT_SCORE)
    bars = len(score.progression)
    bar = seconds / bars
    fade = max(0.05, min(score.swell * bar, bar / 2 - 0.01))

    ffmpeg = _require_ffmpeg("audio")
    inputs: list[str] = []
    chains: list[str] = []
    labels: list[str] = []
    n = 0

    for index, (root, shape) in enumerate(score.progression):
        start_ms = int(index * bar * 1000)
        # The bass an octave below the root, then the triad above it. Voices are
        # attenuated as they climb so the chord reads as one sound rather than as
        # three tones of equal weight.
        voices = [(_hz(root - 12), score.bass_gain)]
        voices += [
            (_hz(root + semitones), score.pad_gain / (voice + 1))
            for voice, semitones in enumerate(shape)
        ]
        for frequency, gain in voices:
            inputs += ["-f", "lavfi", "-i", f"sine=frequency={frequency:.3f}:duration={bar:.4f}"]
            chains.append(
                f"[{n}:a]afade=t=in:st=0:d={fade:.4f},"
                f"afade=t=out:st={bar - fade:.4f}:d={fade:.4f},"
                f"volume={gain:.4f},adelay={start_ms}|{start_ms}[v{n}]"
            )
            labels.append(f"[v{n}]")
            n += 1

    chains.append(
        "".join(labels) + f"amix=inputs={len(labels)}:normalize=0,"
        # Rolled off at both ends: below 60 Hz is rumble a phone speaker cannot
        # reproduce but a loudness meter still counts, and the top is where the
        # bed would otherwise compete with whatever the ad is actually saying.
        f"highpass=f=60,lowpass=f={score.lowpass_hz},"
        "aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[out]"
    )

    path = _write_temp(b"", "m4a")
    try:
        proc = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                *inputs,
                "-filter_complex",
                ";".join(chains),
                "-map",
                "[out]",
                "-t",
                f"{seconds:.4f}",
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
                f"could not synthesise a {mood} bed: "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}"
            )
        with open(path, "rb") as fh:
            data = fh.read()
    finally:
        os.unlink(path)

    return AudioBed(
        data=data,
        title=f"{str(mood).replace('_', ' ')} bed",
        source="synthesised by adml.audio.generated_bed",
        licence="generated by this project — no third-party rights",
        generated=True,
    )


# --- A library of real, licensed music -------------------------------------


#: Audio containers the mixer will accept from a library directory.
LIBRARY_SUFFIXES = (".m4a", ".mp3", ".wav", ".aac", ".ogg", ".flac")


@dataclass(frozen=True)
class LicensedTrack:
    """A bed from the library, with the moods it suits.

    ``moods`` is advisory. A track that lists none is eligible for every mood
    rather than for none — the common case is a user who dropped in one piece of
    music and wants it used, and refusing it because they did not fill in a field
    would be the tool being clever at their expense.
    """

    bed: AudioBed
    moods: frozenset[str] = frozenset()

    def suits(self, mood: str) -> bool:
        return not self.moods or str(mood) in self.moods


def load_library(directory: str | os.PathLike[str]) -> list[LicensedTrack]:
    """Read every track in ``directory`` that has its licence recorded beside it.

    Each audio file needs a JSON sidecar of the same stem::

        bright-morning.m4a
        bright-morning.json   {"title": ..., "source": ..., "licence": ...,
                               "url": ..., "moods": ["warm_lifestyle"]}

    A file **without** a sidecar is skipped, not loaded with blanks filled in.
    This is the same rule :class:`AudioBed` enforces at construction, applied one
    step earlier: the failure mode being prevented is a track that got into a
    published deliverable because a directory scan was permissive, and the person
    who could have caught it never saw a prompt.

    Returns an empty list for a directory that does not exist, which is the normal
    state and not an error — the caller falls back to a synthesised bed.
    """
    root = Path(directory)
    if not root.is_dir():
        return []

    tracks: list[LicensedTrack] = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in LIBRARY_SUFFIXES:
            continue
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            continue
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        try:
            bed = AudioBed(
                data=path.read_bytes(),
                title=str(meta.get("title", "")),
                source=str(meta.get("source", "")),
                licence=str(meta.get("licence", "")),
                url=str(meta.get("url", "")),
                attribution_required=bool(meta.get("attribution_required", False)),
            )
        except (AudioError, OSError):
            # Includes LicenceMissing: a sidecar that exists but leaves the licence
            # blank is the same problem as no sidecar, and skipping is the same
            # answer.
            continue
        moods = meta.get("moods") or []
        tracks.append(
            LicensedTrack(
                bed=bed,
                moods=frozenset(str(m) for m in moods) if isinstance(moods, list) else frozenset(),
            )
        )
    return tracks


def choose_bed(
    library: Sequence[LicensedTrack],
    mood: str,
    seconds: float = 10.0,
) -> AudioBed:
    """The bed for a job: licensed music if there is any, synthesis otherwise.

    Preference order is licensed-and-suited, then licensed-at-all, then generated.
    Real music always beats synthesis — the generated bed exists so that a clip is
    never silent, not because it is better than a track someone chose.

    Never returns ``None``. A delivered advertisement has a soundtrack; the
    question this answers is only which one.
    """
    suited = [track for track in library if track.suits(mood)]
    if suited:
        # Chosen by mood, then by name, so the same job picks the same track every
        # run. A rotating soundtrack would make two runs of one golden job
        # disagree for a reason that has nothing to do with the model.
        return min(suited, key=lambda t: t.bed.title).bed
    if library:
        return min(library, key=lambda t: t.bed.title).bed
    return generated_bed(mood, seconds)


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
