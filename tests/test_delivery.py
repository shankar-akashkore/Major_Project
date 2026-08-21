"""Stage 8: reframing, audio, previews and the bundle.

The claim delivery has to support is that reframing by saliency beats centring, so
the first tests measure that rather than asserting the code ran.  The rest guard the
two ways this stage can be quietly wrong: a bundle that ships without its media, and
audio whose licence nobody recorded.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from adml import audio as A
from adml import crop as C
from adml import video as V
from adschema import (
    AspectRatio,
    CameraAngle,
    Composition,
    JobState,
    Lighting,
    MotionIntent,
    Platform,
)
from adworker import Pipeline, build_bundle, deliver, preview_frame, report_card
from PIL import Image

needs_ffmpeg = pytest.mark.skipif(V.FFMPEG is None, reason="ffmpeg is not installed")


# --- Fixtures ----------------------------------------------------------------


def _offset_subject(
    n: int = 24,
    w: int = 240,
    h: int = 240,
    composition: Composition = Composition.RULE_OF_THIRDS_LEFT,
) -> list[np.ndarray]:
    """Frames from the mock renderer with the subject off-centre.

    The case that matters: the design-space sampler puts the subject off-centre in
    three of its five composition settings, so a centre crop mutilates the majority
    of candidates rather than an unlucky few.

    Rendered rather than drawn as a bright disc on a flat field, and that is not
    fussiness. Spectral-residual saliency *rings* on synthetic hard edges — measured
    on a single disc over a flat background the map came out periodic with its
    column-wise minimum sitting exactly on the disc, so the crop went to the wrong
    side of the frame. The mock renderer produces gradients and grain, and on those
    the map puts 79% of its mass in the correct half. See the caveat in
    ``adml.crop``'s module docstring.
    """
    from adproviders.mock import _render_frame

    frames = []
    for i in range(n):
        image = _render_frame(
            size=(w, h),
            palette=["#2b3a55", "#ce7777", "#f2e7d5"],
            angle=CameraAngle.EYE_LEVEL,
            lighting=Lighting.SOFT_DIFFUSED,
            composition=composition,
            label="",
            t=i / max(1, n - 1),
            motion=MotionIntent.HERO_TURN,
            rng_seed=3 + i,
        )
        frames.append(np.asarray(image))
    return frames


@pytest.fixture
async def delivered(storage, governor, make_request):
    """A completed job with delivery run, using a labelled test tone for audio."""
    import adproviders as P

    bed = A.tone_bed(seconds=5.0) if V.FFMPEG else None
    pipeline = Pipeline(
        storage=storage,
        governor=governor,
        image_provider=P.MockImageProvider(storage),
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
        audio_bed=bed,
    )
    record = await pipeline.run(make_request())
    assert record.state is JobState.COMPLETED, record.error
    return record


# --- The claim: saliency beats centring --------------------------------------


def test_a_saliency_placed_window_keeps_more_than_a_centre_crop():
    """The whole reason this module exists rather than a one-line centre crop."""
    frames = _offset_subject(composition=Composition.RULE_OF_THIRDS_LEFT)
    height, width = frames[0].shape[:2]
    target = AspectRatio.VERTICAL_9_16.ratio

    plan = C.plan(frames, target)
    centre = C.retained_by(frames, C.centre_plan(width, height, target))

    assert plan.retained_salience > centre, (plan.retained_salience, centre)
    # And the window really is displaced toward the subject rather than tying with
    # the centre by accident.
    assert plan.window.x < C.centre_plan(width, height, target).x


def test_an_extreme_aspect_change_pads_rather_than_cropping():
    """A 9:16 source to 16:9 keeps about a third of its height.

    Cropping that hard usually means losing the model's face, so the plan switches
    to letterboxing and says why.
    """
    frames = _offset_subject(w=136, h=240)
    plan = C.plan(frames, AspectRatio.LANDSCAPE_16_9.ratio)
    assert plan.mode == "pad", plan.summary()
    assert "discard" in plan.reason
    # The judgement is driven by the measurement, not hardcoded to the ratio pair.
    assert plan.retained_salience < C.MIN_RETAINED_SALIENCE


def test_an_unchanged_ratio_is_a_noop_despite_even_dimension_rounding():
    """136x240 is not exactly 9:16, and must not trigger a two-pixel re-encode.

    Encoders pad odd dimensions to even, so a nominally 9:16 clip is delivered at
    136x240 = 0.5667 rather than 0.5625. A ratio epsilon tight enough to be useful
    elsewhere calls that a reframe.
    """
    frames = [np.zeros((240, 136, 3), dtype=np.uint8) for _ in range(4)]
    plan = C.plan(frames, AspectRatio.VERTICAL_9_16.ratio)
    assert plan.is_noop
    assert plan.retained_salience == 1.0


@pytest.mark.parametrize("motion", list(MotionIntent))
def test_holding_the_window_still_costs_almost_nothing_on_real_clip_motion(motion):
    """The measurement that justifies a static crop window.

    A subject in an 8-10 s ad works within the frame; it does not cross it. Measured
    on the mock renderer's own motion, re-taken after the motion axis was rewritten
    from camera moves to performances: a per-frame tracker would gain 0.0019 at worst
    (`hero_turn`, whose subject swings 8.0% of the short edge) and 0.0000 at best
    (`product_reveal`, which is vertical and central). Paying that to avoid a
    jittering frame is obviously right, and it is only obviously right because the
    number was taken.

    Parametrised over the whole axis rather than a chosen three, so a motion level
    added later cannot quietly break the assumption the crop window rests on.
    """
    from adproviders.mock import _render_frame

    frames = [
        np.asarray(
            _render_frame(
                size=(240, 240),
                palette=["#2b3a55", "#ce7777", "#f2e7d5"],
                angle=CameraAngle.EYE_LEVEL,
                lighting=Lighting.SOFT_DIFFUSED,
                composition=Composition.CENTERED_HERO,
                label="",
                t=i / 23,
                motion=motion,
                rng_seed=3 + i,
            )
        )
        for i in range(24)
    ]
    plan = C.plan(frames, AspectRatio.VERTICAL_9_16.ratio)
    assert plan.tracking_gain < 0.01, plan.summary()


def test_the_tracking_gain_metric_responds_to_a_subject_that_really_travels():
    """A sensitivity check on the metric, not a claim about crop quality.

    Without this, the ~0 gains above would be equally consistent with a metric that
    simply never reports anything. Synthetic frames are acceptable here precisely
    because the assertion is a *relative* comparison of one metric against itself —
    spectral-residual saliency is unreliable on flat synthetic content
    (see `adml.crop`'s module docstring), so this fixture says nothing about where a
    crop window should actually go.
    """

    def clip(travel: float, n: int = 24, w: int = 240, h: int = 240) -> list[np.ndarray]:
        frames = []
        ys, xs = np.mgrid[0:h, 0:w]
        for i in range(n):
            centre = int(w * (0.5 - travel / 2 + travel * (i / (n - 1))))
            frame = np.full((h, w, 3), 30, dtype=np.uint8)
            frame[(xs - centre) ** 2 + (ys - h // 2) ** 2 <= 22**2] = 245
            frames.append(frame)
        return frames

    still = C.plan(clip(0.0), AspectRatio.VERTICAL_9_16.ratio)
    moving = C.plan(clip(0.7), AspectRatio.VERTICAL_9_16.ratio)

    assert still.tracking_gain < 0.02
    assert moving.tracking_gain > 0.05
    assert moving.centre_travel > still.centre_travel


def test_the_safe_area_is_part_of_the_objective():
    """A window that frames the product but buries it under the caption bar has not
    solved the problem, so safe-area saliency is weighted above in-frame saliency."""
    height, width = 240, 240
    ys, xs = np.mgrid[0:height, 0:width]
    # Two equal blobs: one high in the frame, one in Reels' bottom chrome band.
    frames = []
    for _ in range(8):
        frame = np.full((height, width, 3), 20, dtype=np.uint8)
        frame[(xs - 120) ** 2 + (ys - 60) ** 2 <= 24**2] = 240
        frame[(xs - 120) ** 2 + (ys - 215) ** 2 <= 24**2] = 240
        frames.append(frame)

    safe = tuple(Platform.INSTAGRAM_REELS.safe_area)
    aware = C.plan(frames, 1.0, safe_area=safe)
    naive = C.plan(frames, 1.0, safe_area=None)
    # The safe-area-aware window sits higher, keeping the blob the platform will not
    # cover; without weighting the two blobs are interchangeable.
    assert aware.window.y <= naive.window.y


# --- Rendering ---------------------------------------------------------------


@needs_ffmpeg
def test_a_crop_render_has_the_requested_geometry_and_keeps_its_duration():
    frames = _offset_subject(n=30)
    source = V.encode_mp4(frames, fps=10)
    plan = C.plan(frames, AspectRatio.VERTICAL_9_16.ratio)

    out = V.render_crop(
        source,
        x=plan.window.x,
        y=plan.window.y,
        width=plan.window.width,
        height=plan.window.height,
        out_width=270,
        out_height=480,
    )
    probe = V.probe(out)
    assert (probe.width, probe.height) == (270, 480)
    assert abs(probe.duration_seconds - 3.0) < 0.1


@needs_ffmpeg
def test_a_padded_render_fills_the_target_without_stretching():
    frames = _offset_subject(n=20, w=136, h=240)
    out = V.render_padded(V.encode_mp4(frames, fps=10), out_width=480, out_height=270)
    probe = V.probe(out)
    assert (probe.width, probe.height) == (480, 270)


# --- Audio -------------------------------------------------------------------


def test_a_bed_without_provenance_cannot_be_constructed():
    """An ad is a commercial artefact and a submitted project is a published one."""
    with pytest.raises(A.LicenceMissing, match="licence"):
        A.AudioBed(data=b"audio", title="Track", source="a folder", licence="   ")


def test_a_test_tone_says_it_is_not_music():
    if V.FFMPEG is None:
        pytest.skip("ffmpeg is not installed")
    bed = A.tone_bed(seconds=1.0)
    assert bed.is_test_signal
    assert "not music" in bed.credit


@needs_ffmpeg
def test_a_short_bed_is_looped_to_the_clip_and_normalised_to_target():
    """Loudness is checked by measuring the output, not by trusting the filter.

    ``loudnorm`` runs single-pass by default, which corrects toward the target rather
    than landing exactly on it, so the delivered figure is worth reading back.
    """
    clip = V.encode_mp4([np.full((64, 64, 3), i * 3, dtype=np.uint8) for i in range(60)], fps=10)
    bed = A.tone_bed(seconds=2.0)  # a third of the clip's length

    mixed, report = A.mix(clip, bed)
    probe = V.probe(mixed)

    assert probe.has_audio
    assert abs(probe.duration_seconds - 6.0) < 0.15
    assert abs(A.measure_loudness(mixed) - A.TARGET_LUFS) < 1.5
    assert not report.has_voice


@needs_ffmpeg
def test_a_voiceover_ducks_the_bed():
    clip = V.encode_mp4([np.full((64, 64, 3), i * 3, dtype=np.uint8) for i in range(40)], fps=10)
    mixed, report = A.mix(clip, A.tone_bed(seconds=4.0), voice=A.tone_bed(seconds=4.0, hz=700).data)
    assert report.has_voice
    assert "sidechaincompress" in report.filtergraph
    assert V.probe(mixed).has_audio


def test_a_caption_too_long_for_the_clip_is_caught_before_any_mixing():
    short = "Glow that lasts"
    long = (
        "Glow that lasts all day and into the night with our new advanced hydrating "
        "serum formula clinically proven across twelve weeks of continuous use"
    )
    assert A.caption_fits(short, 9.0)
    assert not A.caption_fits(long, 9.0)
    assert A.voice_duration_estimate(long) > 9.0


# --- Previews ----------------------------------------------------------------


@needs_ffmpeg
async def test_a_preview_draws_the_platforms_own_safe_area(delivered, storage):
    """Bands come from `Platform.safe_area`, the same rectangle the score uses.

    A hand-placed mock would look authoritative while disagreeing with the number
    printed beside it, which is worse than having no preview.
    """
    video = delivered.result.winner.video
    png = preview_frame(video, Platform.INSTAGRAM_REELS, storage, caption="Glow that lasts")
    image = np.asarray(Image.open(io.BytesIO(png)).convert("L"), dtype=float)

    safe = Platform.INSTAGRAM_REELS.safe_area
    height = image.shape[0]
    top_band = image[: int(height * safe.top) // 2].mean()
    middle = image[int(height * 0.4) : int(height * 0.6)].mean()
    # The chrome band is darkened, so it reads darker than the unobscured middle.
    assert top_band < middle, (top_band, middle)


@needs_ffmpeg
async def test_previews_are_produced_for_each_distinct_safe_area(delivered):
    previews = delivered.result.delivery.previews
    assert set(previews) == {p.value for p in __import__("adworker").delivery.PREVIEW_PLATFORMS}


# --- The bundle --------------------------------------------------------------


@needs_ffmpeg
async def test_the_bundle_actually_contains_the_media(delivered, storage):
    """The regression guard for a bug that shipped a 2 kB download.

    ``build_bundle`` used to read the report off ``result.delivery``, which the
    pipeline assigns only *after* delivery returns — so the archive silently held
    report.json and nothing else, with no error and no empty-file warning.
    """
    report = delivered.result.delivery
    data = storage.get_bytes(report.bundle.key)

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        videos = {n for n in names if n.endswith(".mp4")}
        previews = {n for n in names if n.endswith(".png")}

        assert len(videos) == len(report.renders), names
        assert len(previews) == len(report.previews), names
        assert {"report.json", "README.txt"} <= names
        # Every clip in the zip is a real, probeable video at its stated ratio.
        for name in videos:
            probe = V.probe(archive.read(name))
            assert probe.duration_seconds > 0
            assert probe.width > 0 and probe.height > 0

    assert len(data) > 50_000, f"bundle is only {len(data)} bytes"


@needs_ffmpeg
async def test_the_bundle_readme_states_the_scores_are_not_click_rates(delivered, storage):
    """The bundle outlives the conversation that produced it."""
    data = storage.get_bytes(delivered.result.delivery.bundle.key)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        readme = archive.read("README.txt").decode()
    assert "NOT PREDICTED CLICK RATES" in readme
    assert "STUB" in readme  # heuristic-0 with no trained model present


@needs_ffmpeg
async def test_a_report_card_carries_the_model_that_produced_the_score(delivered):
    """A card is the artefact most likely to be read out of context."""
    ranked = delivered.result.ranking[0]
    card = report_card(ranked, delivered.request)
    assert card["prediction"]["model_version"] == "heuristic-0"
    assert card["prediction"]["is_stub"] is True
    assert card["generation"]["seed_honoured"] in (True, False)
    assert card["rank"] == 1


@needs_ffmpeg
async def test_the_report_json_round_trips(delivered, storage):
    data = storage.get_bytes(delivered.result.delivery.bundle.key)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        payload = json.loads(archive.read("report.json"))
    assert payload["job_id"] == delivered.result.job_id
    assert len(payload["candidates"]) == len(delivered.result.ranking)
    assert payload["delivery"]["renders"]


# --- Reporting the loss ------------------------------------------------------


@needs_ffmpeg
async def test_a_lossy_reframe_is_warned_about_not_silently_shipped(delivered):
    """Reframing is lossy and the person publishing needs to be told."""
    report = delivered.result.delivery
    worst = report.worst_reframe
    assert worst is not None
    if worst.retained_salience < 0.80:
        assert any(worst.aspect_ratio in w for w in report.warnings), report.warnings


@needs_ffmpeg
async def test_the_smart_crop_benefit_is_reported_as_a_comparison(delivered):
    """A benefit is a comparison. "Retains 84%" alone is not a claim about anything."""
    lossy = [r for r in delivered.result.delivery.renders if r.mode != "none"]
    assert lossy
    for render in lossy:
        assert render.centre_crop_salience is not None
        assert render.improvement_over_centre is not None


@needs_ffmpeg
async def test_a_job_with_no_music_library_still_delivers_a_soundtrack(
    storage, governor, make_request
):
    """An advertisement with no sound is not a deliverable.

    This test used to assert the opposite — that no bed meant no audio and a note
    explaining why — and it passed for as long as it took someone to watch the
    output. The reasoning behind it was sound and is unchanged: this project ships
    no music, and `AudioBed` refuses audio whose licence is not recorded. The
    conclusion drawn from it was not. "We own no music" is answered by synthesising
    some, not by shipping silence and describing it.
    """
    import adproviders as P

    pipeline = Pipeline(
        storage=storage,
        governor=governor,
        image_provider=P.MockImageProvider(storage),
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
        audio_bed=None,
        audio_library=[],
    )
    record = await pipeline.run(make_request())
    audio = record.result.delivery.audio

    assert audio.attached
    # Synthesised, and saying so. A manifest that leaves the reader to assume the
    # soundtrack was licensed is the same provenance failure as an unrecorded
    # licence, pointing the other way.
    assert audio.generated
    assert not audio.is_test_signal
    assert "no third-party rights" in audio.credit
    assert audio.measured_lufs is not None and -17.0 < audio.measured_lufs < -11.0


@needs_ffmpeg
async def test_every_delivered_clip_has_the_soundtrack_not_only_the_winner(
    storage, governor, make_request
):
    """The runner-up is playable in the UI, so it cannot be the silent one.

    Delivery reframes the winner only, deliberately — nine re-encodes for output
    nobody asked for is not a saving worth making. Audio is the exception because
    the video track is *copied*, so mixing a runner-up costs a stream copy rather
    than a generation.
    """
    import adproviders as P
    from adml import video as V

    pipeline = Pipeline(
        storage=storage,
        governor=governor,
        image_provider=P.MockImageProvider(storage),
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
    )
    record = await pipeline.run(make_request())
    videos = record.result.videos
    assert len(videos) > 1, "this test needs a runner-up to be about anything"

    for video in videos:
        mixed = video.platform_renders.get("audio")
        assert mixed is not None, f"clip {video.index} was delivered without audio"
        assert V.probe(storage.get_bytes(mixed.key)).has_audio


@needs_ffmpeg
def test_delivery_called_without_a_bed_still_says_why_it_is_silent(storage, delivered):
    """The direct entry point keeps its old contract, and explains itself.

    `deliver()` is called by scripts and by tests as well as by the pipeline, and a
    caller that passes no bed gets silence — the bed is chosen upstream now, so this
    is the one path where `None` still reaches the mixer. What it must not do is
    return a report indistinguishable from a successful mix.
    """
    report = deliver(delivered.result, delivered.request, storage, bed=None)

    assert not report.audio.attached
    assert "silent" in report.audio.note


@needs_ffmpeg
async def test_delivery_costs_nothing(delivered, ledger):
    """Every part of stage 8 is local. A delivery run must not move the ledger."""
    assert await ledger.total_spent() == 0.0
    assert delivered.result.total_cost_usd == 0.0


def test_bundling_a_job_with_no_ranking_reports_rather_than_raising(storage, make_request):
    from adschema import JobResult

    request = make_request()
    result = JobResult(job_id=request.job_id)
    report = deliver(result, request, storage)
    assert report.renders == []
    assert any("nothing was delivered" in w for w in report.warnings)
    # And a bundle is still buildable, so a failed job still hands something over.
    assert build_bundle(result, request, storage, report)


# --- The music library -------------------------------------------------------
#
# A synthesised bed is the fallback, not the goal. When the user supplies real
# music it has to win, and when they supply music with no recorded licence it has
# to lose — silently skipping is the right answer there, because the alternative
# is a rights claim against a submitted project.


def _write_track(directory, stem: str, meta: dict | None) -> None:
    """A tiny valid audio file plus its sidecar, or the file alone."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.m4a").write_bytes(A.tone_bed(seconds=1.0).data)
    if meta is not None:
        (directory / f"{stem}.json").write_text(json.dumps(meta))


@needs_ffmpeg
def test_a_licensed_track_beats_the_synthesised_one(tmp_path):
    _write_track(
        tmp_path,
        "bright-morning",
        {
            "title": "Bright Morning",
            "source": "Pixabay",
            "licence": "Pixabay Content Licence",
            "url": "https://pixabay.test/bright-morning",
            "moods": ["warm_lifestyle"],
        },
    )
    library = A.load_library(tmp_path)
    assert len(library) == 1

    bed = A.choose_bed(library, "warm_lifestyle", 9.0)
    assert bed.title == "Bright Morning"
    assert not bed.generated
    assert "Pixabay Content Licence" in bed.credit


@needs_ffmpeg
def test_a_track_without_a_recorded_licence_is_skipped_not_shipped(tmp_path):
    """The failure this prevents is a file reaching a deliverable because a
    directory scan was permissive and nobody was ever asked."""
    _write_track(tmp_path, "found-in-downloads", None)
    _write_track(tmp_path, "blank-licence", {"title": "Untitled", "source": "?", "licence": ""})

    assert A.load_library(tmp_path) == []
    # And the job still gets a soundtrack, from the one source that needs no licence.
    assert A.choose_bed(A.load_library(tmp_path), "calm_premium", 9.0).generated


@needs_ffmpeg
def test_a_track_listing_no_moods_is_eligible_for_all_of_them(tmp_path):
    """The common case is one track and a user who wants it used."""
    _write_track(tmp_path, "anything", {"title": "Anything", "source": "own", "licence": "CC0"})
    library = A.load_library(tmp_path)

    for mood in ("calm_premium", "warm_lifestyle", "bold_confident", "high_energy"):
        assert A.choose_bed(library, mood, 9.0).title == "Anything"


@needs_ffmpeg
def test_an_absent_library_directory_is_normal_rather_than_an_error(tmp_path):
    assert A.load_library(tmp_path / "no-such-directory") == []


@needs_ffmpeg
def test_the_bed_is_scored_to_the_mood_and_is_reproducible():
    """Four moods must not all produce the same music, and one mood must always
    produce the same music — a soundtrack that varied between runs would make two
    replays of one golden job disagree for a reason unrelated to the model."""
    beds = {mood: A.generated_bed(mood, 6.0).data for mood in A._MOOD_SCORES}

    assert len(set(beds.values())) == len(beds), "moods are not musically distinct"
    for mood, data in beds.items():
        assert A.generated_bed(mood, 6.0).data == data, f"{mood} is not deterministic"


@needs_ffmpeg
def test_an_unknown_mood_gets_a_bed_rather_than_an_exception():
    """Adding a Mood to the enum must not make delivery start throwing."""
    bed = A.generated_bed("mood_invented_next_semester", 4.0)
    assert bed.generated and bed.data
