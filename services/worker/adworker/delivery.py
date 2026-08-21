"""Stage 8: platform renders, mockup previews, report cards and the bundle.

Everything here is free — no provider is called, nothing leaves the machine — which
is why it is worth doing properly rather than treating as packaging. Three of the
four outputs answer a question the ranked list cannot.

**The reframe report answers "what did this cost?"**  Delivering a 9:16 ad as 16:9
throws away most of the frame. :mod:`adml.crop` places the window by saliency instead
of centring it, and the report quotes both what the smart crop kept and what a centre
crop would have kept — a benefit is a comparison, not a lone number.

**The mockup preview answers "will the platform cover the product?"**  A safe-area
compliance score is a number; a frame with Instagram's caption bar drawn over it is
the same fact in the form the person publishing will actually act on.

**The report card answers "why did this win?"**  It carries the explanation, the
component scores, the model that produced them and whether that model is a stub.

The bundle is then just a zip of the above, and exists so the whole thing can be
handed over in one file.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass

from adml import audio as A
from adml import crop as C
from adml import video as V
from adproviders import Storage
from adschema import (
    AdJobRequest,
    AspectRatio,
    AudioReport,
    DeliveryReport,
    JobResult,
    Platform,
    RankedCandidate,
    ReframeReport,
    VideoCandidate,
)
from PIL import Image, ImageDraw

#: Long edge of a delivered render. 720 keeps a 9 s clip small enough to hand over
#: while staying above the point where platform re-encoding becomes visible.
DELIVERY_LONG_EDGE = 720
#: Long edge of a mockup preview PNG.
PREVIEW_LONG_EDGE = 540

#: Aspect ratios every job is delivered in, regardless of its target platform. These
#: three cover the placements the plan names: vertical for Reels/Shorts/TikTok,
#: portrait for feed, landscape for pre-roll.
DELIVERY_RATIOS: tuple[AspectRatio, ...] = (
    AspectRatio.VERTICAL_9_16,
    AspectRatio.PORTRAIT_4_5,
    AspectRatio.LANDSCAPE_16_9,
)

#: Platforms whose chrome is drawn as a preview. One per distinct safe area, since a
#: second platform with identical geometry would produce an identical picture.
PREVIEW_PLATFORMS: tuple[Platform, ...] = (
    Platform.INSTAGRAM_REELS,
    Platform.INSTAGRAM_FEED,
    Platform.YOUTUBE_SHORTS,
)

#: Retained-salience level below which a reframe is called out as compromised.
#: Deliberately above `adml.crop.MIN_RETAINED_SALIENCE`: that constant decides
#: *whether to pad*, this one decides *whether to mention it*, and a padded render
#: is still worth mentioning.
WARN_RETAINED_SALIENCE = 0.80


def _even(value: int) -> int:
    return value - (value % 2)


def _delivery_size(ratio: AspectRatio) -> tuple[int, int]:
    width, height = ratio.pixel_size(DELIVERY_LONG_EDGE)
    return _even(width), _even(height)


@dataclass(frozen=True)
class RenderedVariant:
    report: ReframeReport
    data: bytes


def reframe(
    video: VideoCandidate,
    request: AdJobRequest,
    storage: Storage,
    ratio: AspectRatio,
) -> RenderedVariant:
    """Produce one aspect-ratio variant of one clip, with its cost measured.

    The crop is planned against the *sampled* frames :mod:`adml.video` returns, which
    are the same frames the scorer measured — so the window is chosen against the
    clip that was ranked, not against a different decode of it.
    """
    data = storage.get_bytes(video.asset.key)
    clip = V.decode(data)
    safe = tuple(request.platform.safe_area)

    plan = C.plan(clip.frames, ratio.ratio, safe_area=safe)
    width, height = _delivery_size(ratio)
    key = f"delivery/{request.job_id}/{video.index}_{ratio.value.replace(':', 'x')}.mp4"

    if plan.is_noop:
        # Still re-encoded to the delivery size, but no reframe decision was made, so
        # there is no salience to report as lost.
        out = V.render_crop(
            data,
            x=0,
            y=0,
            width=_even(clip.probe.width),
            height=_even(clip.probe.height),
            out_width=width,
            out_height=height,
        )
        centre_salience = None
    elif plan.mode == "pad":
        out = V.render_padded(data, out_width=width, out_height=height)
        centre_salience = C.retained_by(
            clip.frames,
            C.centre_plan(clip.probe.width, clip.probe.height, ratio.ratio),
            safe_area=safe,
        )
    else:
        out = V.render_crop(
            data,
            x=plan.window.x,
            y=plan.window.y,
            width=plan.window.width,
            height=plan.window.height,
            out_width=width,
            out_height=height,
        )
        centre_salience = C.retained_by(
            clip.frames,
            C.centre_plan(clip.probe.width, clip.probe.height, ratio.ratio),
            safe_area=safe,
        )

    asset = storage.put_bytes(key, out, "video/mp4")
    asset.width, asset.height = width, height
    return RenderedVariant(
        report=ReframeReport(
            aspect_ratio=ratio.value,
            asset=asset,
            mode=plan.mode,
            width=width,
            height=height,
            retained_salience=round(plan.retained_salience, 4),
            centre_crop_salience=(None if centre_salience is None else round(centre_salience, 4)),
            tracking_gain=round(plan.tracking_gain, 4),
            note=plan.reason,
        ),
        data=out,
    )


# --- Mockup previews ---------------------------------------------------------

#: Chrome is drawn at these opacities so the underlying frame stays readable. The
#: point is to show what will be *covered*, which needs both visible.
_CHROME_FILL = (12, 14, 18, 150)
_CHROME_EDGE = (255, 255, 255, 90)


def _plated_text(draw: ImageDraw.ImageDraw, at: tuple[int, int], text: str) -> None:
    """Draw text over a dark plate sized to the text itself."""
    x, y = at
    left, top, right, bottom = draw.textbbox((x, y), text)
    draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=(0, 0, 0, 190))
    draw.text((x, y), text, fill=(255, 255, 255, 240))


def preview_frame(
    video: VideoCandidate,
    platform: Platform,
    storage: Storage,
    *,
    caption: str = "",
) -> bytes:
    """A PNG of the clip's middle frame with the platform's chrome drawn over it.

    The middle frame, matching the video scorer, so what is previewed is what was
    measured. The first frame would be the generated still the image stage already
    showed the user.

    Bands are drawn from :attr:`adschema.Platform.safe_area` rather than from a
    hand-placed mock, so the picture and the ``safe_area_compliance`` score are
    describing the same rectangle. A drifting mock would be worse than none: it would
    look authoritative while disagreeing with the number beside it.
    """
    clip = V.decode(storage.get_bytes(video.asset.key))
    middle = clip.frames[len(clip.frames) // 2]

    frame = Image.fromarray(middle).convert("RGBA")
    scale = PREVIEW_LONG_EDGE / max(frame.size)
    if scale < 1.0:
        frame = frame.resize(
            (max(2, round(frame.width * scale)), max(2, round(frame.height * scale))),
            Image.LANCZOS,
        )
    width, height = frame.size

    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    safe = platform.safe_area

    top = round(height * safe.top)
    bottom = round(height * safe.bottom)
    left = round(width * safe.left)
    right = round(width * safe.right)

    for box in (
        (0, 0, width, top),
        (0, height - bottom, width, height),
        (0, 0, left, height),
        (width - right, 0, width, height),
    ):
        if box[2] > box[0] and box[3] > box[1]:
            draw.rectangle(box, fill=_CHROME_FILL)

    # The boundary of what survives, so the safe region reads as a region.
    draw.rectangle(
        (left, top, width - right - 1, height - bottom - 1), outline=_CHROME_EDGE, width=2
    )

    # Both labels get a backing plate. The frame underneath is arbitrary imagery —
    # and mock frames carry their own burnt-in design-point caption — so white text
    # alone is legible only by luck.
    _plated_text(draw, (8, 6), platform.value.replace("_", " "))
    if caption:
        # Placed inside the bottom band, where the platform would put it.
        text = caption if len(caption) <= 48 else caption[:45] + "..."
        _plated_text(draw, (8, height - max(bottom, 20) + 4), text)

    out = io.BytesIO()
    Image.alpha_composite(frame, overlay).convert("RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()


# --- Report cards ------------------------------------------------------------


def report_card(ranked: RankedCandidate, request: AdJobRequest) -> dict:
    """The per-candidate record that ships with the download.

    Carries the model that produced the score and whether it is a stub, because a
    report card is the artefact most likely to be read out of context — a number
    quoted from here with no provenance attached is how a heuristic baseline becomes
    "the predictor" in a write-up.
    """
    video = ranked.video
    score = video.score
    return {
        "rank": ranked.rank,
        "candidate": chr(ord("A") + video.source_image_index),
        "explanation": ranked.explanation,
        "image_stage_rank": ranked.image_stage_rank,
        "rank_shift": ranked.rank_shift,
        "prediction": {
            "overall": None if score is None else score.overall,
            "components": {} if score is None else score.components(),
            "model_version": None if score is None else score.model_version,
            "is_stub": True if score is None else score.is_stub,
        },
        "generation": {
            "provider": video.provider,
            "tier": video.tier.value,
            "seed": video.seed,
            "seed_honoured": video.seed_honoured,
            "duration_seconds": video.duration_seconds,
            "requested_duration_seconds": video.requested_duration_seconds,
            "was_chained": video.was_chained,
            "seam_consistency": video.seam_consistency,
            "cost_usd": video.cost_usd,
        },
        "brief": {
            "design_point": video.brief.design_point.axes(),
            "motion_prompt": video.brief.motion_prompt,
        },
        "request": {
            "product_name": request.product_name,
            "vertical": request.vertical.value,
            "platform": request.platform.value,
            "caption": request.caption,
        },
    }


# --- The bundle --------------------------------------------------------------


def build_bundle(
    result: JobResult,
    request: AdJobRequest,
    storage: Storage,
    report: DeliveryReport | None = None,
) -> bytes:
    """Zip every render, preview and report card into one download.

    Stored uncompressed for the video entries: H.264 is already compressed and
    deflating it again costs CPU to save nothing. The JSON is deflated.

    ``report`` is passed explicitly rather than read from ``result.delivery``, and
    that is a fix rather than a preference.  Reading it off the result created an
    ordering dependency on an assignment that happens *after* this function runs, so
    the bundle silently shipped with no videos and no previews in it — 2 kB of JSON
    where half a megabyte of media should have been, with nothing to indicate a
    problem. An explicit argument cannot be got wrong in that direction.
    """
    buffer = io.BytesIO()
    report = report if report is not None else result.delivery
    with zipfile.ZipFile(buffer, "w") as archive:
        if report is not None:
            for render in report.renders:
                name = f"video/{render.aspect_ratio.replace(':', 'x')}.mp4"
                archive.writestr(
                    zipfile.ZipInfo(name), storage.get_bytes(render.asset.key), zipfile.ZIP_STORED
                )
            for platform, asset in report.previews.items():
                archive.writestr(
                    zipfile.ZipInfo(f"previews/{platform}.png"),
                    storage.get_bytes(asset.key),
                    zipfile.ZIP_STORED,
                )

        cards = [report_card(r, request) for r in result.ranking]
        archive.writestr(
            "report.json",
            json.dumps(
                {
                    "job_id": result.job_id,
                    "winner": cards[0]["candidate"] if cards else None,
                    "total_cost_usd": result.total_cost_usd,
                    "candidates": cards,
                    "delivery": None if report is None else report.model_dump(mode="json"),
                },
                indent=2,
            ),
            zipfile.ZIP_DEFLATED,
        )
        archive.writestr(
            "README.txt", _bundle_readme(result, request, report), zipfile.ZIP_DEFLATED
        )
    return buffer.getvalue()


def _bundle_readme(result: JobResult, request: AdJobRequest, report: DeliveryReport | None) -> str:
    """A plain-text note in the zip, saying what the numbers mean.

    Included because the bundle outlives this conversation. Someone opening it in
    week 16 needs to know that a 0.74 is a within-set position from a possibly-stub
    model, not a click-through rate.
    """
    winner = result.winner
    lines = [
        f"Ad candidates for {request.product_name}",
        f"job {result.job_id}",
        "",
        f"{len(result.ranking)} candidates, ranked. Winner: "
        + (f"candidate {chr(ord('A') + winner.video.source_image_index)}" if winner else "none"),
        "",
        "SCORES ARE RELATIVE, NOT PREDICTED CLICK RATES.",
        "The ranker is trained on pairwise human preference, which fixes an order and",
        "not a scale, so a score is a position within these candidates and nothing more.",
    ]
    if winner is not None and winner.video.score is not None:
        score = winner.video.score
        lines += [
            "",
            f"Scored by: {score.model_version}"
            + ("  (STUB — not a validated predictor)" if score.is_stub else ""),
        ]
    if report is not None:
        lines += ["", "RENDERS"]
        for render in report.renders:
            note = f"  {render.note}" if render.note else ""
            lines.append(
                f"  {render.aspect_ratio:>5}  {render.width}x{render.height}  "
                f"via {render.mode}, keeps {render.retained_salience:.0%} of the "
                f"salient content{note}"
            )
        if report.audio.attached:
            lines += ["", "AUDIO", f"  {report.audio.credit}"]
            if report.audio.licence:
                lines.append(f"  licence: {report.audio.licence}")
            if report.audio.is_test_signal:
                lines.append("  NOTE: a synthesised test tone, not music. Replace before use.")
        if report.warnings:
            lines += ["", "WARNINGS"] + [f"  - {w}" for w in report.warnings]
    return "\n".join(lines) + "\n"


# --- Orchestration -----------------------------------------------------------


def deliver(
    result: JobResult,
    request: AdJobRequest,
    storage: Storage,
    *,
    bed: A.AudioBed | None = None,
    ratios: tuple[AspectRatio, ...] = DELIVERY_RATIOS,
    platforms: tuple[Platform, ...] = PREVIEW_PLATFORMS,
) -> DeliveryReport:
    """Produce everything for the winning candidate.

    The winner only, deliberately.  Reframing three clips into three ratios each is
    nine re-encodes for output nobody asked for, and the runner-up's value is its
    *score* — evidence for the ranking — not its 16:9 variant. The other candidates
    keep their native render and appear in the report cards.
    """
    report = DeliveryReport()
    winner = result.winner
    if winner is None:
        report.warnings.append("no ranked candidate, so nothing was delivered")
        return report

    video = winner.video
    for ratio in ratios:
        try:
            variant = reframe(video, request, storage, ratio)
        except V.ClipDecodeError as exc:
            report.warnings.append(f"{ratio.value} render failed: {exc}")
            continue
        report.renders.append(variant.report)
        video.platform_renders[ratio.value] = variant.report.asset
        if variant.report.retained_salience < WARN_RETAINED_SALIENCE:
            report.warnings.append(
                f"{ratio.value} keeps only {variant.report.retained_salience:.0%} of the "
                f"salient content — check the product survived the reframe"
            )

    for platform in platforms:
        try:
            png = preview_frame(video, platform, storage, caption=request.caption)
        except V.ClipDecodeError as exc:
            report.warnings.append(f"{platform.value} preview failed: {exc}")
            continue
        key = f"delivery/{request.job_id}/preview_{platform.value}.png"
        report.previews[platform.value] = storage.put_bytes(key, png, "image/png")

    # Audio goes on **every** clip, not only the winner, which is the one place
    # this stage departs from "the winner only". Reframing a runner-up means nine
    # re-encodes for output nobody asked for; mixing a runner-up means one stream
    # copy, because the video track is copied rather than re-encoded. And the UI
    # lets the user play all of them — a runner-up that plays silent next to a
    # winner that does not reads as a broken clip rather than as a deliberate
    # economy.
    for candidate in result.videos:
        attached = _attach_audio(
            candidate, request, storage, bed, report, primary=candidate is video
        )
        if candidate is video:
            report.audio = attached

    # `report` explicitly, not `result.delivery` — the caller has not assigned that
    # yet, and reading it here shipped a bundle with no media in it.
    bundle = build_bundle(result, request, storage, report)
    report.bundle = storage.put_bytes(
        f"delivery/{request.job_id}/bundle.zip", bundle, "application/zip"
    )
    return report


def _attach_audio(
    video: VideoCandidate,
    request: AdJobRequest,
    storage: Storage,
    bed: A.AudioBed | None,
    report: DeliveryReport,
    *,
    primary: bool = True,
) -> AudioReport:
    """Mix audio onto the native render.

    ``bed`` is now chosen upstream and is effectively never ``None`` — the pipeline
    takes a licensed track from the library when one is there and synthesises one
    when it is not. The branch stays because delivery is called directly in tests
    and by scripts, and a silent clip should say why it is silent rather than
    leaving the reader of a manifest to guess.

    The reason it used to be ``None`` in a real job was not a bug: this project
    ships no music, and :class:`adml.audio.AudioBed` refuses audio with no recorded
    licence. Both still hold. What changed is that "we have no licensed music" is
    answered by generating some rather than by delivering an advertisement with no
    sound.
    """
    if bed is None:
        return AudioReport(
            attached=False,
            note="no music bed supplied, so this clip ships silent. The pipeline "
            "normally passes one — see adml.audio.choose_bed.",
        )

    # Once per job, not once per clip. Whether the caption fits the duration is a
    # fact about the job — every clip is the same length — and repeating it per
    # candidate would turn one finding into a list.
    if primary and not A.caption_fits(request.caption, video.duration_seconds):
        estimate = A.voice_duration_estimate(request.caption)
        report.warnings.append(
            f"the caption reads in about {estimate:.1f}s, which does not fit a "
            f"{video.duration_seconds:.1f}s clip — no voiceover was attached"
        )

    try:
        mixed, mix_report = A.mix(storage.get_bytes(video.asset.key), bed)
        measured = A.measure_loudness(mixed)
    except A.AudioError as exc:
        report.warnings.append(
            f"the audio mix failed for clip {video.index}, so it ships silent: {exc}"
        )
        return AudioReport(attached=False, note=str(exc))

    key = f"delivery/{request.job_id}/{video.index}_with_audio.mp4"
    asset = storage.put_bytes(key, mixed, "video/mp4")
    video.platform_renders["audio"] = asset

    return AudioReport(
        attached=True,
        credit=bed.credit,
        licence=bed.licence,
        target_lufs=mix_report.target_lufs,
        measured_lufs=round(measured, 2),
        has_voiceover=mix_report.has_voice,
        is_test_signal=bed.is_test_signal,
        generated=bed.generated,
    )
