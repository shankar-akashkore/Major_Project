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

import numpy as np
from adml import audio as A
from adml import crop as C
from adml import video as V
from adproviders import Storage
from adschema import (
    END_CARD_SECONDS,
    AdJobRequest,
    AspectRatio,
    AudioReport,
    DeliveryReport,
    EndCardReport,
    JobResult,
    Platform,
    RankedCandidate,
    ReframeReport,
    VideoCandidate,
)
from PIL import Image, ImageDraw, ImageFont

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
    *,
    end_card: bytes | None = None,
) -> RenderedVariant:
    """Produce one aspect-ratio variant of one clip, with its cost measured.

    The crop is planned against the *sampled* frames :mod:`adml.video` returns, which
    are the same frames the scorer measured — so the window is chosen against the
    clip that was ranked, not against a different decode of it.

    ``end_card`` is appended after the reframe and before the store, so the crop is
    still planned against advertisement frames only — a held still has no salience
    to track, and letting it into the planner would drag the window towards the
    middle of a black rectangle. Baking it in before the single ``put_bytes`` also
    keeps the returned :class:`ReframeReport`'s ``sha256`` describing the bytes that
    were actually written.
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

    if end_card is not None:
        out = V.append_still(out, end_card, END_CARD_SECONDS)

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


# --- Closing brand slate -----------------------------------------------------

#: Share of the short edge the logo is allowed to occupy on the end card.
#:
#: Was 0.30, which read as a splash screen rather than a sign-off — the mark filled
#: the frame and the address under it looked like a footnote. A slate is glanced at,
#: not studied, and the two lines want to read as one block.
_CARD_LOGO_SHARE = 0.25
#: Share of the short edge used as the starting point for the address's type size,
#: and the share of the width it must fit inside before the size is reduced.
_CARD_TEXT_SHARE = 0.058
_CARD_TEXT_MAX_WIDTH = 0.82
#: The search glyph, as shares of the address's own ink height: the box it is drawn
#: into, and the space between it and the first letter. Tied to the text rather than
#: to the canvas so the pair stays a matched set at any render size.
_CARD_ICON_SHARE = 0.92
_CARD_ICON_GAP = 0.42
#: Where the logo-and-URL block is centred vertically. Slightly above the middle,
#: which is where the eye expects a title to sit and where the prototype put it.
_CARD_BLOCK_CENTRE = 0.46


def _fitted_font(text: str, start_px: int, max_width: int) -> ImageFont.FreeTypeFont:
    """The largest default-family size at which ``text`` fits ``max_width``.

    ``ImageFont.load_default(size=...)`` returns a scalable FreeType face on Pillow
    10.1 and later, which is what makes this possible without shipping a font file
    and taking on its licence — a real consideration for a project that already
    refuses unlicensed audio.

    Shrinking rather than truncating, because the string is a web address: an
    ellipsis in the middle of a URL does not degrade, it misinforms.
    """
    size = max(8, start_px)
    while size > 8:
        font = ImageFont.load_default(size=size)
        left, _, right, _ = font.getbbox(text)
        if right - left <= max_width:
            return font
        size -= 2
    return ImageFont.load_default(size=size)


#: Mean luminance (0-255, measured inside the alpha mask) below which a logo is
#: treated as a dark mark and redrawn light. 89 is 35% grey.
_DARK_LOGO_LUMINANCE = 89.0


def _legible_on_black(logo: Image.Image) -> Image.Image:
    """Lighten a dark logo so it is visible on the slate.

    Most brand marks are supplied black-on-transparent, because most of the places
    they are used are white. Pasted unaltered onto a black slate such a mark
    composites perfectly and shows nothing at all — the delivery report says
    ``has_logo: true`` and the viewer sees an empty frame. That is exactly what a
    live job produced: an Apple mark whose opaque pixels measured luminance 0.0.

    Only genuinely dark marks are touched, and only their brightness — the alpha
    channel carries the shape and is left alone, so the silhouette is still the real
    one, and the scaling below preserves hue so a brand that chose a colour keeps it.

    Luminance is the wrong test on its own and this nearly shipped with it: red is
    weighted 0.2126, so a perfectly visible saturated red mark measures 78 and reads
    as "dark". Scaling to the peak channel rather than painting white is what makes
    the rule safe for those.
    """
    pixels = np.asarray(logo).astype(np.float32)
    opaque = pixels[..., 3] > 8
    if not opaque.any():
        return logo
    luminance = 0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]
    if luminance[opaque].mean() >= _DARK_LOGO_LUMINANCE:
        return logo

    # Scale every channel by one factor until the brightest reaches full, rather than
    # painting the mark white. That keeps the hue: a dark navy mark comes back bright
    # navy, not white, and a brand that chose a colour keeps it. One factor for the
    # whole mark rather than per pixel, so internal shading survives too.
    #
    # A mark that is *pure* black has no hue to preserve and no peak to scale from,
    # so it becomes white — which is the common case and the one that failed.
    peak = float(pixels[..., :3][opaque].max())
    if peak <= 0.0:
        pixels[..., :3] = 255.0
    else:
        pixels[..., :3] = np.clip(pixels[..., :3] * (255.0 / peak), 0.0, 255.0)
    return Image.fromarray(pixels.astype(np.uint8), mode="RGBA")


def _search_glyph(draw: ImageDraw.ImageDraw, x: int, y: int, size: int) -> None:
    """A magnifying glass, drawn into a ``size``-square box with its top-left at (x, y).

    Two vector primitives rather than the U+1F50D character, because the default
    font is not guaranteed to carry it and a missing glyph does not raise — it draws
    a tofu box, which would composite cleanly and ship. A circle and a line cannot
    go missing.
    """
    if size < 6:
        # Below this the stroke and the lens are the same pixel and the glyph reads
        # as a smudge. A bare address is better than a blur beside it.
        return
    stroke = max(2, round(size * 0.11))
    lens = round(size * 0.72)
    draw.ellipse((x, y, x + lens, y + lens), outline=(255, 255, 255), width=stroke)

    # The handle leaves the rim at 45 degrees, so it meets the circle tangentially
    # and reads as attached rather than as a line crossing a ring.
    centre = lens / 2.0
    reach = centre * 0.7071
    draw.line(
        (round(x + centre + reach), round(y + centre + reach), x + size, y + size),
        fill=(255, 255, 255),
        width=stroke,
    )


def end_card_png(request: AdJobRequest, storage: Storage, width: int, height: int) -> bytes:
    """The closing slate: the advertiser's logo over their web address, on black.

    Drawn here rather than asked of the generator, and that is the whole point of
    the feature. A video model cannot spell a URL — it renders plausible-looking
    text — and it redraws a logo rather than reproducing it. Both failures are
    invisible to every automated check in this pipeline and obvious to a viewer.
    Compositing the slate makes it exact, free, and reproducible from the request.
    """
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    short = min(width, height)

    logo = None
    if request.logo_image is not None:
        try:
            logo = Image.open(io.BytesIO(storage.get_bytes(request.logo_image.key)))
            logo = _legible_on_black(logo.convert("RGBA"))
            box = round(short * _CARD_LOGO_SHARE)
            logo.thumbnail((box, box), Image.LANCZOS)
        except (OSError, FileNotFoundError, ValueError):
            # A slate with the URL alone is still a usable slate. Losing the whole
            # delivery because a logo file is unreadable would be the worse trade.
            logo = None

    # The host alone, not the stored URL. See ``AdJobRequest.website_display``.
    site = request.website_display
    font = None
    text_left = text_top = text_width = text_height = 0
    if site:
        start_px = round(short * _CARD_TEXT_SHARE)
        # The glyph and its gap share the line with the text, so the text is fitted
        # to what is left of the width rather than to all of it. Reserved from the
        # *starting* size, which is the largest the glyph can turn out to be.
        reserved = round(start_px * (_CARD_ICON_SHARE + _CARD_ICON_GAP))
        font = _fitted_font(site, start_px, round(width * _CARD_TEXT_MAX_WIDTH) - reserved)
        text_left, text_top, text_right, text_bottom = font.getbbox(site)
        text_width = text_right - text_left
        text_height = text_bottom - text_top

    icon = round(text_height * _CARD_ICON_SHARE) if font is not None else 0
    icon_gap = round(text_height * _CARD_ICON_GAP) if font is not None else 0
    line_width = icon + icon_gap + text_width
    line_height = max(icon, text_height)

    gap = round(short * 0.06) if (logo is not None and font is not None) else 0
    block = (logo.height if logo is not None else 0) + gap + line_height
    y = round(height * _CARD_BLOCK_CENTRE) - block // 2

    if logo is not None:
        canvas.paste(logo, ((width - logo.width) // 2, y), logo)
        y += logo.height + gap

    if font is not None:
        draw = ImageDraw.Draw(canvas)
        x = (width - line_width) // 2
        _search_glyph(draw, x, y + (line_height - icon) // 2, icon)
        # ``anchor="la"`` puts the pen at the ascender and at the origin, and the ink
        # is at neither: the ascender clears the tallest letter, and the first glyph
        # carries a left side bearing. The layout above measured ink, so subtracting
        # both offsets is what makes the two agree. Without the vertical half of it
        # the block is laid out against one origin and drawn against another —
        # measured on a delivered 1080x1920 slate the address landed at 57% of the
        # height instead of 46%. Without the horizontal half the glyph sits too far
        # from a letter that starts with a bearing, and too close to one that does not.
        draw.text(
            (
                x + icon + icon_gap - text_left,
                y + (line_height - text_height) // 2 - text_top,
            ),
            site,
            fill=(255, 255, 255),
            font=font,
            anchor="la",
        )

    out = io.BytesIO()
    canvas.save(out, format="PNG", optimize=True)
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

    if request.has_end_card:
        report.end_card = EndCardReport(
            seconds=END_CARD_SECONDS,
            website=request.website_url,
            has_logo=request.logo_image is not None,
        )

    video = winner.video
    for ratio in ratios:
        # Drawn per ratio rather than once, because the slate has to be the variant's
        # own geometry — a 9:16 card letterboxed into a 16:9 render would put black
        # bars around a frame that is already black, and centre the logo in the wrong
        # rectangle.
        #
        # Caught separately from the reframe, and that separation is the point: an
        # advertisement that has been generated, ranked and paid for should ship
        # unbranded rather than not at all, so a slate that cannot be drawn costs the
        # slate and nothing else.
        card = None
        if request.has_end_card:
            try:
                card = end_card_png(request, storage, *_delivery_size(ratio))
            except (OSError, ValueError) as exc:
                report.warnings.append(
                    f"the {ratio.value} end card could not be drawn, so that render "
                    f"ships without one: {exc}"
                )

        try:
            variant = reframe(video, request, storage, ratio, end_card=card)
        except V.ClipDecodeError as exc:
            report.warnings.append(f"{ratio.value} render failed: {exc}")
            continue
        if card is not None:
            report.end_card.attached = True
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
        source = _carded_native(candidate, request, storage, report)
        attached = _attach_audio(
            candidate, request, storage, bed, report, primary=candidate is video, source=source
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


def _carded_native(
    video: VideoCandidate,
    request: AdJobRequest,
    storage: Storage,
    report: DeliveryReport,
) -> bytes | None:
    """The native-geometry clip with the closing slate on it, or ``None``.

    Stored under its own key rather than written back over ``video.asset``. The
    generated asset is the record of what the provider produced and what was billed
    for, and its ``duration_seconds`` describes the advertisement; overwriting it
    with a longer file would leave the record contradicting the bytes.
    """
    if not request.has_end_card:
        return None
    try:
        raw = storage.get_bytes(video.asset.key)
        info = V.probe(raw)
        card = end_card_png(request, storage, info.width, info.height)
        carded = V.append_still(raw, card, END_CARD_SECONDS)
    except (V.ClipDecodeError, FileNotFoundError, OSError, ValueError) as exc:
        report.warnings.append(
            f"the end card could not be added to clip {video.index}, so it ships without one: {exc}"
        )
        return None

    key = f"delivery/{request.job_id}/{video.index}_end_card.mp4"
    video.platform_renders["end_card"] = storage.put_bytes(key, carded, "video/mp4")
    report.end_card.attached = True
    return carded


def _attach_audio(
    video: VideoCandidate,
    request: AdJobRequest,
    storage: Storage,
    bed: A.AudioBed | None,
    report: DeliveryReport,
    *,
    primary: bool = True,
    source: bytes | None = None,
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
        # The carded clip when there is one, so the bed runs under the slate and
        # fades at the true end. `A.mix` trims the bed to the video's own duration,
        # so this needs no extra audio handling — mixing first and appending after
        # would have ended the music 1.5 s early.
        mixed, mix_report = A.mix(
            source if source is not None else storage.get_bytes(video.asset.key), bed
        )
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
