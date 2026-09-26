"""The closing brand slate: the last 1.5 s of every delivered ad.

The feature exists because a generated advertisement ends on whatever frame the
video model happened to produce, with nothing saying whose product it is or where
to buy it.  The obvious implementation — describe the slate in the motion prompt —
is the one that cannot work: a diffusion model renders plausible-looking text
rather than the URL it was given, and redraws a logo rather than reproducing it.
Both failures are invisible to every automated check in this pipeline and obvious
to a viewer.  So the slate is composited with ffmpeg from the request instead.

Two things here are easy to get quietly wrong, and both have a test aimed at them.

The first is the arithmetic.  ``duration_seconds`` is what the viewer receives, so
the slate is carved *out* of it rather than added to it — 8.5 s of advertisement
plus 1.5 s of card, not 10 s plus 1.5.  Read the other way the delivered file
lands outside the 8-10 s window the project commits to, and nothing in the older
suite would have noticed, because the duration check runs against the generated
clip rather than against the delivered one.

The second is that a job which never asked for a slate must be completely
unaffected — no trim, no appended black, no change in length.  Frozen golden
bundles were recorded before ``website_url`` existed and replay through this same
code path.
"""

from __future__ import annotations

import io

import adworker.delivery as D
import numpy as np
import pytest
from adml import video as V
from adschema import END_CARD_SECONDS, MAX_DURATION_S, MIN_DURATION_S, AdJobRequest, JobState
from adworker import Pipeline, end_card_png
from PIL import Image, ImageDraw

needs_ffmpeg = pytest.mark.skipif(V.FFMPEG is None, reason="ffmpeg is not installed")

LONG_URL = "https://shop.example-brand-with-a-long-name.co.uk/collections/aurora"


# --- What the request says ---------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "stored"),
    [
        ("acme.com", "https://acme.com"),
        ("  acme.com  ", "https://acme.com"),
        ("https://acme.com", "https://acme.com"),
        ("http://acme.com", "http://acme.com"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_a_typed_address_is_stored_as_a_whole_url(make_request, typed, stored):
    """Normalised at the edge, so the manifest records a link rather than a fragment.

    Someone filling in a form types ``acme.com``; the record should hold
    ``https://acme.com``, because that is where the ad sends the viewer. What the
    slate draws is narrower — see the display tests below.
    """
    assert make_request(website_url=typed).website_url == stored


@pytest.mark.parametrize(
    ("stored", "drawn"),
    [
        ("acme.com", "acme.com"),
        ("https://acme.com", "acme.com"),
        ("www.acme.com", "acme.com"),
        ("https://www.acme.com", "acme.com"),
        ("WWW.Acme.com", "Acme.com"),
        ("acme.com/collections/aurora", "acme.com"),
        ("acme.com/shop?ref=ad#top", "acme.com"),
        ("shop.acme.co.uk", "shop.acme.co.uk"),
        ("", ""),
    ],
)
def test_the_slate_draws_the_host_and_nothing_else(make_request, stored, drawn):
    """``https://www.acme.com/collections/aurora`` is drawn ``acme.com``.

    The scheme and the path carry no brand and cost width the address needs. Only
    the leading ``www.`` is dropped, not every subdomain: ``shop.acme.co.uk`` is a
    real address a viewer would type, and truncating it to ``acme.co.uk`` would send
    them somewhere the advertiser did not ask for.

    Case is left alone. Lowercasing is a safe transform for a hostname and a
    destructive one for a brand — ``Acme`` chose its capital.
    """
    assert make_request(website_url=stored).website_display == drawn


def test_a_job_with_nothing_to_show_runs_full_length(make_request):
    """The backward-compatibility guarantee, stated as an equality.

    No website and no logo means no slate, which means no trim and no change to
    any duration. This is what lets a golden bundle frozen before the feature
    existed replay unchanged.
    """
    request = make_request()
    assert not request.has_end_card
    assert request.ad_seconds == request.duration_seconds


@pytest.mark.parametrize("duration", [MIN_DURATION_S, 9.0, MAX_DURATION_S])
def test_the_slate_is_carved_out_of_the_duration_not_added_to_it(make_request, duration):
    """The arithmetic the whole feature turns on.

    A 10 s job is 8.5 s of advertisement and 1.5 s of card. Read the other way it
    would be an 11.5 s file, which is outside the window the project commits to
    and outside what the user asked for.
    """
    request = make_request(website_url="acme.com", duration_seconds=duration)

    assert request.has_end_card
    assert request.ad_seconds == pytest.approx(duration - END_CARD_SECONDS)
    assert request.ad_seconds + END_CARD_SECONDS == pytest.approx(duration)


def test_a_logo_alone_is_enough_to_earn_a_slate(make_request, storage):
    """An advertiser with a mark and no URL still gets branded.

    ``has_end_card`` keys on having something to draw rather than on a flag, so
    neither input is privileged over the other.
    """
    import adproviders as P

    logo = P.mock_reference_asset(storage, "uploads/t/logo.png", seed=5)
    assert make_request(logo_image=logo).has_end_card
    assert not make_request().has_end_card


# --- What the slate looks like -----------------------------------------------


def _card(request: AdJobRequest, storage, width: int = 480, height: int = 854) -> np.ndarray:
    png = end_card_png(request, storage, width, height)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def test_the_slate_is_black_with_the_address_legible_on_it(make_request, storage):
    """Mostly black, but not entirely — which is the whole difference between a
    brand slate and a bug that appended dead video."""
    pixels = _card(make_request(website_url="acme.com"), storage)

    assert pixels.shape == (854, 480, 3)
    assert pixels.mean() < 12, "the slate should read as black"
    assert pixels.max() > 200, "the address should be drawn in something legible"


def test_a_long_address_is_shrunk_rather_than_truncated(make_request, storage):
    """An ellipsis in the middle of a URL does not degrade, it misinforms.

    So the type size comes down until the string fits. The test that it *fits* is
    that no light pixel touches either edge — text running off the canvas is the
    failure this guards, and it is invisible in a report that only checks the file
    was written.
    """
    pixels = _card(make_request(website_url=LONG_URL), storage)
    lit = pixels.max(axis=2) > 128

    assert lit.any(), "the address was not drawn at all"
    assert not lit[:, 0].any() and not lit[:, -1].any(), "the address ran off the slate"


def _lit_column_groups(pixels: np.ndarray) -> list[tuple[int, int]]:
    """Runs of columns carrying light ink, split on fully dark columns."""
    lit = (pixels.max(axis=2) > 100).any(axis=0)
    groups, start = [], None
    for x, on in enumerate(lit):
        if on and start is None:
            start = x
        elif not on and start is not None:
            groups.append((start, x - 1))
            start = None
    if start is not None:
        groups.append((start, len(lit) - 1))
    return groups


def test_a_search_glyph_is_drawn_to_the_left_of_the_address(make_request, storage):
    """The address reads as something to look up, not as decoration.

    Tested as a *ring* rather than as "some ink is present", because the failure
    this guards is the glyph degrading into a blob or a tofu box — both of which
    would light pixels in the right place and neither of which is a magnifier. A
    horizontal cut through a ring crosses ink twice with a gap between; a filled
    shape crosses once.
    """
    pixels = _card(make_request(website_url="apple.in"), storage, width=1080, height=1920)
    groups = _lit_column_groups(pixels)

    assert len(groups) >= 2, "nothing was drawn beside the address"
    left, right = groups[0]
    band = pixels[:, left : right + 1].max(axis=2) > 100
    rows = np.where(band.any(axis=1))[0]
    height = rows[-1] - rows[0]

    assert abs((right - left) - height) <= max(3, height // 5), (
        f"the leftmost mark is {right - left}x{height}, which is not a lens"
    )
    mid = band[(rows[0] + rows[-1]) // 2]
    runs = np.count_nonzero(mid[1:] & ~mid[:-1]) + int(mid[0])
    assert runs == 2, f"a cut through the lens crossed ink {runs} times, not 2 — it is filled"


def test_the_scheme_and_path_never_reach_the_pixels(make_request, storage):
    """Stated as an equality between renders, which is the only form that cannot
    pass by accident.

    Three requests that differ only in the parts the slate is supposed to discard
    must produce the same bytes. A test that merely counted lit pixels would go on
    passing if ``https://`` were drawn in a smaller size to make it fit.
    """
    bare = _card(make_request(website_url="acme.com"), storage)
    schemed = _card(make_request(website_url="https://acme.com"), storage)
    prefixed = _card(make_request(website_url="https://www.acme.com/shop"), storage)

    assert np.array_equal(bare, schemed)
    assert np.array_equal(bare, prefixed)


def _mark(colour: tuple[int, int, int], size: int = 256) -> bytes:
    """A logo: a filled disc in ``colour`` on transparency, like any real mark."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((16, 16, size - 16, size - 16), fill=(*colour, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_a_black_logo_is_redrawn_light_so_it_can_be_seen(make_request, storage):
    """The bug a live job shipped: ``has_logo: true`` and an empty frame.

    Brand marks are usually supplied black-on-transparent, because most places they
    are used are white. Pasted onto a black slate such a mark composites perfectly
    and shows nothing — the Apple logo in that job measured luminance 0.0 inside its
    own alpha mask. Nothing in the pipeline noticed, because compositing *succeeded*.
    """
    from adschema import AssetRef

    storage.put_bytes("uploads/t/black.png", _mark((0, 0, 0)), "image/png")
    request = make_request(
        website_url="acme.com",
        logo_image=AssetRef(key="uploads/t/black.png", url="/x", mime_type="image/png"),
    )
    with_logo = _card(request, storage)
    without = _card(make_request(website_url="acme.com"), storage)

    assert with_logo.max() > 200
    assert (with_logo > 128).sum() > (without > 128).sum() * 2, (
        "the logo added no visible pixels — it is black on black again"
    )


def test_a_logo_with_its_own_colour_is_left_alone(make_request, storage):
    """Only genuinely dark marks are touched.

    Where a brand supplies colour, that colour is the point, and flattening a bright
    mark to white would be the same kind of silent damage in the other direction.
    """
    from adschema import AssetRef

    storage.put_bytes("uploads/t/red.png", _mark((220, 40, 40)), "image/png")
    request = make_request(
        logo_image=AssetRef(key="uploads/t/red.png", url="/x", mime_type="image/png")
    )
    pixels = _card(request, storage)
    lit = pixels[pixels.max(axis=2) > 100]

    assert len(lit), "the mark did not render at all"
    # Still recognisably red: the red channel dominates rather than being flattened.
    assert lit[:, 0].mean() > lit[:, 1].mean() + 40


@pytest.mark.parametrize("with_logo", [True, False], ids=["logo+url", "url only"])
def test_the_slate_content_sits_where_the_layout_says(make_request, storage, with_logo):
    """A second silent one: measured, not eyeballed.

    PIL's text anchors position by the *ascender*, which sits well above the tallest
    ink in a lowercase-heavy address, while the layout measures ink. The two
    disagreed, and on a delivered 1080x1920 slate the address landed at 57% of the
    height against an intended 46% — off by a fifth of the frame, and invisible to
    every test that only asked whether the card was black.
    """
    from adschema import AssetRef

    overrides = {"website_url": "acme.com"}
    if with_logo:
        storage.put_bytes("uploads/t/white.png", _mark((255, 255, 255)), "image/png")
        overrides["logo_image"] = AssetRef(
            key="uploads/t/white.png", url="/x", mime_type="image/png"
        )

    pixels = _card(make_request(**overrides), storage, width=1080, height=1920)
    lit_rows = np.where(pixels.max(axis=(1, 2)) > 100)[0]
    centre = (lit_rows.min() + lit_rows.max()) / 2 / pixels.shape[0]

    assert abs(centre - D._CARD_BLOCK_CENTRE) < 0.02, (
        f"the slate's content is centred at {centre:.1%}, not {D._CARD_BLOCK_CENTRE:.0%}"
    )


def test_an_unreadable_logo_costs_the_logo_and_not_the_delivery(make_request, storage):
    """Degrade to the URL alone rather than failing the job.

    The alternative is losing an entire generated advertisement — paid for, ranked
    and otherwise deliverable — because one PNG would not open.
    """
    from adschema import AssetRef

    storage.put_bytes("uploads/t/broken.png", b"not a png at all", "image/png")
    request = make_request(
        website_url="acme.com",
        logo_image=AssetRef(key="uploads/t/broken.png", url="/x", mime_type="image/png"),
    )

    pixels = _card(request, storage)
    assert pixels.max() > 200, "the address should still be there"


# --- What reaches the file ---------------------------------------------------


@needs_ffmpeg
def test_a_trimmed_clip_and_a_held_still_land_exactly_on_the_promise():
    """The two primitives composed, measured off the container.

    This is the shape a live job takes on Kling, whose image-to-video endpoint
    offers ``{5, 10}`` and nothing between: a request for 8.5 s arrives as a 10 s
    file, is trimmed back, and the slate makes up the difference.
    """
    fps = 24.0
    frames = [np.full((362, 642, 3), i % 256, dtype=np.uint8) for i in range(int(10 * fps))]
    generated = V.encode_mp4(frames, fps)
    assert V.probe(generated).duration_seconds == pytest.approx(10.0, abs=0.1)

    ad = V.trim(generated, MAX_DURATION_S - END_CARD_SECONDS)
    assert V.probe(ad).duration_seconds == pytest.approx(8.5, abs=V.DURATION_TOLERANCE_S)

    card = Image.new("RGB", (642, 362), (0, 0, 0))
    import io

    buf = io.BytesIO()
    card.save(buf, format="PNG")

    delivered = V.append_still(ad, buf.getvalue(), END_CARD_SECONDS)
    assert V.probe(delivered).duration_seconds == pytest.approx(
        MAX_DURATION_S, abs=V.DURATION_TOLERANCE_S
    )


@needs_ffmpeg
async def test_a_delivered_ad_measures_what_the_job_promised(storage, governor, make_request):
    """End to end, off the bytes.

    The arithmetic and the unit tests can all be right while the delivered file is
    still the wrong length, because the duration check in the pipeline runs against
    the *generated* clip and never sees the appended slate. This is the test that
    watches the file the user downloads.
    """
    import adproviders as P

    pipeline = Pipeline(
        storage=storage,
        governor=governor,
        image_provider=P.MockImageProvider(storage),
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
    )
    request = make_request(website_url="acme.com", duration_seconds=MAX_DURATION_S)
    record = await pipeline.run(request)
    assert record.state is JobState.COMPLETED, record.error

    report = record.result.delivery
    assert report.end_card.attached
    assert report.end_card.seconds == END_CARD_SECONDS
    assert report.end_card.website == "https://acme.com"

    assert report.renders, "nothing was delivered to measure"
    for render in report.renders:
        measured = V.probe(storage.get_bytes(render.asset.key)).duration_seconds
        assert measured == pytest.approx(MAX_DURATION_S, abs=0.2), (
            f"the {render.aspect_ratio} render runs {measured:.2f}s, not {MAX_DURATION_S:.1f}s"
        )
        assert MIN_DURATION_S <= measured <= MAX_DURATION_S + V.DURATION_TOLERANCE_S


@needs_ffmpeg
async def test_the_delivered_ad_ends_on_the_slate(storage, governor, make_request):
    """The last thing on screen is the brand, not the advertisement.

    Measured on the final decoded frame rather than asserted from the code path,
    because every step between here and there — the fade, the concat, the reframe —
    could succeed and still leave the card off the end.
    """
    import adproviders as P

    pipeline = Pipeline(
        storage=storage,
        governor=governor,
        image_provider=P.MockImageProvider(storage),
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
    )
    record = await pipeline.run(make_request(website_url="acme.com"))
    assert record.state is JobState.COMPLETED, record.error

    render = record.result.delivery.renders[0]
    frames = V.decode(storage.get_bytes(render.asset.key)).frames

    assert frames[-1].mean() < 20, "the ad did not end on the slate"
    assert frames[0].mean() > 20, "the ad should not *start* black"


@needs_ffmpeg
async def test_every_playable_clip_carries_the_slate(storage, governor, make_request):
    """Including the runner-up, which the UI also lets you play.

    A branded winner beside an unbranded runner-up reads as a broken render rather
    than as a deliberate economy — the same reasoning that puts audio on every clip.
    """
    import adproviders as P

    pipeline = Pipeline(
        storage=storage,
        governor=governor,
        image_provider=P.MockImageProvider(storage),
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
    )
    record = await pipeline.run(make_request(website_url="acme.com"))
    videos = record.result.videos
    assert len(videos) > 1, "this test needs a runner-up to be about anything"

    for video in videos:
        carded = video.platform_renders.get("end_card")
        assert carded is not None, f"clip {video.index} shipped without the slate"
        measured = V.probe(storage.get_bytes(carded.key)).duration_seconds
        assert measured == pytest.approx(video.duration_seconds + END_CARD_SECONDS, abs=0.2)


@needs_ffmpeg
async def test_a_job_without_a_website_is_delivered_exactly_as_before(
    storage, governor, make_request
):
    """The guard on the frozen golden bundles.

    They were recorded before ``website_url`` existed and replay through this code
    path. If the slate ever became unconditional they would gain 1.5 s of black and
    every frozen comparison in the write-up would shift under it.
    """
    import adproviders as P

    pipeline = Pipeline(
        storage=storage,
        governor=governor,
        image_provider=P.MockImageProvider(storage),
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
    )
    record = await pipeline.run(make_request(duration_seconds=9.0))
    assert record.state is JobState.COMPLETED, record.error

    report = record.result.delivery
    assert not report.end_card.attached
    for video in record.result.videos:
        assert "end_card" not in video.platform_renders
    for render in report.renders:
        measured = V.probe(storage.get_bytes(render.asset.key)).duration_seconds
        assert measured == pytest.approx(9.0, abs=0.2)


@needs_ffmpeg
async def test_delivery_survives_a_slate_it_cannot_be_drawn(
    storage, governor, make_request, monkeypatch
):
    """A slate that will not draw costs the slate, not the job.

    Consistent with how reframe and audio failures are already handled, and for a
    sharper reason than consistency: by this point the advertisement has been
    generated, ranked and paid for. Losing it because a logo would not open — or a
    font would not load — would throw away real money to avoid shipping an
    unbranded clip.
    """
    import adproviders as P
    import adworker.delivery as D

    def _explode(*args, **kwargs):
        raise ValueError("no canvas, no slate")

    monkeypatch.setattr(D, "end_card_png", _explode)
    pipeline = Pipeline(
        storage=storage,
        governor=governor,
        image_provider=P.MockImageProvider(storage),
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
    )
    record = await pipeline.run(make_request(website_url="acme.com"))

    assert record.state is JobState.COMPLETED, record.error
    report = record.result.delivery
    assert report.renders, "the advertisement should still have been delivered"
    assert not report.end_card.attached
    assert any("end card" in w for w in report.warnings), report.warnings
