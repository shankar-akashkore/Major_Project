"""Intake preprocessing: reference validation, product cutout, palette extraction.

The tests worth having here are the ones that would catch intake quietly doing
nothing.  A cutout that silently falls back to the original photo, a palette
extracted from the backdrop instead of the product, a face box invented when no
detector was present — each of those still produces a completed job and a
plausible-looking report, so only a test that checks the *substance* catches them.
"""

from __future__ import annotations

import io

import adproviders as P
import numpy as np
import pytest
from adml import features as F
from adschema import AspectRatio, JobState, ThemeSpec
from adworker import Pipeline, preprocess
from adworker.intake import MIN_EDGE_PX, MIN_SHARPNESS, PALETTE_SIZE
from PIL import Image, ImageDraw, ImageFilter

#: The colours ``mock_product_asset`` actually draws the product with. Extraction
#: is judged against these rather than against whatever it happens to return.
TRUE_PRODUCT_COLOURS = ["#2b3a55", "#ce7777", "#f2e7d5"]


def _store(storage, key: str, img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return storage.put_bytes(key, buf.getvalue(), "image/png")


def _resized_product(storage, key: str, edge: int):
    """A product upload at an arbitrary size, for the resolution checks."""
    original = P.mock_product_asset(storage, "tmp/full.png", AspectRatio.SQUARE_1_1, seed=22)
    with Image.open(io.BytesIO(storage.get_bytes(original.key))) as img:
        return _store(storage, key, img.resize((edge, edge), Image.LANCZOS))


def _product_on_busy_backdrop(storage, key: str, size: int = 512):
    """A product in a room-like scene, which flood-fill must decline.

    Large blocks of differing colour rather than noise: noise blurs into flat grey
    and would make a busy backdrop measure as uniform, which is how a fixture can
    end up validating nothing.
    """
    img = Image.new("RGB", (size, size), (206, 190, 168))
    d = ImageDraw.Draw(img)
    d.rectangle([0, round(size * 0.67), size, size], fill=(122, 92, 66))
    d.rectangle(
        [round(size * 0.06), round(size * 0.19), round(size * 0.39), round(size * 0.67)],
        fill=(88, 110, 96),
    )
    d.rectangle(
        [round(size * 0.67), round(size * 0.09), size, round(size * 0.47)], fill=(230, 226, 214)
    )
    img = img.filter(ImageFilter.GaussianBlur(6))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        [round(size * 0.33), round(size * 0.23), round(size * 0.66), round(size * 0.78)],
        radius=round(size * 0.05),
        fill=(43, 58, 85),
    )
    return _store(storage, key, img)


def _mean_delta_e(palette_hex: list[str], targets_hex: list[str]) -> float:
    """Mean ΔE from each extracted colour to the nearest true product colour."""
    targets = F.rgb_to_lab(
        np.array([F._hex_to_rgb_tuple(h) for h in targets_hex], dtype=np.float64)
    )
    total = 0.0
    for hex_value in palette_hex:
        lab = F.rgb_to_lab(np.array(F._hex_to_rgb_tuple(hex_value), dtype=np.float64))
        total += min(float(np.linalg.norm(lab - t)) for t in targets)
    return total / max(1, len(palette_hex))


class _RecordingImageProvider(P.MockImageProvider):
    """Records which reference keys each generation call actually received."""

    def __init__(self, storage):
        super().__init__(storage)
        self.reference_keys: list[list[str]] = []

    async def generate(self, request):
        self.reference_keys.append([r.key for r in request.references])
        return await super().generate(request)


# --- Reference validation ---------------------------------------------------


async def test_an_undersized_upload_is_refused_before_any_spend(
    storage, ledger, make_request, pipeline
):
    """The whole point of validating at intake: refuse before paying for anything."""
    small = _resized_product(storage, "uploads/t/tiny.png", 180)
    request = make_request(product_image=small)

    report = preprocess(request, storage)
    assert report.blocking_reason is not None
    assert "resolution" in report.blocking_reason

    record = await pipeline.run(request)
    assert record.state is JobState.FAILED
    assert "No generator can recover detail" in record.error
    assert await ledger.total_spent() == 0.0
    # Refused before the design space was even sampled.
    assert record.result.briefs is None
    assert record.result.images == []


def test_resolution_is_the_only_blocking_check(storage, make_request):
    """Focus is measured and reported, but must not refuse a job yet.

    The focus threshold was calibrated against vector fixtures whose hard edges
    produce unrealistically high Laplacian variance, so it is not yet trustworthy
    enough to block on. If someone promotes it to a blocking check, this test
    should fail and send them to look at MIN_SHARPNESS first.
    """
    request = make_request()
    report = preprocess(request, storage)

    for ref in report.references:
        assert [c.name for c in ref.checks] == ["resolution"]
        assert "focus" in [m.name for m in ref.measurements]

    human = next(r for r in report.references if r.role == "human_model")
    # The synthetic portrait is flat-shaded, so it measures soft — and is still usable.
    assert human.measurements[0].value < 1.0
    assert human.usable


def test_a_soft_upload_is_advised_about_rather_than_refused(storage, make_request):
    blurry = _resized_product(storage, "uploads/t/blurry.png", MIN_EDGE_PX + 40)
    with Image.open(io.BytesIO(storage.get_bytes(blurry.key))) as img:
        soft = img.filter(ImageFilter.GaussianBlur(4))
    blurry = _store(storage, "uploads/t/blurry.png", soft)

    report = preprocess(make_request(product_image=blurry), storage)
    product = next(r for r in report.references if r.role == "product")

    assert product.usable, "a soft upload must not be refused"
    focus = next(m for m in product.measurements if m.name == "focus")
    assert focus.value < MIN_SHARPNESS
    assert not focus.passed
    assert any("focus" in note for note in product.advisories)


# --- Product cutout ---------------------------------------------------------


def test_cutout_separates_the_product_from_a_flat_backdrop(storage, make_request):
    report = preprocess(make_request(), storage)
    cutout = report.cutout

    assert cutout.accepted
    assert cutout.method in {"rembg", "flood-fill"}
    assert cutout.asset is not None
    # The product occupies a modest share of the frame; anything near 0 or 1 means
    # the matte removed the subject or removed nothing.
    assert 0.05 < cutout.coverage < 0.6

    rgba = F.load_rgba(storage.get_bytes(cutout.asset.key))
    assert rgba.shape[-1] == 4
    # The frame corners are backdrop, so they must be transparent.
    for y, x in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert rgba[y, x, 3] == 0
    # The centre is product, so it must not be.
    mid = rgba.shape[0] // 2, rgba.shape[1] // 2
    assert rgba[mid[0], mid[1], 3] > 0


def test_cutout_is_declined_rather_than_trusted_on_a_busy_backdrop(storage, make_request):
    """The fallback's precondition is what makes it safe to ship at all.

    Skipped when rembg is installed: a trained matting model is *expected* to
    handle a busy backdrop, so declining would be the wrong behaviour there.
    """
    try:
        import rembg  # noqa: F401

        pytest.skip("rembg installed; the flood-fill precondition does not apply")
    except ImportError:
        pass

    busy = _product_on_busy_backdrop(storage, "uploads/t/busy.png")
    report = preprocess(make_request(product_image=busy), storage)

    assert not report.cutout.accepted
    assert report.cutout.asset is None
    assert report.cutout.border_uniformity < 0.35
    assert "too varied" in report.cutout.detail
    # And the original is what generation receives.
    assert report.product_reference(busy).key == busy.key


async def test_the_cutout_is_what_generation_actually_receives(storage, governor, make_request):
    """Producing a cutout and then not using it would be invisible otherwise."""
    recorder = _RecordingImageProvider(storage)
    pipeline = Pipeline(
        storage=storage,
        governor=governor,
        image_provider=recorder,
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
    )
    request = make_request()
    record = await pipeline.run(request)

    assert record.state is JobState.COMPLETED
    cutout = record.result.intake.cutout
    assert cutout.accepted, "this test only means anything when a cutout was produced"

    assert recorder.reference_keys, "no generation calls were recorded"
    for keys in recorder.reference_keys:
        assert cutout.asset.key in keys
        assert request.product_image.key not in keys


# --- Palette --------------------------------------------------------------


def test_palette_is_extracted_when_the_user_supplies_none(storage, make_request):
    request = make_request(theme=ThemeSpec(palette=[]))
    report = preprocess(request, storage)

    assert report.palette
    assert len(report.palette) <= PALETTE_SIZE
    assert report.palette_source in {"product-cutout", "product-frame"}
    assert len(report.palette_weights) == len(report.palette)
    assert report.palette_weights == sorted(report.palette_weights, reverse=True)


def test_a_supplied_palette_is_never_overwritten(storage, make_request):
    request = make_request(theme=ThemeSpec(palette=["#112233", "#445566"]))
    report = preprocess(request, storage)

    assert report.palette == ["#112233", "#445566"]
    assert report.palette_source == "user"
    assert not request.theme.palette_auto_extracted


def test_masking_by_the_cutout_gives_a_truer_palette_than_the_whole_frame(storage, make_request):
    """The reason the cutout exists at all, stated as a measurement.

    Reading dominant colours off the whole photograph describes the backdrop,
    because the backdrop covers most of the pixels. Reading them from inside the
    product mask describes the product. This asserts the difference rather than
    trusting it.
    """
    request = make_request()
    report = preprocess(request, storage)
    assert report.cutout.accepted

    product_rgb = F.load_image(storage.get_bytes(request.product_image.key))
    rgba = F.load_rgba(storage.get_bytes(report.cutout.asset.key))
    mask = rgba[..., 3] > 8

    from_frame = [c for c, _ in F.extract_palette(product_rgb, k=PALETTE_SIZE)]
    from_mask = [c for c, _ in F.extract_palette(product_rgb, k=PALETTE_SIZE, mask=mask)]

    frame_error = _mean_delta_e(from_frame, TRUE_PRODUCT_COLOURS)
    mask_error = _mean_delta_e(from_mask, TRUE_PRODUCT_COLOURS)
    assert mask_error < frame_error, (
        f"masked palette {from_mask} (ΔE {mask_error:.1f}) should be closer to the "
        f"product's real colours than {from_frame} (ΔE {frame_error:.1f})"
    )


async def test_an_extracted_palette_reaches_the_gate_and_is_flagged_as_derived(
    storage, make_request, pipeline
):
    request = make_request(theme=ThemeSpec(palette=[]))
    record = await pipeline.run(request)

    assert record.state is JobState.COMPLETED
    assert record.request.theme.palette, "the derived palette should be written back"
    assert record.request.theme.palette_auto_extracted is True
    assert record.request.theme.palette == record.result.intake.palette

    # And the gate actually scored against it rather than skipping the check.
    for candidate in record.result.images:
        check = next(c for c in candidate.gate.checks if c.name == "palette_adherence")
        assert check.implemented


# --- Face detection -------------------------------------------------------


def test_face_detection_never_reports_what_it_did_not_measure(storage, make_request):
    report = preprocess(make_request(), storage)
    face = report.face

    if not face.implemented:
        assert face.detector == "unavailable"
        assert face.box is None
        assert face.faces_found == 0
        assert face.box_area_fraction is None
        assert face.detail, "an unavailable detector must say why"
    else:
        assert face.detector == "opencv-haar"
        if face.faces_found:
            assert face.box is not None and len(face.box) == 4
            assert 0.0 < face.box_area_fraction <= 1.0
        else:
            assert face.box is None


def test_a_missed_face_does_not_refuse_the_job(storage, make_request):
    """A detector with uneven error rates across faces must not gate anyone out."""
    faceless = _resized_product(storage, "uploads/t/faceless.png", 512)
    report = preprocess(make_request(human_model_image=faceless), storage)

    assert report.blocking_reason is None
    human = next(r for r in report.references if r.role == "human_model")
    assert human.usable


# --- Feature-level guards --------------------------------------------------


def test_sharpness_survives_a_deliberately_blurred_background():
    """Shallow depth of field is good photography, not a defect.

    Guards the tiled implementation: a global Laplacian variance would score the
    bokeh frame far below the flat-background one, and reject good product shots.
    """
    size = 512
    scene = Image.new("RGB", (size, size), (150, 150, 150))
    d = ImageDraw.Draw(scene)
    d.rectangle([0, 340, size, size], fill=(120, 90, 60))
    d.rectangle([40, 90, 200, 340], fill=(80, 120, 100))
    blurred_bg = scene.filter(ImageFilter.GaussianBlur(14))

    def with_sharp_subject(base: Image.Image) -> np.ndarray:
        img = base.copy()
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle([180, 130, 340, 400], radius=22, fill=(43, 58, 85))
        draw.rectangle([205, 210, 315, 270], fill=(240, 236, 226))
        return np.asarray(img)

    bokeh = F.sharpness(with_sharp_subject(blurred_bg))
    flat = F.sharpness(with_sharp_subject(Image.new("RGB", (size, size), (240, 240, 240))))

    assert bokeh > MIN_SHARPNESS
    assert bokeh > 0.6 * flat, (
        f"a blurred background cost too much sharpness: bokeh {bokeh:.3f} vs flat {flat:.3f}"
    )


def test_flood_mask_requires_connectivity_to_the_frame_edge():
    """Colour similarity alone would punch a hole through the product.

    The product here carries a label the same colour as the backdrop. A threshold
    on colour would remove it; requiring a path back to the border does not.
    """
    size = 512
    backdrop = (245, 245, 245)
    img = Image.new("RGB", (size, size), backdrop)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([170, 120, 340, 400], radius=24, fill=(43, 58, 85))
    d.rectangle([200, 200, 310, 260], fill=backdrop)  # label, same colour as backdrop

    rgb = np.asarray(img)
    mask = F.background_mask_by_flood(rgb)

    assert mask[0, 0], "the corner is backdrop and must be masked"
    assert not mask[230, 255], "the label is enclosed by product and must survive"
    assert not mask[160, 255], "the product body must survive"


def test_alpha_coverage_detects_a_matte_that_removed_everything():
    opaque = np.dstack([np.full((32, 32, 3), 128, np.uint8), np.full((32, 32), 255, np.uint8)])
    empty = np.dstack([np.full((32, 32, 3), 128, np.uint8), np.zeros((32, 32), np.uint8)])

    assert F.alpha_coverage(opaque) == 1.0
    assert F.alpha_coverage(empty) == 0.0
    assert F.alpha_coverage(np.full((32, 32, 3), 128, np.uint8)) == 1.0
