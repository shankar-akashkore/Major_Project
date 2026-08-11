"""The research tier: clips generated free on Colab, served from a manifest.

Two properties carry most of the weight.  The fingerprint has to agree with the copy
of that function living inside ``notebooks/colab_video.ipynb`` — a drift there costs
a whole GPU session and is invisible until every lookup misses.  And a miss has to
raise rather than fall back to the mock renderer, because a research-tier run
quietly filled with placeholder frames would make the tier comparison meaningless.
"""

from __future__ import annotations

import hashlib
import json

import adproviders as P
import numpy as np
import pytest
from adml import video as V
from adproviders.prerendered import MANIFEST_NAME, MANIFEST_VERSION
from adschema import AspectRatio, MotionIntent, ProviderMode, Tier

# --- Fixtures ----------------------------------------------------------------


@pytest.fixture
def clip_bytes() -> bytes:
    """A tiny real MP4, so the provider serves something that actually decodes."""
    if V.FFMPEG is None:
        pytest.skip("ffmpeg is not installed")
    frames = [np.full((32, 32, 3), i * 12 % 256, dtype=np.uint8) for i in range(20)]
    return V.encode_mp4(frames, fps=10)


@pytest.fixture
def research_corpus(storage, clip_bytes, references):
    """A manifest and one clip, laid out the way the notebook's zip would be."""
    human, _ = references
    digest = hashlib.sha256(storage.get_bytes(human.key)).hexdigest()
    fingerprint = P.clip_fingerprint(digest, MotionIntent.SLOW_DOLLY_IN.value, 9.0)

    storage.put_bytes("research/item-0.mp4", clip_bytes, "video/mp4")
    manifest = {
        "version": MANIFEST_VERSION,
        "notebook": "colab_video.ipynb",
        "generated_at": "2026-08-11",
        "clips": [
            {
                "fingerprint": fingerprint,
                "key": "research/item-0.mp4",
                "model": "ltx-video",
                "duration_seconds": 9.0,
                "fps": 24,
                "seed": 7,
                "seed_honoured": True,
            }
        ],
    }
    path = storage.root / "research"
    path.mkdir(parents=True, exist_ok=True)
    (path / MANIFEST_NAME).write_text(json.dumps(manifest))
    return path, human, fingerprint


def _request(brief, start, *, duration=9.0, key="generations/j/vid_0.mp4"):
    return P.VideoGenRequest(
        brief=brief,
        start_image=start,
        duration_seconds=duration,
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        seed=7,
        output_key=key,
    )


@pytest.fixture
def dolly_brief():
    from adschema import BackgroundTreatment, CameraAngle, Composition, Lighting
    from adschema.brief import DesignPoint, ShotBrief

    return ShotBrief(
        index=0,
        design_point=DesignPoint(
            index=0,
            seed=7,
            angle=CameraAngle.EYE_LEVEL,
            lighting=Lighting.SOFT_DIFFUSED,
            composition=Composition.CENTERED_HERO,
            motion=MotionIntent.SLOW_DOLLY_IN,
            background=BackgroundTreatment.SOFT_GRADIENT,
        ),
        image_prompt="a prompt",
        motion_prompt="slow dolly in",
    )


# --- The notebook contract ---------------------------------------------------


def test_the_notebook_fingerprint_matches_the_provider():
    """Pins the exact payload the notebook's duplicated copy must produce.

    The notebook cannot import this repo, so the function exists twice. This test is
    what makes that safe: it hard-codes the payload shape rather than calling the
    provider's own function, so a change to either copy alone fails here instead of
    silently missing every lookup after four hours of GPU time.
    """
    payload = json.dumps(
        {"start": "abc123", "motion": "slow_dolly_in", "duration": 9.0},
        sort_keys=True,
    )
    expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    assert P.clip_fingerprint("abc123", "slow_dolly_in", 9.0) == expected


def test_the_fingerprint_uses_the_motion_intent_not_the_prompt_text(research_corpus, dolly_brief):
    """Prompt wording must not be part of the key.

    ``compile_briefs`` expands briefs through an LLM, so the prompt text is not
    reproducible between runs. If it were fingerprinted, every clip would be
    orphaned the moment the compiler phrased the same design point differently.
    """
    path, human, _ = research_corpus
    provider = P.PrerenderedVideoProvider(P.LocalStorage(path.parent), path)
    reworded = dolly_brief.model_copy(
        update={"motion_prompt": "a completely different sentence about dollying"}
    )
    assert provider.fingerprint_for(_request(dolly_brief, human)) == provider.fingerprint_for(
        _request(reworded, human)
    )


def test_duration_rounding_does_not_split_one_request_in_two():
    a = P.clip_fingerprint("d", "static_subtle", 9.0)
    b = P.clip_fingerprint("d", "static_subtle", 9.0000001)
    assert a == b


# --- Serving -----------------------------------------------------------------


async def test_a_prerendered_clip_is_served_free_at_the_research_tier(
    research_corpus, dolly_brief, storage
):
    path, human, _ = research_corpus
    provider = P.PrerenderedVideoProvider(storage, path)

    result = await provider.generate(_request(dolly_brief, human))

    assert result.cost_usd == 0.0
    assert result.tier is Tier.RESEARCH
    assert result.model == "ltx-video"
    assert result.seed_honoured  # unlike Kling, which has no seed at all
    assert result.cache_hit
    # Copied to the job's key, so a research run leaves the same layout as a paid one.
    assert result.asset.key == "generations/j/vid_0.mp4"
    assert V.container_of(storage.get_bytes(result.asset.key)) == "mp4"


async def test_a_miss_refuses_rather_than_substituting_a_mock_clip(
    research_corpus, dolly_brief, storage, references
):
    """The refusal that keeps the tier comparison meaningful.

    A silent fallback would put synthetic placeholder frames into a run labelled
    "research tier", and the report would be comparing Kling against a rounded
    rectangle.
    """
    path, human, _ = research_corpus
    _, product = references
    provider = P.PrerenderedVideoProvider(storage, path)

    # A different start frame, so the fingerprint cannot match.
    with pytest.raises(P.ProviderError, match="no pre-rendered clip"):
        await provider.generate(_request(dolly_brief, product))


async def test_a_manifest_entry_pointing_at_a_missing_file_says_so(
    research_corpus, dolly_brief, storage
):
    path, human, _ = research_corpus
    (path / "item-0.mp4").unlink()
    provider = P.PrerenderedVideoProvider(storage, path)
    with pytest.raises(P.ProviderError, match="not fully extracted|not in storage"):
        await provider.generate(_request(dolly_brief, human))


# --- Manifest handling -------------------------------------------------------


def test_a_missing_manifest_explains_how_to_produce_one(storage, tmp_path):
    provider = P.PrerenderedVideoProvider(storage, tmp_path / "nothing-here")
    with pytest.raises(P.ProviderUnavailable, match="colab_video.ipynb"):
        _ = provider.manifest


def test_an_old_manifest_version_is_refused(storage, tmp_path):
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"version": 0, "clips": []}))
    with pytest.raises(P.ProviderError, match="version"):
        P.ClipManifest.load(tmp_path)


def test_an_incomplete_clip_entry_names_the_missing_fields(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps({"version": MANIFEST_VERSION, "clips": [{"fingerprint": "x"}]})
    )
    with pytest.raises(P.ProviderError, match="missing"):
        P.ClipManifest.load(tmp_path)


def test_the_manifest_reports_which_generator_made_each_clip(tmp_path):
    """One session can legitimately mix LTX-Video and Wan output.

    The report needs to know which clip came from which, so the model is recorded
    per clip rather than per manifest.
    """

    def entry(name: str, model: str) -> dict:
        return {"fingerprint": name, "key": f"k{name}", "model": model, "duration_seconds": 9}

    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "version": MANIFEST_VERSION,
                "clips": [
                    entry("a", "ltx-video"),
                    entry("b", "ltx-video"),
                    entry("c", "wan-2.1-i2v"),
                ],
            }
        )
    )
    manifest = P.ClipManifest.load(tmp_path)
    assert manifest.models == {"ltx-video": 2, "wan-2.1-i2v": 1}
    assert "ltx-video x2" in manifest.summary()


# --- Registry wiring ---------------------------------------------------------


def test_the_free_tier_does_not_require_flipping_the_money_switch(storage, tmp_path):
    """Mock mode's guarantee is "no spend", not "no provider".

    Requiring ``AD_PROVIDER_MODE=live`` to serve clips that were generated for free
    would also arm the paid image provider, which is the opposite of safe.
    """
    settings = P.Settings(
        provider_mode=ProviderMode.MOCK, video_provider="ltx-video", clip_manifest=tmp_path
    )
    provider = P.get_video_provider(settings, storage)
    assert isinstance(provider, P.PrerenderedVideoProvider)
    assert provider.estimate_cost(10.0) == 0.0


def test_a_paid_provider_is_still_forced_to_mock_in_mock_mode(storage):
    settings = P.Settings(provider_mode=ProviderMode.MOCK, video_provider="kling-2.5-turbo-pro")
    assert isinstance(P.get_video_provider(settings, storage), P.MockVideoProvider)


def test_every_free_video_provider_really_is_priced_at_zero():
    """The allowlist and the price table must not disagree.

    A name on the free list that turned out to cost money would let mock mode spend.
    """
    from adproviders.registry import _FREE_VIDEO_PROVIDERS

    for name in _FREE_VIDEO_PROVIDERS:
        if name in P.VIDEO_PRICES:
            assert P.VIDEO_PRICES[name].usd_per_second == 0.0, name
