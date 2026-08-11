"""The live fal adapters, driven entirely by a stub transport.

**No test here touches the network and none of them can spend money.** Every
request is served by an in-process `httpx.MockTransport`, which also means these
tests verify the adapters against fal's *documented* contract rather than its
actual behaviour. That distinction matters and is stated in `adproviders.fal`
too: the documentation could be wrong, and one cheap real call per adapter is
budgeted to find out before any bulk run.

What these do establish is everything that would otherwise only be discovered
while paying: that the request matches the documented schema, that the duration
enum is respected, that a queued job is polled to completion, that failures map
onto the pipeline's error types rather than escaping as HTTP noise, and that a
missing key fails before a job starts instead of halfway through one.
"""

from __future__ import annotations

import base64
import json

import adproviders as P
import httpx
import pytest
from adproviders.fal import KLING_I2V_ENDPOINT, SEEDREAM_EDIT_ENDPOINT, as_data_uri
from adschema import AspectRatio, DesignPoint, ShotBrief
from adschema.enums import CameraAngle, Composition, Lighting, MotionIntent

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32


def _brief(index: int = 0) -> ShotBrief:
    return ShotBrief(
        index=index,
        design_point=DesignPoint(
            index=index,
            angle=CameraAngle.EYE_LEVEL,
            lighting=Lighting.SOFT_DIFFUSED,
            composition=Composition.CENTERED_HERO,
            motion=MotionIntent.SLOW_DOLLY_IN,
            seed=7,
        ),
        image_prompt="Advertising photograph for Aurora Serum.",
        negative_prompt="distorted face, extra fingers",
        motion_prompt="slow smooth dolly in toward the subject",
        concept="Calm premium hero shot.",
    )


class _Recorder:
    """Captures every request the adapter makes, and serves canned responses."""

    def __init__(
        self, *, queue: bool = True, fail_with: int | None = None, status: str = "COMPLETED"
    ):
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []
        self.queue = queue
        self.fail_with = fail_with
        self.status = status
        self.status_polls = 0
        # The queue's response_url does not name the model, so which result to
        # serve has to be remembered from the submit call.
        self.submitted_endpoint = ""

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)

        if request.method == "POST":
            self.bodies.append(json.loads(request.content))
            self.submitted_endpoint = url
            if self.fail_with:
                return httpx.Response(self.fail_with, text="nope")
            if not self.queue:
                return httpx.Response(200, json=self._result_body(url))
            return httpx.Response(
                200,
                json={
                    "request_id": "req-123",
                    "status_url": "https://queue.fal.run/x/requests/req-123/status",
                    "response_url": "https://queue.fal.run/x/requests/req-123",
                },
            )

        if url.endswith("/status"):
            self.status_polls += 1
            # First poll is still running, so the polling loop is genuinely exercised.
            state = "IN_PROGRESS" if self.status_polls < 2 else self.status
            return httpx.Response(200, json={"status": state, "error": "boom"})

        if url.endswith("/assets/image.png"):
            return httpx.Response(200, content=PNG_BYTES, headers={"content-type": "image/png"})
        if url.endswith("/assets/video.mp4"):
            return httpx.Response(200, content=MP4_BYTES, headers={"content-type": "video/mp4"})

        return httpx.Response(200, json=self._result_body(self.submitted_endpoint))

    @staticmethod
    def _result_body(url: str) -> dict:
        if "kling" in url or "video" in url:
            return {
                "video": {
                    "url": "https://cdn.fal.test/assets/video.mp4",
                    "content_type": "video/mp4",
                    "file_name": "video.mp4",
                    "file_size": len(MP4_BYTES),
                }
            }
        return {
            "images": [
                {
                    "url": "https://cdn.fal.test/assets/image.png",
                    "content_type": "image/png",
                    "width": 864,
                    "height": 1536,
                }
            ],
            "seed": 4242,
        }


def _client(recorder: _Recorder, key: str = "test-key") -> P.FalClient:
    # Zero poll interval so the polling loop runs without a real delay.
    return P.FalClient(key, transport=recorder.transport(), poll_interval_s=0.0)


# --- Image adapter ---------------------------------------------------------


async def test_image_request_matches_the_documented_seedream_schema(storage, references):
    human, product = references
    recorder = _Recorder()
    provider = P.FalImageProvider(storage, client=_client(recorder))

    result = await provider.generate(
        P.ImageGenRequest(
            brief=_brief(),
            references=[human, product],
            aspect_ratio=AspectRatio.VERTICAL_9_16,
            seed=7,
            output_key="generations/j/img_0.png",
        )
    )

    body = recorder.bodies[0]
    assert body["prompt"].startswith("Advertising photograph")
    assert len(body["image_urls"]) == 2
    assert all(u.startswith("data:image/") for u in body["image_urls"])
    assert body["num_images"] == 1
    assert body["seed"] == 7
    assert body["enable_safety_checker"] is True
    # 9:16 is not one of fal's named presets, and the platform geometry is not
    # negotiable, so explicit pixels are sent instead.
    assert body["image_size"]["width"] < body["image_size"]["height"]

    assert str(recorder.requests[0].url).endswith(SEEDREAM_EDIT_ENDPOINT)
    assert recorder.requests[0].headers["authorization"] == "Key test-key"

    assert storage.get_bytes(result.asset.key) == PNG_BYTES
    assert result.cost_usd == pytest.approx(0.04)
    assert result.seed == 4242
    assert result.model == "seedream-4.5-edit"


async def test_too_many_references_fails_before_the_call(storage, references):
    human, product = references
    recorder = _Recorder()
    provider = P.FalImageProvider(storage, client=_client(recorder))

    with pytest.raises(P.ProviderError, match="reference images"):
        await provider.generate(
            P.ImageGenRequest(
                brief=_brief(),
                references=[human, product] * 6,  # 12 > the documented 10
                aspect_ratio=AspectRatio.VERTICAL_9_16,
                seed=7,
                output_key="generations/j/img_0.png",
            )
        )
    assert recorder.requests == [], "nothing should have been sent"


# --- Video adapter ---------------------------------------------------------


async def test_a_nine_second_request_is_sent_and_reported_as_ten(storage, references):
    """The single most consequential finding of the provider spike.

    Kling's image-to-video endpoint takes duration as the enum {"5", "10"}. The
    project's default request is 9 s, which is not requestable, so it snaps up —
    and both the payload and the reported duration have to say 10, because that
    is what the file contains and what the invoice reflects.
    """
    _, product = references
    recorder = _Recorder()
    provider = P.FalVideoProvider(storage, client=_client(recorder))

    result = await provider.generate(
        P.VideoGenRequest(
            brief=_brief(),
            start_image=product,
            duration_seconds=9.0,
            aspect_ratio=AspectRatio.VERTICAL_9_16,
            seed=7,
            output_key="generations/j/vid_0.mp4",
        )
    )

    body = recorder.bodies[0]
    assert body["duration"] == "10", "duration is a string enum, not a number"
    assert body["image_url"].startswith("data:image/")
    assert body["prompt"] == "slow smooth dolly in toward the subject"
    assert body["negative_prompt"]
    assert "seed" not in body, "the endpoint has no seed parameter"

    assert result.duration_seconds == 10.0
    assert result.cost_usd == pytest.approx(0.70)
    assert result.seed_honoured is False
    assert not result.was_chained
    assert storage.get_bytes(result.asset.key) == MP4_BYTES


async def test_a_five_second_request_stays_five(storage, references):
    _, product = references
    recorder = _Recorder()
    provider = P.FalVideoProvider(storage, client=_client(recorder))

    result = await provider.generate(
        P.VideoGenRequest(
            brief=_brief(),
            start_image=product,
            duration_seconds=5.0,
            aspect_ratio=AspectRatio.VERTICAL_9_16,
            seed=7,
            output_key="generations/j/vid_0.mp4",
        )
    )
    assert recorder.bodies[0]["duration"] == "5"
    assert result.duration_seconds == 5.0
    assert result.cost_usd == pytest.approx(0.35)


# --- Client behaviour ------------------------------------------------------


async def test_a_queued_job_is_polled_until_it_completes(storage, references):
    _, product = references
    recorder = _Recorder()
    provider = P.FalVideoProvider(storage, client=_client(recorder))

    await provider.generate(
        P.VideoGenRequest(
            brief=_brief(),
            start_image=product,
            duration_seconds=10.0,
            aspect_ratio=AspectRatio.VERTICAL_9_16,
            seed=7,
            output_key="generations/j/vid_0.mp4",
        )
    )
    assert recorder.status_polls >= 2, "the first poll returns IN_PROGRESS"


async def test_a_failed_queue_job_raises_rather_than_returning_nothing(storage, references):
    _, product = references
    recorder = _Recorder(status="FAILED")
    provider = P.FalVideoProvider(storage, client=_client(recorder))

    with pytest.raises(P.ProviderError, match="FAILED"):
        await provider.generate(
            P.VideoGenRequest(
                brief=_brief(),
                start_image=product,
                duration_seconds=10.0,
                aspect_ratio=AspectRatio.VERTICAL_9_16,
                seed=7,
                output_key="generations/j/vid_0.mp4",
            )
        )


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, P.ProviderUnavailable),
        (403, P.ProviderUnavailable),
        (402, P.ProviderUnavailable),
        (500, P.ProviderError),
    ],
)
async def test_http_failures_map_onto_the_pipelines_error_types(
    storage, references, status_code, expected
):
    """A bad key and a flaky server need different handling.

    ProviderUnavailable means "do not retry" — retrying a rejected key wastes
    time and retrying a 402 while out of credit cannot possibly work. Only
    ProviderError is retryable, and the governor caps that at one attempt.
    """
    human, product = references
    recorder = _Recorder(fail_with=status_code)
    provider = P.FalImageProvider(storage, client=_client(recorder))

    with pytest.raises(expected):
        await provider.generate(
            P.ImageGenRequest(
                brief=_brief(),
                references=[human, product],
                aspect_ratio=AspectRatio.VERTICAL_9_16,
                seed=7,
                output_key="generations/j/img_0.png",
            )
        )


async def test_a_missing_api_key_fails_before_any_request(storage, references):
    human, product = references
    recorder = _Recorder()
    provider = P.FalImageProvider(storage, client=_client(recorder, key=""))

    with pytest.raises(P.ProviderUnavailable, match="fal API key"):
        await provider.generate(
            P.ImageGenRequest(
                brief=_brief(),
                references=[human, product],
                aspect_ratio=AspectRatio.VERTICAL_9_16,
                seed=7,
                output_key="generations/j/img_0.png",
            )
        )
    assert recorder.requests == []


async def test_a_synchronous_response_is_accepted_without_a_queue_id(storage, references):
    """Not every fal deployment queues; taking the result is better than failing."""
    human, product = references
    recorder = _Recorder(queue=False)
    provider = P.FalImageProvider(storage, client=_client(recorder))

    result = await provider.generate(
        P.ImageGenRequest(
            brief=_brief(),
            references=[human, product],
            aspect_ratio=AspectRatio.VERTICAL_9_16,
            seed=7,
            output_key="generations/j/img_0.png",
        )
    )
    assert storage.get_bytes(result.asset.key) == PNG_BYTES


def test_local_assets_are_inlined_because_fal_cannot_reach_localhost(storage, references):
    human, _ = references
    uri = as_data_uri(storage, human)

    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == storage.get_bytes(human.key)


# --- Registry and mode safety ---------------------------------------------


def test_mock_mode_cannot_resolve_a_live_provider(tmp_path):
    """The money switch, restated as a test.

    Even with a key present and a live provider named, mock mode must hand back
    the mock — this is what makes it safe to develop all day against real config.
    """
    settings = P.Settings(
        provider_mode="mock",
        image_provider="seedream-4.5-edit",
        video_provider="kling-2.5-turbo-pro",
        fal_api_key="test-key",
        storage_root=tmp_path,
    )
    assert isinstance(P.get_image_provider(settings), P.MockImageProvider)
    assert isinstance(P.get_video_provider(settings), P.MockVideoProvider)


def test_live_mode_without_a_key_refuses_to_build_a_provider(tmp_path):
    settings = P.Settings(
        provider_mode="live",
        image_provider="seedream-4.5-edit",
        video_provider="kling-2.5-turbo-pro",
        fal_api_key="",
        storage_root=tmp_path,
    )
    with pytest.raises(P.ProviderUnavailable, match="AD_FAL_API_KEY"):
        P.get_image_provider(settings)
    with pytest.raises(P.ProviderUnavailable, match="AD_FAL_API_KEY"):
        P.get_video_provider(settings)


def test_live_mode_with_a_key_resolves_the_fal_adapters(tmp_path):
    settings = P.Settings(
        provider_mode="live",
        image_provider="seedream-4.5-edit",
        video_provider="kling-2.5-turbo-pro",
        fal_api_key="test-key",
        storage_root=tmp_path,
    )
    image = P.get_image_provider(settings)
    video = P.get_video_provider(settings)

    assert isinstance(image, P.FalImageProvider)
    assert isinstance(video, P.FalVideoProvider)
    assert video.supports_duration(9.0)
    assert video.deliverable_duration(9.0) == 10.0
    assert video.honours_seed is False
    assert KLING_I2V_ENDPOINT.endswith("image-to-video")
