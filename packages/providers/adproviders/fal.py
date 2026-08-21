"""fal.ai adapters — the live image and video providers.

Both live models sit behind one platform on purpose.  fal hosts Seedream (the
multi-reference image model) and Kling (the image-to-video model), so the project
needs one account, one API key, one auth scheme and one billing statement instead
of two of each.  For a solo build with a $35 ceiling, that is worth more than
squeezing the last cent out of per-model pricing.

**Nothing here has been run against the live API.**  Every test drives a stub
transport, and no request has been sent, so what is verified is that the adapters
build the documented request, parse the documented response, and map failures onto
the pipeline's error types.  What is *not* verified is that the documentation is
accurate.  Budget one cheap call per adapter to confirm that before any bulk run —
see ``scripts/smoke_live.py``.

Two provider facts shaped this module more than anything else:

* **Kling takes ``duration`` as the enum {"5", "10"}.**  There is no 9.  A request
  inside the project's 8-10 s window has to snap up to a real option, and the cost
  has to be estimated on what will be billed.  See ``VideoPrice.snap_duration``.
* **Kling's image-to-video endpoint has no seed.**  The image stage is
  reproducible and the video stage is not, which the evaluation has to state
  rather than assume away.  ``VideoGenResult.seed_honoured`` carries it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import Any

import httpx
from adml import video as V
from adschema import AspectRatio, AssetRef, Tier
from PIL import Image, UnidentifiedImageError

from .base import (
    ImageGenRequest,
    ImageGenResult,
    ImageProvider,
    ProviderError,
    ProviderUnavailable,
    VideoGenRequest,
    VideoGenResult,
    VideoProvider,
)
from .pricing import IMAGE_PRICES, VIDEO_PRICES
from .storage import Storage

FAL_RUN_BASE = "https://fal.run"
FAL_QUEUE_BASE = "https://queue.fal.run"

#: Endpoint ids, kept beside the price-table keys they correspond to.
SEEDREAM_EDIT_ENDPOINT = "fal-ai/bytedance/seedream/v4.5/edit"
KLING_I2V_ENDPOINT = "fal-ai/kling-video/v2.5-turbo/pro/image-to-video"

#: How closely Kling is asked to follow the prompt. fal documents the range as
#: 0-1 and defaults it to 0.5; higher means stick closer to what was written.
#:
#: Raised from the default because the failure this project hit was precisely the
#: prompt not being followed — a detailed action description came back as a slow
#: zoom. Unlike everything else in the motion fix this is a judgement call rather
#: than a correction of something demonstrably wrong, so it lives here as one
#: named number: A/B it against 0.5 on a single clip when there is budget to
#: spare, and push it no higher without looking at the result, since guidance
#: turned all the way up trades motion for artefacts.
KLING_CFG_SCALE = 0.7

#: A single generation is slow — tens of seconds for an image, minutes for video.
IMAGE_TIMEOUT_S = 180.0
VIDEO_TIMEOUT_S = 900.0
#: How often to ask the queue whether a job is done.
POLL_INTERVAL_S = 3.0
#: Per-HTTP-request timeout. Separate from the overall job timeout above: a slow
#: generation is normal, a slow *response to a status check* is not.
HTTP_TIMEOUT_S = 60.0
#: How many times a submission is re-sent after fal asks it to wait.
#:
#: This exists because the pipeline submits its slots concurrently. A 429 is not a
#: failure of the generation — nothing was generated, and nothing was charged — but
#: the slot it lands on has already passed the budget check, so treating it as an
#: error converts a rate limit into a candidate the user paid attention to and
#: never received. Retrying a request that was *refused* cannot double-charge,
#: which is what makes this safe to do automatically where a failed generation
#: would not be.
RATE_LIMIT_RETRIES = 4
#: Backoff base, in seconds: 2, 4, 8, 16. Doubling rather than fixed because a
#: rate limit that is still there after two seconds is unlikely to clear on the
#: same schedule that hit it.
RATE_LIMIT_BACKOFF_S = 2.0


class _RateLimited(ProviderError):
    """fal asked for the request to be sent again later.

    Internal to this module: it never escapes ``FalClient.run``, which either
    succeeds after waiting or re-raises it as an ordinary :class:`ProviderError`
    once the retries are spent.
    """

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """The server's requested wait, when it sent one and it is a number.

    Only the delta-seconds form is honoured. ``Retry-After`` also permits an HTTP
    date, and parsing that correctly means trusting the client's clock against the
    server's — a backoff we choose ourselves is better than a wait computed from
    two disagreeing clocks.
    """
    raw = response.headers.get("retry-after", "").strip()
    try:
        seconds = float(raw)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


class FalClient:
    """Thin async client for fal's queue API.

    Deliberately small. The official `fal-client` package would work, but this
    project's whole cost discipline rests on knowing exactly what leaves the
    machine and when, and an SDK that retries internally would sit between the
    cost governor and the thing it is governing.
    """

    def __init__(
        self,
        api_key: str | None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        poll_interval_s: float = POLL_INTERVAL_S,
    ):
        self._api_key = (api_key or "").strip()
        self._transport = transport
        self.poll_interval_s = poll_interval_s

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def _require_key(self, model: str) -> None:
        if not self.configured:
            raise ProviderUnavailable(
                f"{model} needs a fal API key. Set AD_FAL_API_KEY in .env, or keep "
                "AD_PROVIDER_MODE=mock to run for free."
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Key {self._api_key}",
            "Content-Type": "application/json",
        }

    async def run(self, endpoint: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        """Submit a job and wait for its result.

        Uses the queue endpoints rather than the synchronous one because video
        generation routinely outlives any sensible HTTP timeout.

        The *submission* is retried on a 429; the polling is not, because a job
        already in the queue is not a request the service is asking us to slow
        down. See :data:`RATE_LIMIT_RETRIES`.
        """
        self._require_key(endpoint)
        deadline = time.monotonic() + timeout_s

        async with httpx.AsyncClient(
            transport=self._transport, timeout=HTTP_TIMEOUT_S, headers=self._headers()
        ) as client:
            submitted = await self._submit(client, endpoint, payload, deadline)
            request_id = submitted.get("request_id")
            if not request_id:
                # Some deployments answer the submit call with the finished
                # payload. Take it rather than failing on a missing queue id.
                if "images" in submitted or "video" in submitted:
                    return submitted
                raise ProviderError(f"fal did not return a request_id for {endpoint}: {submitted}")

            status_url = submitted.get("status_url") or (
                f"{FAL_QUEUE_BASE}/{endpoint}/requests/{request_id}/status"
            )
            response_url = submitted.get("response_url") or (
                f"{FAL_QUEUE_BASE}/{endpoint}/requests/{request_id}"
            )

            while True:
                status = await self._get(client, status_url)
                state = str(status.get("status", "")).upper()
                if state == "COMPLETED":
                    return await self._get(client, response_url)
                if state in {"FAILED", "ERROR", "CANCELLED"}:
                    raise ProviderError(
                        f"fal job {request_id} for {endpoint} ended as {state}: "
                        f"{status.get('error') or status}"
                    )
                if time.monotonic() >= deadline:
                    raise ProviderError(
                        f"fal job {request_id} for {endpoint} did not finish within "
                        f"{timeout_s:.0f}s (last status {state or 'unknown'})"
                    )
                await asyncio.sleep(self.poll_interval_s)

    async def _submit(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        payload: dict[str, Any],
        deadline: float,
    ) -> dict[str, Any]:
        """POST the job, waiting out a rate limit rather than failing on one.

        Bounded twice over: by the retry count, and by the caller's deadline. A
        backoff that outlives the job timeout would turn a rate limit into a slot
        that hangs for the full fifteen minutes a video is allowed and then fails
        anyway, which is worse than failing at once.
        """
        url = f"{FAL_QUEUE_BASE}/{endpoint}"
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                return await self._post(client, url, payload)
            except _RateLimited as limited:
                if attempt == RATE_LIMIT_RETRIES:
                    raise ProviderError(
                        f"fal kept rate-limiting {endpoint} after "
                        f"{RATE_LIMIT_RETRIES + 1} attempts. Lower "
                        "AD_MAX_CONCURRENT_GENERATIONS and run the job again — "
                        "nothing was generated, so nothing was charged."
                    ) from limited
                wait = limited.retry_after
                if wait is None:
                    wait = RATE_LIMIT_BACKOFF_S * (2**attempt)
                if time.monotonic() + wait >= deadline:
                    raise ProviderError(
                        f"fal rate-limited {endpoint} and the backoff would outlast "
                        "the job timeout, so the slot is failing now rather than "
                        "holding the pipeline open to fail later."
                    ) from limited
                await asyncio.sleep(wait)
        raise AssertionError("unreachable: the retry loop returns or raises")

    async def _post(
        self, client: httpx.AsyncClient, url: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"fal request to {url} failed: {exc}") from exc
        return self._decode(response, url)

    async def _get(self, client: httpx.AsyncClient, url: str) -> dict[str, Any]:
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            raise ProviderError(f"fal request to {url} failed: {exc}") from exc
        return self._decode(response, url)

    @staticmethod
    def _decode(response: httpx.Response, url: str) -> dict[str, Any]:
        # 401/403 are configuration faults, and retrying a bad key just burns
        # time; 402 means the account is out of credit, which retrying makes
        # worse. Everything else is retryable once, which the governor caps.
        if response.status_code in {401, 403}:
            raise ProviderUnavailable(f"fal rejected the API key ({response.status_code}) at {url}")
        if response.status_code == 402:
            raise ProviderUnavailable(f"fal reports insufficient credit (402) at {url}")
        if response.status_code == 429:
            # Separated from the generic error below so `run` can wait rather than
            # fail. Carries the server's own Retry-After when it sent one; guessing
            # a backoff when the service has stated one is just being wrong politely.
            raise _RateLimited(
                f"fal rate-limited the request at {url}",
                retry_after=_retry_after_seconds(response),
            )
        if response.status_code >= 400:
            raise ProviderError(f"fal returned {response.status_code} at {url}: {response.text}")
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderError(f"fal returned non-JSON at {url}: {response.text[:200]}") from exc

    async def fetch_bytes(self, url: str) -> bytes:
        """Download a generated asset from the URL fal hands back."""
        async with httpx.AsyncClient(transport=self._transport, timeout=HTTP_TIMEOUT_S) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ProviderError(f"could not download the generated asset: {exc}") from exc
            return response.content


#: MIME types by file extension, for encoding a local asset as a data URI.
_MIME_BY_SUFFIX = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}

#: PIL's format name for the encodings a generation endpoint might return.
_MIME_BY_FORMAT = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}


def as_data_uri(storage: Storage, asset: AssetRef) -> str:
    """Encode a stored asset as a ``data:`` URI.

    fal takes inputs as URLs, and this project's media sits in local storage
    served on ``localhost`` — which fal obviously cannot reach. Inlining the bytes
    avoids needing a public bucket before the first real generation. It costs
    roughly a third in base64 overhead, so this is the thing to replace with
    signed Supabase URLs once storage moves off the laptop.
    """
    mime = asset.mime_type or _MIME_BY_SUFFIX.get(asset.key.rsplit(".", 1)[-1].lower(), "image/png")
    payload = base64.b64encode(storage.get_bytes(asset.key)).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _measure_image(data: bytes) -> tuple[int | None, int | None, str | None]:
    """Read an image's real dimensions and encoding out of its own bytes.

    The first live call is what made this necessary. Two things the adapter had
    been believing turned out to be false:

    * Seedream's response carries **no** ``width`` or ``height`` field at all, so
      ``images[0].get("width")`` was silently ``None`` and the asset record said
      nothing about the image's size.
    * It returned **JPEG** bytes for a request whose ``output_key`` ends in
      ``.png``, so the recorded mime type was a guess that happened to be wrong.

    Both would have gone into the delivery manifest as fact. The video path
    already reads geometry from the file with :func:`adml.video.probe` rather than
    trusting a field; this makes the image path consistent with it.

    Best-effort by design. A generation that has already been *paid for* must not
    be thrown away because it arrived in an encoding Pillow does not know — the
    bytes are on disk either way, and an unmeasured asset is recoverable while a
    discarded one is not.
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.width, im.height, _MIME_BY_FORMAT.get((im.format or "").upper())
    except (UnidentifiedImageError, OSError, ValueError):
        return None, None, None


def _image_size(aspect: AspectRatio, long_edge: int = 1536) -> dict[str, int]:
    """Explicit pixel dimensions for the target aspect ratio.

    Passed as width/height rather than one of fal's named presets, because the
    presets do not include 9:16 and the platform geometry is not negotiable —
    a Reels ad that comes back 4:3 is a wasted generation.

    What the first live call established: Seedream treats this as the *shape* to
    match, not the size to produce. A request for 864x1536 came back 1920x3416 —
    the aspect ratio honoured to within 0.07%, the resolution its own choice.
    That is the useful half. Downstream only needs the ratio (delivery crops to
    platform safe areas from whatever it is given), so this asks for the ratio and
    :func:`_measure_image` records what actually arrived rather than what was asked
    for. Do not treat the returned size as predictable.
    """
    width, height = aspect.pixel_size(long_edge)
    return {"width": width, "height": height}


class FalImageProvider(ImageProvider):
    """Multi-reference composition via Seedream 4.5 edit."""

    name = "fal"
    model = "seedream-4.5-edit"

    def __init__(
        self,
        storage: Storage,
        api_key: str | None = None,
        client: FalClient | None = None,
    ):
        self.storage = storage
        self.client = client or FalClient(api_key)

    def _payload(self, request: ImageGenRequest) -> dict[str, Any]:
        references = request.references[: self.max_reference_images]
        if len(request.references) > self.max_reference_images:
            raise ProviderError(
                f"{self.model} accepts {self.max_reference_images} reference images, "
                f"got {len(request.references)}"
            )
        return {
            "prompt": request.brief.image_prompt,
            "image_urls": [as_data_uri(self.storage, ref) for ref in references],
            "image_size": _image_size(request.aspect_ratio),
            "num_images": 1,
            "seed": request.seed,
            "enable_safety_checker": True,
        }

    async def generate(self, request: ImageGenRequest) -> ImageGenResult:
        started = time.monotonic()
        payload = self._payload(request)
        raw = await self.client.run(SEEDREAM_EDIT_ENDPOINT, payload, IMAGE_TIMEOUT_S)

        images = raw.get("images") or []
        if not images or not images[0].get("url"):
            raise ProviderError(f"{self.model} returned no image: {raw}")

        data = await self.client.fetch_bytes(images[0]["url"])
        # Measured, not reported. See `_measure_image` for what the first live
        # call found wrong with the reported values.
        width, height, mime = _measure_image(data)
        asset = self.storage.put_bytes(
            request.output_key,
            data,
            mime or images[0].get("content_type") or "image/png",
        )
        asset.width, asset.height = width, height

        return ImageGenResult(
            asset=asset,
            model=self.model,
            tier=Tier(IMAGE_PRICES[self.model].tier),
            cost_usd=self.estimate_cost(1),
            latency_ms=round((time.monotonic() - started) * 1000),
            seed=raw.get("seed", request.seed),
            raw=raw,
        )


class FalVideoProvider(VideoProvider):
    """Image-to-video via Kling 2.5 Turbo Pro."""

    name = "fal"
    model = "kling-2.5-turbo-pro"

    def __init__(
        self,
        storage: Storage,
        api_key: str | None = None,
        client: FalClient | None = None,
    ):
        self.storage = storage
        self.client = client or FalClient(api_key)

    def _payload(self, request: VideoGenRequest) -> dict[str, Any]:
        duration = self.deliverable_duration(request.duration_seconds)
        return {
            "prompt": request.brief.motion_prompt,
            "image_url": as_data_uri(self.storage, request.start_image),
            # The API takes duration as a *string* enum, not a number.
            "duration": str(int(duration)),
            # The video negatives, not the image ones. This used to send
            # `brief.negative_prompt` — a list of image artefacts that never
            # mentions motion — so nothing in the payload ever discouraged the
            # model from simply panning across the still it was given.
            "negative_prompt": (
                request.brief.video_negative_prompt
                or request.brief.negative_prompt
                or "blur, distort, and low quality"
            ),
            "cfg_scale": KLING_CFG_SCALE,
        }

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        started = time.monotonic()
        delivered = self.deliverable_duration(request.duration_seconds)
        payload = self._payload(request)
        raw = await self.client.run(KLING_I2V_ENDPOINT, payload, VIDEO_TIMEOUT_S)

        video = raw.get("video") or {}
        if not video.get("url"):
            raise ProviderError(f"{self.model} returned no video: {raw}")

        data = await self.client.fetch_bytes(video["url"])
        asset = self.storage.put_bytes(
            request.output_key, data, video.get("content_type", "video/mp4")
        )

        # The duration this adapter used to report was `delivered` — the enum value
        # it *asked* for. That is a claim about someone else's API, and the project
        # guarantees every delivered clip is 8-10 s. A guarantee checked against the
        # request rather than the file is not checked at all, so read it off the
        # bytes and let the pipeline's duration gate see the truth. Falls back to the
        # requested value only when the container will not give one up.
        measured = delivered
        try:
            probe = V.probe(data)
            if probe.duration_measured and probe.duration_seconds > 0:
                measured = probe.duration_seconds
        except V.ClipDecodeError:
            pass

        return VideoGenResult(
            asset=asset,
            model=self.model,
            tier=Tier(VIDEO_PRICES[self.model].tier),
            cost_usd=self.estimate_cost(request.duration_seconds),
            latency_ms=round((time.monotonic() - started) * 1000),
            seed=request.seed,
            # Measured off the file, not requested — a 9 s ask comes back 10 s, and
            # whether it really does is a question only the bytes can answer.
            duration_seconds=measured,
            fps=request.fps,
            was_chained=False,
            seed_honoured=self.honours_seed,
            raw=raw,
        )
