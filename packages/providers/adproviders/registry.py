"""Provider resolution.

The registry is the one place that maps a configuration string to a concrete
provider.  Live adapters are registered here as they land (week 5 for images,
week 12 for premium video); until then a request for a live provider fails with
an explicit message rather than silently falling back to mock — a silent
fallback would let a "live" experiment quietly produce mock data.
"""

from __future__ import annotations

from adschema import ProviderMode

from .base import ImageProvider, LLMProvider, ProviderUnavailable, VideoProvider
from .fal import FalImageProvider, FalVideoProvider
from .mock import MockImageProvider, MockLLMProvider, MockVideoProvider
from .prerendered import PrerenderedVideoProvider
from .settings import Settings, get_settings
from .storage import Storage, get_storage

_IMAGE_BUILDERS: dict[str, type[ImageProvider]] = {
    "mock": MockImageProvider,
    "seedream-4.5-edit": FalImageProvider,
}
_VIDEO_BUILDERS: dict[str, type[VideoProvider]] = {
    "mock": MockVideoProvider,
    "kling-2.5-turbo-pro": FalVideoProvider,
    # The research tier. Not a hosted endpoint: clips are generated on a Colab
    # GPU and served from a manifest, which is why one name covers both open-weights
    # models — the manifest records per clip which one produced it.
    "prerendered": PrerenderedVideoProvider,
    "ltx-video": PrerenderedVideoProvider,
    "wan-2.1-i2v": PrerenderedVideoProvider,
}
_LLM_BUILDERS: dict[str, type[LLMProvider]] = {"mock": MockLLMProvider}

#: Live adapters that are planned but not yet implemented. Naming them lets the
#: error message say "not wired yet" instead of "unknown provider".
_PLANNED_IMAGE = {"gemini-flash-image", "seedream-4", "flux-2-pro", "flux-kontext-dev"}
_PLANNED_VIDEO = {"kling-3-pro", "seedance-2.0", "veo-3.1-fast"}
_PLANNED_LLM = {"claude-haiku", "claude-sonnet", "gemini-flash"}

#: Providers that need a fal key, so the registry can fail with the actual reason
#: rather than letting the first generation blow up mid-job.
_NEEDS_FAL_KEY = {FalImageProvider, FalVideoProvider}
#: ...and the one that needs a clip manifest instead of a credential.
_NEEDS_CLIP_MANIFEST = {PrerenderedVideoProvider}


def _unavailable(kind: str, name: str, planned: set[str]) -> ProviderUnavailable:
    if name in planned:
        return ProviderUnavailable(
            f"{kind} provider {name!r} is planned but not implemented yet. "
            f"Keep AD_{kind.upper()}_PROVIDER=mock until its adapter lands."
        )
    return ProviderUnavailable(f"unknown {kind} provider {name!r}; known: {sorted(_known(kind))}")


def _known(kind: str) -> set[str]:
    return {
        "image": set(_IMAGE_BUILDERS),
        "video": set(_VIDEO_BUILDERS),
        "llm": set(_LLM_BUILDERS),
    }[kind]


def _build(builder: type, settings: Settings, storage: Storage):
    """Instantiate a provider, passing credentials only to the ones that need them."""
    if builder in _NEEDS_FAL_KEY:
        if not settings.has_fal_key:
            raise ProviderUnavailable(
                f"{builder.__name__} needs AD_FAL_API_KEY, which is not set. "
                "Add it to .env, or keep AD_PROVIDER_MODE=mock to run for free."
            )
        return builder(storage, settings.fal_api_key)
    if builder in _NEEDS_CLIP_MANIFEST:
        # The manifest is not read here — it is loaded lazily on first use, so a
        # worker can start with the research tier selected before the notebook's
        # output has been copied in.
        return builder(storage, settings.clip_manifest_dir)
    return builder(storage)


def get_image_provider(
    settings: Settings | None = None, storage: Storage | None = None
) -> ImageProvider:
    settings = settings or get_settings()
    name = "mock" if settings.provider_mode is ProviderMode.MOCK else settings.image_provider
    builder = _IMAGE_BUILDERS.get(name)
    if builder is None:
        raise _unavailable("image", name, _PLANNED_IMAGE)
    storage = storage or get_storage(settings.storage_backend, settings.storage_root)
    return _build(builder, settings, storage)


#: Video providers that mock mode permits, because they cannot spend.
#:
#: Mock mode's guarantee is "no spend", not "no provider" — and the research tier
#: serves clips that were generated for free on someone else's GPU. Requiring
#: ``AD_PROVIDER_MODE=live`` to use them would mean flipping the money switch in
#: order to run the *free* corpus, which also arms the paid image provider. The
#: price table is the check that keeps this honest: every name here is $0/second.
_FREE_VIDEO_PROVIDERS = frozenset({"prerendered", "ltx-video", "wan-2.1-i2v"})


def get_video_provider(
    settings: Settings | None = None, storage: Storage | None = None
) -> VideoProvider:
    settings = settings or get_settings()
    requested = settings.video_provider
    if settings.provider_mode is ProviderMode.MOCK and requested not in _FREE_VIDEO_PROVIDERS:
        requested = "mock"
    name = requested
    builder = _VIDEO_BUILDERS.get(name)
    if builder is None:
        raise _unavailable("video", name, _PLANNED_VIDEO)
    storage = storage or get_storage(settings.storage_backend, settings.storage_root)
    return _build(builder, settings, storage)


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    name = "mock" if settings.provider_mode is ProviderMode.MOCK else settings.llm_provider
    builder = _LLM_BUILDERS.get(name)
    if builder is None:
        raise _unavailable("llm", name, _PLANNED_LLM)
    return builder()  # type: ignore[call-arg]
