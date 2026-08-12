"""Generation providers, cost governance and media storage.

Import surface is deliberately narrow: the pipeline talks to providers through
the ABCs in :mod:`adproviders.base` and never to a vendor SDK directly.
"""

from .base import (
    ImageGenRequest,
    ImageGenResult,
    ImageProvider,
    LLMProvider,
    ProviderError,
    ProviderUnavailable,
    VideoGenRequest,
    VideoGenResult,
    VideoProvider,
)
from .fal import FalClient, FalImageProvider, FalVideoProvider
from .golden import (
    GOLDEN_VERSION,
    FrozenAsset,
    FrozenClip,
    FrozenFrame,
    GoldenBundle,
    GoldenDrift,
    GoldenError,
    GoldenExpectation,
    GoldenImageProvider,
    GoldenLLMProvider,
    GoldenMiss,
    GoldenRecorder,
    GoldenSession,
    GoldenVideoProvider,
    RecordingImageProvider,
    RecordingLLMProvider,
    RecordingVideoProvider,
    build_golden_bundle,
    compare_replay,
    list_bundles,
)
from .governor import BudgetExceeded, CostGovernor, RetryBudgetExceeded
from .ledger import InMemoryLedger, LedgerStore, SqlLedger
from .mock import (
    MockImageProvider,
    MockLLMProvider,
    MockVideoProvider,
    mock_portrait_asset,
    mock_product_asset,
    mock_reference_asset,
)
from .prerendered import (
    ClipEntry,
    ClipManifest,
    PrerenderedVideoProvider,
    clip_fingerprint,
)
from .pricing import (
    IMAGE_PRICES,
    LLM_PRICES,
    VIDEO_PRICES,
    estimate_image_cost,
    estimate_job_cost,
    estimate_llm_cost,
    estimate_video_cost,
)
from .registry import (
    ProviderSet,
    get_image_provider,
    get_llm_provider,
    get_providers,
    get_video_provider,
    golden_bundle_path,
)
from .settings import Settings, get_settings
from .storage import LocalStorage, Storage, get_storage

__all__ = [
    # interfaces
    "ImageProvider",
    "VideoProvider",
    "LLMProvider",
    "ImageGenRequest",
    "ImageGenResult",
    "VideoGenRequest",
    "VideoGenResult",
    "ProviderError",
    "ProviderUnavailable",
    # governance
    "CostGovernor",
    "BudgetExceeded",
    "RetryBudgetExceeded",
    "LedgerStore",
    "InMemoryLedger",
    "SqlLedger",
    # pricing
    "IMAGE_PRICES",
    "VIDEO_PRICES",
    "LLM_PRICES",
    "estimate_image_cost",
    "estimate_video_cost",
    "estimate_llm_cost",
    "estimate_job_cost",
    # mock
    "MockImageProvider",
    "MockVideoProvider",
    "MockLLMProvider",
    "mock_reference_asset",
    "mock_portrait_asset",
    "mock_product_asset",
    "FalClient",
    "FalImageProvider",
    "FalVideoProvider",
    # research tier
    "PrerenderedVideoProvider",
    "ClipManifest",
    "ClipEntry",
    "clip_fingerprint",
    # golden demo set
    "GOLDEN_VERSION",
    "GoldenBundle",
    "GoldenExpectation",
    "GoldenDrift",
    "GoldenError",
    "GoldenMiss",
    "GoldenRecorder",
    "GoldenSession",
    "GoldenImageProvider",
    "GoldenVideoProvider",
    "GoldenLLMProvider",
    "RecordingImageProvider",
    "RecordingVideoProvider",
    "RecordingLLMProvider",
    "FrozenAsset",
    "FrozenFrame",
    "FrozenClip",
    "build_golden_bundle",
    "compare_replay",
    "list_bundles",
    # wiring
    "ProviderSet",
    "get_providers",
    "get_image_provider",
    "get_video_provider",
    "get_llm_provider",
    "golden_bundle_path",
    "Settings",
    "get_settings",
    "Storage",
    "LocalStorage",
    "get_storage",
]
