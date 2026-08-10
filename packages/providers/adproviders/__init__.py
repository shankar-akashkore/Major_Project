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
from .governor import BudgetExceeded, CostGovernor, RetryBudgetExceeded
from .ledger import InMemoryLedger, LedgerStore, SqlLedger
from .mock import MockImageProvider, MockLLMProvider, MockVideoProvider, mock_reference_asset
from .pricing import (
    IMAGE_PRICES,
    LLM_PRICES,
    VIDEO_PRICES,
    estimate_image_cost,
    estimate_job_cost,
    estimate_llm_cost,
    estimate_video_cost,
)
from .registry import get_image_provider, get_llm_provider, get_video_provider
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
    # wiring
    "get_image_provider",
    "get_video_provider",
    "get_llm_provider",
    "Settings",
    "get_settings",
    "Storage",
    "LocalStorage",
    "get_storage",
]
