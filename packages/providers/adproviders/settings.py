"""Runtime configuration.

The single most important setting here is ``provider_mode``.  It defaults to
``mock``, and flipping it to ``live`` is the only way this project can spend
money.  Everything else — caps, caching, retry limits — exists to make sure
that even in live mode a bug cannot run away with the budget.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from adschema import ProviderMode
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- The money switch ---
    provider_mode: ProviderMode = Field(
        default=ProviderMode.MOCK,
        description="mock = free and deterministic; live = real API calls.",
    )

    # --- Cost governor ---
    budget_total_usd: float = Field(default=35.00, ge=0.0)
    #: A verified default job is $2.22 (see docs/provider-spike.md), so the cap
    #: sits just above it rather than at a round number — close enough to catch a
    #: runaway before it has cost a whole extra job.
    budget_per_job_usd: float = Field(default=2.50, ge=0.0)
    max_retries_per_slot: int = Field(default=1, ge=0, le=3)

    # --- Persistence ---
    database_url: str = "sqlite+aiosqlite:///./adgen.db"

    # --- Storage ---
    storage_backend: str = Field(default="local", description="'local' | 'supabase'")
    storage_root: Path = Field(default=Path("./fixtures"))

    # --- Provider selection (names resolved by the registry) ---
    image_provider: str = "mock"
    video_provider: str = "mock"
    llm_provider: str = "mock"

    # --- Credentials ---
    # One key covers both live providers: fal hosts Seedream for multi-reference
    # composition and Kling for image-to-video, so there is one account to fund
    # and one statement to reconcile against the ledger.
    fal_api_key: str = Field(default="", description="fal.ai API key. Blank keeps live mode off.")

    @property
    def is_live(self) -> bool:
        return self.provider_mode is ProviderMode.LIVE

    def describe(self) -> str:
        """One-line banner, printed on worker startup.

        Making the mode loudly visible is deliberate: the failure we most want
        to avoid is spending real money while believing we are in mock mode.
        """
        if self.is_live:
            return (
                f"⚠️  LIVE MODE — real spend enabled. "
                f"budget=${self.budget_total_usd:.2f} per_job=${self.budget_per_job_usd:.2f} "
                f"image={self.image_provider} video={self.video_provider}"
            )
        return "✅ MOCK MODE — no external calls, no spend."

    @property
    def has_fal_key(self) -> bool:
        return bool(self.fal_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
