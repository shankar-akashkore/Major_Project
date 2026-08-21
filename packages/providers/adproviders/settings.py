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
    #: How many generations may be in flight at once.
    #:
    #: The pipeline used to run its slots one after another, and the first live job
    #: measured what that costs: three Seedream edits at ~50 s each took 152 s of
    #: wall clock to produce 151 s of generation. Nothing was overlapping. At five
    #: candidates it is four minutes of a user watching a progress bar for no
    #: reason — the provider is a queue, and waiting for one's turn to wait is not
    #: a constraint, it is an omission.
    #:
    #: Five rather than unlimited, and five rather than four, because five is the
    #: default candidate count: a limit of four would put the fifth image in a wave
    #: of its own and charge a whole extra generation's latency for it. Bounded at
    #: all because fan-out is limited by what a rate limiter tolerates rather than
    #: by what asyncio will start — a 429 lands on a slot that has already cleared
    #: the budget check. `FalClient` waits one out rather than failing the slot, so
    #: the cost of guessing this too high is time, not a lost candidate. Lower it
    #: if fal starts refusing submissions.
    max_concurrent_generations: int = Field(default=5, ge=1, le=8)

    #: Image-prompt template. An ablation axis, kept here rather than on the
    #: request because the person submitting a job has no view on prompt
    #: verbosity. Default stays `full` until the compact template has been
    #: measured against a real generator rather than against assertions.
    prompt_style: str = Field(default="full", description="'full' | 'compact'")

    # --- Persistence ---
    database_url: str = "sqlite+aiosqlite:///./adgen.db"

    # --- Storage ---
    storage_backend: str = Field(default="local", description="'local' | 'supabase'")
    storage_root: Path = Field(default=Path("./fixtures"))

    # --- Soundtrack ---
    #: Where licensed music lives. Each track needs a JSON sidecar of the same
    #: stem recording its title, source and licence; a file without one is
    #: skipped rather than shipped with the provenance left blank. Empty or
    #: absent is the normal state, and delivery then synthesises a bed scored to
    #: the job's mood — see ``adml.audio.choose_bed``.
    audio_library: Path | None = Field(
        default=None,
        description="Directory of licensed music beds. Defaults to <storage_root>/audio.",
    )

    # --- Provider selection (names resolved by the registry) ---
    image_provider: str = "mock"
    video_provider: str = "mock"
    llm_provider: str = "mock"

    # --- Research tier ---
    #: Where the Colab notebook's clip output was unzipped. Read by
    #: `PrerenderedVideoProvider`, which serves free clips generated off-machine.
    #: Defaults under the storage root so the usual case needs no configuration.
    clip_manifest: Path | None = Field(
        default=None,
        description="Directory holding clips.json from notebooks/colab_video.ipynb. "
        "Defaults to <storage_root>/research.",
    )

    # --- Credentials ---
    # One key covers both live providers: fal hosts Seedream for multi-reference
    # composition and Kling for image-to-video, so there is one account to fund
    # and one statement to reconcile against the ledger.
    fal_api_key: str = Field(default="", description="fal.ai API key. Blank keeps live mode off.")

    # --- Golden demo set ---
    #: Which frozen bundle ``AD_PROVIDER_MODE=replay`` serves. A slug, matching a
    #: directory under ``<storage_root>/golden/``. The viva runs this way: no
    #: network, no credential, no spend, and the exact candidates that were
    #: generated when the budget was still available.
    golden_set: str | None = Field(
        default=None,
        description="Slug of the golden bundle to replay. Required when AD_PROVIDER_MODE=replay.",
    )

    @property
    def clip_manifest_dir(self) -> Path:
        return self.clip_manifest or (self.storage_root / "research")

    @property
    def audio_root(self) -> Path:
        return self.audio_library or (self.storage_root / "audio")

    @property
    def golden_root(self) -> Path:
        return self.storage_root

    @property
    def is_live(self) -> bool:
        return self.provider_mode is ProviderMode.LIVE

    @property
    def is_replay(self) -> bool:
        return self.provider_mode is ProviderMode.REPLAY

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
        if self.is_replay:
            return (
                f"📼 REPLAY MODE — serving frozen golden set "
                f"{self.golden_set or '(none configured)'}. No calls, no spend."
            )
        return "✅ MOCK MODE — no external calls, no spend."

    @property
    def has_fal_key(self) -> bool:
        return bool(self.fal_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
