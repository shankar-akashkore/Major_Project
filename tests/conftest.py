"""Shared fixtures.

Every test runs against the mock providers and an isolated temporary storage root,
so the suite cannot touch the network or the real spend ledger.
"""

from __future__ import annotations

import adproviders as P
import pytest
from adschema import (
    AdJobRequest,
    AspectRatio,
    ConsentAttestation,
    Mood,
    Platform,
    ThemeSpec,
    Vertical,
)
from adworker import Pipeline

BRAND_PALETTE = ["#2b3a55", "#ce7777", "#f2e7d5"]


@pytest.fixture
def storage(tmp_path) -> P.LocalStorage:
    return P.LocalStorage(tmp_path)


@pytest.fixture
def settings() -> P.Settings:
    """Mock mode, so no test can spend money even if a live adapter regresses."""
    return P.Settings(provider_mode="mock", storage_backend="local")


@pytest.fixture
def ledger() -> P.InMemoryLedger:
    return P.InMemoryLedger()


@pytest.fixture
def governor(ledger, settings) -> P.CostGovernor:
    return P.CostGovernor(ledger, settings)


@pytest.fixture
def references(storage) -> tuple:
    human = P.mock_reference_asset(
        storage, "uploads/t/human.png", AspectRatio.PORTRAIT_4_5, seed=11
    )
    product = P.mock_reference_asset(
        storage, "uploads/t/product.png", AspectRatio.SQUARE_1_1, seed=22
    )
    return human, product


@pytest.fixture
def make_request(references):
    """Factory so individual tests can vary one field without repeating the rest."""
    human, product = references

    def _make(**overrides) -> AdJobRequest:
        payload = {
            "human_model_image": human,
            "product_image": product,
            "product_name": "Aurora Serum",
            "caption": "Glow that lasts",
            "cta_text": "Shop now",
            "vertical": Vertical.BEAUTY,
            "platform": Platform.INSTAGRAM_REELS,
            "mood": Mood.CALM_PREMIUM,
            "theme": ThemeSpec(palette=list(BRAND_PALETTE)),
            "duration_seconds": 9.0,
            "candidate_count": 3,
            "seed": 7,
            "consent": ConsentAttestation(has_model_release=True, not_a_public_figure=True),
        }
        payload.update(overrides)
        return AdJobRequest(**payload)

    return _make


@pytest.fixture
def pipeline(storage, governor) -> Pipeline:
    return Pipeline(
        storage=storage,
        governor=governor,
        image_provider=P.MockImageProvider(storage),
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
    )
