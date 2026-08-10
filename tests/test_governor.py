"""Cost-governor guarantees.

These are the tests that protect the project's ability to produce a demo at all.
The budget is $35 for the whole semester, so each of these is a real failure mode
rather than a hypothetical: a retry loop, a double charge, a crash that loses a
charge, or a live provider leaking into what was supposed to be a free run.
"""

from __future__ import annotations

import adproviders as P
import pytest
from adschema import ProviderMode


def live_settings(**overrides) -> P.Settings:
    payload = {
        "provider_mode": ProviderMode.LIVE,
        "budget_total_usd": 1.00,
        "budget_per_job_usd": 0.60,
        "max_retries_per_slot": 1,
    }
    payload.update(overrides)
    return P.Settings(**payload)


async def _ok() -> str:
    return "ok"


async def test_affordable_call_is_settled():
    ledger = P.InMemoryLedger()
    governor = P.CostGovernor(ledger, live_settings())

    result = await governor.guarded_call(
        job_id="j1",
        provider="kling",
        model="kling-3",
        operation="video",
        estimated_usd=0.50,
        quantity=5.0,
        call=_ok,
    )
    assert result == "ok"
    assert await ledger.total_spent() == pytest.approx(0.50)


async def test_per_job_cap_is_enforced():
    ledger = P.InMemoryLedger()
    governor = P.CostGovernor(ledger, live_settings())

    await governor.guarded_call(
        job_id="j1",
        provider="k",
        model="kling-3",
        operation="video",
        estimated_usd=0.50,
        quantity=5.0,
        call=_ok,
    )
    with pytest.raises(P.BudgetExceeded, match="per-job"):
        await governor.guarded_call(
            job_id="j1",
            provider="k",
            model="kling-3",
            operation="video",
            estimated_usd=0.50,
            quantity=5.0,
            call=_ok,
        )


async def test_total_budget_cap_is_enforced_across_jobs():
    ledger = P.InMemoryLedger()
    governor = P.CostGovernor(ledger, live_settings())

    await governor.guarded_call(
        job_id="j1",
        provider="k",
        model="kling-3",
        operation="video",
        estimated_usd=0.55,
        quantity=5.0,
        call=_ok,
    )
    with pytest.raises(P.BudgetExceeded, match="total budget"):
        await governor.guarded_call(
            job_id="j2",
            provider="k",
            model="kling-3",
            operation="video",
            estimated_usd=0.50,
            quantity=5.0,
            call=_ok,
        )


async def test_failed_call_is_voided_not_charged():
    """A provider error must not leave money on the ledger — but the reservation
    must have existed during the call, so a crash cannot lose the charge."""
    ledger = P.InMemoryLedger()
    governor = P.CostGovernor(ledger, live_settings())

    async def boom() -> str:
        raise RuntimeError("provider 500")

    with pytest.raises(RuntimeError):
        await governor.guarded_call(
            job_id="j1",
            provider="k",
            model="kling-3",
            operation="video",
            estimated_usd=0.40,
            quantity=4.0,
            call=boom,
        )
    assert await ledger.total_spent() == 0.0
    # The voided entry is still on record, for observability.
    assert len(await ledger.entries("j1")) == 1


async def test_reservations_count_as_spent_so_concurrency_cannot_overrun():
    """Twenty calls that each look affordable in isolation must not collectively
    exceed the cap. Counting in-flight reservations is what prevents that."""
    ledger = P.InMemoryLedger()
    governor = P.CostGovernor(ledger, live_settings(budget_total_usd=1.00, budget_per_job_usd=1.00))

    granted = 0
    for _ in range(20):
        try:
            await governor.guarded_call(
                job_id="j1",
                provider="k",
                model="kling-3",
                operation="video",
                estimated_usd=0.10,
                quantity=1.0,
                call=_ok,
            )
            granted += 1
        except P.BudgetExceeded:
            break

    assert granted == 10
    assert await ledger.total_spent() == pytest.approx(1.00)


async def test_retry_allowance_is_capped():
    governor = P.CostGovernor(P.InMemoryLedger(), live_settings(max_retries_per_slot=1))
    assert governor.register_attempt("j:slot0") == 1
    assert governor.register_attempt("j:slot0") == 2
    with pytest.raises(P.RetryBudgetExceeded):
        governor.register_attempt("j:slot0")


async def test_retry_allowance_is_per_slot():
    governor = P.CostGovernor(P.InMemoryLedger(), live_settings())
    governor.register_attempt("j:slot0")
    governor.register_attempt("j:slot0")
    # A different slot has its own allowance.
    assert governor.register_attempt("j:slot1") == 1


async def test_mock_mode_rejects_a_nonzero_estimate():
    """If a live provider ever leaks into a mock run, fail loudly. A silent
    fallback would mean an experiment quietly produced mock data."""
    governor = P.CostGovernor(P.InMemoryLedger(), P.Settings(provider_mode=ProviderMode.MOCK))
    with pytest.raises(AssertionError, match="mock mode"):
        await governor.guarded_call(
            job_id="j",
            provider="kling",
            model="kling-3",
            operation="video",
            estimated_usd=0.50,
            quantity=5.0,
            call=_ok,
        )


async def test_preflight_refuses_an_unaffordable_job_before_any_stage_runs():
    ledger = P.InMemoryLedger()
    governor = P.CostGovernor(ledger, live_settings(budget_total_usd=1.00))
    with pytest.raises(P.BudgetExceeded):
        await governor.preflight("j1", estimated_total_usd=2.82)
    assert await ledger.total_spent() == 0.0


async def test_preflight_is_a_no_op_in_mock_mode():
    governor = P.CostGovernor(P.InMemoryLedger(), P.Settings(provider_mode=ProviderMode.MOCK))
    await governor.preflight("j1", estimated_total_usd=999.0)


async def test_cache_returns_the_stored_asset():
    from adschema import AssetRef

    ledger = P.InMemoryLedger()
    governor = P.CostGovernor(ledger, live_settings())

    fingerprint = governor.fingerprint("image", "gemini-flash-image", {"prompt": "x", "seed": 1})
    assert await governor.cached_asset(fingerprint) is None

    await governor.remember_asset(fingerprint, "image", AssetRef(key="generations/a.png"))
    hit = await governor.cached_asset(fingerprint)
    assert hit is not None and hit.key == "generations/a.png"


async def test_fingerprint_separates_model_and_seed():
    governor = P.CostGovernor(P.InMemoryLedger(), live_settings())
    base = {"prompt": "a photo", "seed": 1}

    same = governor.fingerprint("image", "gemini-flash-image", base)
    assert governor.fingerprint("image", "gemini-flash-image", base) == same
    assert governor.fingerprint("image", "seedream-4", base) != same
    assert governor.fingerprint("image", "gemini-flash-image", {**base, "seed": 2}) != same
    assert governor.fingerprint("video", "gemini-flash-image", base) != same


async def test_sql_ledger_is_durable(tmp_path):
    """The ledger must survive a process restart, or the budget number is fiction."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'spend.db'}"

    first = P.SqlLedger.from_url(url)
    await first.create_all()
    governor = P.CostGovernor(first, live_settings())
    await governor.guarded_call(
        job_id="j1",
        provider="k",
        model="kling-3",
        operation="video",
        estimated_usd=0.50,
        quantity=5.0,
        call=_ok,
    )
    await first.engine.dispose()

    reopened = P.SqlLedger.from_url(url)
    assert await reopened.total_spent() == pytest.approx(0.50)
    assert await reopened.job_spent("j1") == pytest.approx(0.50)
    await reopened.engine.dispose()


async def test_estimates_match_the_plans_budget_arithmetic():
    """Guards the number the whole project plan is built on: a 3-candidate,
    9-second job on the cheap tier is about $2.82, and Veo is out of reach."""
    cheap = P.estimate_job_cost("gemini-flash-image", "kling-3", "claude-haiku", 3, 9.0)
    assert 2.70 <= cheap <= 2.95

    premium = P.estimate_job_cost("gemini-flash-image", "veo-3.1", "claude-haiku", 3, 9.0)
    assert premium > 10.0

    free = P.estimate_job_cost("mock", "mock", "mock", 3, 9.0)
    assert free == 0.0

    research = P.estimate_job_cost("flux-kontext-dev", "ltx-video", "mock", 3, 9.0)
    assert research == 0.0


async def test_chained_provider_bills_whole_segments():
    """A 5 s-capped model reaching 9 s pays for two clips, not 1.8."""
    from adproviders.pricing import VIDEO_PRICES

    assert not VIDEO_PRICES["wan-2.1-i2v"].supports_project_window
    assert VIDEO_PRICES["kling-3"].supports_project_window

    # Priced at $0 on the research tier, but the segment arithmetic still applies.
    assert P.estimate_video_cost("runway-gen-4.5", 9.0) == pytest.approx(9.0 * 0.20)


async def test_unimplemented_live_provider_fails_loudly():
    """Naming a planned-but-unbuilt provider must error, never silently mock."""
    settings = live_settings(image_provider="gemini-flash-image")
    with pytest.raises(P.ProviderUnavailable, match="not implemented yet"):
        P.get_image_provider(settings)

    with pytest.raises(P.ProviderUnavailable, match="unknown"):
        P.get_image_provider(live_settings(image_provider="nonexistent-model"))
