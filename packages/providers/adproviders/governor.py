"""The cost governor.

Every paid call is wrapped by this class.  With a $35 total budget, an
ungoverned retry loop is not a theoretical risk — it is the single most likely
way this project loses the ability to produce its demo.  So the governor is
deliberately strict:

* It **refuses loudly** rather than degrading silently.  A refusal raises
  :class:`BudgetExceeded`; it never quietly returns a lower-quality result,
  because a silent downgrade is a bug you discover in the write-up.
* It **checks before every call**, not once per job, and counts in-flight
  reservations as spent.
* It **caps retries per slot**, so a candidate that keeps failing the quality
  gate costs at most two generations.
* It **serves the cache first**, so an identical resubmission is free.
* In mock mode it charges nothing and asserts the estimate was zero — which is
  what makes "the full pipeline runs at exactly $0" a testable claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import TypeVar

from adschema import AssetRef, BudgetStatus, SpendEntry

from .ledger import LedgerStore
from .settings import Settings

T = TypeVar("T")


class BudgetExceeded(RuntimeError):
    """A call was refused because it would push spend past a cap.

    Carries the numbers so the refusal message is actionable rather than just
    "over budget".
    """

    def __init__(self, requested_usd: float, status: BudgetStatus, scope: str):
        self.requested_usd = requested_usd
        self.status = status
        self.scope = scope
        super().__init__(
            f"refused: {scope} cap would be exceeded — "
            f"call costs ${requested_usd:.4f}, "
            f"${status.remaining_usd:.4f} of ${status.total_budget_usd:.2f} remains "
            f"(spent ${status.spent_usd:.4f})"
        )


class RetryBudgetExceeded(RuntimeError):
    """A candidate slot exhausted its retry allowance."""


class CostGovernor:
    """Gatekeeper for every call that could cost money."""

    def __init__(self, ledger: LedgerStore, settings: Settings, *, use_cache: bool = True):
        self.ledger = ledger
        self.settings = settings
        #: Whether to consult the generation cache at all.
        #:
        #: On by default and it should stay that way — serving a repeated request
        #: for free is what stops a re-run costing $2.22 again. It is off for a
        #: golden replay, and that turned out to matter: the cache is checked
        #: *before* the provider, and the frozen uploads hash to what they hashed
        #: to at freeze time, so every fingerprint hits and the replay serves
        #: nothing from the bundle at all. The first replay written here reported
        #: a clean pass having read zero frozen bytes — correct output produced by
        #: local database state, on a code path that would not exist on a fresh
        #: machine. Which bytes a replay serves must not depend on what happens to
        #: be in the ledger.
        self.use_cache = use_cache
        self._attempts: dict[str, int] = {}

    # --- Budget introspection ------------------------------------------------

    async def status(self) -> BudgetStatus:
        return BudgetStatus(
            total_budget_usd=self.settings.budget_total_usd,
            spent_usd=await self.ledger.total_spent(),
            per_job_cap_usd=self.settings.budget_per_job_usd,
        )

    async def preflight(self, job_id: str, estimated_total_usd: float) -> None:
        """Refuse an unaffordable job *before* stage 1 runs.

        Failing up front matters: a job that dies at the video stage has already
        paid for images, and that money buys nothing.
        """
        if not self.settings.is_live:
            return
        status = await self.status()
        if not status.can_afford(estimated_total_usd):
            raise BudgetExceeded(estimated_total_usd, status, scope="total budget")
        if estimated_total_usd > self.settings.budget_per_job_usd:
            raise BudgetExceeded(estimated_total_usd, status, scope="per-job")

    # --- Retry accounting ---------------------------------------------------

    def attempt_number(self, slot_key: str) -> int:
        return self._attempts.get(slot_key, 0) + 1

    def register_attempt(self, slot_key: str) -> int:
        """Claim one attempt for a slot, or refuse if the allowance is spent."""
        used = self._attempts.get(slot_key, 0)
        allowed = 1 + self.settings.max_retries_per_slot
        if used >= allowed:
            raise RetryBudgetExceeded(
                f"slot {slot_key!r} exhausted its {allowed} attempt(s); "
                "rejecting rather than paying for another generation"
            )
        self._attempts[slot_key] = used + 1
        return used + 1

    def reset_attempts(self, prefix: str = "") -> None:
        if not prefix:
            self._attempts.clear()
            return
        for key in [k for k in self._attempts if k.startswith(prefix)]:
            del self._attempts[key]

    # --- The wrapper every paid call goes through ---------------------------

    @staticmethod
    def fingerprint(operation: str, model: str, payload: dict) -> str:
        """Cache key for one generation call.

        Includes the model, because the same prompt on a different model is a
        different result, and the seed, because reproducibility is what makes
        the ablations comparable.
        """
        blob = json.dumps(
            {"op": operation, "model": model, **payload}, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:48]

    async def guarded_call(
        self,
        *,
        job_id: str,
        provider: str,
        model: str,
        operation: str,
        estimated_usd: float,
        quantity: float,
        call: Callable[[], Awaitable[T]],
        actual_cost_of: Callable[[T], float] | None = None,
        note: str = "",
    ) -> T:
        """Run ``call`` with budget enforcement and durable accounting.

        Order of operations is the point: check, reserve, call, settle.  A crash
        anywhere after the reservation leaves the money accounted for.
        """
        # Every mode except live must estimate zero. Written as "not live" rather
        # than "is mock" so that adding a free mode cannot accidentally add a
        # spending one: a replay of a frozen premium job runs providers whose model
        # names are priced in dollars, and the assertion below is what forces those
        # providers to declare the cost of *this* run rather than the original.
        if not self.settings.is_live:
            if estimated_usd > 0:
                raise AssertionError(
                    f"{self.settings.provider_mode.value} mode must never estimate a "
                    f"non-zero cost (got ${estimated_usd:.4f} for {provider}/{operation}) — "
                    "this means a live provider leaked into a free run"
                )
            return await call()

        status = await self.status()
        if not status.can_afford(estimated_usd):
            raise BudgetExceeded(estimated_usd, status, scope="total budget")

        job_so_far = await self.ledger.job_spent(job_id)
        if job_so_far + estimated_usd > self.settings.budget_per_job_usd:
            raise BudgetExceeded(
                estimated_usd,
                BudgetStatus(
                    total_budget_usd=self.settings.budget_per_job_usd,
                    spent_usd=job_so_far,
                    per_job_cap_usd=self.settings.budget_per_job_usd,
                ),
                scope=f"per-job cap for {job_id}",
            )

        entry = SpendEntry(
            job_id=job_id,
            provider=provider,
            operation=operation,
            model=model,
            quantity=quantity,
            unit_cost_usd=round(estimated_usd / quantity, 6) if quantity else 0.0,
            cost_usd=estimated_usd,
            note=note,
        )
        entry_id = await self.ledger.reserve(entry, estimated_usd)

        try:
            result = await call()
        except Exception:
            # The call never produced output, so it should not be charged.
            await self.ledger.void(entry_id)
            raise

        actual = actual_cost_of(result) if actual_cost_of else estimated_usd
        await self.ledger.settle(entry_id, actual)
        return result

    # --- Cache ---------------------------------------------------------------

    async def cached_asset(self, fingerprint: str) -> AssetRef | None:
        if not self.use_cache:
            return None
        return await self.ledger.lookup_cache(fingerprint)

    async def remember_asset(self, fingerprint: str, operation: str, asset: AssetRef) -> None:
        # Writing is disabled with reading, not only for symmetry: a replay's assets
        # live under a throwaway job id, and caching them would point a later real
        # run at a directory that gets cleaned up.
        if not self.use_cache:
            return
        await self.ledger.store_cache(fingerprint, operation, asset)
