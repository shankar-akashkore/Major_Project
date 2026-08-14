"""The spend ledger and the generation cache.

Two ideas make this durable rather than decorative:

1. **Reserve before calling, settle after.**  A charge is written to the ledger
   *before* the HTTP request goes out.  If the process dies mid-call, the money
   is still accounted for — the alternative (record on success) silently loses
   spend exactly when you most need the number to be right.
2. **Reserved money counts as spent.**  ``total_spent`` sums settled *and*
   in-flight reservations, so twenty concurrent calls cannot each individually
   look affordable and collectively blow the cap.

The generation cache lives here too, because a cache hit is the cheapest
possible outcome and belongs on the same code path as the charge it avoids.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from adschema import AssetRef, SpendEntry
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    MetaData,
    String,
    Table,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncEngine

from .db import open_database, upsert, write

METADATA = MetaData()

#: A reservation that was never settled or voided is still counted as spent.
STATE_RESERVED = "reserved"
STATE_SETTLED = "settled"
STATE_VOIDED = "voided"

spend_entries = Table(
    "spend_entries",
    METADATA,
    Column("entry_id", String(32), primary_key=True),
    Column("job_id", String(32), index=True, nullable=True),
    Column("provider", String(64), nullable=False),
    Column("operation", String(16), nullable=False),
    Column("model", String(64), nullable=False, default=""),
    Column("quantity", Float, nullable=False, default=1.0),
    Column("unit_cost_usd", Float, nullable=False, default=0.0),
    Column("cost_usd", Float, nullable=False, default=0.0),
    Column("estimated_usd", Float, nullable=False, default=0.0),
    Column("state", String(16), nullable=False, default=STATE_RESERVED, index=True),
    Column("cache_hit", Boolean, nullable=False, default=False),
    Column("at", DateTime(timezone=True), nullable=False),
    Column("note", String(512), nullable=False, default=""),
)

generation_cache = Table(
    "generation_cache",
    METADATA,
    Column("fingerprint", String(64), primary_key=True),
    Column("operation", String(16), nullable=False),
    Column("asset_json", String(2048), nullable=False),
    Column("at", DateTime(timezone=True), nullable=False),
)


@runtime_checkable
class LedgerStore(Protocol):
    """What the cost governor needs from a ledger."""

    async def total_spent(self) -> float: ...

    async def job_spent(self, job_id: str) -> float: ...

    async def reserve(self, entry: SpendEntry, estimated_usd: float) -> str: ...

    async def settle(self, entry_id: str, actual_usd: float) -> None: ...

    async def void(self, entry_id: str) -> None: ...

    async def lookup_cache(self, fingerprint: str) -> AssetRef | None: ...

    async def store_cache(self, fingerprint: str, operation: str, asset: AssetRef) -> None: ...

    async def entries(self, job_id: str | None = None) -> list[SpendEntry]: ...


class InMemoryLedger:
    """Non-durable ledger for tests and mock runs."""

    def __init__(self) -> None:
        self._rows: dict[str, dict] = {}
        self._cache: dict[str, AssetRef] = {}

    async def total_spent(self) -> float:
        return round(
            sum(r["cost_usd"] for r in self._rows.values() if r["state"] != STATE_VOIDED), 6
        )

    async def job_spent(self, job_id: str) -> float:
        return round(
            sum(
                r["cost_usd"]
                for r in self._rows.values()
                if r["job_id"] == job_id and r["state"] != STATE_VOIDED
            ),
            6,
        )

    async def reserve(self, entry: SpendEntry, estimated_usd: float) -> str:
        self._rows[entry.entry_id] = {
            **entry.model_dump(),
            "cost_usd": estimated_usd,
            "estimated_usd": estimated_usd,
            "state": STATE_RESERVED,
        }
        return entry.entry_id

    async def settle(self, entry_id: str, actual_usd: float) -> None:
        row = self._rows[entry_id]
        row["cost_usd"] = actual_usd
        row["state"] = STATE_SETTLED

    async def void(self, entry_id: str) -> None:
        row = self._rows[entry_id]
        row["cost_usd"] = 0.0
        row["state"] = STATE_VOIDED

    async def lookup_cache(self, fingerprint: str) -> AssetRef | None:
        return self._cache.get(fingerprint)

    async def store_cache(self, fingerprint: str, operation: str, asset: AssetRef) -> None:
        self._cache[fingerprint] = asset

    async def entries(self, job_id: str | None = None) -> list[SpendEntry]:
        rows = [r for r in self._rows.values() if job_id is None or r["job_id"] == job_id]
        return [
            SpendEntry(**{k: v for k, v in r.items() if k in SpendEntry.model_fields}) for r in rows
        ]


class SqlLedger:
    """Durable ledger backed by SQLAlchemy (SQLite locally, Postgres in deploy)."""

    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    @classmethod
    def from_url(cls, url: str) -> SqlLedger:
        return cls(open_database(url))

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(METADATA.create_all)

    async def total_spent(self) -> float:
        stmt = select(func.coalesce(func.sum(spend_entries.c.cost_usd), 0.0)).where(
            spend_entries.c.state != STATE_VOIDED
        )
        async with self.engine.connect() as conn:
            return round(float((await conn.execute(stmt)).scalar_one()), 6)

    async def job_spent(self, job_id: str) -> float:
        stmt = select(func.coalesce(func.sum(spend_entries.c.cost_usd), 0.0)).where(
            spend_entries.c.job_id == job_id, spend_entries.c.state != STATE_VOIDED
        )
        async with self.engine.connect() as conn:
            return round(float((await conn.execute(stmt)).scalar_one()), 6)

    async def reserve(self, entry: SpendEntry, estimated_usd: float) -> str:
        stmt = spend_entries.insert().values(
            entry_id=entry.entry_id,
            job_id=entry.job_id,
            provider=entry.provider,
            operation=entry.operation,
            model=entry.model,
            quantity=entry.quantity,
            unit_cost_usd=entry.unit_cost_usd,
            cost_usd=estimated_usd,
            estimated_usd=estimated_usd,
            state=STATE_RESERVED,
            cache_hit=entry.cache_hit,
            at=entry.at,
            note=entry.note,
        )
        await write(self.engine, stmt)
        return entry.entry_id

    async def settle(self, entry_id: str, actual_usd: float) -> None:
        stmt = (
            spend_entries.update()
            .where(spend_entries.c.entry_id == entry_id)
            .values(cost_usd=actual_usd, state=STATE_SETTLED)
        )
        await write(self.engine, stmt)

    async def void(self, entry_id: str) -> None:
        stmt = (
            spend_entries.update()
            .where(spend_entries.c.entry_id == entry_id)
            .values(cost_usd=0.0, state=STATE_VOIDED)
        )
        await write(self.engine, stmt)

    async def lookup_cache(self, fingerprint: str) -> AssetRef | None:
        stmt = select(generation_cache.c.asset_json).where(
            generation_cache.c.fingerprint == fingerprint
        )
        async with self.engine.connect() as conn:
            row = (await conn.execute(stmt)).scalar_one_or_none()
        return AssetRef.model_validate(json.loads(row)) if row else None

    async def store_cache(self, fingerprint: str, operation: str, asset: AssetRef) -> None:
        """Record a generation against its fingerprint, first writer wins.

        An upsert that overwrites nothing: the same fingerprint means the same
        request, so a second write has nothing new to say and the original `at` is
        the more useful timestamp. Written as one statement for the reason in
        :mod:`adproviders.db` — the read-then-write it replaces is what SQLite
        refuses under concurrency, and this runs once per generation.
        """
        await write(
            self.engine,
            upsert(
                self.engine.dialect,
                generation_cache,
                {
                    "fingerprint": fingerprint,
                    "operation": operation,
                    "asset_json": asset.model_dump_json(),
                    "at": datetime.now(UTC),
                },
                key="fingerprint",
                update=[],
            ),
        )

    async def entries(self, job_id: str | None = None) -> list[SpendEntry]:
        stmt = select(spend_entries)
        if job_id is not None:
            stmt = stmt.where(spend_entries.c.job_id == job_id)
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt.order_by(spend_entries.c.at))).mappings().all()
        return [
            SpendEntry(**{k: v for k, v in row.items() if k in SpendEntry.model_fields})
            for row in rows
        ]
