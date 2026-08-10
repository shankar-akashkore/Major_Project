"""Job persistence and the live event bus.

Jobs are stored as a JSON document rather than a normalised schema.  The job
contract in :mod:`adschema` is still moving weekly, and a migration per field
change would slow that down for no benefit at this stage — the query patterns are
"fetch one by id" and "list recent", neither of which needs columns.  When the
annotation tool needs to query candidates across jobs (week 7), that will be the
moment to normalise.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime

from adschema import JobRecord, StageEvent
from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, select
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

METADATA = MetaData()

jobs = Table(
    "jobs",
    METADATA,
    Column("job_id", String(32), primary_key=True),
    Column("state", String(32), nullable=False, index=True),
    Column("document", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


class JobStore:
    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    @classmethod
    def from_url(cls, url: str) -> JobStore:
        return cls(create_async_engine(url, future=True))

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(METADATA.create_all)

    async def save(self, record: JobRecord) -> None:
        now = datetime.now(UTC)
        document = record.model_dump_json()
        async with self.engine.begin() as conn:
            existing = await conn.execute(
                select(jobs.c.job_id).where(jobs.c.job_id == record.job_id)
            )
            if existing.scalar_one_or_none() is None:
                await conn.execute(
                    jobs.insert().values(
                        job_id=record.job_id,
                        state=record.state.value,
                        document=document,
                        created_at=record.request.created_at,
                        updated_at=now,
                    )
                )
            else:
                await conn.execute(
                    jobs.update()
                    .where(jobs.c.job_id == record.job_id)
                    .values(state=record.state.value, document=document, updated_at=now)
                )

    async def get(self, job_id: str) -> JobRecord | None:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(select(jobs.c.document).where(jobs.c.job_id == job_id))
            ).scalar_one_or_none()
        return JobRecord.model_validate_json(row) if row else None

    async def list_recent(self, limit: int = 25) -> list[JobRecord]:
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(jobs.c.document).order_by(jobs.c.created_at.desc()).limit(limit)
                )
            ).scalars()
        return [JobRecord.model_validate_json(r) for r in rows]


class EventBus:
    """Fan-out of stage events to SSE subscribers.

    Events are also buffered per job so a client that connects mid-run receives
    the history before the live tail — otherwise a page refresh would show a
    progress bar starting from wherever it happened to reconnect.
    """

    def __init__(self) -> None:
        self._history: dict[str, list[StageEvent]] = defaultdict(list)
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._finished: set[str] = set()

    async def publish(self, event: StageEvent) -> None:
        self._history[event.job_id].append(event)
        for queue in list(self._subscribers[event.job_id]):
            queue.put_nowait(event)

    def mark_finished(self, job_id: str) -> None:
        self._finished.add(job_id)
        for queue in list(self._subscribers[job_id]):
            queue.put_nowait(None)

    def is_finished(self, job_id: str) -> bool:
        return job_id in self._finished

    def history(self, job_id: str) -> list[StageEvent]:
        return list(self._history[job_id])

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[job_id].append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        if queue in self._subscribers[job_id]:
            self._subscribers[job_id].remove(queue)
