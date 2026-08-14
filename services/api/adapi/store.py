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

from adproviders import open_database, upsert, write
from adschema import JobRecord, StageEvent
from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, func, select
from sqlalchemy.ext.asyncio import AsyncEngine

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
        return cls(open_database(url))

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(METADATA.create_all)

    async def save(self, record: JobRecord) -> None:
        """Insert or update in one statement, which is not a micro-optimisation.

        This used to read the row to choose between an insert and an update, then
        write it, in one transaction. Under concurrency that read-then-upgrade is
        what SQLite refuses outright — see :mod:`adproviders.db` — and it lost ten
        of sixteen simultaneous jobs to ``database is locked``, reported on screen
        as a job failing at intake.

        ``created_at`` is written on insert and not in the update set, so a save
        does not keep moving the row to the top of the board.
        """
        await write(
            self.engine,
            upsert(
                self.engine.dialect,
                jobs,
                {
                    "job_id": record.job_id,
                    "state": record.state.value,
                    "document": record.model_dump_json(),
                    "created_at": record.request.created_at,
                    "updated_at": datetime.now(UTC),
                },
                key="job_id",
                update=["state", "document", "updated_at"],
            ),
        )

    async def get(self, job_id: str) -> JobRecord | None:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(select(jobs.c.document).where(jobs.c.job_id == job_id))
            ).scalar_one_or_none()
        return JobRecord.model_validate_json(row) if row else None

    #: Newest first, with ``job_id`` breaking ties. Without the tiebreak two jobs
    #: created in the same clock tick have no order defined between them, and a row
    #: can appear on two consecutive pages or on neither.
    _NEWEST_FIRST = (jobs.c.created_at.desc(), jobs.c.job_id.desc())

    async def list_recent(self, limit: int = 25) -> list[JobRecord]:
        """The newest ``limit`` jobs. What a script wants: rows, no bookkeeping."""
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(jobs.c.document).order_by(*self._NEWEST_FIRST).limit(limit)
                )
            ).scalars()
        return [JobRecord.model_validate_json(r) for r in rows]

    async def page(self, limit: int = 25, offset: int = 0) -> tuple[list[JobRecord], int]:
        """One page for the board, and the total it was taken from.

        A separate method rather than an extra return value on
        :meth:`list_recent`, which is what this was first written as — three scripts
        call that and every one of them broke silently, unpacking a two-tuple as if
        it were the list. A signature that changes shape under callers who never
        asked for the new part is a trap, and the type checker does not run on
        ``scripts/``.

        Both queries share one connection so the count cannot be taken after a job
        the page did not see was inserted; a total smaller than the rows returned
        would read as a bug in the board.
        """
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    select(jobs.c.document)
                    .order_by(*self._NEWEST_FIRST)
                    .limit(limit)
                    .offset(offset)
                )
            ).scalars()
            records = [JobRecord.model_validate_json(r) for r in rows]
            total = (await conn.execute(select(func.count()).select_from(jobs))).scalar_one()
        return records, int(total)


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
