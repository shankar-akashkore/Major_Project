"""One way to open the database, because SQLite's defaults lose writes under load.

**The bug this exists to fix.**  Sixteen jobs were started at once against the dev
server.  Ten of them died — not on generation, not on a budget refusal, but on
``sqlite3.OperationalError: database is locked`` while persisting a stage event.
The API writes the whole job document after *every* progress event, so a handful of
concurrent jobs is a steady stream of short write transactions, and SQLite's
out-of-the-box behaviour is the worst possible response to that:

* the default journal mode takes a lock over the entire database file for a write,
  so one job's 5 ms update blocks every other job's, including plain reads; and
* the default ``busy_timeout`` is **zero**, so a connection that finds the file
  locked does not wait a millisecond before raising.

The result was a failure that looks exactly like a pipeline bug on screen — a job
marked ``failed`` at the intake stage — and is nothing of the sort.  Worse, the
error escaped from ``JobStore.save`` while it was recording *another* stage's
progress, so the job's own error field was written by the thing that was trying to
write the job.

:func:`open_database` sets three pragmas on every new connection:

``journal_mode=WAL``
    Readers stop blocking the writer and the writer stops blocking readers, which
    is most of the contention: the job board polls while jobs are running.
    Persistent — it is a property of the file, not the connection — but set on each
    connect anyway, because the first process to open a fresh database is the one
    that decides, and that must not depend on which process got there first.

``busy_timeout=5000``
    The actual fix for the remaining writer-versus-writer case.  Five seconds is far
    longer than any write here (each is one row of JSON) and far shorter than a user
    waiting for a job, so a contended write waits instead of failing.

``synchronous=NORMAL``
    The safe companion to WAL: durable across a process crash, and only at risk from
    an OS-level crash mid-checkpoint.  The content is regenerable job documents and
    a spend ledger that is reconciled against the provider's own statement, so the
    trade is worth one less fsync per transaction.

Postgres ignores all of this, deliberately — Supabase is the deployment target and
none of these pragmas are meaningful there.  The check is on the URL rather than on
a settings flag so that it cannot be configured wrong.

**The pragmas were not enough on their own**, which is the more interesting half.
``JobStore.save`` read the row to decide between an insert and an update, then wrote
it, both inside one transaction.  In WAL mode a connection that has taken a read
snapshot and then asks to write gets ``SQLITE_BUSY_SNAPSHOT`` the moment any other
connection has committed in between — and SQLite returns that one **immediately,
without consulting ``busy_timeout`` at all**, because waiting on it could deadlock
two transactions that each hold a read and want a write.  So the pragma fixed the
plain writer-versus-writer case and left the actual failure untouched.

:func:`upsert` is half the fix: one statement that inserts or updates, so there is
no read to upgrade from.  It also closes a race the old code had regardless of
locking — two saves of the same job could both find no row and both try to insert.

:func:`write` is the other half, and it is the one that finally made eight
concurrent jobs pass.  **``busy_timeout`` cannot work here, by construction.**  It
blocks inside SQLite's C code, on the worker thread ``aiosqlite`` gave that
connection, and never yields the event loop.  But the connection *holding* the lock
is an async one: it has issued its statement and is waiting on the loop to run its
``COMMIT``.  The mock pipeline draws images and shells out to ffmpeg
synchronously, so with several jobs running the loop is busy for seconds at a
time — and a waiter that refuses to yield to the loop is waiting for something only
the loop can deliver.  Five seconds of that is five seconds of guaranteed failure.

The async analogue is to retry around ``await asyncio.sleep``, which hands control
back so the holder can commit.  ``busy_timeout`` stays anyway: it is still the right
mechanism for the case it *can* serve, another OS process — a script running
alongside the dev server.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import Table, event
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Dialect
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.sql import Executable, Insert

#: Milliseconds a blocked writer waits before giving up. See the module docstring.
BUSY_TIMEOUT_MS = 5_000

#: Attempts :func:`write` makes before giving up, and the first backoff in seconds.
#: Doubling from 20 ms, seven attempts wait about 2.5 s in total — longer than any
#: single stage of the mock pipeline holds the loop, and short enough that a
#: genuinely stuck database still surfaces as an error rather than a hang.
WRITE_ATTEMPTS = 7
WRITE_BACKOFF_S = 0.02


def open_database(url: str) -> AsyncEngine:
    """Build an engine for ``url``, tuned for concurrency if it is SQLite."""
    engine = create_async_engine(url, future=True)
    if not url.startswith("sqlite"):
        return engine

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_connection, _record) -> None:  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()

    return engine


def _is_locked(exc: OperationalError) -> bool:
    """Whether this is contention rather than a real fault.

    Matched on the message because the driver does not expose SQLite's extended
    result code through SQLAlchemy, and the alternative — retrying every
    ``OperationalError`` — would sit in a backoff loop over a missing table.
    """
    text = str(exc.orig).lower()
    return "locked" in text or "busy" in text


async def write(engine: AsyncEngine, statement: Executable) -> None:
    """Run one write, retrying while the database is merely busy.

    Yielding to the event loop between attempts is the entire point; see the module
    docstring for why ``busy_timeout`` cannot do this job.
    """
    for attempt in range(WRITE_ATTEMPTS):
        try:
            async with engine.begin() as conn:
                await conn.execute(statement)
            return
        except OperationalError as exc:
            if attempt == WRITE_ATTEMPTS - 1 or not _is_locked(exc):
                raise
            await asyncio.sleep(WRITE_BACKOFF_S * 2**attempt)


def upsert(dialect: Dialect, table: Table, values: dict, *, key: str, update: list[str]) -> Insert:
    """An insert-or-update for ``table`` as one statement.

    ``update`` names the columns a conflict overwrites, which is how a column like
    ``created_at`` stays at its original value: it is written on insert and simply
    not listed here.  An empty ``update`` means first-writer-wins, and becomes ``DO
    NOTHING`` — ``DO UPDATE SET`` with nothing to set is not valid SQL.

    Both dialects spell this the same way but reach it through different modules,
    so it is chosen on the connection's dialect rather than on configuration —
    there is no way for the two to disagree.
    """
    module = sqlite if dialect.name == "sqlite" else postgresql
    statement = module.insert(table).values(**values)
    if not update:
        return statement.on_conflict_do_nothing(index_elements=[key])
    return statement.on_conflict_do_update(
        index_elements=[key],
        set_={name: statement.excluded[name] for name in update},
    )
