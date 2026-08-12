"use client";

/**
 * Data loading, kept small on purpose — no query library.
 *
 * `useJobStream` is the only one with any real content. It reads the SSE progress
 * stream, and the load-bearing detail is that the server sends its buffered history
 * before the live tail (`stream_events` in `adapi.main`), so a page opened halfway
 * through a job still draws the stages that already happened. A client that assumed
 * it was joining live would show a job at "video gen" with no history behind it.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, eventsUrl, getJob } from "./api.ts";
import type { JobRecord, StageEvent } from "./contract.ts";
import { isTerminal } from "./presentation.ts";

export type Async<T> = {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
};

/** One fetch, with a manual reload and a guard against setting state after unmount. */
export function useAsync<T>(load: () => Promise<T>, deps: unknown[] = []): Async<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let live = true;
    setLoading(true);
    load()
      .then((value) => {
        if (!live) return;
        setData(value);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (!live) return;
        setError(cause instanceof Error ? cause.message : String(cause));
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { data, error, loading, reload };
}

/** The same, on an interval. Used for the budget, which any job can move. */
export function usePolled<T>(load: () => Promise<T>, intervalMs: number): Async<T> {
  const state = useAsync(load, []);
  const reload = state.reload;
  useEffect(() => {
    const timer = setInterval(reload, intervalMs);
    return () => clearInterval(timer);
  }, [reload, intervalMs]);
  return state;
}

export type JobStream = {
  record: JobRecord | null;
  events: StageEvent[];
  error: string | null;
  /** True while the SSE connection is open. */
  streaming: boolean;
};

/**
 * A job's record plus its live event stream.
 *
 * The record is re-fetched when the stream closes rather than being assembled from
 * the events: the events say what happened, and the record is what was persisted.
 * Building the result client-side from progress messages would put a second,
 * subtly different idea of the job's outcome on screen.
 */
export function useJobStream(jobId: string): JobStream {
  const [record, setRecord] = useState<JobRecord | null>(null);
  const [events, setEvents] = useState<StageEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const seen = useRef(new Set<string>());

  const refresh = useCallback(async () => {
    try {
      setRecord(await getJob(jobId));
      setError(null);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    }
  }, [jobId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    // Nothing to stream once the job has stopped moving; the record is enough, and
    // an EventSource against a finished job just reconnects forever.
    if (record && isTerminal(record)) return;

    const source = new EventSource(eventsUrl(jobId));
    setStreaming(true);

    source.onmessage = (message) => {
      try {
        const event = JSON.parse(message.data) as StageEvent;
        // The history replay can overlap the live tail, so events are de-duplicated
        // on their own content rather than trusted to arrive once.
        const key = `${event.stage}|${event.state}|${event.at}|${event.message}`;
        if (seen.current.has(key)) return;
        seen.current.add(key);
        setEvents((previous) => [...previous, event]);
      } catch {
        // A malformed frame is not worth tearing the stream down for.
      }
    };

    source.addEventListener("done", () => {
      source.close();
      setStreaming(false);
      void refresh();
    });

    source.onerror = () => {
      source.close();
      setStreaming(false);
      // Not an error the user can act on: the stream also ends this way when the
      // job finishes between the record fetch and the connection opening.
      void refresh();
    };

    return () => {
      source.close();
      setStreaming(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, record?.state, refresh]);

  return { record, events, error, streaming };
}
