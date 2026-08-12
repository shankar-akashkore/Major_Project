"use client";

/**
 * The eight stages, and where the job is.
 *
 * Two things are shown that a plain spinner would hide. The **warnings** in the
 * event log, which is where a golden replay reports that its output drifted from
 * what was frozen — a regression has to be visible in the run that found it, not in
 * a script's stdout. And the **stage names**, because "quality gate" failing is a
 * different conversation from "video gen" failing, and during a demo the difference
 * decides what you say next.
 */

import type { Stage, StageEvent } from "@/lib/contract.ts";
import { STAGE_VALUES } from "@/lib/contract.ts";
import { humanise, progressFrom } from "@/lib/presentation.ts";
import { Meter } from "./ui.tsx";

const LABEL: Record<string, string> = {
  intake: "intake",
  brief: "briefs",
  image_gen: "images",
  quality_gate: "gate",
  image_rank: "rank 1",
  video_gen: "videos",
  video_rank: "rank 2",
  delivery: "delivery",
};

export function StageProgress({
  events,
  state,
  streaming,
}: {
  events: StageEvent[];
  state: string;
  streaming: boolean;
}) {
  const progress = progressFrom(events);
  const reached = new Set(events.map((event) => event.stage));
  const failed = events.some((event) => event.state === "failed");
  const warnings = events.filter((event) => event.state === "warning");
  const currentIndex = progress.stage ? STAGE_VALUES.indexOf(progress.stage) : -1;
  const last = events.length > 0 ? events[events.length - 1] : undefined;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-4 text-xs">
        <span className="text-zinc-300">
          {state === "completed"
            ? "Completed"
            : state === "failed"
              ? "Failed"
              : progress.stage
                ? `Running · ${humanise(progress.stage)}`
                : "Queued"}
        </span>
        <span className="text-zinc-500 tabular">
          {Math.round(progress.fraction * 100)}%
          {streaming ? <span className="ml-2 text-sky-400">live</span> : null}
        </span>
      </div>

      <Meter
        fraction={progress.fraction}
        tone={failed ? "bad" : state === "completed" ? "good" : "info"}
      />

      <ol className="flex flex-wrap gap-1">
        {STAGE_VALUES.map((stage: Stage, index) => {
          const done = index < currentIndex || state === "completed";
          const current = index === currentIndex && state !== "completed";
          return (
            <li
              key={stage}
              className={`rounded px-1.5 py-0.5 text-[10px] ring-1 ring-inset ${
                current
                  ? "bg-sky-950 text-sky-300 ring-sky-500/40"
                  : done || reached.has(stage)
                    ? "bg-zinc-800 text-zinc-300 ring-zinc-700"
                    : "text-zinc-600 ring-zinc-800"
              }`}
              title={humanise(stage)}
            >
              {LABEL[stage] ?? humanise(stage)}
            </li>
          );
        })}
      </ol>

      {last?.message ? (
        <p className="font-mono text-[11px] leading-snug text-zinc-500">{last.message}</p>
      ) : null}

      {warnings.length > 0 ? (
        <ul className="space-y-1 rounded-md border border-warn-500/30 bg-warn-950 p-2.5">
          {warnings.map((warning, index) => (
            <li key={index} className="text-[11px] leading-snug text-warn-400">
              ▲ {warning.message}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

/** The full event log, collapsed by default — useful when a stage misbehaves. */
export function EventLog({ events }: { events: StageEvent[] }) {
  if (events.length === 0) return null;
  return (
    <details className="group">
      <summary className="cursor-pointer text-xs text-zinc-500 hover:text-zinc-300">
        Event log ({events.length})
      </summary>
      <ol className="mt-2 max-h-64 space-y-1 overflow-y-auto rounded border border-zinc-800 bg-zinc-950 p-2">
        {events.map((event, index) => (
          <li key={index} className="flex gap-2 font-mono text-[10px] leading-relaxed">
            <span className="w-16 shrink-0 text-zinc-600">
              {new Date(event.at).toLocaleTimeString()}
            </span>
            <span className="w-24 shrink-0 text-zinc-500">{event.stage}</span>
            <span
              className={
                event.state === "failed"
                  ? "text-bad-400"
                  : event.state === "warning"
                    ? "text-warn-400"
                    : "text-zinc-400"
              }
            >
              {event.message || event.state}
            </span>
          </li>
        ))}
      </ol>
    </details>
  );
}
