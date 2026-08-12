"use client";

/**
 * Score rendering — the one component that could undo the whole project's care.
 *
 * `scoreDisplay` returns a tagged union so the stub case cannot be forgotten; this
 * is where the tag becomes something a person sees. A placeholder shows in the
 * warning colour, with the model version that produced it, and the number is struck
 * through — because the *ordering* the stub produces is real (the pipeline really
 * did rank these three) while the *value* means nothing and must not be read as a
 * predicted click-through rate.
 */

import { Badge, Caveat } from "./ui.tsx";
import { humanise, num, percent, scoreDisplay, topDrivers } from "@/lib/presentation.ts";
import type { ScoreBreakdown } from "@/lib/contract.ts";

export function ScoreChip({ score }: { score: ScoreBreakdown | null }) {
  const display = scoreDisplay(score);
  if (display.kind === "absent") {
    return <Badge tone="neutral">{display.text}</Badge>;
  }
  if (display.kind === "stub") {
    return (
      <Badge tone="warn" title={display.caveat}>
        <span className="line-through decoration-warn-400/60">{display.text}</span>
        <span className="opacity-80">placeholder</span>
      </Badge>
    );
  }
  return <Badge tone="good">{display.text}</Badge>;
}

/** The full breakdown: what the ranker sorted on, and what it was made of. */
export function ScoreDetail({ score }: { score: ScoreBreakdown | null }) {
  const display = scoreDisplay(score);
  if (!score || display.kind === "absent") {
    return <p className="text-[11px] text-zinc-500">{display.text}</p>;
  }
  const drivers = topDrivers(score, 4);
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <ScoreChip score={score} />
        <span className="font-mono text-[10px] text-zinc-500">{score.model_version}</span>
      </div>

      {drivers.length > 0 ? (
        <dl className="space-y-1">
          {drivers.map(([name, value]) => (
            <div key={name} className="flex items-center gap-2 text-[11px]">
              <dt className="w-32 shrink-0 truncate text-zinc-500" title={humanise(name)}>
                {humanise(name)}
              </dt>
              <dd className="flex flex-1 items-center gap-2">
                <span className="h-1 flex-1 overflow-hidden rounded-full bg-zinc-800">
                  <span
                    className="block h-full rounded-full bg-zinc-500"
                    style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
                  />
                </span>
                <span className="w-9 text-right text-zinc-400 tabular">{percent(value)}</span>
              </dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="text-[11px] text-zinc-500">
          No component scores were recorded, so there is nothing to explain the ranking with.
        </p>
      )}

      {display.kind === "stub" ? <Caveat>{display.caveat}</Caveat> : null}
    </div>
  );
}

/**
 * The predicted image-stage order, as a row of slot letters.
 *
 * Shown before the videos exist, which is the point of the stage: this is the
 * ordering a budget-constrained system would have promoted on.
 */
export function OrderStrip({ order, label }: { order: string[]; label: string }) {
  return (
    <div className="flex items-center gap-2 text-[11px] text-zinc-400">
      <span className="w-20 shrink-0">{label}</span>
      <span className="flex items-center gap-1">
        {order.map((slot, index) => (
          <span key={slot} className="flex items-center gap-1">
            {index > 0 ? <span className="text-zinc-600">›</span> : null}
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-zinc-200">{slot}</span>
          </span>
        ))}
      </span>
    </div>
  );
}

export function Numeric({ value, digits = 3 }: { value: number | null; digits?: number }) {
  return <span className="tabular">{num(value, digits)}</span>;
}
