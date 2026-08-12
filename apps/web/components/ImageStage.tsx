"use client";

/**
 * Stage 5: the three candidates, ranked as images, before any video exists.
 *
 * This is the screen the project's claim is about. The order shown here is what a
 * budget-constrained system would have promoted on, and whether it survives
 * animation is the thing being measured.
 *
 * The gate is rendered beside the score and never merged into it. They answer
 * different questions — "is this usable at all" versus "how well will it perform" —
 * and the pipeline keeps them apart precisely so a quality defect cannot be
 * laundered into a slightly lower performance number. Merging them here would undo
 * that on the last hop.
 */

import { mediaUrl } from "@/lib/api.ts";
import type { ImageCandidate } from "@/lib/contract.ts";
import { gateSummary, humanise, num, prose, slotLetter } from "@/lib/presentation.ts";
import { OrderStrip, ScoreDetail } from "./Score.tsx";
import { Badge, Card, Caveat, SectionTitle } from "./ui.tsx";

export function ImageStage({ images }: { images: ImageCandidate[] }) {
  if (images.length === 0) return null;

  const scored = images.filter((image) => image.score !== null);
  const order = [...scored]
    .sort((a, b) => (b.score?.overall ?? 0) - (a.score?.overall ?? 0))
    .map((image) => slotLetter(image.index));
  const rankOf = new Map(order.map((slot, index) => [slot, index + 1]));

  return (
    <section>
      <SectionTitle hint={`${images.length} candidates from ${images.length} design points`}>
        Image stage
      </SectionTitle>

      {order.length > 0 ? (
        <div className="mb-3">
          <OrderStrip order={order} label="Predicted order" />
        </div>
      ) : null}

      <div className="grid gap-4 md:grid-cols-3">
        {images.map((image) => {
          const gate = gateSummary(image.gate);
          const slot = slotLetter(image.index);
          const url = mediaUrl(image.asset);
          return (
            <Card key={image.index} className="overflow-hidden" as="article">
              <div className="relative aspect-4/5 bg-zinc-950">
                {url ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={url}
                    alt={`Candidate ${slot}`}
                    className="size-full object-cover"
                    loading="lazy"
                  />
                ) : (
                  <div className="flex size-full items-center justify-center text-[11px] text-zinc-600">
                    no frame
                  </div>
                )}
                <div className="absolute top-2 left-2 flex items-center gap-1.5">
                  <span className="rounded bg-zinc-950/85 px-1.5 py-0.5 font-mono text-xs text-zinc-100">
                    {slot}
                  </span>
                  {rankOf.has(slot) ? (
                    <span className="rounded bg-zinc-950/85 px-1.5 py-0.5 text-[10px] text-zinc-300">
                      predicted #{rankOf.get(slot)}
                    </span>
                  ) : null}
                </div>
              </div>

              <div className="space-y-3 p-3">
                <p className="text-xs leading-snug text-zinc-300">
                  {image.brief.concept || humanise(image.brief.design_point.angle)}
                </p>

                <div className="flex flex-wrap gap-1">
                  <Badge>{humanise(image.brief.design_point.angle)}</Badge>
                  <Badge>{humanise(image.brief.design_point.lighting)}</Badge>
                  <Badge>{humanise(image.brief.design_point.composition)}</Badge>
                </div>

                <ScoreDetail score={image.score} />

                {gate ? (
                  <div className="border-t border-zinc-800 pt-2.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[11px] text-zinc-500">Quality gate</span>
                      <Badge
                        tone={
                          gate.verdict === "pass"
                            ? gate.pending.length > 0
                              ? "warn"
                              : "good"
                            : gate.verdict === "retry"
                              ? "warn"
                              : "bad"
                        }
                      >
                        {gate.verdict}
                      </Badge>
                    </div>
                    <p className="mt-1 text-[11px] leading-snug text-zinc-400">{gate.text}</p>

                    {gate.failed.length > 0 ? (
                      <ul className="mt-1.5 space-y-0.5">
                        {gate.failed.map((check) => (
                          <li key={check.name} className="text-[11px] text-bad-400">
                            {humanise(check.name)} {num(check.value)} vs {num(check.threshold)}
                          </li>
                        ))}
                      </ul>
                    ) : null}

                    {gate.pending.length > 0 ? (
                      <div className="mt-1.5">
                        <Caveat>
                          {prose(gate.pending.map((check) => humanise(check.name)))} did not run —
                          the model has not landed, and a pending check always passes. Identity is{" "}
                          <strong>not</strong> verified.
                        </Caveat>
                      </div>
                    ) : gate.verifiedIdentity ? (
                      <p className="mt-1.5 text-[11px] text-good-400">
                        Product and face identity both verified.
                      </p>
                    ) : null}
                  </div>
                ) : null}

                <dl className="grid grid-cols-2 gap-x-3 border-t border-zinc-800 pt-2.5 text-[10px] text-zinc-500">
                  <Row label="provider">{image.provider}</Row>
                  <Row label="seed">{image.seed}</Row>
                  <Row label="tier">{image.tier}</Row>
                  <Row label="latency">{image.latency_ms} ms</Row>
                </dl>
              </div>
            </Card>
          );
        })}
      </div>
    </section>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-2">
      <dt>{label}</dt>
      <dd className="text-zinc-400 tabular">{children}</dd>
    </div>
  );
}
