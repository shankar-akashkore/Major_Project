"use client";

/**
 * Stage 5: every candidate, ranked as an image, before any video exists.
 *
 * This is the screen the project's claim is about, and it stopped being a
 * prediction here. The order shown is what the system actually promotes on: at
 * five candidates and two videos, three of these frames never become clips. So
 * the cut is drawn on the cards rather than left to be inferred from a video
 * stage that is shorter than this one — a candidate with no clip is a decision
 * the system made, and unlabelled it reads as something that failed.
 *
 * The gate is rendered beside the score and never merged into it. They answer
 * different questions — "is this usable at all" versus "how well will it perform" —
 * and the pipeline keeps them apart precisely so a quality defect cannot be
 * laundered into a slightly lower performance number. Merging them here would undo
 * that on the last hop.
 *
 * The cards crop to 4:5 with `object-cover`, which is right for comparing five
 * frames and wrong for judging one — the crop is exactly where a clipped hand or a
 * product pushed out of the safe area hides. So each frame opens whole, over the
 * grid rather than in a new tab, with a download that names the file after the
 * candidate. See `Lightbox.tsx`.
 */

import { useState } from "react";

import { mediaUrl } from "@/lib/api.ts";
import type { ImageCandidate } from "@/lib/contract.ts";
import {
  downloadName,
  gateSummary,
  humanise,
  num,
  prose,
  slotLetter,
} from "@/lib/presentation.ts";
import { Lightbox, type LightboxItem } from "./Lightbox.tsx";
import { OrderStrip, ScoreDetail } from "./Score.tsx";
import { Badge, Card, Caveat, SectionTitle } from "./ui.tsx";

export function ImageStage({ images }: { images: ImageCandidate[] }) {
  // Which frame the overlay is showing, as a position in `shots` below. Held here
  // rather than in the Lightbox so the arrow keys can walk the whole set: the
  // overlay is a view onto this grid, not a thing of its own.
  const [open, setOpen] = useState<number | null>(null);

  // Candidates with no asset are skipped, so the position a card opens is a
  // position that exists. Numbering the overlay by `image.index` instead would
  // step onto a missing frame the first time a generation fails.
  const shots: LightboxItem[] = [];
  const shotAt = new Map<number, number>();
  for (const image of images) {
    const url = mediaUrl(image.asset);
    if (!url) continue;
    const slot = slotLetter(image.index);
    shotAt.set(image.index, shots.length);
    shots.push({
      url,
      title: `Candidate ${slot}`,
      caption: image.brief.concept || humanise(image.brief.design_point.angle),
      filename: downloadName(image.asset, slot),
      width: image.asset.width,
      height: image.asset.height,
    });
  }

  if (images.length === 0) return null;

  const scored = images.filter((image) => image.score !== null);
  const order = [...scored]
    .sort((a, b) => (b.score?.overall ?? 0) - (a.score?.overall ?? 0))
    .map((image) => slotLetter(image.index));
  const rankOf = new Map(order.map((slot, index) => [slot, index + 1]));
  const promotedCount = images.filter((image) => image.promoted).length;
  // Only worth saying when something was actually cut. On a job that animates
  // everything this is noise, and the counts already say so.
  const cut = images.length - promotedCount;

  return (
    <section>
      <SectionTitle
        hint={
          cut > 0
            ? `${images.length} candidates · top ${promotedCount} animated, ${cut} not`
            : `${images.length} candidates from ${images.length} design points`
        }
      >
        Image stage
      </SectionTitle>

      {order.length > 0 ? (
        <div className="mb-3">
          <OrderStrip order={order} label="Predicted order" />
        </div>
      ) : null}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {images.map((image) => {
          const gate = gateSummary(image.gate);
          const slot = slotLetter(image.index);
          const url = mediaUrl(image.asset);
          return (
            <Card key={image.index} className="overflow-hidden" as="article">
              <div className="relative aspect-4/5 bg-zinc-950">
                {url ? (
                  // The grid crops to 4:5 with `object-cover`, which is where a
                  // clipped hand or an out-of-safe-area product hides. The whole
                  // frame is the control, because that is what a person aims at.
                  <button
                    type="button"
                    onClick={() => setOpen(shotAt.get(image.index) ?? null)}
                    aria-label={`View candidate ${slot} full size`}
                    className="group absolute inset-0 block cursor-zoom-in"
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={url}
                      alt={`Candidate ${slot}`}
                      className="size-full object-cover"
                      loading="lazy"
                    />
                    <span className="pointer-events-none absolute right-2 bottom-2 inline-flex items-center gap-1.5 rounded-md bg-zinc-950/85 px-2 py-1 text-[11px] font-medium text-zinc-200 ring-1 ring-zinc-700 ring-inset transition group-hover:bg-zinc-100 group-hover:text-zinc-900">
                      <span aria-hidden>⤢</span> Full size
                    </span>
                  </button>
                ) : (
                  <div className="flex size-full items-center justify-center text-[11px] text-zinc-600">
                    no frame
                  </div>
                )}
                <div className="pointer-events-none absolute top-2 left-2 z-10 flex items-center gap-1.5">
                  <span className="rounded bg-zinc-950/85 px-1.5 py-0.5 font-mono text-xs text-zinc-100">
                    {slot}
                  </span>
                  {rankOf.has(slot) ? (
                    <span className="rounded bg-zinc-950/85 px-1.5 py-0.5 text-[10px] text-zinc-300">
                      predicted #{rankOf.get(slot)}
                    </span>
                  ) : null}
                  {cut > 0 ? (
                    <span
                      className={
                        image.promoted
                          ? "rounded bg-emerald-500/90 px-1.5 py-0.5 text-[10px] font-medium text-emerald-950"
                          : "rounded bg-zinc-950/85 px-1.5 py-0.5 text-[10px] text-zinc-400"
                      }
                    >
                      {image.promoted ? "animated" : "not animated"}
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
                          {prose(gate.pending.map((check) => humanise(check.name)))} did not run. A
                          pending check always passes, so it verifies nothing.
                          {gate.identityPending ? (
                            <>
                              {" "}
                              Identity is <strong>not</strong> verified.
                            </>
                          ) : null}
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

      <Lightbox items={shots} index={open} onClose={() => setOpen(null)} onIndex={setOpen} />
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
