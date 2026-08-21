"use client";

/**
 * One job, from queued to downloadable.
 *
 * The page is assembled from the persisted record rather than from the event
 * stream. The events say what happened; the record is what was saved. Rebuilding
 * the result client-side out of progress messages would put a second, subtly
 * different account of the outcome on screen — and the two would diverge exactly
 * when something went wrong, which is when the screen matters most.
 *
 * The delivery panel is fetched separately, as its own route, because it is the
 * expensive part of the result and it 409s until stage 8 has run. Polling it while
 * the job is still going would mean either a stream of handled 409s or a spinner
 * that outlives the job.
 */

import { use, useEffect, useState } from "react";
import Link from "next/link";

import { DeliveryPanel } from "@/components/DeliveryPanel.tsx";
import { ImageStage } from "@/components/ImageStage.tsx";
import { EventLog, StageProgress } from "@/components/StageProgress.tsx";
import { VideoStage } from "@/components/VideoStage.tsx";
import { Badge, Card, Caveat, ErrorNote, SectionTitle } from "@/components/ui.tsx";
import { getDelivery } from "@/lib/api.ts";
import type { DeliveryResponse, IntakeReport } from "@/lib/contract.ts";
import { useJobStream } from "@/lib/hooks.ts";
import { allStub, humanise, isTerminal, percent, seconds, usd } from "@/lib/presentation.ts";

export default function JobPage({ params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = use(params);
  const { record, events, error, streaming } = useJobStream(jobId);
  const [delivery, setDelivery] = useState<DeliveryResponse | null>(null);

  const state = record?.state ?? "queued";
  const done = record ? isTerminal(record) : false;

  useEffect(() => {
    if (!done || state !== "completed") return;
    let live = true;
    getDelivery(jobId)
      .then((response) => {
        if (live) setDelivery(response);
      })
      .catch(() => {
        // A 409 here means stage 8 did not run, which the job's own events already
        // explain. Nothing to add.
      });
    return () => {
      live = false;
    };
  }, [jobId, done, state]);

  if (error && !record) {
    return (
      <div className="space-y-4">
        <ErrorNote>{error}</ErrorNote>
        <Link href="/jobs" className="text-sm text-zinc-400 hover:underline">
          ← back to jobs
        </Link>
      </div>
    );
  }

  if (!record) {
    return <p className="text-sm text-zinc-500">Loading job {jobId}…</p>;
  }

  const request = record.request;
  const result = record.result;
  const intake = result?.intake ?? null;
  const stubbed = allStub([
    ...(result?.images ?? []).map((image) => image.score),
    ...(result?.videos ?? []).map((video) => video.score),
  ]);

  return (
    <div className="space-y-8">
      <div>
        <Link href="/jobs" className="text-xs text-zinc-500 hover:underline">
          ← jobs
        </Link>
        <div className="mt-2 flex flex-wrap items-baseline justify-between gap-3">
          <h1 className="text-xl font-semibold text-zinc-100">{request.product_name}</h1>
          <span className="font-mono text-xs text-zinc-500">{jobId}</span>
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          <Badge>{humanise(request.platform)}</Badge>
          <Badge>{humanise(request.vertical)}</Badge>
          {/* Shown even when unset, and warned about when it is. A job that told
              the generator nothing about how big the product is will produce a
              product of arbitrary size, and the job record is where that has to
              be visible after the fact. */}
          {request.product_scale ? (
            <Badge title="Physical size stated to the generator.">
              {humanise(request.product_scale)}
            </Badge>
          ) : (
            <Badge tone="warn" title="No explicit size; derived from the vertical, which may imply none.">
              size auto
            </Badge>
          )}
          <Badge>{humanise(request.mood)}</Badge>
          <Badge>{seconds(request.duration_seconds)}</Badge>
          <Badge>seed {request.seed}</Badge>
          {request.locked_angle ? (
            <Badge tone="warn" title="All three candidates were shot at one angle.">
              angle locked
            </Badge>
          ) : null}
          {result ? <Badge tone={result.total_cost_usd > 0 ? "info" : "good"}>{usd(result.total_cost_usd)}</Badge> : null}
        </div>
      </div>

      <Card className="p-4">
        <StageProgress events={events} state={state} streaming={streaming} />
        <div className="mt-3">
          <EventLog events={events} />
        </div>
      </Card>

      {record.error ? <ErrorNote>{record.error}</ErrorNote> : null}
      {record.refusal_reason ? (
        <ErrorNote>Refused over budget — {record.refusal_reason}</ErrorNote>
      ) : null}

      {stubbed ? (
        <Card className="border-warn-500/30 bg-warn-950/40 p-3">
          <Caveat>
            Every score on this page is a placeholder. The ordering is what the pipeline actually
            produced; the numbers are not predictions and are not comparable across jobs. A trained
            head needs pairwise human judgements, which do not exist yet.
          </Caveat>
        </Card>
      ) : null}

      {intake ? <IntakePanel intake={intake} /> : null}

      {result?.briefs ? (
        <section>
          <SectionTitle
            hint={`${result.briefs.sampler} · minimum pairwise distance ${result.briefs.min_pairwise_distance} of 4`}
          >
            Design points
          </SectionTitle>
          <p className="mb-3 max-w-3xl text-[11px] leading-relaxed text-zinc-500">
            Diversity comes from sampling the design space geometrically, not from asking a model for
            three different ideas — which is what makes it measurable and what lets the report ablate
            it. A minimum distance of 0 would mean two candidates share every axis.
          </p>
        </section>
      ) : null}

      {result?.images ? <ImageStage images={result.images} /> : null}
      {result?.ranking ? <VideoStage ranking={result.ranking} /> : null}
      {delivery ? <DeliveryPanel jobId={jobId} response={delivery} /> : null}
    </div>
  );
}

function IntakePanel({ intake }: { intake: IntakeReport }) {
  const advisories = [
    ...intake.advisories,
    ...intake.references.flatMap((reference) =>
      reference.advisories.map((note) => `${reference.role}: ${note}`),
    ),
  ];
  return (
    <section>
      <SectionTitle hint={`palette from ${intake.palette_source}`}>Intake</SectionTitle>
      <Card className="space-y-3 p-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            {intake.palette.map((colour, index) => (
              <span
                key={`${colour}-${index}`}
                title={`${colour}${
                  intake.palette_weights[index] !== undefined
                    ? ` · ${percent(intake.palette_weights[index])} coverage`
                    : ""
                }`}
                className="size-6 rounded border border-zinc-700"
                style={{ backgroundColor: colour }}
              />
            ))}
            {intake.palette.length === 0 ? (
              <span className="text-[11px] text-zinc-500">no palette resolved</span>
            ) : null}
          </div>

          <div className="flex flex-wrap gap-1.5">
            <Badge tone={intake.cutout.accepted ? "good" : "warn"}>
              {intake.cutout.accepted ? "product cut out" : "cutout rejected"}
            </Badge>
            <Badge
              tone={
                !intake.face.implemented ? "neutral" : intake.face.faces_found > 0 ? "good" : "warn"
              }
              title={intake.face.detail}
            >
              {intake.face.implemented
                ? intake.face.faces_found > 0
                  ? `${intake.face.faces_found} face(s) detected`
                  : "no face detected"
                : "face detection unavailable"}
            </Badge>
          </div>
        </div>

        {intake.palette_source === "product-frame" ? (
          <Caveat>
            The palette was read off the whole photograph rather than from inside the product mask,
            so it includes the backdrop — a materially weaker constraint.
          </Caveat>
        ) : null}

        {!intake.cutout.accepted ? (
          <Caveat>
            No product cutout was used ({intake.cutout.method}
            {intake.cutout.detail ? `: ${intake.cutout.detail}` : ""}), so generation saw the
            original photograph rather than a transparent PNG.
          </Caveat>
        ) : null}

        {intake.face.implemented && intake.face.faces_found === 0 ? (
          <Caveat>
            No face was detected. This is an advisory, never a refusal: the Haar cascade&apos;s miss
            rate is not uniform across faces, and a detector with uneven error rates must not decide
            whose photograph is acceptable.
          </Caveat>
        ) : null}

        {advisories.length > 0 ? (
          <ul className="space-y-1 border-t border-zinc-800 pt-2.5">
            {advisories.map((advisory) => (
              <li key={advisory} className="text-[11px] leading-snug text-zinc-400">
                {advisory}
              </li>
            ))}
          </ul>
        ) : null}
      </Card>
    </section>
  );
}
