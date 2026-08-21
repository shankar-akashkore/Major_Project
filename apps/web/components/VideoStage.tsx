"use client";

/**
 * Stage 7: the final ranking, with the movement between the two stages made visible.
 *
 * The movement is the project's headline, and this is the only place a person can
 * see it happen on a single job. Two things are said carefully.
 *
 * **"Held" is a result, not a missing value.** A candidate that did not move means
 * the image stage predicted it exactly — which is the finding, if it holds up. A
 * candidate with no image-stage rank at all is a different thing, and says so.
 *
 * **One job is not the measurement.** Three candidates in agreement is an anecdote;
 * the headline is a correlation over many generation sets, which
 * `scripts/stage_agreement.py` withholds until there are enough of them to carry an
 * interval. The note under the strip says so, so a screenshot of one lucky job
 * cannot be read as the result.
 */

import { mediaUrl } from "@/lib/api.ts";
import type { RankedCandidate } from "@/lib/contract.ts";
import { heldOrder, movement, num, seconds, slotLetter, usd } from "@/lib/presentation.ts";
import { OrderStrip, ScoreDetail } from "./Score.tsx";
import { Badge, Card, Caveat, SectionTitle } from "./ui.tsx";

export function VideoStage({ ranking }: { ranking: RankedCandidate[] }) {
  if (ranking.length === 0) return null;

  const ordered = [...ranking].sort((a, b) => a.rank - b.rank);
  const held = heldOrder(ordered);
  const imageOrder = [...ordered]
    .filter((candidate) => candidate.image_stage_rank !== null)
    .sort((a, b) => (a.image_stage_rank ?? 0) - (b.image_stage_rank ?? 0))
    .map((candidate) => slotLetter(candidate.video.source_image_index));
  const videoOrder = ordered.map((candidate) => slotLetter(candidate.video.source_image_index));

  return (
    <section>
      <SectionTitle hint="Ranked after animation">Video stage</SectionTitle>

      <Card className="mb-4 space-y-2 p-3">
        {imageOrder.length > 0 ? <OrderStrip order={imageOrder} label="Image stage" /> : null}
        <OrderStrip order={videoOrder} label="Video stage" />
        <p className="text-[11px] leading-snug text-zinc-500">
          {held === null
            ? "Some candidates have no image-stage rank, so the two orders cannot be compared."
            : held
              ? "The order held: the image stage anticipated the final ranking exactly, on this job."
              : "The order changed: animation reordered the candidates."}{" "}
          One job of {ordered.length} candidates is an illustration, not the measurement — the
          headline is a rank correlation over many generation sets, reported by{" "}
          <span className="font-mono">scripts/stage_agreement.py</span>.
        </p>
      </Card>

      <ol className="space-y-4">
        {ordered.map((candidate) => {
          const move = movement(candidate);
          const video = candidate.video;
          const slot = slotLetter(video.source_image_index);
          // The mixed render, falling back to the native one. `video.asset` is the
          // clip as the provider returned it, which has no audio track at all —
          // playing it directly is why the delivered ads were silent in the browser
          // even after the mix was working and written to storage.
          const url = mediaUrl(video.platform_renders?.audio ?? video.asset);
          const poster = mediaUrl(video.thumbnail) ?? undefined;
          return (
            <Card key={candidate.rank} as="li" className="overflow-hidden">
              <div className="grid gap-4 md:grid-cols-[16rem_1fr]">
                <div className="relative aspect-4/5 bg-black md:aspect-auto">
                  {url ? (
                    <video
                      src={url}
                      poster={poster}
                      controls
                      loop
                      // Deliberately *not* muted. `muted` is what autoplay
                      // policies require, and there is no autoplay here — the user
                      // presses play. All it did was guarantee that a soundtrack
                      // this pipeline paid ffmpeg to produce arrived silent unless
                      // the viewer thought to unmute it.
                      playsInline
                      preload="metadata"
                      className="size-full object-cover"
                    />
                  ) : (
                    <div className="flex size-full items-center justify-center text-[11px] text-zinc-600">
                      no clip
                    </div>
                  )}
                  <span className="absolute top-2 left-2 rounded bg-zinc-950/85 px-1.5 py-0.5 font-mono text-xs text-zinc-100">
                    {slot}
                  </span>
                </div>

                <div className="space-y-3 p-3 md:py-4 md:pr-4 md:pl-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-lg font-semibold text-zinc-100">#{candidate.rank}</span>
                    {candidate.rank === 1 ? <Badge tone="good">winner</Badge> : null}
                    <MovementBadge
                      direction={move.direction}
                      label={move.label}
                      from={candidate.image_stage_rank}
                      to={candidate.rank}
                    />
                  </div>

                  <p className="text-sm leading-relaxed text-zinc-300">{candidate.explanation}</p>

                  <ScoreDetail score={video.score} />

                  <dl className="grid grid-cols-2 gap-x-4 gap-y-1 border-t border-zinc-800 pt-2.5 text-[10px] text-zinc-500 sm:grid-cols-4">
                    <Row label="duration">{seconds(video.duration_seconds)}</Row>
                    <Row label="fps">{video.fps}</Row>
                    <Row label="provider">{video.provider}</Row>
                    <Row label="cost">{usd(video.cost_usd)}</Row>
                  </dl>

                  {video.was_chained ? (
                    <Caveat>
                      Reached {seconds(video.duration_seconds)} by concatenating two clips rather
                      than generating natively. Seam consistency{" "}
                      {video.seam_consistency === null
                        ? "was not measured"
                        : num(video.seam_consistency)}
                      .
                    </Caveat>
                  ) : null}

                  {video.requested_duration_seconds !== null &&
                  video.requested_duration_seconds !== video.duration_seconds ? (
                    <Caveat>
                      {seconds(video.requested_duration_seconds)} was requested;{" "}
                      {seconds(video.duration_seconds)} was delivered and billed — the provider takes
                      duration as a fixed set of values.
                    </Caveat>
                  ) : null}

                  {!video.seed_honoured ? (
                    <Caveat>
                      This provider has no seed parameter, so the clip is not reproducible even
                      though the frame it came from is. An ablation over these is not a controlled
                      comparison.
                    </Caveat>
                  ) : null}
                </div>
              </div>
            </Card>
          );
        })}
      </ol>
    </section>
  );
}

function MovementBadge({
  direction,
  label,
  from,
  to,
}: {
  direction: "up" | "down" | "held" | "unknown";
  label: string;
  from: number | null;
  to: number;
}) {
  const tone = direction === "up" ? "good" : direction === "down" ? "bad" : "neutral";
  const arrow = direction === "up" ? "▲" : direction === "down" ? "▼" : direction === "held" ? "=" : "?";
  const title =
    from === null
      ? "This candidate was not ranked at the image stage."
      : `Image stage #${from} → video stage #${to}`;
  return (
    <Badge tone={tone} title={title}>
      <span aria-hidden>{arrow}</span>
      {label}
      {from !== null ? <span className="opacity-70">(was #{from})</span> : null}
    </Badge>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-2 sm:flex-col sm:justify-start sm:gap-0">
      <dt>{label}</dt>
      <dd className="text-zinc-400 tabular">{children}</dd>
    </div>
  );
}
