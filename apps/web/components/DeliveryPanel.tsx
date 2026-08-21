"use client";

/**
 * Stage 8: the platform renders, what each one cost in content, and the download.
 *
 * Reframing an ad for a platform it was not generated for is lossy, and the loss is
 * a fact about the deliverable that whoever publishes it needs. A 9:16 clip
 * reframed to 16:9 keeps about a third of its height. So each variant reports what
 * it retained, and — where it was measured — how much better than a naive centre
 * crop that was, since an unanchored "82%" means nothing on its own.
 *
 * A padded variant is stated separately. Its retention figure describes what a crop
 * *would* have lost, which is why padding was chosen; printing it as a loss would be
 * reporting damage that did not happen.
 */

import { bundleUrl, mediaUrl } from "@/lib/api.ts";
import type { DeliveryResponse } from "@/lib/contract.ts";
import { humanise, percent, reframeSummary } from "@/lib/presentation.ts";
import { Badge, Card, Caveat, SectionTitle } from "./ui.tsx";

export function DeliveryPanel({
  jobId,
  response,
}: {
  jobId: string;
  response: DeliveryResponse;
}) {
  const delivery = response.delivery;
  const renders = delivery.renders ?? [];
  const previews = Object.entries(delivery.previews ?? {});
  const audio = delivery.audio;

  return (
    <section className="space-y-4">
      <SectionTitle hint={response.summary}>Delivery</SectionTitle>

      {delivery.warnings.length > 0 ? (
        <div className="space-y-1 rounded-md border border-warn-500/30 bg-warn-950 p-3">
          {delivery.warnings.map((warning) => (
            <Caveat key={warning}>{warning}</Caveat>
          ))}
        </div>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="p-4">
          <h3 className="mb-3 text-xs font-semibold tracking-wide text-zinc-400 uppercase">
            Platform renders
          </h3>
          {renders.length === 0 ? (
            <p className="text-[11px] text-zinc-500">No renders were produced.</p>
          ) : (
            <ul className="space-y-2.5">
              {renders.map((render) => {
                const summary = reframeSummary(render);
                return (
                  <li key={summary.ratio} className="space-y-1 border-b border-zinc-800 pb-2.5 last:border-0 last:pb-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono text-xs text-zinc-200">{summary.ratio}</span>
                      <div className="flex items-center gap-1.5">
                        <Badge tone={summary.mode === "none" ? "neutral" : summary.padded ? "info" : "warn"}>
                          {summary.mode}
                        </Badge>
                        <span className="text-[10px] text-zinc-500 tabular">
                          {render.width}×{render.height}
                        </span>
                      </div>
                    </div>

                    {summary.retained === null ? (
                      <p className="text-[11px] text-zinc-500">
                        Generated at this ratio, so nothing was reframed.
                      </p>
                    ) : summary.padded ? (
                      <p className="text-[11px] leading-snug text-zinc-400">
                        Padded, so the whole frame is still visible. A crop would have kept{" "}
                        {percent(summary.retained)} of the salient content, which is why it was not
                        cropped.
                      </p>
                    ) : (
                      <p className="text-[11px] leading-snug text-zinc-400">
                        Kept {percent(summary.retained)} of the salient content
                        {summary.improvement !== null ? (
                          <>
                            {" — "}
                            {summary.improvement >= 0 ? "+" : ""}
                            {percent(summary.improvement)} against a centre crop
                          </>
                        ) : null}
                        .
                      </p>
                    )}

                    {summary.note ? (
                      <p className="text-[10px] leading-snug text-zinc-500">{summary.note}</p>
                    ) : null}
                  </li>
                );
              })}
            </ul>
          )}
        </Card>

        <Card className="p-4">
          <h3 className="mb-3 text-xs font-semibold tracking-wide text-zinc-400 uppercase">
            Audio and download
          </h3>

          {audio.attached ? (
            <dl className="space-y-1 text-[11px] text-zinc-400">
              <Line label="credit">{audio.credit || "—"}</Line>
              <Line label="licence">{audio.licence || "—"}</Line>
              <Line label="loudness">
                {audio.measured_lufs === null ? "—" : `${audio.measured_lufs.toFixed(1)} LUFS`}
                {audio.target_lufs !== null ? ` (target ${audio.target_lufs.toFixed(1)})` : ""}
              </Line>
              <Line label="voiceover">{audio.has_voiceover ? "yes" : "no"}</Line>
            </dl>
          ) : (
            <p className="text-[11px] leading-snug text-zinc-500">
              No audio. The clips are silent until a licensed bed is supplied — an unlicensed one
              cannot ship, so it is refused rather than substituted.
            </p>
          )}

          {audio.is_test_signal ? (
            <div className="mt-2">
              <Caveat>
                The bed is a synthesised tone, not music. It is a development placeholder and must
                not be described as a soundtrack.
              </Caveat>
            </div>
          ) : null}

          {audio.generated && !audio.is_test_signal ? (
            <p className="mt-2 text-[10px] leading-snug text-zinc-500">
              This bed was synthesised for the ad and scored to the mood you chose, so it carries no
              third-party rights and needs no attribution. To use real music instead, drop the track
              plus a JSON file recording its licence into the audio library — a licensed track always
              takes precedence.
            </p>
          ) : null}

          {audio.note ? (
            <p className="mt-2 text-[10px] leading-snug text-zinc-500">{audio.note}</p>
          ) : null}

          <div className="mt-4 border-t border-zinc-800 pt-3">
            {response.download_url ? (
              <a
                href={bundleUrl(jobId)}
                className="inline-flex items-center gap-2 rounded-md bg-zinc-100 px-3 py-2 text-sm font-medium text-zinc-900 transition hover:bg-white"
              >
                Download bundle
              </a>
            ) : (
              <p className="text-[11px] text-zinc-500">No bundle was written for this job.</p>
            )}
            <p className="mt-2 text-[10px] leading-snug text-zinc-500">
              Every render, preview and report card, in one zip.
            </p>
          </div>
        </Card>
      </div>

      {previews.length > 0 ? (
        <Card className="p-4">
          <h3 className="mb-1 text-xs font-semibold tracking-wide text-zinc-400 uppercase">
            Platform previews
          </h3>
          <p className="mb-3 text-[11px] leading-snug text-zinc-500">
            The frame with each platform&apos;s chrome overlaid, so a product sitting under the
            caption bar is visible before publishing rather than after.
          </p>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {previews.map(([platform, asset]) => {
              const url = mediaUrl(asset);
              return (
                <figure key={platform} className="space-y-1.5">
                  <div className="overflow-hidden rounded border border-zinc-800 bg-black">
                    {url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={url} alt={platform} className="w-full" loading="lazy" />
                    ) : null}
                  </div>
                  <figcaption className="text-[10px] text-zinc-500">{humanise(platform)}</figcaption>
                </figure>
              );
            })}
          </div>
        </Card>
      ) : null}
    </section>
  );
}

function Line({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-zinc-500">{label}</dt>
      <dd className="text-right text-zinc-300">{children}</dd>
    </div>
  );
}
