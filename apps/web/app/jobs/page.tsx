"use client";

/**
 * The board: what has been run, and what can be replayed.
 *
 * The golden set is here rather than tucked away because it is the demo path. A
 * frozen bundle replays the exact frames and clips that were generated once, with
 * no network and no spend, which is what makes a viva demo survive a bad WiFi
 * connection or an empty budget. `bundle.verify()` runs server-side on every
 * listing, so a bundle whose media has gone missing says so here instead of
 * failing when someone clicks it in front of an audience.
 */

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorNote,
  SectionTitle,
  type Tone,
} from "@/components/ui.tsx";
import { listGolden, listJobs, replayGolden } from "@/lib/api.ts";
import { useAsync } from "@/lib/hooks.ts";
import { slotLetter, usd } from "@/lib/presentation.ts";

const STATE_TONE: Record<string, Tone> = {
  completed: "good",
  running: "info",
  queued: "neutral",
  failed: "bad",
  refused_over_budget: "bad",
};

export default function JobsPage() {
  const router = useRouter();
  const jobs = useAsync(listJobs, []);
  const golden = useAsync(listGolden, []);
  const [replaying, setReplaying] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function replay(slug: string) {
    setReplaying(slug);
    setError(null);
    try {
      const launched = await replayGolden(slug);
      router.push(`/jobs/${launched.job_id}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setReplaying(null);
    }
  }

  return (
    <div className="space-y-8">
      <div className="flex items-end justify-between gap-4">
        <h1 className="text-xl font-semibold text-zinc-100">Jobs</h1>
        <Button variant="ghost" onClick={jobs.reload}>
          Refresh
        </Button>
      </div>

      {error ? <ErrorNote>{error}</ErrorNote> : null}

      <section>
        <SectionTitle hint="Frozen output. No network, no spend.">Golden demo set</SectionTitle>
        {golden.error ? <ErrorNote>{golden.error}</ErrorNote> : null}
        {golden.data && golden.data.length === 0 ? (
          <Empty>
            Nothing frozen yet. <span className="font-mono">scripts/freeze_golden.py</span> writes a
            bundle; <span className="font-mono">scripts/replay_golden.py</span> checks it.
          </Empty>
        ) : null}
        <div className="grid gap-3 sm:grid-cols-2">
          {(golden.data ?? []).map((bundle) => (
            <Card key={bundle.slug} className="p-4" as="article">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="text-sm font-medium text-zinc-100">{bundle.title}</h3>
                  <p className="mt-0.5 font-mono text-[11px] text-zinc-500">{bundle.slug}</p>
                </div>
                {bundle.replayable ? (
                  <Badge tone="good">replayable</Badge>
                ) : (
                  <Badge tone="bad" title={bundle.problems.join("; ")}>
                    {bundle.problems.length} problem(s)
                  </Badge>
                )}
              </div>

              <p className="mt-2 text-xs leading-relaxed text-zinc-400">{bundle.summary}</p>

              <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] text-zinc-500">
                <Pair label="frames">{bundle.frames}</Pair>
                <Pair label="clips">{bundle.clips}</Pair>
                <Pair label="image model">{bundle.image_model}</Pair>
                <Pair label="video model">{bundle.video_model}</Pair>
                <Pair label="cost to freeze">{usd(bundle.original_cost_usd)}</Pair>
                <Pair label="expected winner">
                  {bundle.winner_slot === null ? "—" : slotLetter(bundle.winner_slot)}
                </Pair>
              </dl>

              {!bundle.replayable ? (
                <ul className="mt-3 space-y-1 text-[11px] text-bad-400">
                  {bundle.problems.map((problem) => (
                    <li key={problem}>{problem}</li>
                  ))}
                </ul>
              ) : null}

              <div className="mt-4">
                <Button
                  onClick={() => replay(bundle.slug)}
                  disabled={!bundle.replayable || replaying !== null}
                  title={
                    bundle.replayable
                      ? "Runs the real pipeline over the frozen bytes"
                      : "This bundle cannot be replayed until its media is restored"
                  }
                >
                  {replaying === bundle.slug ? "Replaying…" : "Replay"}
                </Button>
              </div>
            </Card>
          ))}
        </div>
      </section>

      <section>
        <SectionTitle hint={jobs.data ? `${jobs.data.length} most recent` : undefined}>
          Recent jobs
        </SectionTitle>
        {jobs.error ? <ErrorNote>{jobs.error}</ErrorNote> : null}
        {jobs.data && jobs.data.length === 0 ? (
          <Empty>No jobs yet. Start one from “New job”.</Empty>
        ) : null}
        {jobs.data && jobs.data.length > 0 ? (
          <Card className="overflow-hidden">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-zinc-800 bg-zinc-900/60 text-[11px] tracking-wide text-zinc-400 uppercase">
                <tr>
                  <th className="px-4 py-2 font-medium">Product</th>
                  <th className="px-4 py-2 font-medium">Platform</th>
                  <th className="px-4 py-2 font-medium">State</th>
                  <th className="px-4 py-2 font-medium">Winner</th>
                  <th className="px-4 py-2 text-right font-medium">Cost</th>
                  <th className="px-4 py-2 font-medium">Started</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800">
                {jobs.data.map((job) => (
                  <tr key={job.job_id} className="transition hover:bg-zinc-900">
                    <td className="px-4 py-2">
                      <Link href={`/jobs/${job.job_id}`} className="hover:underline">
                        <span className="text-zinc-100">{job.product_name}</span>
                        <span className="ml-2 font-mono text-[11px] text-zinc-500">
                          {job.job_id.slice(0, 8)}
                        </span>
                      </Link>
                    </td>
                    <td className="px-4 py-2 text-zinc-400">{job.platform.replace(/_/g, " ")}</td>
                    <td className="px-4 py-2">
                      <Badge tone={STATE_TONE[job.state] ?? "neutral"}>
                        {job.state.replace(/_/g, " ")}
                      </Badge>
                    </td>
                    <td className="px-4 py-2 text-zinc-400">
                      {job.winner === null ? "—" : slotLetter(job.winner)}
                    </td>
                    <td className="px-4 py-2 text-right text-zinc-400 tabular">
                      {usd(job.cost_usd)}
                    </td>
                    <td className="px-4 py-2 text-[11px] text-zinc-500">
                      {new Date(job.created_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        ) : null}
      </section>
    </div>
  );
}

function Pair({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-2">
      <dt>{label}</dt>
      <dd className="text-zinc-300 tabular">{children}</dd>
    </div>
  );
}
