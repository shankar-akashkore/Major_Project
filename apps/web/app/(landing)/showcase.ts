/**
 * The showcase manifest, read off disk when the server renders the page.
 *
 * The landing page shows five real generated frames and the two that were promoted
 * to video. That media is built by `scripts/build_showcase.py` from a job that
 * already ran, and it is **gitignored** — a fresh checkout has none of it. So this
 * module answers one question: is there a real run to show, or not?
 *
 * Read here rather than fetched from the browser, and that ordering is the whole
 * point. A client-side fetch would render the abstract treatment, then swap in the
 * real one a moment later — a flash of one design becoming another on the first
 * screen a stranger sees. Deciding on the server means the page is only ever one of
 * the two, and it is decided before a single byte reaches the browser.
 *
 * There is no failure mode where the page is blank. A missing directory, an
 * unreadable file and a manifest from a future version all resolve to `null`, and
 * `null` is a complete design rather than an empty one.
 *
 * **Ordering under `next build`.** The landing page is prerendered, so this runs
 * once at build time rather than per request. Build the showcase first and the app
 * second, or the production bundle bakes in the fallback. `next dev` re-renders on
 * every request, so a demo run from the dev server picks up a rebuilt showcase
 * immediately — which is how the viva will actually be run.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

/** Bumped when the shape below changes, so a stale directory is ignored rather
 * than rendered with missing fields. Must match `MANIFEST_VERSION` in the builder. */
const SUPPORTED_VERSION = 1;

export type ShowcaseCheck = {
  name: string;
  value: number;
  threshold: number;
  passed: boolean;
};

export type ShowcaseCandidate = {
  slot: number;
  /** Public URL of the still, e.g. `/showcase/still-0.webp`. */
  still: string;
  /** The design point in the brief builder's own words. */
  concept: string;
  gate: { verdict: string; checks: ShowcaseCheck[] };
  /** `null` when the candidate was never scored — a gate rejection, or a run that
   * stopped before it got there. Rendered as an absence, never as a zero. */
  score: number | null;
  promoted: boolean;
  /** Present only for promoted candidates. Three of five have no clip, and that
   * absence is the page's central claim rather than missing data. */
  clip: string | null;
  poster: string | null;
  rank: number | null;
  image_stage_rank: number | null;
  explanation: string;
};

export type Showcase = {
  version: number;
  job_id: string;
  product_name: string;
  generated_at: string;
  cost: { images: number; videos: number };
  total_cost_usd: number;
  /** True when the ranking came from the heuristic baseline rather than a trained
   * model — which is every job on disk today. The page must say so wherever it
   * shows a score. */
  scores_are_stub: boolean;
  candidates: ShowcaseCandidate[];
};

/**
 * The showcase, or `null` when there is no run to show.
 *
 * `process.cwd()` is the web app's own directory under `next dev` and `next build`,
 * which is where `public/` lives.
 */
export function loadShowcase(): Showcase | null {
  let raw: string;
  try {
    raw = readFileSync(join(process.cwd(), "public", "showcase", "manifest.json"), "utf8");
  } catch {
    // The ordinary case on a fresh checkout, not an error. Nothing is logged,
    // because a build that prints a warning every time trains people to ignore it.
    return null;
  }

  try {
    const parsed = JSON.parse(raw) as Showcase;
    // A manifest that parses but describes something else is more dangerous than
    // one that does not parse: it renders, with holes.
    if (parsed.version !== SUPPORTED_VERSION) return null;
    if (!Array.isArray(parsed.candidates) || parsed.candidates.length === 0) return null;
    return parsed;
  } catch {
    return null;
  }
}

/**
 * The slots the ranking promoted, in slot order.
 *
 * Pulled out because the replay's geometry needs it as a plain array and because
 * the answer is data, not a rule: a different job promotes different slots, and a
 * page that assumed "the first two" would be lying about this one, where the
 * ranking kept slots 1 and 2.
 */
export function promotedSlots(showcase: Showcase | null, fallback: readonly number[]): number[] {
  if (!showcase) return [...fallback];
  return showcase.candidates.filter((c) => c.promoted).map((c) => c.slot);
}
