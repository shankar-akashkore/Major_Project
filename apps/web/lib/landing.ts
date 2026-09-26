/**
 * Pure geometry for the landing page's pinned pipeline.
 *
 * This lives in `lib/` rather than beside the component so the project's test runner
 * picks it up — and it is worth pinning, because it is the one piece of the landing
 * page whose behaviour cannot be checked by looking at the page. Which step is live
 * only reveals itself while scrolling, and a browser that is not rendering produces
 * neither frames nor observer callbacks, so "it looked right" was never available as
 * evidence here.
 */

/**
 * Which step a reader is level with, given the runway's position in the viewport.
 *
 * The runway is `count` screens tall with a sticky child one screen high, so the
 * distance actually travelled while the section is pinned is its height less one
 * viewport. `rect.top` runs from 0 at the moment it sticks to `-(travel)` at the
 * moment it lets go; dividing that by `travel` gives progress in 0..1, and the floor
 * of `progress * count` gives the step.
 *
 * Clamped at both ends, so scrolling past in either direction leaves the first or
 * last step showing rather than an index nothing renders for.
 */
export function stepAtOffset(
  rect: { top: number; height: number },
  viewport: number,
  count: number,
): number {
  if (count <= 1) return 0;

  // The floor of the continuous form. Delegating rather than repeating the
  // arithmetic is what keeps the pinned pipeline and the replay reading the same
  // scroll position — two copies of this would drift by a boundary and nobody would
  // see it, because the two sections are never on screen at the same time.
  const index = Math.floor(progressAtOffset(rect, viewport) * count);
  return Math.min(count - 1, Math.max(0, index));
}

// --- The replay ---------------------------------------------------------------

/**
 * The five acts of the replay, in order. Exported because the component, the CSS
 * and the tests all have to agree on how many there are, and a count that lives in
 * three places is a count that will disagree in two of them.
 */
export const ACTS = ["inputs", "compose", "gate", "rank", "animate"] as const;

export type Act = (typeof ACTS)[number];

/**
 * How far a candidate is through the run, as one of six states.
 *
 * A union rather than a set of booleans because the states are genuinely exclusive
 * and the ordering between them is the claim the page is making — a candidate that
 * could be `scored` and `waiting` at once is a bug the type should refuse to hold.
 */
export type CandidateState =
  | "waiting"
  | "composing"
  | "gated"
  | "scored"
  | "promoted"
  | "cut";

/**
 * Where within an act each candidate's beat falls, as a share of that act.
 *
 * Stopping at 0.72 rather than running to 1 leaves a rest at the end of every act:
 * the last frame lands, and then there is a moment where nothing changes before the
 * next act starts. Without it the five arrivals run straight into the next act's
 * first arrival and the sequence reads as one continuous shuffle.
 */
const CUE_SPAN = 0.72;

/**
 * When the rank act resolves promotions — after every candidate has been scored,
 * which `CUE_SPAN` guarantees. This is the beat the whole page is built around, so
 * it happens to all five at once rather than one at a time.
 */
const PROMOTION_CUE = 0.85;

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function cueFor(slot: number, count: number): number {
  return count <= 1 ? 0 : (slot / count) * CUE_SPAN;
}

/**
 * How far the reader is through the runway, in 0..1.
 *
 * The continuous form of `stepAtOffset`, which now floors this. Same geometry:
 * a runway `count` screens tall pins a one-screen child, so the distance travelled
 * while pinned is its height less one viewport.
 */
export function progressAtOffset(rect: { top: number; height: number }, viewport: number): number {
  const travel = rect.height - viewport;
  // A runway shorter than the viewport never pins, so there is no progress to read.
  if (travel <= 0) return 0;
  return clamp01(-rect.top / travel);
}

/**
 * Which act the reader is in, and how far through it.
 *
 * `local` is what drives everything inside an act — the stagger of the five frames,
 * the fill of a meter — so it is returned rather than recomputed by each caller
 * from `progress` and the act count, which is the same arithmetic written five
 * times with five chances to get the boundary wrong.
 */
export function actAt(progress: number, count: number): { index: number; local: number } {
  if (count <= 1) return { index: 0, local: clamp01(progress) };

  const scaled = clamp01(progress) * count;
  // At progress 1 the floor is `count`, one past the last act — the same off-by-one
  // `stepAtOffset` guards, and the reason `local` is derived from the clamped index.
  const index = Math.min(count - 1, Math.floor(scaled));
  return { index, local: Math.min(1, scaled - index) };
}

/**
 * What has happened to one candidate by this point in the replay.
 *
 * `promoted` carries the slots the ranking kept, so the page never hard-codes which
 * two survive — the showcase manifest says, and a different job promotes different
 * slots.
 *
 * The ordering is the page's whole argument and is enforced here rather than in the
 * markup: nothing reaches `promoted` or `cut` before the rank act has scored it,
 * because a page that showed a candidate being dropped before it showed the number
 * that dropped it would be illustrating a decision it had not made yet.
 */
export function candidateStateAt(
  slot: number,
  progress: number,
  { count, promoted }: { count: number; promoted: readonly number[] },
): CandidateState {
  const { index, local } = actAt(progress, ACTS.length);
  const cue = cueFor(slot, count);
  const verdict: CandidateState = promoted.includes(slot) ? "promoted" : "cut";

  switch (index) {
    case 0:
      return "waiting";
    case 1:
      return local >= cue ? "composing" : "waiting";
    case 2:
      return local >= cue ? "gated" : "composing";
    case 3:
      if (local >= PROMOTION_CUE) return verdict;
      return local >= cue ? "scored" : "gated";
    default:
      return verdict;
  }
}

/**
 * The spend meter, in dollars, at this point in the replay.
 *
 * Money accrues where it is actually spent: the images during the compose act, the
 * video during the animate act, and nothing at all in between — which is the shape
 * of the argument. The gap between the two ramps is the gate and the ranking doing
 * their work for free, and it is the most legible thing on the page precisely
 * because the number is not moving.
 */
export function spendAt(progress: number, cost: { images: number; videos: number }): number {
  const { index, local } = actAt(progress, ACTS.length);

  if (index < 1) return 0;
  if (index === 1) return cost.images * local;
  if (index < 4) return cost.images;
  return cost.images + cost.videos * local;
}

// --- Watching the scroll -------------------------------------------------------

/**
 * The part of `window` a scroll watcher touches.
 *
 * Named as an interface so a test can supply one. That is not ceremony: the bug
 * this file exists to prevent is a scroll handler that stops responding, and the
 * only way to demonstrate it is to control whether frames are delivered — which no
 * real browser will let you do, and which this environment demonstrates by
 * accident.
 */
export type ScrollHost = {
  addEventListener(type: string, listener: () => void, options?: { passive: boolean }): void;
  removeEventListener(type: string, listener: () => void): void;
  requestAnimationFrame(callback: () => void): number;
  cancelAnimationFrame(handle: number): void;
  performance: { now(): number };
};

/**
 * How long to wait for a frame that was asked for before giving up on it.
 *
 * `requestAnimationFrame` is a request, not a promise. A page that reports itself
 * visible but is never composited — an offscreen pane, an embedded webview, a
 * window fully occluded on some platforms — queues the callback and never runs it.
 */
export const STALE_FRAME_MS = 250;

/**
 * Call `measure` once immediately, then at most once per painted frame while the
 * reader scrolls or resizes. Returns the teardown.
 *
 * Scroll fires far faster than the screen refreshes, so a burst of events has to
 * collapse into one layout read; otherwise every pinned section on the page forces
 * a reflow several times per frame.
 *
 * The staleness check is the part that is not obvious, and it was found by watching
 * this fail rather than by reading it. Coalescing with a bare `if (queued) return`
 * means that when the promised frame never arrives, `queued` stays set forever and
 * **every later scroll event is dropped** — the pinned section freezes wherever it
 * happened to be, on a page the reader can see and is actively scrolling, and it
 * cannot recover. Waiting `STALE_FRAME_MS` and then measuring anyway costs one
 * comparison per event and removes the failure completely.
 */
export function watchScroll(host: ScrollHost, measure: () => void): () => void {
  let queued = 0;
  let queuedAt = 0;

  const run = () => {
    queued = 0;
    measure();
  };

  const arm = (now: number) => {
    queuedAt = now;
    queued = host.requestAnimationFrame(run);
  };

  const onScroll = () => {
    const now = host.performance.now();
    if (queued) {
      if (now - queuedAt < STALE_FRAME_MS) return;
      // The frame is not coming. Measure without it — and then arm again anyway,
      // which is the half a first attempt at this got wrong: giving up once and
      // waiting for the *next* event to re-arm means a host that never paints
      // measures on every second event instead of on the frame budget, and the
      // reader watches the section update in jerks. Re-arming here keeps the fast
      // path exactly as it was and makes the slow path steady.
      host.cancelAnimationFrame(queued);
      queued = 0;
      measure();
    }
    arm(now);
  };

  measure();
  host.addEventListener("scroll", onScroll, { passive: true });
  host.addEventListener("resize", onScroll, { passive: true });

  return () => {
    host.removeEventListener("scroll", onScroll);
    host.removeEventListener("resize", onScroll);
    if (queued) host.cancelAnimationFrame(queued);
  };
}

/* -------------------------------------------------------------------- prose */

const NUMERALS = [
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
] as const;

/**
 * A small count as a word, for a sentence that also carries a figure.
 *
 * The replay's lede describes one specific run — how many candidates were made,
 * how many were animated — with the money for each read from the manifest. Written
 * as literals, the counts would keep saying "five" and "two" for a showcase built
 * from a job of some other shape, and they would be doing it inside the sentence
 * that claims every number on the page was measured rather than chosen.
 */
export function spell(n: number): string {
  return NUMERALS[n] ?? String(n);
}
