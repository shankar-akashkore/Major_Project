import assert from "node:assert/strict";
import { test } from "node:test";

import {
  actAt,
  candidateStateAt,
  progressAtOffset,
  spell,
  spendAt,
  stepAtOffset,
  STALE_FRAME_MS,
  watchScroll,
  type CandidateState,
} from "./landing.ts";

// One screen per step: eight steps at 900px is a 7200px runway, of which 6300px is
// actually travelled while the section is pinned.
const RUNWAY = { top: 0, height: 7200 };
const VIEWPORT = 900;
const COUNT = 8;

test("the first step is live the moment the section pins", () => {
  assert.equal(stepAtOffset(RUNWAY, VIEWPORT, COUNT), 0);
});

test("the last step is live at the end of the travel", () => {
  assert.equal(stepAtOffset({ ...RUNWAY, top: -6300 }, VIEWPORT, COUNT), 7);
});

test("each step gets an equal share of the travel", () => {
  const share = 6300 / COUNT;
  for (let i = 0; i < COUNT; i++) {
    const middle = -(i * share + share / 2);
    assert.equal(stepAtOffset({ ...RUNWAY, top: middle }, VIEWPORT, COUNT), i);
  }
});

test("the steps advance in order, never skipping or repeating out of sequence", () => {
  const seen: number[] = [];
  for (let top = 0; top >= -6300; top -= 25) {
    const step = stepAtOffset({ ...RUNWAY, top }, VIEWPORT, COUNT);
    if (seen.at(-1) !== step) seen.push(step);
  }
  assert.deepEqual(seen, [0, 1, 2, 3, 4, 5, 6, 7]);
});

test("scrolling above the section holds the first step rather than going negative", () => {
  assert.equal(stepAtOffset({ ...RUNWAY, top: 4000 }, VIEWPORT, COUNT), 0);
});

test("scrolling past the section holds the last step rather than overrunning", () => {
  // The boundary case that a bare floor() gets wrong: at exactly the end, progress is
  // 1.0 and `floor(1 * 8)` is 8 — one past the last step, which renders nothing.
  assert.equal(stepAtOffset({ ...RUNWAY, top: -6300 }, VIEWPORT, COUNT), COUNT - 1);
  assert.equal(stepAtOffset({ ...RUNWAY, top: -20000 }, VIEWPORT, COUNT), COUNT - 1);
});

test("a runway shorter than the viewport reports the first step, not a division by zero", () => {
  assert.equal(stepAtOffset({ top: -50, height: 400 }, VIEWPORT, COUNT), 0);
  assert.equal(stepAtOffset({ top: -50, height: VIEWPORT }, VIEWPORT, COUNT), 0);
});

test("a single step is always the live one", () => {
  assert.equal(stepAtOffset({ ...RUNWAY, top: -3000 }, VIEWPORT, 1), 0);
});

// --- The replay ---------------------------------------------------------------

// Five candidates, of which the ranking kept the first and the third. Deliberately
// not [0, 1]: a page that hard-coded "the top two" would look right against a
// promoted set that happens to be contiguous and wrong against a real one.
const COUNT_C = 5;
const PROMOTED_SLOTS = [0, 2] as const;
const OPTS = { count: COUNT_C, promoted: PROMOTED_SLOTS };

/** Every distinct state a slot passes through, in the order it passes through them. */
function trail(slot: number): CandidateState[] {
  const seen: CandidateState[] = [];
  for (let p = 0; p <= 1.0001; p += 0.001) {
    const state = candidateStateAt(slot, p, OPTS);
    if (seen.at(-1) !== state) seen.push(state);
  }
  return seen;
}

test("progress runs 0 to 1 across the travel and clamps outside it", () => {
  assert.equal(progressAtOffset(RUNWAY, VIEWPORT), 0);
  assert.equal(progressAtOffset({ ...RUNWAY, top: -3150 }, VIEWPORT), 0.5);
  assert.equal(progressAtOffset({ ...RUNWAY, top: -6300 }, VIEWPORT), 1);
  assert.equal(progressAtOffset({ ...RUNWAY, top: 4000 }, VIEWPORT), 0);
  assert.equal(progressAtOffset({ ...RUNWAY, top: -20000 }, VIEWPORT), 1);
  // A runway shorter than the viewport never pins, so there is nothing to read.
  assert.equal(progressAtOffset({ top: -50, height: 400 }, VIEWPORT), 0);
});

test("the step index stays the floor of the progress, everywhere", () => {
  // `stepAtOffset` delegates to `progressAtOffset` now. This is the guard against
  // someone re-inlining the arithmetic: the two sections are never on screen
  // together, so a boundary that drifted by one would go unnoticed by eye.
  for (let top = 200; top >= -7000; top -= 17) {
    const rect = { ...RUNWAY, top };
    const expected = Math.min(COUNT - 1, Math.floor(progressAtOffset(rect, VIEWPORT) * COUNT));
    assert.equal(stepAtOffset(rect, VIEWPORT, COUNT), expected, `top ${top}`);
  }
});

test("acts divide the runway evenly and the last one holds at the end", () => {
  assert.deepEqual(actAt(0, 5), { index: 0, local: 0 });
  assert.deepEqual(actAt(0.2, 5), { index: 1, local: 0 });
  assert.equal(actAt(0.3, 5).index, 1);
  assert.equal(Math.round(actAt(0.3, 5).local * 100), 50);
  // At the very end the floor is 5 — one past the last act, which renders nothing.
  assert.deepEqual(actAt(1, 5), { index: 4, local: 1 });
  assert.deepEqual(actAt(2, 5), { index: 4, local: 1 });
  assert.deepEqual(actAt(-1, 5), { index: 0, local: 0 });
});

test("a promoted candidate passes through every state in order", () => {
  assert.deepEqual(trail(0), ["waiting", "composing", "gated", "scored", "promoted"]);
});

test("a cut candidate passes through the same states and ends cut", () => {
  // Same trail up to the verdict. A candidate that were dimmed early — skipping
  // `scored` — would be a page saying it dropped a frame it had never measured.
  assert.deepEqual(trail(3), ["waiting", "composing", "gated", "scored", "cut"]);
  assert.deepEqual(trail(4), ["waiting", "composing", "gated", "scored", "cut"]);
});

test("the ranking decides who survives, not the slot number", () => {
  assert.equal(candidateStateAt(2, 1, OPTS), "promoted");
  assert.equal(candidateStateAt(1, 1, OPTS), "cut");
});

test("nothing is promoted or cut before the act that scores it", () => {
  // The ordering claim the whole page rests on. Scoring lives in act 4 of 5, so
  // any verdict before 0.6 is the page illustrating a decision it has not made.
  for (let slot = 0; slot < COUNT_C; slot++) {
    for (let p = 0; p < 0.6; p += 0.002) {
      const state = candidateStateAt(slot, p, OPTS);
      assert.notEqual(state, "promoted", `slot ${slot} at ${p.toFixed(3)}`);
      assert.notEqual(state, "cut", `slot ${slot} at ${p.toFixed(3)}`);
    }
  }
});

test("the five frames arrive one at a time rather than together", () => {
  // Mid-compose, some have landed and some have not. Without the stagger the whole
  // act is a single cut and there is nothing to watch.
  // A fifth of the way into the compose act.
  const mid = 0.2 + 0.2 * 0.2;
  const landed = [0, 1, 2, 3, 4].filter((s) => candidateStateAt(s, mid, OPTS) !== "waiting");

  assert.ok(landed.length > 0 && landed.length < COUNT_C, `landed ${landed.length}`);
  assert.deepEqual(landed, [0, 1]);
});

test("spend only moves where money is actually spent", () => {
  const cost = { images: 0.2, videos: 1.4 };

  // Money in floating point: 0.2 + 1.4 lands on 1.5999999999999999. The function
  // returns dollars for a caller that will format them, so the tolerance belongs
  // here rather than a round() inside that would quietly change what it returns.
  const close = (actual: number, expected: number) =>
    assert.ok(Math.abs(actual - expected) < 1e-9, `${actual} is not ${expected}`);

  close(spendAt(0, cost), 0);
  close(spendAt(0.1, cost), 0);
  // Flat across the gate and the rank acts: the two stages that decide everything
  // are the two that cost nothing, and the still meter is how the page says so.
  close(spendAt(0.45, cost), 0.2);
  close(spendAt(0.75, cost), 0.2);
  close(spendAt(1, cost), 1.6);
});

test("spend never goes down", () => {
  const cost = { images: 0.2, videos: 1.4 };
  let previous = -1;
  for (let p = 0; p <= 1.0001; p += 0.001) {
    const now = spendAt(p, cost);
    assert.ok(now >= previous, `spend fell at ${p.toFixed(3)}: ${now} < ${previous}`);
    previous = now;
  }
});

// --- Watching the scroll -------------------------------------------------------

/** A `window` whose frames are delivered only when the test says so. */
function fakeHost() {
  const pending: (() => void)[] = [];
  const listeners: Record<string, (() => void)[]> = {};
  let clock = 0;

  const host = {
    addEventListener(type: string, fn: () => void) {
      (listeners[type] ??= []).push(fn);
    },
    removeEventListener(type: string, fn: () => void) {
      listeners[type] = (listeners[type] ?? []).filter((l) => l !== fn);
    },
    requestAnimationFrame(fn: () => void) {
      pending.push(fn);
      return pending.length;
    },
    cancelAnimationFrame() {},
    performance: { now: () => clock },
  };

  return {
    host,
    /** Deliver every frame the watcher asked for. */
    paint() {
      const due = pending.splice(0);
      for (const fn of due) fn();
    },
    scroll() {
      for (const fn of listeners["scroll"] ?? []) fn();
    },
    advance(ms: number) {
      clock += ms;
    },
    listenerCount: () => (listeners["scroll"] ?? []).length + (listeners["resize"] ?? []).length,
    frameCount: () => pending.length,
  };
}

test("the first measurement happens without waiting for a scroll", () => {
  // The section may already be on screen when the component mounts.
  const fake = fakeHost();
  let measured = 0;
  watchScroll(fake.host, () => measured++);

  assert.equal(measured, 1);
});

test("a burst of scroll events costs one measurement, not one each", () => {
  const fake = fakeHost();
  let measured = 0;
  watchScroll(fake.host, () => measured++);

  for (let i = 0; i < 20; i++) fake.scroll();
  assert.equal(measured, 1, "the burst measured before a frame was painted");

  fake.paint();
  assert.equal(measured, 2);
});

test("a frame that never arrives does not freeze the section forever", () => {
  // The failure this guards, watched happening: with a bare `if (queued) return`,
  // one undelivered frame swallows every scroll event after it and the pinned
  // section stops responding on a page the reader can see and is scrolling.
  const fake = fakeHost();
  let measured = 0;
  watchScroll(fake.host, () => measured++);
  assert.equal(measured, 1);

  fake.scroll();
  assert.equal(measured, 1, "still waiting for the frame, correctly");

  // The frame is never painted. Time passes; the reader keeps scrolling.
  fake.advance(STALE_FRAME_MS + 1);
  fake.scroll();
  assert.equal(measured, 2, "the watcher gave up on the frame and measured anyway");

  fake.advance(STALE_FRAME_MS + 1);
  fake.scroll();
  assert.equal(measured, 3, "and it keeps working, rather than recovering once");
});

test("giving up early would defeat the coalescing, so it does not", () => {
  const fake = fakeHost();
  let measured = 0;
  watchScroll(fake.host, () => measured++);

  fake.scroll();
  fake.advance(STALE_FRAME_MS - 1);
  for (let i = 0; i < 5; i++) fake.scroll();

  assert.equal(measured, 1, "measured during the frame budget it should have waited out");
});

test("teardown removes both listeners", () => {
  const fake = fakeHost();
  const stop = watchScroll(fake.host, () => {});

  assert.equal(fake.listenerCount(), 2);
  stop();
  assert.equal(fake.listenerCount(), 0);
});

// --- Prose ------------------------------------------------------------------

test("counts a reader would read aloud are spelled, and larger ones are not", () => {
  assert.equal(spell(0), "zero");
  assert.equal(spell(2), "two");
  assert.equal(spell(5), "five");
  assert.equal(spell(10), "ten");
  assert.equal(spell(11), "11");
});

test("a count out of range falls back rather than reading undefined", () => {
  // `noUncheckedIndexedAccess` makes this a type error rather than a runtime
  // surprise, but the lede interpolates the result straight into a sentence, so
  // the fallback is what stops "the other undefined" reaching a visitor.
  assert.equal(spell(-1), "-1");
  assert.equal(spell(1.5), "1.5");
});
