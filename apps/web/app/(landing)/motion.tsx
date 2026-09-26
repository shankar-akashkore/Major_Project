"use client";

/**
 * The two pieces of the landing page that need JavaScript, and nothing else.
 *
 * Everything continuous — the progress bar, the hero's recede, the marquee — is a
 * CSS scroll timeline running on the compositor. What is left here is the motion CSS
 * genuinely cannot express: a reveal that must fire *once* and stay fired, and a
 * pinned section that needs to know which step the reader is level with.
 *
 * Both degrade to readable, finished content when this script does not run.
 */

import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";

import {
  ACTS,
  actAt,
  candidateStateAt,
  progressAtOffset,
  spell,
  spendAt,
  stepAtOffset,
  type Act,
  type CandidateState,
  watchScroll,
} from "@/lib/landing.ts";
import { num, usd } from "@/lib/presentation.ts";

/**
 * One observer for every reveal on the page, not one each.
 *
 * A page this long has a few dozen of them, and each `IntersectionObserver` carries
 * its own callback and bookkeeping. Sharing one costs a module-level variable and
 * saves all of that. Targets unobserve themselves on first entry, so the set drains
 * as the reader descends.
 */
let shared: IntersectionObserver | null = null;

function reveal(node: Element) {
  shared ??= new IntersectionObserver(
    (entries, self) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add("is-in");
        self.unobserve(entry.target);
      }
    },
    // Fire a little before the element reaches the bottom edge, so text is already
    // settled by the time it is comfortably readable rather than arriving late.
    { rootMargin: "0px 0px -10% 0px", threshold: 0.15 },
  );
  shared.observe(node);
}

/**
 * Fades and lifts its children into place the first time they are scrolled to.
 *
 * The hidden state is applied by the ref callback, not by the stylesheet, and the
 * ordering is the whole point: ref callbacks run during commit, before the browser
 * paints, so there is no flash of visible-then-hidden content. More importantly, a
 * reader whose JavaScript failed gets a plain, fully visible page — the CSS only
 * hides what this component has already promised to bring back.
 */
export function Reveal({
  children,
  delay = 0,
  className = "",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  const ref = useCallback((node: HTMLDivElement | null) => {
    if (!node || typeof IntersectionObserver === "undefined") return;
    node.dataset.armed = "1";
    reveal(node);
  }, []);

  return (
    <div
      ref={ref}
      className={`home-reveal ${className}`}
      style={delay ? ({ "--d": `${delay}ms` } as CSSProperties) : undefined}
    >
      {children}
    </div>
  );
}

export type Step = { n: string; title: string; body: string; note: string };

/**
 * The pipeline, held still while the reader scrolls through its stages.
 *
 * The section is as tall as its step count, with a sticky child that stays put for
 * the whole descent. Which step is live comes from `stepAtOffset`, measuring the
 * runway's own position in the viewport.
 *
 * An IntersectionObserver over sentinel blocks was the first attempt, and it is the
 * more fashionable answer, but it hinges on `rootMargin: -50% 0 -50%` collapsing the
 * root to a zero-height line — behaviour at the edge of what the spec pins down, and
 * not something this browser could be made to demonstrate either way. Arithmetic on
 * a rect is behaviour I can state, and test, without a running browser at all. The
 * listener is passive and coalesced into one rAF, so a burst of scroll events costs
 * a single measurement.
 *
 * Below 768px the pin is dropped in CSS and the steps become an ordinary list. Eight
 * screens of scroll-jacking on a phone is a way to lose a reader, not to impress one.
 */
export function Pipeline({ steps }: { steps: Step[] }) {
  const root = useRef<HTMLDivElement | null>(null);
  const [active, setActive] = useState(0);

  useEffect(() => {
    const host = root.current;
    if (!host) return;

    return watchScroll(window, () => {
      setActive(stepAtOffset(host.getBoundingClientRect(), window.innerHeight, steps.length));
    });
  }, [steps.length]);

  return (
    <div
      ref={root}
      className="home-runway relative"
      style={{ "--steps": steps.length } as CSSProperties}
    >
      <div className="home-pin">
        <div className="wrap grid w-full gap-8 md:grid-cols-[8rem_1fr] md:gap-16">
          <div className="hidden md:block">
            <p className="eyebrow">Stage</p>
            <p className="num mt-3 tabular-nums">{steps[active]?.n}</p>
            <p className="mt-5 text-[0.6875rem] tracking-wider text-[var(--l-3)] tabular-nums">
              {active + 1} / {steps.length}
            </p>
          </div>

          <ol className="flex flex-col gap-4 md:gap-5">
            {steps.map((step, i) => (
              <li
                key={step.n}
                className="home-step"
                data-active={i === active ? "1" : "0"}
                aria-current={i === active ? "step" : undefined}
              >
                <div className="h-px w-full bg-white/15">
                  <div className="home-step-bar h-px w-full bg-white" />
                </div>

                <div className="flex items-baseline gap-4 pt-3 md:gap-6">
                  <span className="eyebrow step-num tabular-nums">{step.n}</span>
                  <h3 className="flex-1 text-[1.25rem] leading-none md:text-[2rem]">
                    {step.title}
                  </h3>
                  <span className="step-note hidden text-[0.6875rem] text-[var(--l-3)] md:block">
                    {step.note}
                  </span>
                </div>

                {/*
                  Only the live step carries its explanation. Collapsing the rest is
                  what keeps all of them on one screen without shrinking the type to
                  something nobody would read — a pinned section that overflows the
                  viewport strands its own last line permanently out of sight.
                */}
                <div className="home-step-body">
                  <div className="overflow-hidden">
                    <p className="max-w-[52ch] pt-2 text-[0.8125rem] leading-relaxed text-[var(--l-2)] md:pl-[3.75rem]">
                      {step.body}
                    </p>
                  </div>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </div>
  );
}

// --- The replay ---------------------------------------------------------------

export type ReplayAct = { key: Act; n: string; title: string; body: string };

export type ReplayCandidate = {
  slot: number;
  /** Public URL of the generated frame, or `null` when there is no showcase on disk
   * and the card falls back to an abstract plate. */
  still: string | null;
  concept: string;
  checks: readonly { name: string; value: number; threshold: number; passed: boolean }[];
  score: number | null;
  promoted: boolean;
  clip: string | null;
  poster: string | null;
  rank: number | null;
};

/**
 * How finely the scroll position is sampled into React state.
 *
 * Storing the raw float would re-render five cards on every scroll event, several
 * times per painted frame. 400 steps across the whole runway is finer than the
 * spend meter can display and far finer than anyone can follow, and it turns a
 * continuous stream of updates into a few hundred.
 */
const QUANTA = 400;

/** A→E rather than 0→4, matching `slotLetter` in the console. */
const letter = (slot: number) => String.fromCharCode("A".charCodeAt(0) + slot);

/**
 * One job, replayed under the reader's scroll.
 *
 * The section is five screens tall with a sticky child that holds for the whole
 * descent — the same geometry as `Pipeline`, and the same reasoning: the arithmetic
 * lives in `lib/landing.ts` where it can be tested, because scroll behaviour cannot
 * be verified by looking at a page that is not being rendered.
 *
 * What is on screen is derived, never stored. `candidateStateAt` decides what has
 * happened to each candidate and `spendAt` decides what has been spent, both from
 * the single scroll position — so there is no sequence of `setState` calls that can
 * get out of order, and no state in which a candidate has been promoted by one part
 * of the page and not yet scored by another.
 */
export function Replay({
  acts,
  candidates,
  cost,
  stub,
}: {
  acts: readonly ReplayAct[];
  candidates: readonly ReplayCandidate[];
  cost: { images: number; videos: number };
  stub: boolean;
}) {
  const root = useRef<HTMLDivElement | null>(null);
  const pin = useRef<HTMLDivElement | null>(null);
  // Starts finished, not empty. If this script never runs, the reader gets the
  // resolved state — five frames, two promoted — which is a complete picture. The
  // other initial value is act one, which is five empty boxes and no explanation.
  const [progress, setProgress] = useState(1);
  // Starts false, so the first paint — and any reader whose JavaScript never runs —
  // gets the unpinned layout: every act written out, every candidate resolved. The
  // scrubbed version is the enhancement, and it is the one that hides four fifths of
  // the narration behind a gesture.
  const [scrubbing, setScrubbing] = useState(false);

  useEffect(() => {
    const host = root.current;
    const held = pin.current;
    if (!host || !held) return;

    return watchScroll(window, () => {
      const rect = host.getBoundingClientRect();
      const viewport = window.innerHeight;
      // Whether the section is pinned is a question for the layout, not for a
      // breakpoint guessed at from here. Below 768px the stylesheet drops the pin
      // and the whole replay becomes an ordinary block that scrolls past — there is
      // nothing to scrub, so it holds its finished state. Inferring that from the
      // runway's height instead looked equivalent and was not: unpinned, the runway
      // is its content, which on a phone is still taller than the screen, so the
      // scrub would run while the frames scrolled away underneath it.
      const pinned = getComputedStyle(held).position === "sticky" && rect.height > viewport;
      const next = pinned ? progressAtOffset(rect, viewport) : 1;
      setScrubbing(pinned);
      setProgress((prev) =>
        Math.round(prev * QUANTA) === Math.round(next * QUANTA) ? prev : next,
      );
    });
  }, []);

  const promoted = candidates.filter((c) => c.promoted).map((c) => c.slot);
  const { index: act } = actAt(progress, ACTS.length);
  const spend = spendAt(progress, cost);
  const live = acts[act];

  return (
    <div
      ref={root}
      className="home-runway relative"
      style={{ "--steps": ACTS.length } as CSSProperties}
    >
      <div ref={pin} className="home-pin">
        <div className="wrap w-full">
          <div className="flex flex-wrap items-end justify-between gap-x-10 gap-y-4">
            <div>
              <p className="eyebrow">
                {scrubbing
                  ? `${live?.n} · ${live?.title}`
                  : `${spell(ACTS.length)} acts, ${ACTS.length} of ${ACTS.length}`}
              </p>
              <p className="mt-3 max-w-[54ch] text-[0.9375rem] leading-relaxed text-[var(--l-2)] md:text-[1.0625rem]">
                {scrubbing
                  ? live?.body
                  : `${spell(candidates.length)} frames composed and checked, ${spell(promoted.length)} promoted to video. Every act is written out below, because there is no scroll to hold them one at a time here.`}
              </p>
            </div>

            {/*
              The meter is the argument in one number. It moves in act two, stops
              dead through the gate and the ranking — the two stages that decide
              everything and cost nothing — and moves again only for the two clips
              that were earned. The stillness in the middle is the point.
            */}
            <div className="shrink-0 text-right">
              <p className="eyebrow">Spent</p>
              {/*
                Four decimal places, through the console's own `usd`. It looks like
                a lot of digits for a landing page and it is the right number: at
                two, a genuinely free stage and a real fraction-of-a-cent charge
                print identically, and the whole meter exists to tell those apart.
              */}
              <p className="num mt-2 text-[clamp(1.5rem,3.4vw,2.5rem)] tabular-nums">
                {usd(spend)}
              </p>
            </div>
          </div>

          <ol className="home-stage mt-8 md:mt-10">
            {candidates.map((candidate) => (
              <Frame
                key={candidate.slot}
                candidate={candidate}
                state={candidateStateAt(candidate.slot, progress, {
                  count: candidates.length,
                  promoted,
                })}
                act={act}
                stub={stub}
              />
            ))}
          </ol>

          {scrubbing ? (
            <div className="home-acts mt-7 flex">
              {acts.map((step, i) => (
                <span key={step.key} className="home-acts-tick" data-done={i <= act ? "1" : "0"} />
              ))}
            </div>
          ) : (
            // Unpinned, so nothing is holding the acts one at a time. Written out in
            // full rather than collapsed to the last one — which is what reading the
            // live act alone would give a phone, and it is the act that only makes
            // sense after the four before it.
            <ol className="mt-10 grid gap-6 sm:grid-cols-2">
              {acts.map((step) => (
                <li key={step.key}>
                  <p className="eyebrow">
                    {step.n} · {step.title}
                  </p>
                  <p className="mt-2 text-[0.8125rem] leading-relaxed text-[var(--l-2)]">
                    {step.body}
                  </p>
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * One candidate, in whatever state the scroll has reached.
 *
 * The readout under each frame changes with the state rather than accumulating, so
 * at any moment the row is showing one kind of information about all five — five
 * concepts, then five measurements, then five scores. A column that grew as the
 * replay ran would be five different things at once and legible as none of them.
 */
function Frame({
  candidate,
  state,
  act,
  stub,
}: {
  candidate: ReplayCandidate;
  state: CandidateState;
  act: number;
  stub: boolean;
}) {
  // Mounted an act early so it has buffered by the time it is uncovered. Two clips
  // at a few hundred kilobytes, and only for candidates that were actually animated.
  const clip = candidate.clip && act >= 3 ? candidate.clip : null;
  const playing = state === "promoted" && act >= 4;

  return (
    <li className="home-cand" data-state={state}>
      <div className="home-cand-frame">
        {candidate.still ? (
          <img
            src={candidate.still}
            alt={candidate.concept || `Candidate ${letter(candidate.slot)}`}
            className="home-cand-still"
            loading="lazy"
            decoding="async"
          />
        ) : (
          // No showcase on disk. A plate carrying the design point rather than a
          // grey box: the claim being made is that the five briefs differ, and that
          // survives having no pictures to prove it with.
          <span className="home-cand-plate">{candidate.concept}</span>
        )}

        {clip ? (
          <video
            className="home-cand-clip"
            data-playing={playing ? "1" : "0"}
            src={clip}
            poster={candidate.poster ?? undefined}
            autoPlay
            muted
            loop
            playsInline
            preload="metadata"
            aria-hidden
          />
        ) : null}

        <span className="home-cand-slot">{letter(candidate.slot)}</span>
        {state === "promoted" && candidate.rank ? (
          <span className="home-cand-rank">#{candidate.rank}</span>
        ) : null}
      </div>

      <div className="home-cand-read">
        <Readout candidate={candidate} state={state} stub={stub} />
      </div>
    </li>
  );
}

function Readout({
  candidate,
  state,
  stub,
}: {
  candidate: ReplayCandidate;
  state: CandidateState;
  stub: boolean;
}) {
  if (state === "waiting") return <span className="home-cand-idle">—</span>;

  if (state === "composing") {
    return <p className="home-cand-concept">{candidate.concept}</p>;
  }

  if (state === "gated") {
    // Two of the five, because five sets of five numbers is a table nobody reads
    // at a glance. These two are the ones that actually reject candidates.
    const shown = candidate.checks.filter((c) =>
      ["palette_adherence", "safe_area"].includes(c.name),
    );
    return (
      <dl className="home-cand-checks">
        {(shown.length ? shown : candidate.checks.slice(0, 2)).map((check) => (
          <div key={check.name}>
            <dt>{check.name.replace(/_/g, " ")}</dt>
            <dd data-passed={check.passed ? "1" : "0"}>
              {num(check.value)} / {num(check.threshold)}
            </dd>
          </div>
        ))}
      </dl>
    );
  }

  if (candidate.score === null) {
    // An absence, not a zero. The two look identical at a glance and mean opposite
    // things — this candidate was never scored.
    return <span className="home-cand-idle">not scored</span>;
  }

  return (
    <p className="home-cand-score">
      <span className="home-cand-figure">{num(candidate.score)}</span>
      {stub ? <span className="home-cand-stub">placeholder</span> : null}
      {state === "cut" ? <span className="home-cand-verdict">not animated</span> : null}
      {state === "promoted" ? <span className="home-cand-verdict">animated</span> : null}
    </p>
  );
}
