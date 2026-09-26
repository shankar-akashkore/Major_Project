/**
 * The landing page, and the root of the project. Not the product — the argument
 * for it, made by replaying a run rather than by describing one.
 *
 * Every claim on this page is one the system actually makes: five candidates, two
 * animations, ranked at both stages, a gate that refuses, a ledger that stops. There
 * are no testimonials, no customer logos and no adoption figures, because this is a
 * final-year project and inventing social proof for it would be the one thing on the
 * page that could not survive a question in the viva.
 *
 * It is black and white with no hue at all, which is a constraint with teeth: with no
 * colour to signal hierarchy, the work falls to type size, rules and space, and
 * anything decorative shows up immediately as noise. The generated frames in the
 * replay are the one exception, and they earn it: they arrive desaturated and are
 * given their colour back at the moment the ranking promotes them, so the only hue
 * on the page is doing a job.
 */

import Link from "next/link";

import { spell } from "@/lib/landing.ts";
import { usd } from "@/lib/presentation.ts";

import { Pipeline, Replay, Reveal, type ReplayAct, type ReplayCandidate, type Step } from "./motion.tsx";
import { loadShowcase, type Showcase } from "./showcase.ts";

/** Custom properties need the cast; doing it once keeps it out of the markup. */
const delay = (ms: number) => ({ "--d": `${ms}ms` }) as React.CSSProperties;

/**
 * The replay's narration, one line per act.
 *
 * Written as captions on something the reader is watching rather than as claims
 * they have to take on trust — which is the difference between this page and the
 * one it replaces. The order is `ACTS`, and the type ties them together.
 */
const ACT_COPY: ReplayAct[] = [
  {
    key: "inputs",
    n: "01",
    title: "Inputs",
    body: "A person, a product, and an attestation that there are rights to use both. The product is cut from its background and its palette extracted, all of it locally and none of it costing anything.",
  },
  {
    key: "compose",
    n: "02",
    title: "Compose",
    body: "Five briefs from five well-separated points in the design space — angle, lighting, background, composition — each with its own fixed seed. Different by construction, not by asking a prompt for variety.",
  },
  {
    key: "gate",
    n: "03",
    title: "Gate",
    body: "Every frame measured against a threshold: palette drift, safe-area compliance, focal clarity, exposure, contrast, product scale. A hard pass or fail, with the number that decided it.",
  },
  {
    key: "rank",
    n: "04",
    title: "Rank",
    body: "Scored and ordered while they are still images. This is the decision the whole project exists to test, and it is taken at the one moment when being wrong costs nothing.",
  },
  {
    key: "animate",
    n: "05",
    title: "Animate",
    body: "Only what earned it becomes video. Three frames were generated, measured and then not paid for — which is the saving, stated as a thing you watched happen.",
  },
];

/**
 * The design points a job samples, in the sampler's own words.
 *
 * Shown when there is no showcase media on disk, which is every fresh checkout.
 * The claim being made in that act is that the five briefs genuinely differ, and
 * that survives having no pictures to prove it with — so the fallback carries the
 * real vocabulary rather than five grey rectangles.
 */
const FALLBACK_CONCEPTS = [
  "Tight close-up with the camera moved near the product, soft diffused studio lighting.",
  "Shot at eye level with bright high-key lighting, subject on the right third.",
  "Three-quarter view with rim backlighting separating the subject from the background.",
  "Side profile with hard directional key light, subject on the left third.",
  "Slightly high angle looking down, warm golden-hour light across the set.",
];

/** Which slots the illustration promotes when there is no run to read it from. */
const FALLBACK_PROMOTED = [1, 2];

/**
 * The replay's candidates: a real run when one has been built, the shape of one
 * otherwise.
 *
 * The fallback carries no scores at all rather than plausible-looking ones. A
 * number invented for a diagram is indistinguishable from a measurement once it is
 * on screen, and this page's whole argument is that the numbers on it are real.
 */
function toCandidates(showcase: Showcase | null): ReplayCandidate[] {
  if (showcase) {
    return showcase.candidates.map((c) => ({
      slot: c.slot,
      still: c.still,
      concept: c.concept,
      checks: c.gate.checks,
      score: c.score,
      promoted: c.promoted,
      clip: c.clip,
      poster: c.poster,
      rank: c.rank,
    }));
  }

  return FALLBACK_CONCEPTS.map((concept, slot) => ({
    slot,
    still: null,
    concept,
    checks: [],
    score: null,
    promoted: FALLBACK_PROMOTED.includes(slot),
    clip: null,
    poster: null,
    rank: FALLBACK_PROMOTED.indexOf(slot) + 1 || null,
  }));
}

const STEPS: Step[] = [
  {
    n: "01",
    title: "Intake",
    note: "local, free",
    body: "The product is cut from its background, its palette extracted, the model image checked for a person. Consent and rights are attested here or the job never starts.",
  },
  {
    n: "02",
    title: "Brief",
    note: "sampler, not vibes",
    body: "A design-space sampler picks five well-separated points across angle, lighting, background and composition. Diversity is measured and ablatable, rather than asked for in a prompt.",
  },
  {
    n: "03",
    title: "Compose",
    note: "five candidates",
    body: "Five reference-conditioned generations place this person with this product, each from its own brief and its own fixed seed, so any run can be reproduced exactly.",
  },
  {
    n: "04",
    title: "Gate",
    note: "hard pass or fail",
    body: "Palette drift, safe-area compliance, focal clarity, exposure, contrast and product scale are measured, not assumed. A failure buys one stricter retry, then the slot is rejected.",
  },
  {
    n: "05",
    title: "Rank the stills",
    note: "before any spend",
    body: "All five frames are scored and ordered while they are still images. This is the decision the project exists to test — and the point where the cost of being wrong is zero.",
  },
  {
    n: "06",
    title: "Animate",
    note: "8–10 seconds",
    body: "The frames that earn it become video: native long-form image-to-video, with a motion brief drawn from the same design point that produced the still.",
  },
  {
    n: "07",
    title: "Rank again",
    note: "the finished film",
    body: "Temporal consistency, first-second hook strength, on-screen product time and motion energy produce the final order — and a measurement of how well the still-stage ranking predicted it.",
  },
  {
    n: "08",
    title: "Deliver",
    note: "platform-ready",
    body: "Saliency-aware crops for Reels, Feed and YouTube, an audio bed mixed locally, and a report card per candidate explaining the placement in plain language.",
  },
];

const FEATURES = [
  {
    title: "A gate that refuses",
    body: "Quality control and performance prediction are separate components on purpose. One is a hard pass or fail with a measured threshold; the other is a learned score. Conflating them is how these systems become unfalsifiable.",
    icon: (
      <>
        <path d="M4 12h6l2-5 3 10 2-5h3" />
      </>
    ),
    wide: true,
  },
  {
    title: "A ledger that stops",
    body: "Every provider call passes a cost governor reading a spend ledger. Over the per-job cap or the global budget, it refuses loudly rather than degrading quietly.",
    icon: (
      <>
        <rect x="3" y="6" width="18" height="12" rx="2" />
        <path d="M3 10h18" />
      </>
    ),
  },
  {
    title: "Mock, replay, live",
    body: "Three modes behind one interface. Development runs on procedural placeholders, demos replay a frozen bundle, and only live mode can spend. The mode is stamped on every screen.",
    icon: (
      <>
        <circle cx="12" cy="12" r="8" />
        <path d="M12 4v16" />
      </>
    ),
  },
  {
    title: "Reasons, not just numbers",
    body: "A rank nobody can interrogate is a rank nobody should trust. Each placement carries the checks that ran, the checks that did not, and what each one measured.",
    icon: (
      <>
        <path d="M5 5h14M5 10h14M5 15h9" />
      </>
    ),
  },
  {
    title: "Reproducible by seed",
    body: "Same inputs and same seed, same ranking. It is what makes an ablation an experiment instead of an anecdote, and it is covered by a determinism test.",
    icon: (
      <>
        <path d="M20 12a8 8 0 1 1-2.3-5.6" />
        <path d="M20 4v5h-5" />
      </>
    ),
  },
  {
    title: "Nothing pretends to be finished",
    body: "A check whose model has not landed reports as pending and says so, because a pending check always passes and therefore verifies nothing. Placeholder scores are marked wherever they appear.",
    icon: (
      <>
        <circle cx="12" cy="12" r="8" />
        <path d="M12 8v5M12 16h.01" />
      </>
    ),
    wide: true,
  },
];

const GUARDRAILS = [
  ["Consent is mandatory", "A job carrying a person's likeness is refused outright without both attestations. Not a checkbox in a settings page — a validation error."],
  ["No public figures", "The intake gate declines recognisable public-figure likenesses before a single generation is requested."],
  ["Provenance is on the page", "Mock, replayed and live output look identical in a screenshot. So the mode is written on every screen that shows a frame."],
  ["Identity checks stay advisory", "Face detection misses unevenly across skin tones and poses. A miss is recorded as an advisory and never as grounds for refusing someone's photograph."],
];

/**
 * Where each still-stage rank ended up once the film was scored. An illustration of
 * the shape of the result, not a measurement — the real figures come out of the
 * evaluation harness, and none of them are quoted on a marketing page.
 *
 * Typed as tuples rather than `number[][]` so the destructure below is checked.
 */
/** How many of the five stills are promoted to video. Matches the shape of a job. */
const PROMOTED = 2;

const RANK_PAIRS: ReadonlyArray<readonly [number, number]> = [
  [0, 0],
  [1, 2],
  [2, 1],
  [3, 3],
  [4, 4],
];

/**
 * The page's sections, numbered once.
 *
 * The gutter numbers and the header's table of contents read from this object, so
 * the two cannot drift apart about what section 04 is — which is the failure mode of
 * every hand-numbered page that has ever had a section inserted into the middle of it.
 */
const SECTIONS = {
  replay: { n: "01", label: "One job, replayed", short: "The replay" },
  shape: { n: "02", label: "The shape of a job", short: "Shape" },
  pipeline: { n: "03", label: "Eight stages", short: "Pipeline" },
  claim: { n: "04", label: "The research claim", short: "The claim" },
  built: { n: "05", label: "What is actually built", short: "Built" },
  guardrails: { n: "06", label: "Guardrails", short: "Guardrails" },
} as const;

type SectionKey = keyof typeof SECTIONS;

/** The anchors the header offers. Deliberately not all six — a nav is not an index. */
const NAV: readonly SectionKey[] = ["replay", "pipeline", "claim", "guardrails"];

/**
 * A section's number and running label, hung in the left gutter.
 *
 * The number is sticky, so it rides down the margin for as long as its section is on
 * screen and is swapped for the next one at the boundary. It is a reading-position
 * indicator that cannot get out of step with the page, because it is not tracking the
 * page — it is part of it.
 */
function Spine({ id, children }: { id: SectionKey; children: React.ReactNode }) {
  const { n, label } = SECTIONS[id];
  return (
    <div className="home-spine">
      <div className="home-spine-rail">
        <p className="home-spine-n">
          <b>{n}</b>
          <span>{label}</span>
        </p>
      </div>
      <div className="min-w-0">{children}</div>
    </div>
  );
}

/** The hairline that opens a section, drawn in by the scroll. Decorative throughout. */
function Sweep() {
  return <div className="home-sweep" aria-hidden />;
}

export default function HomePage() {
  // Read here, on the server, so the page is only ever one of its two designs. A
  // client fetch would paint the abstract treatment and then swap the real one in,
  // which is a flash of one page becoming another on the first screen a stranger
  // sees. See `showcase.ts`.
  const showcase = loadShowcase();
  const candidates = toCandidates(showcase);
  // Counted off the candidates rather than off the showcase, so the ledger and the
  // replay below it are describing the same five things. With a run on disk these are
  // that run's figures; without one they are the shape the illustration is drawn in.
  const promotedCount = candidates.filter((c) => c.promoted).length;
  const cutCount = candidates.length - promotedCount;

  return (
    <div className="home home-grain">
      <div className="home-progress" aria-hidden />

      {/* ------------------------------------------------------------- header */}
      <header className="fixed inset-x-0 top-0 z-40 border-b border-[var(--rule-soft)] bg-black/70 backdrop-blur-xl">
        <div className="wrap flex h-16 items-center justify-between gap-6">
          <Link href="/" className="inline-flex min-h-11 items-center gap-2.5" aria-label="AdGen home">
            <svg viewBox="0 0 24 24" className="size-4" aria-hidden>
              <rect x="1.5" y="4" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.5" />
              <path d="M10 8.5 20.5 14 10 19.5Z" fill="currentColor" />
            </svg>
            <span className="text-[0.8125rem] font-medium tracking-[0.2em]">ADGEN</span>
          </Link>

          {/*
            The nav is the same index as the gutter, so the numbers are load-bearing
            rather than ornamental: a reader who has seen `04` beside a heading can
            find their way back to it from up here.
          */}
          <nav className="hidden items-center gap-7 md:flex">
            {NAV.map((key) => (
              <a
                key={key}
                href={`#${key}`}
                className="home-underline inline-flex min-h-11 items-center gap-2 text-[0.75rem] text-[var(--l-2)] hover:text-white"
              >
                <span className="tabular text-[0.625rem] tracking-[0.18em] text-[var(--l-3)]">
                  {SECTIONS[key].n}
                </span>
                {SECTIONS[key].short}
              </a>
            ))}
          </nav>

          <Link
            href="/console"
            className="inline-flex min-h-11 items-center border border-white bg-white px-4 text-[0.75rem] font-medium text-black transition-colors hover:bg-transparent hover:text-white"
          >
            Open the console
          </Link>
        </div>
      </header>

      {/* ----------------------------------------------------------- masthead */}
      <section className="home-recede relative flex min-h-svh flex-col justify-center pt-20 pb-10">
        <div className="wrap">
          {/*
            A dateline, in the newspaper sense: what this is, in its own vocabulary,
            before the headline gets to be short about it.
          */}
          <div
            className="home-fade home-dateline flex flex-wrap items-center gap-x-5 gap-y-2 border-y border-[var(--rule)] py-2.5"
            style={delay(100)}
          >
            <span>Multi-candidate multimodal advertisement generation</span>
            <span aria-hidden className="hidden h-3 w-px bg-[var(--rule)] sm:block" />
            <span className="hidden sm:inline">Learning-based performance prediction</span>
            <span className="text-[var(--l-2)] md:ml-auto">Final-year major project</span>
          </div>

          {/*
            Each line sits on its own rule, so the three of them stack into one plate.
            At 9.5rem, type floating in open space has nothing to be measured against
            and reads as a template; anchored to hairlines it reads as a nameplate.
          */}
          <h1 className="shout mt-8">
            <span className="home-mast-line block pb-[0.08em]">
              <span className="home-line">
                <span style={delay(240)}>Make five.</span>
              </span>
            </span>
            <span className="home-mast-line block py-[0.08em]">
              <span className="home-line">
                <span style={delay(360)}>Rank them</span>
              </span>
            </span>
            <span className="block pt-[0.08em]">
              <span className="home-line">
                <span className="italic" style={delay(480)}>
                  before you pay.
                </span>
              </span>
            </span>
          </h1>

          <div className="mt-10 grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,19rem)] lg:gap-14">
            <div>
              <p className="lede home-fade" style={delay(820)}>
                Upload a model and a product. The system composes five ad frames from five
                deliberately different briefs, measures each one, and predicts which will
                perform — then animates only what earns it, and ranks the finished films again.
              </p>

              <div className="home-fade mt-8 flex flex-wrap items-center gap-3" style={delay(940)}>
                <Link
                  href="/console"
                  className="inline-flex min-h-11 items-center border border-white bg-white px-7 py-3.5 text-[0.8125rem] font-medium text-black transition-colors hover:bg-transparent hover:text-white"
                >
                  Open the console
                </Link>
                <a
                  href="#pipeline"
                  className="inline-flex min-h-11 items-center border border-[var(--rule)] px-7 py-3.5 text-[0.8125rem] text-[var(--l-2)] transition-colors hover:border-white hover:text-white"
                >
                  See how it ranks
                </a>
              </div>
            </div>

            {/*
              The ledger is the one panel on the page that keeps a border, now that the
              cards have given theirs up — it is a block of readings rather than a
              passage of prose, which is the only thing a box should still mean here.

              With no run on disk it carries counts and no money at all. A plausible
              figure in a ledger is indistinguishable from a measured one once it is on
              screen, and this page's entire argument is that its numbers are real.
            */}
            <aside className="home-fade home-ledger p-5" style={delay(1040)}>
              <p className="home-dateline border-b border-[var(--rule)] pb-3.5">
                {showcase ? "Last run, settled" : "The shape of a run"}
              </p>

              <dl className="mt-4">
                <dt className="text-[var(--l-2)]">Candidates composed</dt>
                <dd className="text-[var(--ink)]">{candidates.length}</dd>

                <dt className="text-[var(--l-2)]">Promoted to video</dt>
                <dd className="text-[var(--ink)]">{promotedCount}</dd>

                <dt className="text-[var(--l-2)]">Measured, then cut</dt>
                <dd className="text-[var(--ink)]">{cutCount}</dd>

                {showcase ? (
                  <>
                    <dt className="mt-1 border-t border-[var(--rule)] pt-3 text-[var(--l-2)]">Stills</dt>
                    <dd className="mt-1 border-t border-[var(--rule)] pt-3 text-[var(--ink)]">
                      {usd(showcase.cost.images)}
                    </dd>

                    <dt className="text-[var(--l-2)]">Animation</dt>
                    <dd className="text-[var(--ink)]">{usd(showcase.cost.videos)}</dd>

                    {/* Ruled off above, so the last row reads as the sum of the two over it. */}
                    <dt className="mt-1 border-t border-[var(--rule)] pt-3 text-[var(--l-1)]">Settled</dt>
                    <dd className="mt-1 border-t border-[var(--rule)] pt-3 text-[var(--ink)]">
                      {usd(showcase.total_cost_usd)}
                    </dd>
                  </>
                ) : null}
              </dl>

              {!showcase ? (
                <p className="mt-5 border-t border-[var(--rule)] pt-4 text-[0.6875rem] leading-relaxed text-[var(--l-3)]">
                  No showcase media in this checkout, so there are no figures to quote and
                  none have been invented.
                </p>
              ) : null}
            </aside>
          </div>

          <div
            className="home-fade mt-8 flex items-center gap-4 text-[0.6875rem] tracking-[0.18em] text-[var(--l-3)] uppercase"
            style={delay(1160)}
          >
            <span className="h-px w-14 bg-[var(--rule)]" />
            Scroll
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------- ticker */}
      <div className="home-ticker overflow-hidden border-y border-[var(--rule)] py-4">
        <div className="home-marquee">
          {[0, 1].map((copy) => (
            <div key={copy} className="flex items-center" aria-hidden={copy === 1}>
              {[
                "Image candidates",
                "Design-space sampler",
                "Quality gate",
                "Performance prediction",
                "Intelligent ranking",
                "8–10s video",
                "Platform delivery",
              ].map((word) => (
                <span
                  key={word}
                  className="flex items-center gap-8 px-8 text-[0.75rem] tracking-[0.2em] whitespace-nowrap text-[var(--l-2)] uppercase"
                >
                  {word}
                  <span className="size-1 rotate-45 bg-[var(--l-4)]" />
                </span>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* -------------------------------------------------------------- replay */}
      <section id="replay" className="scroll-mt-16">
        <div className="wrap pt-[var(--gap)] pb-10">
          <Spine id="replay">
            <Reveal>
              <h2 className="title">Watch the money not get spent.</h2>
            </Reveal>

            <Reveal delay={120}>
              {showcase ? (
                <>
                  <p className="lede mt-8">
                    Every frame below came out of one real run, and every number on it was
                    measured rather than chosen. It composed and checked{" "}
                    {spell(showcase.candidates.length)} candidates for {usd(showcase.cost.images)},
                    animated the {spell(promotedCount)} that earned it for{" "}
                    {usd(showcase.cost.videos)}
                    {cutCount > 0 ? `, and never paid for the other ${spell(cutCount)}` : ""}.
                  </p>
                  <p className="mt-6 text-[0.75rem] leading-relaxed text-[var(--l-3)]">
                    {showcase.product_name} — job {showcase.job_id.slice(0, 8)},{" "}
                    {showcase.generated_at}, {usd(showcase.total_cost_usd)} settled.
                    {showcase.scores_are_stub
                      ? " Scores come from the heuristic baseline, not a trained model, and are marked as placeholders wherever they appear."
                      : ""}
                  </p>
                </>
              ) : (
                <>
                  <p className="lede mt-8">
                    Five candidates are composed and checked; two are promoted to video. The
                    other three are generated, measured, and then not paid for — which is where
                    the saving comes from.
                  </p>
                  <p className="mt-6 max-w-[58ch] text-[0.75rem] leading-relaxed text-[var(--l-3)]">
                    This is the shape of a run rather than a run: no showcase media is present in
                    this checkout, so the frames and their measurements are absent rather than
                    invented. <code>scripts/build_showcase.py --job &lt;id&gt;</code> replays a real
                    one.
                  </p>
                </>
              )}
            </Reveal>
          </Spine>
        </div>

        <Replay
          acts={ACT_COPY}
          candidates={candidates}
          cost={showcase?.cost ?? { images: 0.2, videos: 1.4 }}
          stub={showcase?.scores_are_stub ?? false}
        />
      </section>

      {/* ------------------------------------------------------------ numbers */}
      <Sweep />
      <section className="wrap py-[var(--gap)]">
        <Spine id="shape">
          <Reveal>
            <h2 className="title max-w-[20ch]">
              Video is the expensive part. So the decision happens before it.
            </h2>
          </Reveal>

          <div className="mt-14 grid gap-x-8 gap-y-12 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ["5", "image candidates", "one per sampled design point"],
              ["2", "animated", "the ones the still-stage ranking promoted"],
              ["8–10s", "each", "native duration, no chained seams"],
              ["×2", "rankings", "stills first, finished film second"],
            ].map(([big, label, note], i) => (
              <Reveal key={label} delay={i * 90}>
                <div className="home-stat h-full">
                  <p className="num whitespace-nowrap">{big}</p>
                  <p className="mt-5 text-[0.8125rem] text-[var(--l-1)]">{label}</p>
                  <p className="mt-1.5 text-[0.75rem] leading-relaxed text-[var(--l-3)]">{note}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </Spine>
      </section>

      {/* ----------------------------------------------------------- pipeline */}
      <Sweep />
      <section id="pipeline" className="scroll-mt-16">
        <div className="wrap pt-[var(--gap)] pb-10 md:pb-0">
          <Spine id="pipeline">
            <Reveal>
              <h2 className="title max-w-[16ch]">Upload to platform-ready, without a black box.</h2>
            </Reveal>
          </Spine>
        </div>

        <Pipeline steps={STEPS} />
      </section>

      {/* -------------------------------------------------------------- claim */}
      <Sweep />
      <section id="claim" className="scroll-mt-16">
        <div className="wrap py-[var(--gap)]">
          <Spine id="claim">
            <div className="grid gap-14 lg:grid-cols-[1fr_1fr] lg:gap-20">
              <div>
                <Reveal>
                  <h2 className="title max-w-none">
                    Does the still-stage ranking survive contact with the video?
                  </h2>
                </Reveal>

                <Reveal delay={120}>
                  <p className="lede mt-8">
                    Every job produces a paired observation: the order predicted from five still
                    frames, and the order measured on the finished films. The gap between them is
                    the number this project stands on — reported with a confidence interval,
                    against random, aesthetic-only and CLIP-only baselines.
                  </p>
                </Reveal>

                <Reveal delay={200}>
                  <p className="mt-6 max-w-[46ch] text-[0.8125rem] leading-relaxed text-[var(--l-3)]">
                    It reports the counterfactual too. Had only the top-ranked still been
                    promoted, how often would the human-preferred film have survived? That is the
                    saving stated honestly, rather than a cut nobody had to make.
                  </p>
                </Reveal>
              </div>

              {/*
                The diagram is decorative in the strict sense — it illustrates a claim
                stated in full in the prose beside it — so it is hidden from assistive
                technology rather than given a description that would only repeat text
                the reader has already been given.
              */}
              <Reveal delay={160}>
                <figure className="border border-[var(--rule)] p-8">
                  <div className="flex items-baseline justify-between text-[0.6875rem] tracking-[0.18em] text-[var(--l-3)] uppercase">
                    <span>Still rank</span>
                    <span>Film rank</span>
                  </div>

                  <svg viewBox="0 0 320 260" className="mt-6 w-full" aria-hidden>
                    {RANK_PAIRS.map(([from, to]) => (
                      <path
                        key={from}
                        d={`M64 ${26 + from * 52} C 150 ${26 + from * 52}, 170 ${26 + to * 52}, 256 ${26 + to * 52}`}
                        fill="none"
                        stroke="#fff"
                        strokeOpacity={from === to ? 0.5 : 0.22}
                        strokeWidth={from === to ? 1.25 : 1}
                        strokeDasharray={from === to ? undefined : "3 4"}
                      />
                    ))}
                    {[0, 1, 2, 3, 4].map((i) => (
                      <g key={`l${i}`}>
                        <rect
                          x="18"
                          y={14 + i * 52}
                          width="46"
                          height="24"
                          fill={i < PROMOTED ? "#fff" : "none"}
                          stroke="#fff"
                          strokeOpacity="0.35"
                        />
                        <text
                          x="41"
                          y={30 + i * 52}
                          fill={i < PROMOTED ? "#000" : "#fff"}
                          fontSize="11"
                          fontFamily="monospace"
                          textAnchor="middle"
                        >
                          {String.fromCharCode(65 + i)}
                        </text>
                      </g>
                    ))}
                    {[0, 1, 2, 3, 4].map((i) => (
                      <g key={`r${i}`}>
                        <rect x="256" y={14 + i * 52} width="46" height="24" fill="none" stroke="#fff" strokeOpacity="0.35" />
                        <text x="279" y={30 + i * 52} fill="#fff" fontSize="11" fontFamily="monospace" textAnchor="middle">
                          {i + 1}
                        </text>
                      </g>
                    ))}
                  </svg>

                  <figcaption className="mt-6 border-t border-[var(--rule)] pt-5 text-[0.75rem] leading-relaxed text-[var(--l-3)]">
                    Solid lines are candidates the still-stage ranking placed correctly; dashed
                    lines are the ones the finished film moved. The filled cells are the two the
                    still-stage ranking promoted — chosen before any video existed, which is the
                    whole point.
                  </figcaption>
                </figure>
              </Reveal>
            </div>
          </Spine>
        </div>
      </section>

      {/* ----------------------------------------------------------- features */}
      <Sweep />
      <section>
        <div className="wrap py-[var(--gap)]">
          <Spine id="built">
            <Reveal>
              <h2 className="title">The engineering the claim rests on.</h2>
            </Reveal>

            <div className="mt-14 grid gap-x-8 gap-y-12 md:grid-cols-2 lg:grid-cols-3">
              {FEATURES.map((feature, i) => (
                <Reveal
                  key={feature.title}
                  delay={(i % 3) * 90}
                  className={feature.wide ? "lg:col-span-2" : undefined}
                >
                  <article className="home-card flex h-full flex-col">
                    <svg
                      viewBox="0 0 24 24"
                      className="size-5 text-white"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.25"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden
                    >
                      {feature.icon}
                    </svg>
                    <h3 className="mt-6 text-[1.375rem]">{feature.title}</h3>
                    <p className="mt-3 max-w-[42ch] text-[0.8125rem] leading-relaxed text-[var(--l-2)]">
                      {feature.body}
                    </p>
                  </article>
                </Reveal>
              ))}
            </div>
          </Spine>
        </div>
      </section>

      {/* ---------------------------------------------------------- guardrails */}
      <Sweep />
      <section id="guardrails" className="scroll-mt-16">
        <div className="wrap py-[var(--gap)]">
          <Spine id="guardrails">
            <Reveal>
              <h2 className="title max-w-[24ch]">
                A system that puts a real person beside a product owes them something.
              </h2>
            </Reveal>

            <div className="mt-14">
              {GUARDRAILS.map(([title, body], i) => (
                <Reveal key={title} delay={i * 80}>
                  <div className="grid gap-2 border-t border-[var(--rule)] py-7 md:grid-cols-[minmax(0,17rem)_1fr] md:gap-10">
                    <h3 className="text-[1.0625rem] tracking-tight">{title}</h3>
                    <p className="max-w-[62ch] text-[0.8125rem] leading-relaxed text-[var(--l-2)]">
                      {body}
                    </p>
                  </div>
                </Reveal>
              ))}
            </div>
          </Spine>
        </div>
      </section>

      {/* ----------------------------------------------------------------- cta */}
      <Sweep />
      <section>
        <div className="wrap py-[calc(var(--gap)*1.15)] text-center">
          <Reveal>
            <h2 className="shout">
              <span className="home-mast-line mx-auto block w-fit pb-[0.08em]">Five candidates.</span>
              <span className="mx-auto block w-fit pt-[0.08em] italic">One answer.</span>
            </h2>
          </Reveal>

          <Reveal delay={140}>
            <p className="lede mx-auto mt-10 text-center">
              The console runs on procedural placeholders by default. You can walk the whole
              pipeline end to end without a provider key and without spending anything.
            </p>
          </Reveal>

          <Reveal delay={220}>
            <div className="mt-12 flex flex-wrap justify-center gap-3">
              <Link
                href="/console"
                className="inline-flex min-h-11 items-center border border-white bg-white px-8 py-4 text-[0.8125rem] font-medium text-black transition-colors hover:bg-transparent hover:text-white"
              >
                Start a job
              </Link>
              <Link
                href="/jobs"
                className="inline-flex min-h-11 items-center border border-[var(--rule)] px-8 py-4 text-[0.8125rem] text-[var(--l-2)] transition-colors hover:border-white hover:text-white"
              >
                Browse past jobs
              </Link>
            </div>
          </Reveal>
        </div>
      </section>

      {/* -------------------------------------------------------------- footer */}
      <footer className="border-t border-[var(--rule)]">
        <div className="wrap flex flex-col gap-6 py-10 text-[0.75rem] text-[var(--l-3)] md:flex-row md:items-center md:justify-between">
          <p className="max-w-[52ch] leading-relaxed">
            Multi-Candidate Multimodal Advertisement Generation with Learning-Based Performance
            Prediction and Intelligent Ranking — a final-year major project.
          </p>
          <p className="leading-relaxed">
            Candidate scores are predictions, not measurements.
          </p>
        </div>
      </footer>
    </div>
  );
}
