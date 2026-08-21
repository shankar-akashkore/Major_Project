/**
 * Tests for the components, not for the honesty layer.
 *
 * `lib/presentation.test.ts` proves that `scoreDisplay` tags a placeholder and that
 * `reframeSummary` does not call padding a loss. It cannot prove that anything on
 * screen *reads* those answers — `docs/ui-protocol.md` said so under Known
 * Limitations: "a component that stops calling `scoreDisplay` and formats a number
 * directly would pass everything here."  This file closes that.
 *
 * These render the real components with `react-dom/server` and assert on the markup,
 * so what is being checked is what a person would see. Effects do not run, which
 * costs nothing here: none of these components fetch, and the ones that do
 * (`Shell`, the pages) are verified in a browser instead.
 *
 * JSX needs a transform, which `tsx-loader.mjs` provides using the TypeScript
 * compiler already installed — still no test framework and still no second module
 * graph that can disagree with the one Next builds.
 *
 * The assertions are deliberately about *sentences*, not structure. A refactor that
 * moves a caveat into a different element should pass; one that drops the caveat
 * should not.
 */

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { renderToStaticMarkup } from "react-dom/server";

import type {
  DeliveryReport,
  DeliveryResponse,
  GateCheck,
  GateResult,
  ImageCandidate,
  RankedCandidate,
  ReframeReport,
  ScoreBreakdown,
} from "@/lib/contract.ts";
import { DeliveryPanel } from "./DeliveryPanel.tsx";
import { ImageStage } from "./ImageStage.tsx";
import { ScoreChip, ScoreDetail } from "./Score.tsx";
import { VideoStage } from "./VideoStage.tsx";

const html = (element: React.ReactElement) => renderToStaticMarkup(element);

/** Markup with the tags taken out, so an assertion can read a whole sentence. */
const text = (element: React.ReactElement) =>
  html(element)
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();

// --- Fixtures ------------------------------------------------------------
//
// Cast rather than fully populated: every field the component reads is set here,
// and spelling out the forty it ignores would bury what each test is varying.

function score(overall: number, isStub: boolean, extra: Record<string, unknown> = {}) {
  return {
    overall,
    aesthetic: 0.7,
    prompt_alignment: 0.6,
    product_salience: null,
    composition: null,
    palette_adherence: null,
    temporal_consistency: null,
    motion_quality: null,
    hook_strength: null,
    product_screen_time: null,
    safe_area_compliance: null,
    model_version: isStub ? "pairwise-linear-20260811" : "pairwise-v1",
    is_stub: isStub,
    ...extra,
  } as unknown as ScoreBreakdown;
}

function check(name: string, passed: boolean, implemented = true): GateCheck {
  return {
    name,
    value: passed ? 0.9 : 0.2,
    threshold: 0.5,
    passed,
    higher_is_better: true,
    detail: "",
    implemented,
  } as GateCheck;
}

function image(index: number, overrides: Partial<ImageCandidate> = {}): ImageCandidate {
  return {
    index,
    asset: { key: `generations/j/img_${index}.png`, url: `/media/img_${index}.png` },
    brief: {
      slot: index,
      concept: `Concept ${index}`,
      design_point: { angle: "eye_level", lighting: "soft_daylight", composition: "rule_of_thirds" },
    },
    score: score(0.5 - index * 0.1, true),
    gate: {
      verdict: "pass",
      attempt: 1,
      checks: [check("product_identity", true), check("palette_delta_e", true)],
    } as GateResult,
    provider: "mock",
    seed: 7 + index,
    tier: "mock",
    latency_ms: 12,
    ...overrides,
  } as unknown as ImageCandidate;
}

function candidate(
  rank: number,
  imageStageRank: number | null,
  overrides: Record<string, unknown> = {},
): RankedCandidate {
  return {
    rank,
    image_stage_rank: imageStageRank,
    explanation: `Ranked ${rank}.`,
    video: {
      source_image_index: rank - 1,
      asset: { key: `clip_${rank}.mp4`, url: `/media/clip_${rank}.mp4` },
      thumbnail: null,
      score: score(0.5, true),
      duration_seconds: 9,
      fps: 24,
      provider: "mock",
      cost_usd: 0,
      was_chained: false,
      seam_consistency: null,
      requested_duration_seconds: 9,
      seed_honoured: true,
      ...overrides,
    },
  } as unknown as RankedCandidate;
}

function reframe(overrides: Partial<ReframeReport> = {}): ReframeReport {
  return {
    aspect_ratio: "4:5",
    asset: { key: "render.mp4", url: "/media/render.mp4" },
    mode: "crop",
    width: 1024,
    height: 1280,
    retained_salience: 0.84,
    centre_crop_salience: 0.75,
    tracking_gain: 0,
    note: "",
    ...overrides,
  } as ReframeReport;
}

function delivery(report: Partial<DeliveryReport> = {}): DeliveryResponse {
  return {
    job_id: "j0",
    summary: "1 renders, 0 previews",
    download_url: null,
    delivery: {
      renders: [reframe()],
      previews: {},
      audio: {
        attached: false,
        credit: "",
        licence: "",
        target_lufs: null,
        measured_lufs: null,
        has_voiceover: false,
        is_test_signal: false,
        note: "",
      },
      bundle: null,
      warnings: [],
      ...report,
    },
  } as unknown as DeliveryResponse;
}

// --- A stub is marked, on screen ------------------------------------------

test("a placeholder score is struck through and labelled, never shown as a plain number", () => {
  const markup = html(<ScoreChip score={score(0.496, true)} />);
  // The strike-through is asserted by class because that *is* the signal: the
  // number is on screen and has to be visibly disowned.
  assert.match(markup, /line-through/);
  assert.match(markup, /placeholder/);
  assert.match(markup, /0\.496/);
});

test("a real prediction is not labelled as a placeholder", () => {
  const markup = html(<ScoreChip score={score(0.712, false)} />);
  assert.doesNotMatch(markup, /placeholder/);
  assert.doesNotMatch(markup, /line-through/);
  assert.match(markup, /0\.712/);
});

test("the score panel names the model version that produced a placeholder", () => {
  const body = text(<ScoreDetail score={score(0.496, true)} />);
  assert.match(body, /pairwise-linear-20260811/);
  assert.match(body, /not a trained prediction/);
});

test("an unscored candidate says so instead of rendering a zero", () => {
  const body = text(<ScoreDetail score={null} />);
  assert.equal(body, "not scored");
  assert.doesNotMatch(body, /0\.000/);
});

// --- The gate is not a score, and a pending check is not a pass -----------

test("a gate with an unimplemented check refuses to claim identity was verified", () => {
  const gate = {
    verdict: "pass",
    attempt: 1,
    checks: [
      check("product_identity", true, false),
      check("face_identity", true, false),
      check("nsfw", true, false),
      check("palette_delta_e", true),
    ],
  } as GateResult;

  const body = text(<ImageStage images={[image(0, { gate })]} />);
  assert.match(body, /did not run/);
  assert.match(body, /Identity is not verified/);
  // The prose helper, not " and " three times over.
  assert.match(body, /product identity, face identity and nsfw/);
  assert.doesNotMatch(body, /both verified/);
});

test("an unmeasurable check does not claim a model is missing", () => {
  // `product_scale` is pending because the frame could not be measured, not
  // because a model has not landed — and the caveat has to say the right one.
  const gate = {
    verdict: "pass",
    attempt: 1,
    checks: [check("product_scale", true, false), check("palette_delta_e", true)],
  } as GateResult;

  const body = text(<ImageStage images={[image(0, { gate })]} />);
  assert.match(body, /product scale did not run/);
  assert.match(body, /verifies nothing/);
  // No identity check is pending, so the identity sentence must not appear.
  assert.doesNotMatch(body, /Identity is not verified/);
});

test("a gate whose identity checks really ran says they were verified", () => {
  const gate = {
    verdict: "pass",
    attempt: 1,
    checks: [check("product_identity", true), check("face_identity", true)],
  } as GateResult;

  const body = text(<ImageStage images={[image(0, { gate })]} />);
  assert.match(body, /Product and face identity both verified/);
  assert.doesNotMatch(body, /did not run/);
});

test("a failed check is shown with the number it failed on", () => {
  // "reject", not "fail" — the verdict vocabulary is `GateVerdict`, and the
  // generated union is what caught the wrong word here rather than a passing test
  // asserting on a badge that would never render.
  const gate = {
    verdict: "reject",
    attempt: 2,
    checks: [check("product_identity", false), check("face_identity", true)],
  } as GateResult;

  const body = text(<ImageStage images={[image(0, { gate })]} />);
  assert.match(body, /product identity 0\.200 vs 0\.500/);
});

test("the image stage shows the predicted order, which is what the claim is about", () => {
  const body = text(<ImageStage images={[image(0), image(1), image(2)]} />);
  assert.match(body, /Predicted order A › B › C/);
  assert.match(body, /predicted #1/);
});

// --- Seeing the frame whole -----------------------------------------------
//
// The grid crops every candidate to 4:5 with `object-cover`, and the crop is
// exactly where a clipped hand or a product pushed out of the safe area hides.

test("every frame can be opened full size, named so the control says which one", () => {
  const markup = html(<ImageStage images={[image(0), image(1)]} />);
  assert.match(markup, /aria-label="View candidate A full size"/);
  assert.match(markup, /aria-label="View candidate B full size"/);
  assert.match(text(<ImageStage images={[image(0)]} />), /Full size/);
});

test("a candidate with no frame offers nothing to open", () => {
  const missing = image(1, { asset: { key: "", url: null } as unknown as ImageCandidate["asset"] });
  const markup = html(<ImageStage images={[image(0), missing]} />);
  assert.match(markup, /no frame/);
  assert.match(markup, /aria-label="View candidate A full size"/);
  assert.doesNotMatch(markup, /aria-label="View candidate B full size"/);
});

test("the overlay is closed until something is clicked, so the grid renders alone", () => {
  // `renderToStaticMarkup` runs no effects and has no document. A lightbox that
  // needed either to render its closed state would break every page that mounts
  // this component on the server.
  const markup = html(<ImageStage images={[image(0), image(1)]} />);
  assert.doesNotMatch(markup, /role="dialog"/);
  assert.doesNotMatch(markup, /Close preview/);
});

// --- One job is not the measurement ---------------------------------------

test("the video stage says a single job is an illustration, whatever the outcome", () => {
  for (const ranking of [
    [candidate(1, 1), candidate(2, 2), candidate(3, 3)],
    [candidate(1, 2), candidate(2, 1), candidate(3, 3)],
  ]) {
    const body = text(<VideoStage ranking={ranking} />);
    assert.match(body, /is an illustration, not the measurement/);
    assert.match(body, /stage_agreement\.py/);
  }
});

// --- The soundtrack has to survive the trip to the browser -----------------
//
// Delivery mixes a bed onto every clip, normalises it, writes it to storage and
// records the credit. None of that is audible if the player points at the wrong
// file or mutes itself, and both of those were true: the element played
// `video.asset` — the provider's own render, which has no audio track at all —
// and carried a hard-coded `muted`. The user's report was "the video does not
// support sound", and they were right twice over.

test("the player prefers the mixed render over the provider's silent one", () => {
  const mixed = candidate(1, 1, {
    platform_renders: { audio: { key: "1_with_audio.mp4", url: "/media/1_with_audio.mp4" } },
  });
  // `html`, not `text`: the bug lived entirely in attributes, and `text` strips
  // exactly the part of the markup this is about.
  const body = html(<VideoStage ranking={[mixed]} />);

  assert.match(body, /src="[^"]*\/1_with_audio\.mp4"/);
  assert.doesNotMatch(body, /src="[^"]*\/clip_1\.mp4"/);
});

test("a clip with no mix still plays, from the native render", () => {
  // Delivery can fail the mix and say so. Falling back to a silent clip is the
  // right answer; rendering "no clip" because one key was absent is not.
  const body = html(<VideoStage ranking={[candidate(1, 1)]} />);
  assert.match(body, /src="[^"]*\/clip_1\.mp4"/);
});

test("the player is not muted, because nothing here autoplays", () => {
  const body = html(<VideoStage ranking={[candidate(1, 1)]} />);
  assert.doesNotMatch(body, /muted/);
  // And the controls are still there, so the viewer can mute it themselves.
  assert.match(body, /controls/);
});

test("an order that held is reported as a finding, not as a missing comparison", () => {
  const body = text(<VideoStage ranking={[candidate(1, 1), candidate(2, 2)]} />);
  assert.match(body, /The order held/);
  assert.match(body, /= held \(was #1\)/);
});

test("movement between the stages is rendered with its direction and its origin", () => {
  const body = text(<VideoStage ranking={[candidate(1, 2), candidate(2, 1)]} />);
  assert.match(body, /The order changed/);
  assert.match(body, /▲ up 1 \(was #2\)/);
  assert.match(body, /▼ down 1 \(was #1\)/);
});

test("a candidate the image stage never ranked is not described as having held", () => {
  const body = text(<VideoStage ranking={[candidate(1, null), candidate(2, 2)]} />);
  assert.match(body, /no image-stage rank/);
  assert.match(body, /cannot be compared/);
  assert.doesNotMatch(body, /The order held/);
});

test("a clip with no seed is flagged as unreproducible", () => {
  const body = text(<VideoStage ranking={[candidate(1, 1, { seed_honoured: false })]} />);
  assert.match(body, /no seed parameter/);
  assert.match(body, /not a controlled comparison/);
});

test("a chained clip says it was concatenated rather than generated at length", () => {
  const body = text(
    <VideoStage ranking={[candidate(1, 1, { was_chained: true, seam_consistency: null })]} />,
  );
  assert.match(body, /concatenating two clips/);
  assert.match(body, /Seam consistency was not measured/);
});

test("a delivered duration that is not the requested one says which was billed", () => {
  const body = text(
    <VideoStage
      ranking={[candidate(1, 1, { requested_duration_seconds: 9, duration_seconds: 10 })]}
    />,
  );
  assert.match(body, /9\.0s was requested; 10\.0s was delivered and billed/);
});

test("money is rendered at four decimal places, where a free run is distinguishable", () => {
  const body = text(<VideoStage ranking={[candidate(1, 1, { cost_usd: 0 })]} />);
  assert.match(body, /\$0\.0000/);
  assert.doesNotMatch(body, /\$0\.00 /);
});

test("a missing duration renders as an absence, not as zero seconds", () => {
  const body = text(<VideoStage ranking={[candidate(1, 1, { duration_seconds: null })]} />);
  assert.match(body, /duration —/);
  assert.doesNotMatch(body, /0\.0s/);
});

// --- A padded reframe is not a loss ---------------------------------------

test("a padded render is described as keeping everything, with what a crop would have cost", () => {
  const body = text(
    delivered({ renders: [reframe({ aspect_ratio: "16:9", mode: "pad", retained_salience: 0.56 })] }),
  );
  assert.match(body, /Padded, so the whole frame is still visible/);
  assert.match(body, /A crop would have kept 56%/);
  assert.doesNotMatch(body, /Kept 56%/);
});

test("a cropped render reports the loss and the gain over a centre crop", () => {
  const body = text(delivered({ renders: [reframe()] }));
  assert.match(body, /Kept 84%/);
  assert.match(body, /\+9% against a centre crop/);
});

test("a render at its native ratio is not credited with retaining anything", () => {
  const body = text(
    delivered({
      renders: [reframe({ aspect_ratio: "9:16", mode: "none", retained_salience: 1 })],
    }),
  );
  assert.match(body, /nothing was reframed/);
  assert.doesNotMatch(body, /Kept 100%/);
});

test("silent output says why it is silent instead of leaving the panel blank", () => {
  const body = text(delivered({}));
  assert.match(body, /clips are silent until a licensed bed is supplied/);
});

test("a synthesised tone is not described as a soundtrack", () => {
  const body = text(
    delivered({
      audio: {
        attached: true,
        credit: "generated tone",
        licence: "n/a",
        target_lufs: -14,
        measured_lufs: -14.2,
        has_voiceover: false,
        is_test_signal: true,
        note: "",
      },
    } as Partial<DeliveryReport>),
  );
  assert.match(body, /development placeholder/);
  assert.match(body, /must not be described as a soundtrack/);
});

test("a delivery warning is surfaced rather than folded into the summary", () => {
  const body = text(delivered({ warnings: ["safe-area overlap on instagram_reels"] }));
  assert.match(body, /safe-area overlap on instagram_reels/);
});

function delivered(report: Partial<DeliveryReport>) {
  return <DeliveryPanel jobId="j0" response={delivery(report)} />;
}

// --- The one guard worth having on the source -----------------------------

test("no component formats money itself", () => {
  // `usd()` renders four decimal places for a reason recorded in `presentation.ts`:
  // at two, a real $0.0004 charge and a genuinely free run both print $0.00. A
  // component doing its own `toFixed(2)` would put that back, and no rendering test
  // would catch it unless it happened to cover that component.
  //
  // Not a ban on `toFixed` — DeliveryPanel formats LUFS with it, which is a loudness
  // measurement and not money. The check is specific to the currency precisions.
  const roots = ["components", "app"].map((dir) => path.join(HERE, "..", dir));
  const offenders: string[] = [];
  for (const root of roots) {
    for (const file of sources(root)) {
      const source = readFileSync(file, "utf8");
      if (/toFixed\(\s*[24]\s*\)/.test(source)) offenders.push(path.relative(HERE, file));
    }
  }
  assert.deepEqual(offenders, []);
});

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** Every `.ts`/`.tsx` under `dir`, recursively, excluding the tests themselves. */
function sources(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...sources(full));
    else if (/\.tsx?$/.test(entry.name) && !entry.name.includes(".test.")) out.push(full);
  }
  return out;
}
