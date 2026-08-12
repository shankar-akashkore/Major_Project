/**
 * Tests for the honesty layer.
 *
 * Run with `pnpm test`, which is `node --test` — Node 22 strips TypeScript types
 * natively, so this needs no test framework, no transpiler and no config. Same
 * trade as drawing the report figures without matplotlib: the dependency costs
 * more disk than the feature is worth.
 *
 * What is asserted here is not formatting. It is the set of claims this UI must
 * not make: that a placeholder is a prediction, that an absence is a zero, that an
 * unimplemented check verified something, or that a padded render lost content.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  STAGE_VALUES,
  type GateCheck,
  type GateResult,
  type RankedCandidate,
} from "./contract.ts";
import * as P from "./presentation.ts";

// --- Fixtures ------------------------------------------------------------

function score(overall: number, isStub = false, extra: Record<string, unknown> = {}) {
  return {
    overall,
    aesthetic: null,
    prompt_alignment: null,
    product_salience: null,
    composition: null,
    palette_adherence: null,
    temporal_consistency: null,
    motion_quality: null,
    hook_strength: null,
    product_screen_time: null,
    safe_area_compliance: null,
    model_version: isStub ? "stub-0" : "pairwise-1",
    is_stub: isStub,
    ...extra,
  } as never;
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
  };
}

function gate(checks: GateCheck[], verdict = "pass"): GateResult {
  return { verdict, checks, attempt: 1 } as GateResult;
}

function ranked(rank: number, imageStageRank: number | null): RankedCandidate {
  return { rank, image_stage_rank: imageStageRank, explanation: "", video: {} } as RankedCandidate;
}

// --- Absence is not zero -------------------------------------------------

test("a missing number renders as an absence, never as zero", () => {
  for (const absent of [null, undefined, NaN, Infinity]) {
    assert.equal(P.num(absent), "—");
    assert.equal(P.percent(absent), "—");
    assert.equal(P.usd(absent), "—");
    assert.equal(P.seconds(absent), "—");
  }
});

test("a real zero still renders as a zero", () => {
  // The point of the rule above is that these two cases look different.
  assert.equal(P.num(0), "0.000");
  assert.equal(P.percent(0), "0%");
  assert.equal(P.usd(0), "$0.0000");
});

test("money keeps four decimal places", () => {
  // At two, a 0.0004 charge and a free run both print $0.00 — the difference
  // between the cache working and having paid twice.
  assert.equal(P.usd(0.0004), "$0.0004");
  assert.notEqual(P.usd(0.0004), P.usd(0));
});

// --- A stub is not a prediction ------------------------------------------

test("a stub score cannot be rendered as a bare number", () => {
  const display = P.scoreDisplay(score(0.496, true));
  assert.equal(display.kind, "stub");
  assert.equal(display.text, "0.496");
  // The caveat is non-null on this branch, so a caller destructuring it gets
  // something to render rather than an empty string that looks fine.
  assert.ok(display.caveat && display.caveat.includes("not a trained prediction"));
  assert.ok(display.caveat.includes("stub-0"), "the caveat names the model version");
});

test("a trained score carries no caveat", () => {
  const display = P.scoreDisplay(score(0.72, false));
  assert.equal(display.kind, "value");
  assert.equal(display.caveat, null);
});

test("an unscored candidate says so instead of showing zero", () => {
  const display = P.scoreDisplay(null);
  assert.equal(display.kind, "absent");
  assert.equal(display.value, null);
  assert.equal(display.text, "not scored");
  assert.ok(!display.text.includes("0"));
});

test("a NaN overall is an absence, not a score of nothing", () => {
  assert.equal(P.scoreDisplay(score(NaN)).kind, "absent");
});

test("all-stub is detected for the banner, and an empty set is not all-stub", () => {
  assert.equal(P.allStub([score(0.4, true), score(0.5, true)]), true);
  assert.equal(P.allStub([score(0.4, true), score(0.5, false)]), false);
  // Nothing scored is not the same claim as everything being a placeholder.
  assert.equal(P.allStub([]), false);
  assert.equal(P.allStub([null, null]), false);
});

test("a component that did not run is dropped rather than ranked as zero", () => {
  const s = score(0.6, false, { aesthetic: 0.8, hook_strength: 0.0, motion_quality: null });
  const drivers = P.topDrivers(s, 5);
  const names = drivers.map(([k]) => k);
  assert.ok(names.includes("aesthetic"));
  assert.ok(names.includes("hook_strength"), "a measured zero is still a measurement");
  assert.ok(!names.includes("motion_quality"), "an unrun feature group is not a zero");
  assert.deepEqual(drivers[0], ["aesthetic", 0.8]);
});

test("model_version and is_stub are never presented as component scores", () => {
  const drivers = P.topDrivers(score(0.6, true, { aesthetic: 0.5 }), 10).map(([k]) => k);
  assert.ok(!drivers.includes("model_version"));
  assert.ok(!drivers.includes("is_stub"));
  assert.ok(!drivers.includes("overall"));
});

// --- Rank movement -------------------------------------------------------

test("rank shift matches the Python property", () => {
  // RankedCandidate.rank_shift == image_stage_rank - rank
  assert.equal(P.rankShift(ranked(1, 3)), 2);
  assert.equal(P.rankShift(ranked(3, 1)), -2);
  assert.equal(P.rankShift(ranked(2, 2)), 0);
});

test("no image-stage rank gives no shift, not a shift of zero", () => {
  // Zero means "the image stage predicted this exactly" — which is the result the
  // project is measuring. Reporting it for a candidate that was never ranked at
  // the image stage would be inventing an agreement.
  assert.equal(P.rankShift(ranked(1, null)), null);
  assert.equal(P.movement(ranked(1, null)).direction, "unknown");
  assert.notEqual(P.movement(ranked(1, null)).label, "held");
});

test("movement labels read as direction and distance", () => {
  assert.deepEqual(P.movement(ranked(1, 3)), { shift: 2, label: "up 2", direction: "up" });
  assert.deepEqual(P.movement(ranked(3, 1)), { shift: -2, label: "down 2", direction: "down" });
  assert.equal(P.movement(ranked(2, 2)).label, "held");
});

test("a held order over one job is reported, and an unranked set is unknown", () => {
  assert.equal(P.heldOrder([ranked(1, 1), ranked(2, 2), ranked(3, 3)]), true);
  assert.equal(P.heldOrder([ranked(1, 2), ranked(2, 1), ranked(3, 3)]), false);
  assert.equal(P.heldOrder([ranked(1, null), ranked(2, 2)]), null);
});

// --- The quality gate ----------------------------------------------------

test("an unimplemented check is not counted as a pass", () => {
  const summary = P.gateSummary(
    gate([
      check("palette_delta_e", true),
      check("product_identity", true, false),
      check("face_identity", true, false),
    ]),
  );
  assert.ok(summary);
  assert.equal(summary.passed.length, 1, "only the check that ran counts as passed");
  assert.equal(summary.pending.length, 2);
  assert.equal(summary.verifiedIdentity, false);
  assert.ok(summary.text.includes("not yet implemented"));
});

test("identity is verified only when both identity checks really ran and passed", () => {
  const both = P.gateSummary(gate([check("product_identity", true), check("face_identity", true)]));
  assert.equal(both?.verifiedIdentity, true);

  const one = P.gateSummary(gate([check("product_identity", true)]));
  assert.equal(one?.verifiedIdentity, false, "one of two is not verification");

  const failing = P.gateSummary(
    gate([check("product_identity", true), check("face_identity", false)]),
  );
  assert.equal(failing?.verifiedIdentity, false);
});

test("a failure is named with its value and threshold", () => {
  const summary = P.gateSummary(gate([check("palette_delta_e", false)], "reject"));
  assert.ok(summary);
  assert.equal(summary.failed.length, 1);
  assert.ok(summary.text.includes("palette delta e"));
  assert.ok(summary.text.includes("0.200"), "the measured value is shown");
  assert.ok(summary.text.includes("0.500"), "so is the threshold it missed");
});

test("no gate at all is null rather than an empty pass", () => {
  assert.equal(P.gateSummary(null), null);
  assert.equal(P.gateSummary(undefined), null);
});

// --- Progress ------------------------------------------------------------

test("overall progress matches the pipeline's own arithmetic", () => {
  // StageEvent.overall_progress == (stage.index + progress) / len(Stage)
  assert.equal(STAGE_VALUES.length, 8, "eight stages, per adschema.Stage");
  const at = (stage: string, progress: number) =>
    P.progressFrom([{ stage, progress } as never]).fraction;
  assert.equal(at("intake", 0), 0);
  assert.equal(at("intake", 1), 1 / 8);
  assert.equal(at("image_rank", 0), 4 / 8);
  assert.equal(at("delivery", 1), 1);
});

test("progress is clamped and never exceeds one", () => {
  assert.equal(P.progressFrom([{ stage: "delivery", progress: 5 } as never]).fraction, 1);
  assert.equal(P.progressFrom([{ stage: "intake", progress: NaN } as never]).fraction, 0);
});

test("no events is queued at zero, not an error", () => {
  const progress = P.progressFrom([]);
  assert.equal(progress.fraction, 0);
  assert.equal(progress.stage, null);
  assert.equal(progress.label, "queued");
});

test("the last event wins, so a replayed history lands on the right stage", () => {
  const progress = P.progressFrom([
    { stage: "intake", progress: 1 } as never,
    { stage: "video_rank", progress: 0.5 } as never,
  ]);
  assert.equal(progress.stage, "video_rank");
  assert.equal(progress.label, "video rank");
});

test("terminal states close the stream and running states do not", () => {
  for (const state of ["completed", "failed", "refused_over_budget"]) {
    assert.equal(P.isTerminal({ state } as never), true);
  }
  for (const state of ["queued", "running"]) {
    assert.equal(P.isTerminal({ state } as never), false);
  }
});

// --- Delivery ------------------------------------------------------------

test("a padded render is not reported as having lost content", () => {
  // The schema is explicit: for a pad, retained_salience describes what a crop
  // *would* have lost, which is why padding was chosen.
  const summary = P.reframeSummary({
    aspect_ratio: "16:9",
    mode: "pad",
    retained_salience: 0.62,
    centre_crop_salience: 0.4,
    tracking_gain: 0,
    width: 1920,
    height: 1080,
    note: "",
  } as never);
  assert.equal(summary.padded, true);
  assert.equal(summary.improvement, 0.62 - 0.4);
});

test("a no-op reframe has no retention figure to report", () => {
  const summary = P.reframeSummary({
    aspect_ratio: "9:16",
    mode: "none",
    retained_salience: 1,
    centre_crop_salience: null,
    tracking_gain: 0,
    width: 1080,
    height: 1920,
    note: "",
  } as never);
  assert.equal(summary.retained, null, "1.0 by definition is not a measurement");
  assert.equal(summary.improvement, null);
});

test("improvement over a centre crop is absent when it was not measured", () => {
  const summary = P.reframeSummary({
    aspect_ratio: "4:5",
    mode: "crop",
    retained_salience: 0.8,
    centre_crop_salience: null,
    tracking_gain: 0,
    width: 1080,
    height: 1350,
    note: "",
  } as never);
  assert.equal(summary.improvement, null);
  assert.equal(summary.retained, 0.8);
});

// --- Budget --------------------------------------------------------------

test("budget mirrors the Python properties", () => {
  const view = P.budgetView({ total_budget_usd: 35, spent_usd: 7, per_job_cap_usd: 3 } as never);
  assert.equal(view.remaining, 28);
  assert.equal(view.fractionUsed, 0.2);
  assert.equal(view.level, "ok");
});

test("overspend clamps rather than showing a negative remainder", () => {
  const view = P.budgetView({ total_budget_usd: 35, spent_usd: 40, per_job_cap_usd: 3 } as never);
  assert.equal(view.remaining, 0);
  assert.equal(view.fractionUsed, 1);
  assert.equal(view.level, "spent");
});

test("a zero budget is fully used rather than a division by zero", () => {
  const view = P.budgetView({ total_budget_usd: 0, spent_usd: 0, per_job_cap_usd: 0 } as never);
  assert.equal(view.fractionUsed, 1);
  assert.ok(Number.isFinite(view.fractionUsed));
});

test("the tight level warns before the money is gone", () => {
  const view = P.budgetView({ total_budget_usd: 35, spent_usd: 30, per_job_cap_usd: 3 } as never);
  assert.equal(view.level, "tight");
});

// --- Labels --------------------------------------------------------------

test("slot letters match ShotBrief.slot_label", () => {
  assert.equal(P.slotLetter(0), "A");
  assert.equal(P.slotLetter(2), "C");
});

test("field names are readable without being reworded", () => {
  assert.equal(P.humanise("product_screen_time"), "product screen time");
  assert.equal(P.humanise("image_gen"), "image gen");
});

test("a list of three reads as prose, not as two ands", () => {
  assert.equal(P.prose([]), "");
  assert.equal(P.prose(["a"]), "a");
  assert.equal(P.prose(["a", "b"]), "a and b");
  assert.equal(P.prose(["a", "b", "c"]), "a, b and c");
});

// --- Media ---------------------------------------------------------------

const API = "http://localhost:8000";

test("a root-relative media URL is resolved against the API, not this app", () => {
  // LocalStorage.url_for returns "/media/<key>". Served from port 3000 that hits
  // Next, which has no such route, and every frame renders as a broken image.
  const url = P.resolveMediaUrl({ key: "generations/j/img_0.png", url: "/media/x.png" }, API);
  assert.equal(url, `${API}/media/x.png`);
});

test("an absolute URL is left alone, because it is a signed one", () => {
  const signed = "https://bucket.example.com/x.png?sig=abc";
  assert.equal(P.resolveMediaUrl({ key: "x", url: signed }, API), signed);
});

test("a missing URL falls back to the media route over the key", () => {
  assert.equal(
    P.resolveMediaUrl({ key: "generations/j/img_0.png", url: null }, API),
    `${API}/media/generations/j/img_0.png`,
  );
});

test("a trailing slash on the API base does not double up", () => {
  assert.equal(P.resolveMediaUrl({ key: "a.png", url: null }, "http://localhost:8000/"), `${API}/media/a.png`);
});

test("no asset gives no URL rather than a link to nowhere", () => {
  assert.equal(P.resolveMediaUrl(null, API), null);
  assert.equal(P.resolveMediaUrl({ key: "", url: null }, API), null);
});

// --- Platform geometry ---------------------------------------------------

test("a safe area lists only the edges that platform chrome actually covers", () => {
  // Instagram Reels: the right edge carries the action rail, and the wizard used to
  // omit it because the API sent a two-key dict.
  const reels = { top: 0.14, bottom: 0.2, left: 0, right: 0.14 };
  assert.equal(P.safeAreaSummary(reels), "14% top, 20% bottom, 14% right");
});

test("an edge at zero is left out rather than printed as 0%", () => {
  const feed = { top: 0.05, bottom: 0.05, left: 0, right: 0 };
  assert.equal(P.safeAreaSummary(feed), "5% top, 5% bottom");
});

test("a platform with no chrome says none, not an empty string", () => {
  assert.equal(P.safeAreaSummary({ top: 0, bottom: 0, left: 0, right: 0 }), "none");
});
