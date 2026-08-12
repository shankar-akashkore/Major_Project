/**
 * How a value becomes text on screen — and what this app refuses to render.
 *
 * Everything the pipeline is careful about can be undone here in one line. The
 * pipeline records `ScoreBreakdown.is_stub` so a placeholder cannot be mistaken
 * for a prediction; `{score.overall.toFixed(3)}` in a component throws that away
 * and puts `0.496` on a projector in front of an examiner. The report generator
 * exists to stop the same thing happening on paper (`docs/results-protocol.md`).
 * This module is its counterpart for the screen.
 *
 * Four rules, each with a test in `presentation.test.ts`:
 *
 * 1. **A stub score is never plain text.** It comes back tagged, so a caller has
 *    to decide what to do with the tag — the type will not let it be interpolated
 *    as a bare number.
 * 2. **A missing number says so.** `null` and `NaN` render as "—", never as 0%. A
 *    zero is a measurement; an absence is not, and at a glance they look identical.
 * 3. **An unimplemented check is not a pass.** The quality gate passes checks whose
 *    model has not landed (`GateCheck.implemented === false`), so "passed all
 *    quality checks" would be a claim about identity verification that never ran.
 * 4. **Derived values are computed in one place.** The Python `@property` values
 *    are not on the wire (see the header of `contract.ts`), so they are recomputed
 *    here, once, next to a test that pins the formula to the Python.
 */

import {
  type BudgetStatus,
  type GateCheck,
  type GateResult,
  type JobRecord,
  type RankedCandidate,
  type ReframeReport,
  type SafeAreaBox,
  type ScoreBreakdown,
  type Stage,
  type StageEvent,
  STAGE_VALUES,
} from "./contract.ts";

// --- Numbers -------------------------------------------------------------

/**
 * A number that may not exist, as text. "—" for absent, never "0".
 *
 * The em dash is doing real work: a blank cell is indistinguishable from a zero,
 * from a rounding artefact, and from a value someone deleted.
 */
export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

/** A 0–1 fraction as a percentage, or "—". */
export function percent(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

/**
 * Money, at four decimal places.
 *
 * Not two. This project's whole spend is $35 and a mock run costs $0.0000; at two
 * decimals a real 0.0004 charge and a genuinely free run both print as $0.00,
 * which is the difference between "the cache worked" and "we just paid twice".
 */
export function usd(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `$${value.toFixed(4)}`;
}

/** Seconds, with the unit attached, because "9" alone has been read as frames. */
export function seconds(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value.toFixed(1)}s`;
}

// --- Scores --------------------------------------------------------------

export type ScoreDisplay =
  /** A real prediction from a trained head. */
  | { kind: "value"; text: string; value: number; caveat: null }
  /** A placeholder. `caveat` is not optional — a caller cannot forget it. */
  | { kind: "stub"; text: string; value: number; caveat: string }
  /** Nothing was scored. */
  | { kind: "absent"; text: string; value: null; caveat: string };

/**
 * A score, tagged with whether it means anything yet.
 *
 * Returning a union rather than a formatted string is the point: `kind` has to be
 * handled, so the stub case cannot be skipped by accident. The alternative — a
 * boolean second return value — gets dropped at the call site and nothing
 * complains.
 */
export function scoreDisplay(score: ScoreBreakdown | null | undefined): ScoreDisplay {
  if (!score || !Number.isFinite(score.overall)) {
    return {
      kind: "absent",
      text: "not scored",
      value: null,
      caveat: "No prediction was recorded for this candidate.",
    };
  }
  if (score.is_stub) {
    return {
      kind: "stub",
      text: num(score.overall),
      value: score.overall,
      caveat:
        `Placeholder from ${score.model_version}, not a trained prediction. ` +
        "Ordering is real; the value is not comparable to anything.",
    };
  }
  return { kind: "value", text: num(score.overall), value: score.overall, caveat: null };
}

/** True when every score in a job is a placeholder — the banner condition. */
export function allStub(scores: (ScoreBreakdown | null)[]): boolean {
  const present = scores.filter((s): s is ScoreBreakdown => Boolean(s));
  return present.length > 0 && present.every((s) => s.is_stub);
}

/**
 * The component scores that were actually computed, best first.
 *
 * Mirrors `ScoreBreakdown.top_drivers`. A null component means the feature group
 * did not run, which is different from scoring zero, so nulls are dropped rather
 * than sorted to the bottom.
 */
export function topDrivers(score: ScoreBreakdown, n = 3): [string, number][] {
  const skip = new Set(["overall", "model_version", "is_stub"]);
  return Object.entries(score)
    .filter(([k, v]) => !skip.has(k) && typeof v === "number" && Number.isFinite(v))
    .map(([k, v]) => [k, v as number] as [string, number])
    .sort((a, b) => b[1] - a[1])
    .slice(0, n);
}

/** `product_screen_time` -> `product screen time`. */
export function humanise(key: string): string {
  return key.replace(/_/g, " ");
}

/**
 * A list as prose: "a", "a and b", "a, b and c".
 *
 * Three unimplemented gate checks joined with `" and "` read as "product identity
 * and face identity and nsfw", which is the sort of sentence that makes a reader
 * stop trusting the rest of the panel.
 */
export function prose(items: string[]): string {
  if (items.length === 0) return "";
  if (items.length === 1) return items[0]!;
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

// --- The headline: movement between the two stages -----------------------

/**
 * How far a candidate moved between the image-stage and video-stage rankings.
 *
 * Mirrors `RankedCandidate.rank_shift`, which is a Python `@property` and so never
 * arrives over the wire. Positive means it ended up better than the image stage
 * predicted; zero means the image stage already had it right — which is the result
 * this project is trying to measure, not a missing value.
 */
export function rankShift(candidate: RankedCandidate): number | null {
  if (candidate.image_stage_rank === null) return null;
  return candidate.image_stage_rank - candidate.rank;
}

export type Movement = {
  shift: number | null;
  /** "held", "up 1", "down 2", or "unknown" when the image stage did not rank it. */
  label: string;
  direction: "up" | "down" | "held" | "unknown";
};

export function movement(candidate: RankedCandidate): Movement {
  const shift = rankShift(candidate);
  if (shift === null) {
    return { shift: null, label: "no image-stage rank", direction: "unknown" };
  }
  if (shift === 0) return { shift, label: "held", direction: "held" };
  const size = Math.abs(shift);
  return {
    shift,
    label: `${shift > 0 ? "up" : "down"} ${size}`,
    direction: shift > 0 ? "up" : "down",
  };
}

/**
 * Whether the two stages agreed completely, over one job.
 *
 * Deliberately not called "agreement": one job of three candidates is an anecdote,
 * and the project's headline number is a correlation over many generation sets that
 * `scripts/stage_agreement.py` withholds until there are enough of them. This is
 * a description of what is on screen, and the caller says so.
 */
export function heldOrder(ranking: RankedCandidate[]): boolean | null {
  const shifts = ranking.map(rankShift);
  if (shifts.some((s) => s === null)) return null;
  return shifts.every((s) => s === 0);
}

// --- The quality gate ----------------------------------------------------

export type GateSummary = {
  verdict: string;
  /** Checks that really ran and passed. */
  passed: GateCheck[];
  /** Checks that ran and failed. */
  failed: GateCheck[];
  /**
   * Checks whose model has not landed. These *passed* in the pipeline, and saying
   * so without saying they did not run would be claiming a verification.
   */
  pending: GateCheck[];
  /** True only when the identity checks genuinely executed and passed. */
  verifiedIdentity: boolean;
  text: string;
};

const IDENTITY_CHECKS = ["product_identity", "face_identity"];

/**
 * The gate's verdict, with the distinction the pipeline is careful about kept.
 *
 * Mirrors `GateResult.verified_identity` and `GateResult.reason`, both Python
 * properties. The wording matters: "passed all quality checks" is what the
 * pipeline says when the DINOv2 and ArcFace models are not installed, because a
 * pending check always passes.
 */
export function gateSummary(gate: GateResult | null | undefined): GateSummary | null {
  if (!gate) return null;
  const checks = gate.checks ?? [];
  const pending = checks.filter((c) => !c.implemented);
  const ran = checks.filter((c) => c.implemented);
  const failed = ran.filter((c) => !c.passed);
  const passed = ran.filter((c) => c.passed);
  const identityRan = ran.filter((c) => IDENTITY_CHECKS.includes(c.name));
  const verifiedIdentity =
    identityRan.length === IDENTITY_CHECKS.length && identityRan.every((c) => c.passed);

  let text: string;
  if (failed.length > 0) {
    text = failed.map((c) => `${humanise(c.name)} ${num(c.value)} vs ${num(c.threshold)}`).join(", ");
  } else if (pending.length > 0) {
    text = `${passed.length} check(s) passed, ${pending.length} not yet implemented`;
  } else {
    text = `all ${passed.length} checks passed`;
  }
  return { verdict: gate.verdict, passed, failed, pending, verifiedIdentity, text };
}

// --- Progress ------------------------------------------------------------

export type Progress = {
  stage: Stage | null;
  /** 0–1 across the whole pipeline, not within the stage. */
  fraction: number;
  label: string;
};

/**
 * Overall progress from the event stream.
 *
 * Mirrors `StageEvent.overall_progress`: `(stage index + within-stage progress) /
 * eight stages`. Recomputed rather than read because it is a Python property; the
 * test pins the arithmetic so the bar cannot drift from the pipeline's own idea of
 * how far along it is.
 */
export function progressFrom(events: StageEvent[]): Progress {
  const last = events.length > 0 ? events[events.length - 1] : undefined;
  if (!last) return { stage: null, fraction: 0, label: "queued" };
  const index = STAGE_VALUES.indexOf(last.stage);
  const total = STAGE_VALUES.length;
  const within = Number.isFinite(last.progress) ? last.progress : 0;
  const fraction = Math.min(1, Math.max(0, (index + within) / total));
  return { stage: last.stage, fraction, label: humanise(last.stage) };
}

/** Whether a job is still moving, which decides whether to keep the stream open. */
export function isTerminal(record: Pick<JobRecord, "state">): boolean {
  return ["completed", "failed", "refused_over_budget"].includes(record.state);
}

// --- Delivery ------------------------------------------------------------

export type ReframeSummary = {
  ratio: string;
  mode: string;
  /** Null for a no-op reframe, where "retained" is 1.0 by definition. */
  retained: number | null;
  /** How much better than a naive centre crop, when that was measured. */
  improvement: number | null;
  /** True when the frame was padded, so nothing was cropped away at all. */
  padded: boolean;
  note: string;
};

/**
 * One platform render, described honestly.
 *
 * `retained_salience` means two different things depending on `mode`, and the
 * schema says so: for a padded variant everything is still visible, and the figure
 * describes what a crop *would* have lost — which is why padding was chosen. A UI
 * that prints "78% retained" under a padded render is reporting a loss that did
 * not happen.
 */
export function reframeSummary(report: ReframeReport): ReframeSummary {
  const padded = report.mode === "pad";
  const noop = report.mode === "none";
  return {
    ratio: report.aspect_ratio,
    mode: report.mode,
    retained: noop ? null : report.retained_salience,
    improvement:
      report.centre_crop_salience === null
        ? null
        : report.retained_salience - report.centre_crop_salience,
    padded,
    note: report.note,
  };
}

// --- Budget --------------------------------------------------------------

export type BudgetView = {
  remaining: number;
  fractionUsed: number;
  /** "ok" | "tight" under a quarter left | "spent" at zero. */
  level: "ok" | "tight" | "spent";
};

/** Mirrors `BudgetStatus.remaining_usd` and `.fraction_used`, both properties. */
export function budgetView(status: BudgetStatus): BudgetView {
  const remaining = Math.max(0, status.total_budget_usd - status.spent_usd);
  const fractionUsed =
    status.total_budget_usd <= 0 ? 1 : Math.min(1, status.spent_usd / status.total_budget_usd);
  const level = remaining <= 0 ? "spent" : fractionUsed >= 0.75 ? "tight" : "ok";
  return { remaining, fractionUsed, level };
}

/** Candidate slot letters, matching `ShotBrief.slot_label`: 0 -> "A". */
export function slotLetter(index: number): string {
  return String.fromCharCode("A".charCodeAt(0) + index);
}

// --- Platform geometry ---------------------------------------------------

/**
 * A platform's safe area as prose. Mirrors `SafeAreaBox.describe()`.
 *
 * Only the non-zero edges, in the Python's order. Two reasons the zeroes are left
 * out and not printed as "0% left": a true statement that carries no information
 * makes the two that matter harder to read, and this string sits in a form hint
 * where there is room for one line.
 *
 * The wizard printed only top and bottom before the geometry was a model, because
 * the dict it read had only those two keys. Reels chrome also covers 14% of the
 * right edge, which is where a CTA ends up underneath the share button — the field
 * existed in Python the whole time and never crossed the wire.
 */
export function safeAreaSummary(area: SafeAreaBox): string {
  const edges: [string, number][] = [
    ["top", area.top],
    ["bottom", area.bottom],
    ["left", area.left],
    ["right", area.right],
  ];
  const parts = edges
    .filter(([, value]) => value > 0)
    .map(([name, value]) => `${Math.round(value * 100)}% ${name}`);
  return parts.length ? parts.join(", ") : "none";
}

// --- Media ---------------------------------------------------------------

/**
 * Where to actually fetch an asset from.
 *
 * `LocalStorage.url_for` returns a root-relative `/media/<key>`, which is correct
 * for the FastAPI dev harness served from the same origin and wrong here: from the
 * app on port 3000 it resolves against Next, which has no such route, and every
 * image on the results page renders as a broken icon. So a root-relative URL gets
 * the API's origin put back in front of it.
 *
 * An absolute URL is left alone — that is the deployment case, where object storage
 * hands out a signed URL that must not be rewritten.
 */
export function resolveMediaUrl(
  asset: { key: string; url: string | null } | null | undefined,
  apiBase: string,
): string | null {
  if (!asset) return null;
  const base = apiBase.replace(/\/$/, "");
  if (asset.url) {
    if (/^https?:\/\//i.test(asset.url)) return asset.url;
    return `${base}${asset.url.startsWith("/") ? "" : "/"}${asset.url}`;
  }
  if (!asset.key) return null;
  return `${base}/media/${asset.key}`;
}
