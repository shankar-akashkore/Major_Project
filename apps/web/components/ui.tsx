/**
 * The handful of primitives the pages share.
 *
 * Written out rather than pulled from a component library. shadcn/ui is what the
 * plan named, and it is a copy-paste catalogue rather than a dependency — so what
 * it would have contributed here is a CLI and a `components.json`, for eight
 * components that between them are a few hundred lines of Tailwind. Same reasoning
 * as `adml.figures` drawing SVG instead of installing matplotlib: this machine has
 * 19 GB free and the project has already spent its disk on the parts with no
 * alternative.
 */

import Link from "next/link";
import type { ReactNode } from "react";

export type Tone = "neutral" | "good" | "warn" | "bad" | "info";

/**
 * Tone, on a page with no colour in it.
 *
 * These five names used to be five hues — amber for a placeholder, green for a
 * passed check, red for a failure, blue for something still streaming. They were
 * never decoration: `docs/ui-protocol.md` leans on them to keep a stub score from
 * reading as a measurement, and eighteen call sites pass a tone for that reason.
 *
 * The border style and the glyph are the primary carrier, and they were built when
 * the app went fully monochrome. Colour has since come back for the three tones that
 * state a verdict — `good`, `warn`, `bad` — but strictly on top: every one of them
 * is still legible with the hue removed, which is the whole point.
 *
 * That ordering is what makes this stronger than the original. WCAG 1.4.1 asks that
 * colour never be the *only* carrier, and originally it very nearly was — a red badge
 * and a green one differed by hue alone, which is exactly the pair dichromats lose.
 * Now they differ by 2px-solid-✕ versus hairline-✓ as well, and the colour is a
 * shortcut for everyone else rather than the only way in.
 *
 * `info` and `neutral` stay grey. `info` means "running", which is a state rather
 * than a judgement, and `neutral` means "no claim" — a hue would make it look like
 * one.
 *
 * Each tone owns its border width rather than sharing a base `border`, because
 * Tailwind sorts width utilities by its own order and not by the order they appear
 * in the class string — `border-2` beside `border` is a coin toss.
 */
const TONE: Record<Tone, string> = {
  neutral: "border border-[var(--rule)] text-[var(--l-2)]",
  good: "border border-good-500/45 text-good-400",
  warn: "border border-dashed border-warn-500/55 text-warn-400",
  bad: "border-2 border-bad-500 text-bad-400",
  info: "border border-dotted border-[var(--l-3)] text-[var(--l-1)]",
};

/**
 * The second carrier, so the badge survives being printed, screenshotted in
 * greyscale, or read by someone who cannot separate the border styles.
 *
 * `neutral` gets none on purpose: it means "no claim", and a glyph would make it
 * look like one.
 */
const GLYPH: Record<Tone, string | null> = {
  neutral: null,
  good: "✓",
  warn: "▲",
  bad: "✕",
  info: "●",
};

/**
 * The filled treatment, for the one state that must never be missed.
 *
 * A solid red plate is the loudest thing on a page that is otherwise black, white
 * and three small accents. It is deliberately not reachable from `tone`: if any
 * caller could shout, everything would, and the shout would stop meaning anything.
 *
 * The text is black, not white. White on #ef4444 measures 3.76:1 and fails AA at
 * 11px; black measures 5.58:1. The one badge that must be read from across a room
 * is not the one to get that wrong.
 */
const SOLID = "border-2 border-bad-500 bg-bad-500 text-black";

export function Badge({
  children,
  tone = "neutral",
  title,
  glyph,
  solid = false,
}: {
  children: ReactNode;
  tone?: Tone;
  title?: string;
  /**
   * Overrides the tone's glyph; `null` suppresses it entirely.
   *
   * `bad` carries two meanings that a colour could hold at once and a glyph cannot.
   * Red meant both "this failed" and "this is dangerous", so `bad` is correct for a
   * rejected candidate *and* for live mode — but ✕ is only correct for the first.
   * Beside "live mode" it reads as "live mode is off", which is precisely backwards
   * and is the one misreading this app is built to prevent.
   */
  glyph?: string | null;
  /** Reserved for live mode. See `SOLID`. */
  solid?: boolean;
}) {
  const mark = glyph === undefined ? GLYPH[tone] : glyph;
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium ${
        solid ? SOLID : TONE[tone]
      }`}
    >
      {mark ? (
        <span aria-hidden className="text-[9px] leading-none">
          {mark}
        </span>
      ) : null}
      {children}
    </span>
  );
}

export function Card({
  children,
  className = "",
  as: Tag = "div",
}: {
  children: ReactNode;
  className?: string;
  as?: "div" | "section" | "li" | "article";
}) {
  return (
    <Tag className={`rounded-lg border border-[var(--rule)] bg-transparent ${className}`}>
      {children}
    </Tag>
  );
}

export function SectionTitle({ children, hint }: { children: ReactNode; hint?: string }) {
  return (
    <div className="mb-4 flex items-baseline justify-between gap-4">
      <h2 className="text-[12px] font-medium tracking-[0.16em] text-[var(--l-2)] uppercase">
        {children}
      </h2>
      {hint ? <p className="text-[11px] text-[var(--l-3)]">{hint}</p> : null}
    </div>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  type = "button",
  variant = "primary",
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  type?: "button" | "submit";
  variant?: "primary" | "ghost";
  title?: string;
}) {
  // `min-h-11` is 44px — the touch-target floor. The old `py-2` landed at about 34,
  // which is comfortable with a mouse and a miss on a phone.
  const base =
    "inline-flex min-h-11 items-center justify-center gap-2 rounded-md px-4 text-[12px] font-medium tracking-[0.06em] transition disabled:cursor-not-allowed disabled:opacity-40";
  const look =
    variant === "primary"
      ? "bg-[var(--ink)] text-[var(--paper)] hover:opacity-85"
      : "border border-[var(--l-4)] text-[var(--l-2)] hover:border-[var(--ink)] hover:text-[var(--ink)]";
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`${base} ${look}`}
    >
      {children}
    </button>
  );
}

export function Field({
  label,
  hint,
  children,
  required,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
  required?: boolean;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 flex items-baseline gap-1.5 text-[11px] tracking-[0.06em] text-[var(--l-2)]">
        {label}
        {required ? <span className="text-[var(--ink)]">*</span> : null}
      </span>
      {children}
      {/*
        `--l-3`, not the `--l-4` the mockup used for hints. `--l-4` is 0.28 opacity,
        documented as decorative and never text, and at 11px on black it measures
        about 2.3:1 — well under the AA floor. Hints here carry real instructions
        (what a field derives from, what it costs), so they have to be readable.
      */}
      {hint ? (
        <span className="mt-1.5 block text-[11px] leading-snug text-[var(--l-3)]">{hint}</span>
      ) : null}
    </label>
  );
}

const CONTROL =
  "w-full min-h-11 rounded-md border border-[var(--l-4)] bg-transparent px-3 py-2 text-[13px] text-[var(--l-1)] outline-none transition-colors hover:border-[var(--l-3)] focus:border-[var(--ink)]";

export function TextInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={CONTROL} />;
}

export function TextArea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...props} className={`${CONTROL} min-h-[4.5rem] resize-y`} />;
}

/** A select whose options come from a generated `*_VALUES` array, never a literal list. */
export function Select<T extends string>({
  value,
  onChange,
  options,
  format = (v) => String(v).replace(/_/g, " "),
}: {
  value: T;
  onChange: (value: T) => void;
  options: readonly T[];
  format?: (value: T) => string;
}) {
  return (
    <select
      value={value}
      onChange={(event) => onChange(event.target.value as T)}
      className={CONTROL}
    >
      {options.map((option) => (
        // The dropdown itself is drawn by the OS, so the one place the page cannot
        // stay black is here: an unstyled option on a transparent control is
        // unreadable in every browser. Paint it explicitly.
        <option key={option} value={option} className="bg-black text-white">
          {format(option)}
        </option>
      ))}
    </select>
  );
}

export function Checkbox({
  checked,
  onChange,
  children,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  children: ReactNode;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2.5 py-1 text-[12px] leading-relaxed text-[var(--l-2)]">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-0.5 size-4 shrink-0 accent-[var(--ink)]"
      />
      <span className="leading-snug">{children}</span>
    </label>
  );
}

/**
 * A meter with no implied target. Used for progress and for the budget.
 *
 * The fill is always ink. `tone` survives as an outline on the track — dashed when
 * the budget is tight, solid when it is spent — so the prop keeps meaning something
 * without reintroducing a hue. It is outline rather than border so the track stays
 * exactly 6px tall and the row above it does not shift when the level changes.
 */
export function Meter({ fraction, tone = "info" }: { fraction: number; tone?: Tone }) {
  const track: Record<Tone, string> = {
    neutral: "",
    good: "",
    info: "",
    warn: "outline-1 outline-dashed outline-offset-2 outline-[var(--l-4)]",
    bad: "outline-1 outline-solid outline-offset-2 outline-[var(--ink)]",
  };
  return (
    <div
      className={`h-1.5 w-full overflow-hidden rounded-full bg-[var(--rule)] ${track[tone]}`}
    >
      <div
        className="h-full rounded-full bg-[var(--ink)] transition-[width] duration-500"
        style={{ width: `${Math.max(0, Math.min(1, fraction)) * 100}%` }}
      />
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-md border border-dashed border-[var(--rule)] px-4 py-6 text-center text-[13px] text-[var(--l-3)]">
      {children}
    </p>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <p className="flex gap-2 rounded-md border border-[var(--ink)] px-3 py-2.5 text-[13px] text-[var(--ink)]">
      <span aria-hidden>✕</span>
      <span>{children}</span>
    </p>
  );
}

/**
 * A caveat, attached to whatever it qualifies.
 *
 * Deliberately not dismissible and not a tooltip: this is how a stub score, a mock
 * generation and an unimplemented check say what they are, and a caveat that can be
 * clicked away is a caveat that will be, right before the screenshot.
 *
 * It used to be amber. Now it is the ▲ and full-strength ink against body copy set
 * at 0.72 and 0.56 — the glyph was always doing most of the work, which is why the
 * footer explains the ▲ and never explained the colour.
 */
export function Caveat({ children }: { children: ReactNode }) {
  return (
    <p className="flex gap-1.5 text-[11px] leading-snug text-[var(--ink)]">
      <span aria-hidden>▲</span>
      <span>{children}</span>
    </p>
  );
}

export function NavLink({
  href,
  children,
  active = false,
}: {
  href: string;
  children: ReactNode;
  active?: boolean;
}) {
  const look = active
    ? "bg-[var(--ink)] text-[var(--paper)]"
    : "text-[var(--l-3)] hover:bg-[var(--rule-soft)] hover:text-[var(--ink)]";
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={`inline-flex min-h-11 items-center rounded px-3 text-[12px] transition ${look}`}
    >
      {children}
    </Link>
  );
}
