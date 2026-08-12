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

const TONE: Record<Tone, string> = {
  neutral: "bg-zinc-800 text-zinc-300 ring-zinc-700",
  good: "bg-good-950 text-good-400 ring-good-500/30",
  warn: "bg-warn-950 text-warn-400 ring-warn-500/40",
  bad: "bg-bad-950 text-bad-400 ring-bad-500/40",
  info: "bg-sky-950 text-sky-300 ring-sky-500/30",
};

export function Badge({
  children,
  tone = "neutral",
  title,
}: {
  children: ReactNode;
  tone?: Tone;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium ring-1 ring-inset ${TONE[tone]}`}
    >
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
    <Tag className={`rounded-lg border border-zinc-800 bg-zinc-900/50 ${className}`}>{children}</Tag>
  );
}

export function SectionTitle({ children, hint }: { children: ReactNode; hint?: string }) {
  return (
    <div className="mb-3 flex items-baseline justify-between gap-4">
      <h2 className="text-sm font-semibold tracking-wide text-zinc-300 uppercase">{children}</h2>
      {hint ? <p className="text-xs text-zinc-500">{hint}</p> : null}
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
  const base =
    "inline-flex items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40";
  const look =
    variant === "primary"
      ? "bg-zinc-100 text-zinc-900 hover:bg-white"
      : "border border-zinc-700 text-zinc-300 hover:bg-zinc-800";
  return (
    <button type={type} onClick={onClick} disabled={disabled} title={title} className={`${base} ${look}`}>
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
      <span className="mb-1 flex items-baseline gap-1.5 text-xs font-medium text-zinc-400">
        {label}
        {required ? <span className="text-bad-400">*</span> : null}
      </span>
      {children}
      {hint ? <span className="mt-1 block text-[11px] leading-snug text-zinc-500">{hint}</span> : null}
    </label>
  );
}

const CONTROL =
  "w-full rounded-md border border-zinc-700 bg-zinc-950 px-2.5 py-1.5 text-sm text-zinc-100 outline-none focus:border-zinc-500";

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
        <option key={option} value={option}>
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
    <label className="flex cursor-pointer items-start gap-2.5 text-sm text-zinc-300">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="mt-0.5 size-4 shrink-0 accent-zinc-100"
      />
      <span className="leading-snug">{children}</span>
    </label>
  );
}

/** A meter with no implied target. Used for progress and for the budget. */
export function Meter({ fraction, tone = "info" }: { fraction: number; tone?: Tone }) {
  const fill = {
    neutral: "bg-zinc-500",
    good: "bg-good-500",
    warn: "bg-warn-500",
    bad: "bg-bad-500",
    info: "bg-sky-500",
  }[tone];
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
      <div
        className={`h-full rounded-full transition-[width] duration-500 ${fill}`}
        style={{ width: `${Math.max(0, Math.min(1, fraction)) * 100}%` }}
      />
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-md border border-dashed border-zinc-800 px-4 py-6 text-center text-sm text-zinc-500">
      {children}
    </p>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-md border border-bad-500/30 bg-bad-950 px-3 py-2 text-sm text-bad-400">
      {children}
    </p>
  );
}

/**
 * A caveat, in the warning colour, attached to whatever it qualifies.
 *
 * Deliberately not dismissible and not a tooltip: this is how a stub score, a mock
 * generation and an unimplemented check say what they are, and a caveat that can be
 * clicked away is a caveat that will be, right before the screenshot.
 */
export function Caveat({ children }: { children: ReactNode }) {
  return (
    <p className="flex gap-1.5 text-[11px] leading-snug text-warn-400">
      <span aria-hidden>▲</span>
      <span>{children}</span>
    </p>
  );
}

export function NavLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <Link
      href={href}
      className="rounded px-2 py-1 text-sm text-zinc-400 transition hover:bg-zinc-800 hover:text-zinc-100"
    >
      {children}
    </Link>
  );
}
