"use client";

/**
 * Full-size preview for a generated frame, in the window rather than in a new tab.
 *
 * The card grid crops every candidate to 4:5 with `object-cover`, which is the right
 * default for comparing five frames at a glance and the wrong one for judging any of
 * them: the crop is exactly where an out-of-safe-area product or a clipped hand
 * hides. So the frame gets a way to be seen whole, at its generated resolution,
 * without leaving the ranked view that gives it context.
 *
 * **The download goes through a blob, not `<a download>`.** The media lives on the
 * API origin (`localhost:8000`) and the app is served from another
 * (`localhost:3000`), and browsers silently ignore the `download` attribute on a
 * cross-origin href — the click navigates instead, dumping the user on a bare image
 * with their ranked results gone. Fetching the bytes (CORS already allows this
 * origin) and handing the anchor a same-origin blob URL is what makes "download"
 * mean download. It also lets the file be named after the candidate rather than
 * after a storage key.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

export type LightboxItem = {
  /** Absolute media URL, already resolved against the API origin. */
  url: string;
  /** Shown in the header — "Candidate B". */
  title: string;
  /** The shot brief, shown under the frame so the crop can be read against it. */
  caption?: string;
  /** What the saved file is called. */
  filename: string;
  width: number | null;
  height: number | null;
};

type Saving = { state: "idle" | "working" | "failed"; message?: string };

export function Lightbox({
  items,
  index,
  onClose,
  onIndex,
}: {
  items: LightboxItem[];
  /** `null` when closed. Kept by the caller so the grid owns which frame is open. */
  index: number | null;
  onClose: () => void;
  onIndex: (index: number) => void;
}) {
  const open = index !== null && index >= 0 && index < items.length;
  const at = index ?? 0;
  const item = open ? items[at] : null;
  const many = items.length > 1;

  const [saving, setSaving] = useState<Saving>({ state: "idle" });
  const surfaceRef = useRef<HTMLDivElement | null>(null);
  const restoreRef = useRef<HTMLElement | null>(null);

  // A failure notice belongs to one frame. Carrying it across a arrow-key step
  // would blame the next candidate for the previous one's dead URL.
  useEffect(() => {
    setSaving({ state: "idle" });
  }, [index]);

  // Focus moves into the dialog and comes back to the button that opened it, and
  // the page behind stops scrolling. Without the scroll lock a trackpad flick over
  // the backdrop scrolls the results underneath, which reads as the overlay being
  // stuck to the viewport rather than as a modal.
  useEffect(() => {
    if (!open) return;
    restoreRef.current = document.activeElement as HTMLElement | null;
    surfaceRef.current?.focus();
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
      restoreRef.current?.focus?.();
    };
  }, [open]);

  const step = useCallback(
    (delta: number) => {
      if (!many) return;
      onIndex((at + delta + items.length) % items.length);
    },
    [at, items.length, many, onIndex],
  );

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      } else if (event.key === "ArrowRight") {
        step(1);
      } else if (event.key === "ArrowLeft") {
        step(-1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose, step]);

  const save = useCallback(async () => {
    if (!item) return;
    setSaving({ state: "working" });
    try {
      const response = await fetch(item.url);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const blob = await response.blob();
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = item.filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      // Revoked on a delay, not synchronously: WebKit cancels a download whose
      // object URL is released in the same tick as the click.
      window.setTimeout(() => URL.revokeObjectURL(href), 1000);
      setSaving({ state: "idle" });
    } catch (error) {
      setSaving({
        state: "failed",
        message: error instanceof Error ? error.message : "the fetch failed",
      });
    }
  }, [item]);

  if (!open || !item || typeof document === "undefined") return null;

  const size = item.width && item.height ? `${item.width} × ${item.height}` : null;

  return createPortal(
    <div
      ref={surfaceRef}
      role="dialog"
      aria-modal="true"
      aria-label={`${item.title}, full size`}
      tabIndex={-1}
      className="fixed inset-0 z-50 flex flex-col bg-zinc-950/95 outline-none backdrop-blur-sm"
    >
      <header className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-2 border-b border-zinc-800 px-4 py-2.5">
        <h2 className="text-sm font-semibold text-zinc-100">{item.title}</h2>
        {size ? <span className="text-[11px] text-zinc-500 tabular">{size}</span> : null}
        {many ? (
          <span className="text-[11px] text-zinc-500 tabular">
            {at + 1} / {items.length}
          </span>
        ) : null}

        <div className="ml-auto flex items-center gap-2">
          <button
            type="button"
            onClick={save}
            disabled={saving.state === "working"}
            className="inline-flex items-center gap-2 rounded-md bg-zinc-100 px-3 py-2 text-sm font-medium text-zinc-900 transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-40"
          >
            <span aria-hidden>↓</span>
            {saving.state === "working" ? "Saving…" : "Download"}
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close preview"
            className="inline-flex items-center gap-2 rounded-md border border-zinc-700 px-3 py-2 text-sm font-medium text-zinc-300 transition hover:bg-zinc-800"
          >
            <span aria-hidden>✕</span>
            Close
          </button>
        </div>
      </header>

      {saving.state === "failed" ? (
        <p className="shrink-0 border-b border-bad-500/30 bg-bad-950 px-4 py-2 text-[11px] text-bad-400">
          Could not save the file — {saving.message}. The frame is still at{" "}
          <a href={item.url} target="_blank" rel="noreferrer" className="underline">
            {item.url}
          </a>
          .
        </p>
      ) : null}

      {/* Clicking the surround closes; clicking the frame does not. */}
      <div
        className="relative flex min-h-0 flex-1 items-center justify-center p-4"
        onClick={onClose}
      >
        {many ? (
          <ArrowButton side="left" onClick={() => step(-1)} />
        ) : null}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={item.url}
          alt={item.title}
          onClick={(event) => event.stopPropagation()}
          className="max-h-full max-w-full cursor-default object-contain"
        />
        {many ? <ArrowButton side="right" onClick={() => step(1)} /> : null}
      </div>

      {item.caption ? (
        <p className="shrink-0 border-t border-zinc-800 px-4 py-2.5 text-center text-[11px] leading-snug text-zinc-400">
          {item.caption}
        </p>
      ) : null}
    </div>,
    document.body,
  );
}

function ArrowButton({ side, onClick }: { side: "left" | "right"; onClick: () => void }) {
  return (
    <button
      type="button"
      aria-label={side === "left" ? "Previous candidate" : "Next candidate"}
      onClick={(event) => {
        event.stopPropagation();
        onClick();
      }}
      className={`absolute top-1/2 -translate-y-1/2 rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-2 text-lg text-zinc-300 transition hover:bg-zinc-800 hover:text-zinc-100 ${
        side === "left" ? "left-3" : "right-3"
      }`}
    >
      <span aria-hidden>{side === "left" ? "‹" : "›"}</span>
    </button>
  );
}
