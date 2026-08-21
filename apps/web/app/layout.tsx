import type { Metadata } from "next";

import { Shell } from "@/components/Shell.tsx";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ad candidate generation",
  description:
    "Generate three ad candidates, rank them as images, animate each, and rank them again.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      {/*
        `suppressHydrationWarning` is here for browser extensions, not for our own
        mismatches. ColorZilla stamps `cz-shortcut-listen="true"` onto <body> before
        React hydrates, and Grammarly and the password managers do the same with
        attributes of their own — the server sends a bare <body>, the client finds a
        decorated one, and Next raises a hydration error about markup this app never
        wrote. Left alone it is a permanent red overlay on every page, which is worse
        than no overlay: it is how a real mismatch gets waved past.

        It suppresses exactly one element deep — <body>'s own attributes and text —
        so a genuine mismatch anywhere inside <Shell> still reports normally. That
        narrowness is the reason this is acceptable rather than a gag.
      */}
      <body suppressHydrationWarning>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
