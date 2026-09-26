import type { Metadata } from "next";

import { Shell } from "@/components/Shell.tsx";
import { DEFAULT_CANDIDATE_COUNT, DEFAULT_VIDEO_COUNT } from "@/lib/contract.ts";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ad candidate generation",
  description:
    `Generate ${DEFAULT_CANDIDATE_COUNT} ad candidates, rank them as images, animate the ` +
    `${DEFAULT_VIDEO_COUNT} that earn it, and rank them again.`,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      {/*
        The fonts are linked at runtime rather than pulled in through `next/font`,
        which fetches at build time: a build that needs the network to succeed is a
        build that fails on a train, and this project has to survive being
        demonstrated offline. If Google Fonts is unreachable the stacks fall through
        to Menlo and Georgia — the app looks slightly different, never broken.

        Same reasoning, and the same two families, as the landing page used to load
        for itself. It loads here instead, once, for every route.

        `precedence` is not decoration: without it React refuses to hoist a
        stylesheet into <head>, leaves the tag where it was written — as a child of
        <html>, which is invalid HTML — and the page hydrates with an error. With it,
        React hoists and de-duplicates by href.
      */}
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
      <link
        rel="stylesheet"
        precedence="default"
        href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Playfair+Display:ital,wght@0,400;0,500;0,700;1,400&display=swap"
      />
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
