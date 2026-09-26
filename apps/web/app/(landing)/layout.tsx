import type { Metadata } from "next";

import "./home.css";

export const metadata: Metadata = {
  title: "AdGen — rank your ads before you pay for them",
  description:
    "Upload a model and a product. Five ad frames are composed, checked and ranked as stills — then only what earns it is animated, and ranked again.",
};

/**
 * The landing page's own stylesheet.
 *
 * The font links used to be here. They are in the root layout now, because the
 * console pages went monochrome and want the same two families — and two layouts
 * requesting the same Google Fonts URL is one request too many on the one page a
 * stranger sees first.
 *
 * The reasoning that put them in a `<link>` rather than `next/font` still holds and
 * has just moved up a level: `next/font` fetches at build time, and a build that
 * needs the network is a build that fails on a train. If Google Fonts is
 * unreachable the stacks fall through to Iowan Old Style and Georgia — still a
 * high-contrast serif, so the page looks slightly different, never broken.
 */
export default function HomeLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
