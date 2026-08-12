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
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
