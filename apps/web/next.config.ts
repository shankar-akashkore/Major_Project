import type { NextConfig } from "next";

/**
 * The browser talks to FastAPI directly rather than through a Next rewrite.
 *
 * A rewrite would avoid CORS, and it would also sit in the middle of the SSE
 * progress stream — where a proxy that buffers turns a live progress bar into one
 * that jumps to 100% at the end. The API already allows this origin explicitly
 * (`CORSMiddleware`, `allow_origins=["http://localhost:3000"]`), so there is
 * nothing to work around.
 *
 * Media is served from the same API origin in development; in deployment the
 * `AssetRef.url` signed URL takes over and this base is only used for the routes.
 */
const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Generated media is served by the API from local storage, so Next's image
  // optimiser has nothing to add and would only cache stale frames.
  images: { unoptimized: true },
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;
