/**
 * The FastAPI client.
 *
 * Two things here are deliberate.
 *
 * **Errors carry the server's own message.** FastAPI puts the useful text in
 * `detail`, and the two refusals this app can actually provoke are worth reading
 * verbatim: the consent gate's 422 explains which attestation is missing, and the
 * cost governor's refusal explains which cap was hit. A generic "request failed"
 * turns both into a mystery.
 *
 * **Two response shapes are hand-written, and marked.** `/api/config` and
 * `/api/golden` return ad-hoc dicts rather than Pydantic models, so
 * `scripts/export_types.py` has nothing to generate from and these cannot be
 * drift-checked. They are the exception, not the pattern — everything else comes
 * from `contract.ts`.
 */

import type {
  AspectRatio,
  BudgetStatus,
  DeliveryReport,
  JobRecord,
  Platform,
} from "./contract.ts";
import { resolveMediaUrl } from "./presentation.ts";

/**
 * The API origin. Same-origin would need a Next rewrite in front of the SSE
 * stream, and a proxy that buffers turns a live progress bar into one that jumps
 * to 100% at the end. `CORSMiddleware` in `adapi.main` already allows this origin.
 */
export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000").replace(
  /\/$/,
  "",
);

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  } catch {
    // A dead API is the single most likely failure in development, and "fetch
    // failed" does not say which of the two servers is not running.
    throw new ApiError(0, `Cannot reach the API at ${API_BASE}. Is uvicorn running?`);
  }
  if (!response.ok) {
    throw new ApiError(response.status, await errorMessage(response));
  }
  return (await response.json()) as T;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    const detail = (body as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      // Pydantic validation errors: one line each, naming the field.
      return detail
        .map((item) => {
          const loc = Array.isArray(item?.loc) ? item.loc.slice(1).join(".") : "";
          return loc ? `${loc}: ${item?.msg}` : String(item?.msg ?? item);
        })
        .join("; ");
    }
    return JSON.stringify(body);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

// --- Hand-written response shapes ---------------------------------------
// These two routes return dicts, so there is no model to generate from. If a key
// here is wrong, nothing catches it but the screen.

export type SafeAreaConfig = { top: number; bottom: number };

export type PlatformConfig = {
  aspect_ratio: AspectRatio;
  safe_area: SafeAreaConfig;
};

/** `GET /api/config` — the mode the pipeline is in, and what it costs. */
export type AppConfig = {
  provider_mode: "mock" | "live" | "replay";
  is_live: boolean;
  is_replay: boolean;
  image_provider: string;
  video_provider: string;
  golden_set: string | null;
  banner: string;
  platforms: Record<string, PlatformConfig>;
};

/** `GET /api/golden` — one frozen demo bundle. */
export type GoldenBundle = {
  slug: string;
  title: string;
  created_at: string;
  summary: string;
  frames: number;
  clips: number;
  image_model: string;
  video_model: string;
  original_cost_usd: number;
  replayable: boolean;
  problems: string[];
  winner_slot: number | null;
};

/** `GET /api/jobs` — the board's row, which is not the whole record. */
export type JobSummary = {
  job_id: string;
  state: string;
  product_name: string;
  platform: Platform;
  created_at: string;
  cost_usd: number;
  winner: number | null;
};

/**
 * `GET /api/jobs/{id}/delivery`
 *
 * The envelope is hand-written; `delivery` inside it is the generated
 * `DeliveryReport`, so the part with all the fields is still drift-checked.
 */
export type DeliveryResponse = {
  job_id: string;
  summary: string;
  delivery: DeliveryReport;
  download_url: string | null;
};

/** `GET /api/jobs/{id}/ledger` */
export type LedgerResponse = {
  job_id: string;
  total_usd: number;
  entries: Record<string, unknown>[];
};

// --- Routes --------------------------------------------------------------

export const getConfig = () => request<AppConfig>("/api/config");
export const getBudget = () => request<BudgetStatus>("/api/budget");
export const listJobs = (limit = 25) => request<JobSummary[]>(`/api/jobs?limit=${limit}`);
export const getJob = (jobId: string) => request<JobRecord>(`/api/jobs/${jobId}`);
export const getLedger = (jobId: string) => request<LedgerResponse>(`/api/jobs/${jobId}/ledger`);
export const listGolden = () => request<GoldenBundle[]>("/api/golden");

export const getDelivery = (jobId: string) =>
  request<DeliveryResponse>(`/api/jobs/${jobId}/delivery`);

export type Launched = { job_id: string; state: string };

export const replayGolden = (slug: string) =>
  request<Launched>(`/api/jobs/golden/${slug}`, { method: "POST" });

export const createDemoJob = (params: {
  platform?: string;
  duration_seconds?: number;
  seed?: number;
  brand_palette?: boolean;
}) => {
  const query = new URLSearchParams(
    Object.entries(params)
      .filter(([, v]) => v !== undefined)
      .map(([k, v]) => [k, String(v)]),
  );
  return request<Launched>(`/api/jobs/demo?${query}`, { method: "POST" });
};

/**
 * `POST /api/jobs` — multipart, because the uploads go with it.
 *
 * No `Content-Type` header: the browser has to set it so the multipart boundary
 * matches the body it generated. Setting it by hand produces a 422 that reads as a
 * validation failure on fields that are actually present.
 */
export const createJob = (form: FormData) =>
  request<Launched>("/api/jobs", { method: "POST", body: form });

// --- Media and streams ---------------------------------------------------

export const mediaUrl = (asset: { key: string; url: string | null } | null | undefined) =>
  resolveMediaUrl(asset, API_BASE);

export const bundleUrl = (jobId: string) => `${API_BASE}/api/jobs/${jobId}/bundle`;
export const eventsUrl = (jobId: string) => `${API_BASE}/api/jobs/${jobId}/events`;
