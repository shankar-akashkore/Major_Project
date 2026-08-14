/**
 * The FastAPI client.
 *
 * **Errors carry the server's own message.** FastAPI puts the useful text in
 * `detail`, and the two refusals this app can actually provoke are worth reading
 * verbatim: the consent gate's 422 explains which attestation is missing, and the
 * cost governor's refusal explains which cap was hit. A generic "request failed"
 * turns both into a mystery.
 *
 * **Nothing here is hand-written any more.** `/api/config` and `/api/golden` used
 * to return ad-hoc dicts, so their types were transcribed into this file and could
 * not be drift-checked — `docs/ui-protocol.md` §1 listed them as the exception to
 * the rule the rest of the app follows. They are Pydantic models now
 * (`adschema.api`), so every shape below comes from `contract.ts` and a renamed
 * field fails `scripts/export_types.py --check` instead of blanking a panel.
 */

import type {
  AppConfig,
  BudgetStatus,
  DeliveryResponse,
  GoldenSummary,
  JobPage,
  JobRecord,
  Launched,
  LedgerResponse,
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
  /**
   * Assigned in the body rather than declared as a constructor parameter property.
   *
   * `readonly status: number` in the signature is TypeScript that *emits code*, not
   * a type to erase, and Node's strip-only loader refuses it outright. The component
   * tests import this module, so the shorthand would have failed the whole test file
   * with a syntax error a long way from its cause.
   */
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
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

// --- Routes --------------------------------------------------------------

export const getConfig = () => request<AppConfig>("/api/config");
export const getBudget = () => request<BudgetStatus>("/api/budget");
export const listJobs = (limit = 25, offset = 0) =>
  request<JobPage>(`/api/jobs?limit=${limit}&offset=${offset}`);
export const getJob = (jobId: string) => request<JobRecord>(`/api/jobs/${jobId}`);
export const getLedger = (jobId: string) => request<LedgerResponse>(`/api/jobs/${jobId}/ledger`);
export const listGolden = () => request<GoldenSummary[]>("/api/golden");

export const getDelivery = (jobId: string) =>
  request<DeliveryResponse>(`/api/jobs/${jobId}/delivery`);

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
