# Week-5 provider spike

**Question the plan asked:** *does any provider generate 8–10 s of image-to-video
natively, at a price the $35 budget can absorb — or must we chain two 5 s clips?*

**Answer: yes, natively. Chaining is not required.** Several models reach the
window in one call, and the cheapest of them is cheaper than the plan assumed.

Researched 11 August 2026 from vendor documentation. **No API call has been made
and nothing has been spent.** Prices and parameters below are what the vendors
publish; `scripts/smoke_live.py` exists to check those claims against reality for
$0.39 total, and that should happen before any bulk run.

---

## Image-to-video: what reaches 8–10 s

| Model | Native duration | Price/s | 10 s clip | Notes |
|---|---|---|---|---|
| **Kling 2.5 Turbo Pro** | **{5, 10}** | **$0.07** | **$0.70** | Cheapest that reaches the window. **Chosen.** |
| Kling 2.6 Pro | {5, 10} | $0.07 | $0.70 | Same price, newer |
| Kling 3 Pro | 3–15 continuous | $0.112 | $1.12 | 9 s requestable, 1.6× the price |
| Veo 3.1 Fast | {4, 6, 8} | $0.10 | — (8 s = $0.80) | Native audio at $0.15/s |
| Seedance 2.0 | 4–15 | $0.30 | $3.00 | 4× Kling; the plan's $0.12/s was optimistic |
| Sora 2 Pro | {4, 8, 12, 16, 20} | $0.30 | — | Out of reach |
| Veo 3.1 | {4, 6, 8} | $0.20 ($0.40 w/ audio) | — | Out of reach |

**Chosen: Kling 2.5 Turbo Pro** — `fal-ai/kling-video/v2.5-turbo/pro/image-to-video`.
$0.35 for 5 s, then $0.07/s, so a 10 s clip is $0.70.

## Multi-reference image composition

**Chosen: Seedream 4.5 edit** — `fal-ai/bytedance/seedream/v4.5/edit`, $0.04/image,
up to **10 reference images** in one call, and it **honours a seed**.

Both live on fal, which is the point: one account to fund, one key to protect, one
statement to reconcile against the ledger. For a solo build with a $35 ceiling
that is worth more than shaving cents off per-model pricing.

---

## Three findings that changed the code

### 1. Duration is an enum, and 9 is not in it

Kling's image-to-video endpoint takes `duration` as the **string enum `"5"` or
`"10"`**. The project commits to an 8–10 s window and defaults to 9 s — which is
simply not requestable.

The request schema stays continuous (8.0–10.0) because other providers *are*
continuous, and the provider layer resolves it: `VideoPrice.snap_duration` rounds
**up** to the next available option, and `estimate_video_cost` prices what will be
delivered rather than what was asked for. Rounding down would break the stated
8 s floor to save two cents; under-estimating would let the governor reserve less
than the real charge, which is precisely how a budget cap gets quietly exceeded.

A 9 s request therefore delivers and bills at 10 s, and `VideoCandidate` records
both `duration_seconds` (10) and `requested_duration_seconds` (9).

There is a small upside: on a {5, 10} provider an 8 s and a 9 s request produce the
same clip, so the cache is keyed on the delivered duration and they share an entry.

### 2. The video stage cannot be seeded

Kling's image-to-video endpoint has **no seed parameter**. Seedream does, so the
image stage is reproducible and the video stage is not.

This matters to the research claim rather than to the product. Identical inputs
give a different clip each run, so a video-stage ablation is not a controlled
comparison and cannot be reported as one. `VideoPrice.honours_seed` and
`VideoCandidate.seed_honoured` carry it into the data, so the write-up cannot
quietly assume determinism it never had.

Two options when the evaluation gets there, to decide with data rather than now:
generate several clips per condition and report variance, or pay 1.6× for Kling 3
Pro if it turns out to expose a seed.

### 3. fal cannot reach this laptop's storage

fal takes inputs as URLs. Media currently lives in `LocalStorage` served on
`localhost`, which is not reachable from anywhere else, and there is no public
bucket yet.

`as_data_uri()` inlines the bytes as a `data:` URI, which works today and costs
~33% base64 overhead on the upload. Replace it with signed Supabase URLs when
storage moves off the laptop — it is a one-function change, flagged in
`adproviders/fal.py`.

---

## Budget, recomputed against verified prices

A default job is **$2.22**, against the plan's $2.82 — the spike made the project
*cheaper*, because Kling 2.5 Turbo Pro reaches 10 s at $0.07/s.

| Line | Cost |
|---|---|
| 3 images @ $0.04 | $0.12 |
| 3 videos @ 10 s × $0.07 | $2.10 |
| 1 LLM brief call | $0.004 |
| **Per job** | **$2.22** |

| Allocation | Budget | Buys |
|---|---|---|
| Development (mock mode) | $0 | Unlimited |
| Contract smoke tests | $0.39 | One real call per adapter |
| Training image corpus | $12.00 | 300 images @ $0.04 |
| Research-tier video | $0 | ~60 clips, LTX-Video/Wan on Colab free |
| Premium jobs | $17.76 | **8 full jobs** @ $2.22 |
| LLM briefs | $1.00 | ~250 briefs |
| Reserve | $3.85 | Reruns, mistakes |
| **Total** | **$35.00** | |

`AD_BUDGET_PER_JOB_USD` can come down from $3.50 to $2.50 — comfortably above the
$2.22 job cost, and it would catch a runaway before it costs a whole extra job.

---

## Before spending anything in bulk

1. `scripts/smoke_live.py --image --confirm-spend` — $0.04. Confirms the parameter
   names, that the seed is echoed back, and the returned dimensions.
2. `scripts/smoke_live.py --video --confirm-spend` — $0.35 at the 5 s option.
   Confirms the duration enum, the response shape, and **that the file really is
   the length the API claims** — play it and check.
3. Recalibrate the quality gate against real generations. Its thresholds were set
   against the mock renderer, and `THRESH_PALETTE`/`THRESH_SAFE_AREA`/`THRESH_FOCUS`
   have no standing until they have seen real output.
4. Only then bulk-generate the 300-image annotation corpus.

## Sources

- [Best image-to-video APIs 2026 — durations and pricing](https://fal.ai/learn/tools/best-image-to-video-apis-2026)
- [Kling Video V2.5 Turbo Pro image-to-video API](https://fal.ai/models/fal-ai/kling-video/v2.5-turbo/pro/image-to-video/api)
- [Kling Video V2.5 Turbo Pro API reference](https://fal.ai/docs/model-api-reference/video-generation-api/kling-video-v2.5-turbo-pro)
- [Seedream V4 edit API](https://fal.ai/models/fal-ai/bytedance/seedream/v4/edit)
- [Seedream V4.5 API reference](https://fal.ai/docs/model-api-reference/image-generation-api/bytedance-seedream-v4.5)
- [AI video API pricing 2026](https://www.buildmvpfast.com/api-costs/ai-video)
- [Seedream 4.5 pricing](https://openrouter.ai/bytedance-seed/seedream-4.5)
