# Multi-Candidate Multimodal Advertisement Generation
### with Learning-Based Performance Prediction and Intelligent Ranking

A user uploads a human-model photo and a product photo plus creative controls. The
system generates **3 image candidates**, animates each into an **8–10 s video**,
then **predicts and ranks** all three at both stages and delivers platform-ready
output with social previews.

---

## The research claim

Every image becomes a video, so there is no budget-driven selection to defend.
The claim is sharper than that: **early performance prediction.**

The three candidates are ranked at the image stage, ranked again at the video
stage, and the question is whether the first ranking predicts the second. Every
job produces one paired `(image_rank, video_rank)` observation, so the evaluation
dataset builds itself as the system is used.

The cost story is then reported as a counterfactual — *had we promoted only the
top-1 image, how often would we have kept the human-preferred video?* — which
quantifies the saving without having had to make the cut.

The pipeline is built so this is measured honestly rather than reconstructed:
the image-stage ranking is committed to the record **before any video exists**,
and there is a test that enforces the ordering (`test_pipeline.py::
test_image_ranking_is_recorded_before_video_generation`).

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"
```

```bash
.venv/bin/python -m uvicorn adapi.main:app --reload --port 8077
```

Open <http://localhost:8077> and press **Run demo job**. It runs the full
eight-stage pipeline against synthetic references, costs nothing, and needs no
API keys.

```bash
.venv/bin/python -m pytest tests/ -q
```

---

## Money

The entire project budget is **$35**. A 3-candidate, 9-second job on the cheapest
premium tier costs about **$2.82**, so roughly twelve jobs exist in the whole
budget. Everything below follows from that.

| Line | Allocation | Buys |
|---|---|---|
| Development (mock mode) | $0 | Unlimited iteration |
| Training image corpus | $9 | ~300 images → the annotation corpus |
| Research-tier video | $0 | ~60 videos, LTX-Video/Wan on Colab free |
| Premium jobs | $20 | 7 full 3×3 jobs → golden demos |
| LLM brief compilation | $2 | ~200 briefs |
| Reserve | $4 | Reruns |

**Two generation tiers** are what make this work. Research volume comes free from
open weights on a Colab GPU; demo quality comes from the paid API. The comparison
between them is itself a reportable result.

### What stops the budget being lost to a bug

`AD_PROVIDER_MODE=mock` is the default, and flipping it to `live` is the only way
money can be spent. Beyond that, every paid call goes through `CostGovernor`:

- **Reserve before calling, settle after.** A charge is written to the ledger
  *before* the HTTP request. A crash mid-call cannot lose the charge.
- **Reserved money counts as spent**, so concurrent calls cannot each look
  affordable and collectively overrun the cap.
- **Refuses loudly.** Over budget raises `BudgetExceeded`; it never quietly
  degrades, because a silent downgrade is a bug you discover in the write-up.
- **Retries capped per candidate slot**, so a candidate that keeps failing the
  gate costs at most two generations.
- **Cache keyed on `hash(inputs + seed)`** — an identical resubmission is free.
- **Mock mode asserts every estimate is zero**, so a live provider leaking into
  what was meant to be a free run fails immediately instead of silently.

These are enforced by `tests/test_governor.py`, not just intended.

---

## Pipeline

```
intake → brief → images → gate → image rank → videos → video rank → delivery
```

The gate and the image-stage ranking both sit *before* video generation, which is
where nearly all the cost is.

| Stage | What happens |
|---|---|
| 1 **intake** | Validate; consent + rights gate; palette extraction |
| 2 **brief** | Design-space sampler picks 3 separated points; LLM expands each into a shot brief |
| 3 **images** | 3 multi-reference compositions, one per brief, fixed seeds |
| 4 **gate** | Hard pass/fail: palette ΔE, safe area, focal clarity, exposure, contrast |
| 5 **image rank** | Predict performance from the still, **before paying for video** |
| 6 **videos** | 3 image-to-video calls at 8–10 s |
| 7 **video rank** | Hook strength, temporal consistency, motion, screen time → final order + explanations |
| 8 **delivery** | Platform renders, previews, downloads |

### Two design decisions worth knowing

**Diversity is geometric, not prompted.** Asking an LLM for "three different
ideas" gives candidates whose diversity cannot be measured, reproduced, or
ablated. Instead the creative space is four categorical axes (camera angle,
lighting, composition, motion) and a Latin-hypercube construction draws distinct
levels on each. Every pair of candidates then differs on every axis — minimum
pairwise Hamming distance 4, the maximum possible. Verified across 800 seed×mood
combinations.

**The gate and the predictor are kept apart.** The gate answers *is this usable at
all* (hard reject). The predictor answers *how well will it perform* (continuous
score), and only ever runs on candidates that already passed. Expressing an
unusable candidate as a low score would let it rank first on a bad day.

---

## Honesty markers

Scores carry `is_stub=True` and `model_version="heuristic-0"`. What exists today
is a documented linear blend over real numpy features — which is also one of the
baselines the trained model must beat, so it is the control condition rather than
throwaway scaffolding.

Gate checks needing model weights this machine cannot host carry
`implemented=False` and pass by default: `product_identity` (DINOv2),
`face_identity` (ArcFace), `nsfw`. They are visibly distinct from checks that
genuinely verified something, so nothing claims identity verification that never
ran. `GateResult.verified_identity` returns False until they land.

Components that cannot be computed are `None` rather than invented —
`prompt_alignment` needs CLIPScore. A null shows up as absent in the ablation
table instead of quietly diluting a feature group's apparent contribution.

---

## Layout

```
apps/web/            Next.js product UI — not built yet (week 13)
services/api/        FastAPI: jobs, SSE progress, budget, media
  adapi/dev.html     single-file inspection harness (not the product UI)
services/worker/     the pipeline: sampler, briefs, gate, scoring, orchestrator
packages/schema/     the job contract — single source of truth, no heavy deps
packages/providers/  provider ABCs, MockProvider, CostGovernor, ledger, storage
packages/ml/         numpy features now; torch extractors + trained head later
scripts/             calibrate_gate.py
notebooks/           Colab: Stage A pretrain, Stage B calibration, ablations
fixtures/            uploads, generations, golden demo set
tests/               32 tests
```

`packages/schema` imports no torch, no provider SDK and no DB driver, so it stays
importable on an 8 GB laptop and inside a Colab notebook alike.

---

## Thresholds are calibrated, not guessed

```bash
.venv/bin/python scripts/calibrate_gate.py
```

Renders the frame grid twice — once with the brand palette it should honour, once
with a deliberately unrelated palette — and reports the two distributions. Those
are the positive and negative classes; a usable threshold sits between them. If
they overlap, the metric is not discriminating and the threshold is not the thing
to change.

Current separation on `palette_adherence`: on-brand 0.912–0.972, off-brand 0.000.

Two findings from the first run worth recording, because both were metric bugs
rather than threshold bugs:

- The mock renderer's tone curve was applied per channel around mid-grey, which
  **desaturates** — brand navy came out grey-blue. Scaling all channels by a gain
  derived from luminance preserves hue.
- Matching only against exact palette entries is the wrong question. A gradient
  between two brand colours contains midpoints belonging to neither, and both are
  on-brand. The permissible set is now palette colours, blends between them, and
  tints/shades toward white and black — what a brand guideline actually allows.

Re-run against real generations in week 6 before trusting these with money.

---

## Environment constraints

Verified on this machine, and they shaped the architecture:

| Resource | Reality | Consequence |
|---|---|---|
| Apple M1, 8 GB RAM | No local diffusion | Generation via hosted APIs or Colab |
| ~10 GB free disk | Won't fit torch + node_modules + media | **Free ≥40 GB before week 9.** Media in object storage; store embeddings, not files |
| No GPU | No local training | Colab free tier |
| No ffmpeg | Needed for chaining, smart crop, audio mix | `brew install ffmpeg` before week 13. Mock video is animated GIF meanwhile, with real frame timings so duration checks hold |

---

## Open items

- [ ] **Verify a provider does native ≥10 s image-to-video** before committing
      budget (week 5). Most cap at 5 s; the fallback is 5+5 chaining, whose seam
      drift is measured and reported rather than hidden.
- [ ] **Start collecting pairwise annotations by week 8.** This is the critical
      path — if it slips, the trained predictor has no labels and the research
      half of the project collapses.
- [ ] Intake preprocessing: `rembg` product cutout, palette extraction, face
      detection (week 3–4). The cutout matters more than it sounds: a clean
      transparent PNG substantially improves product fidelity versus a cluttered
      source photo.
- [ ] Live image provider adapter + gate recalibration on real generations.
- [ ] Colab: embedding extractors, Stage A pretrain, Stage B Bradley-Terry
      calibration, ablation tables.
- [ ] Next.js product UI, smart crop, ffmpeg audio mix, platform previews.
