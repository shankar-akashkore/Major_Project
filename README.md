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

To collect preference labels, build a corpus and share the comparison tool:

```bash
.venv/bin/python scripts/generate_corpus.py --sets 20 && .venv/bin/python scripts/build_corpus.py --commit
```

Then open <http://localhost:8077/api/annotate/ui>, or share that URL on your
network. `scripts/annotation_report.py` says whether the labels are usable —
see [docs/annotation-protocol.md](docs/annotation-protocol.md).

```bash
.venv/bin/python -m pytest tests/ -q
```

---

## Money

The entire project budget is **$35**. A 3-candidate, 9-second job on the cheapest
premium tier costs about **$2.82**, so roughly twelve jobs exist in the whole
budget. Everything below follows from that.

Verified against vendor pricing in the week-5 spike — a default job is **$2.22**,
down from the plan's $2.82, because Kling 2.5 Turbo Pro reaches 10 s natively at
$0.07/s. Full workings in [docs/provider-spike.md](docs/provider-spike.md).

| Line | Allocation | Buys |
|---|---|---|
| Development (mock mode) | $0 | Unlimited iteration |
| Contract smoke tests | $0.39 | One real call per live adapter |
| Training image corpus | $12 | 300 images @ $0.04 → the annotation corpus |
| Research-tier video | $0 | ~60 videos, LTX-Video/Wan on Colab free |
| Premium jobs | $17.76 | 8 full 3×3 jobs @ $2.22 → golden demos |
| LLM brief compilation | $1 | ~250 briefs |
| Reserve | $3.85 | Reruns |

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
| 1 **intake** | Consent + rights gate; reference validation; product cutout; palette extraction |
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

**Provider limits are modelled, not assumed away.** Kling's image-to-video
endpoint takes `duration` as the enum `{5, 10}` — there is no 9, which is the
project's default. So a request snaps *up* to the next real option, and the cost
is estimated on what will be delivered: rounding down would break the stated 8 s
floor to save two cents, and under-estimating is exactly how a budget cap gets
quietly exceeded. That endpoint also has no seed, so the image stage is
reproducible and the video stage is not; `seed_honoured` carries that into the
data rather than leaving it as a footnote the evaluation might forget.

**The brand palette is read from inside the product, not off the photograph.** The
backdrop covers most of a product shot's pixels, so quantising the whole frame
describes the backdrop. Intake cuts the product out first and reads dominant
colours from inside the mask. Measured on a lifestyle fixture whose true product
colours are known: the frame yields `#6f614b #cabba6 #e6e1d5 #2c3b54 #a15d4d` —
four backdrop colours, product fourth — and the mask yields `#2b3a55 #f0ece2
#c6a05c`, which is exactly the product's three colours. This matters downstream
because that palette then constrains the quality gate.

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

Intake reports what it actually did. `CutoutReport.method` is `rembg`,
`flood-fill` or `none`, and `accepted=False` says in prose that the generator saw
the original photograph instead — a silent fallback would look identical in the
output. `FaceReport.implemented=False` when no detector was installed, with `box`
left `None` rather than guessed. `palette_source` distinguishes a palette read
from inside the product mask from one read off the whole frame, because those are
not equally trustworthy constraints.

**Measurements are separated from blocking checks.** `ReferenceReport.checks`
holds only what can refuse a job; `ReferenceReport.measurements` holds quantities
that are real but whose thresholds are not yet calibrated against real uploads.
`focus` currently sits in the second group — see below.

---

## Layout

```
apps/web/            Next.js product UI — not built yet (week 13)
services/api/        FastAPI: jobs, SSE progress, budget, media, annotation
  adapi/dev.html     single-file inspection harness (not the product UI)
  adapi/annotate.html  the 2AFC comparison tool, shareable, no build step
services/worker/     the pipeline: sampler, briefs, gate, scoring, orchestrator
packages/schema/     the job contract — single source of truth, no heavy deps
packages/providers/  provider ABCs, MockProvider, CostGovernor, ledger, storage
packages/ml/         numpy features, pair design, Bradley-Terry + metrics;
                     torch extractors and the trained head land later
scripts/             calibrate_gate.py, calibrate_intake.py, smoke_live.py,
                     generate_corpus.py, build_corpus.py, annotation_report.py
notebooks/           Colab: Stage A pretrain, Stage B calibration, ablations
fixtures/            uploads, generations, annotation corpus, golden demo set
tests/               127 tests
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

### Intake thresholds

```bash
.venv/bin/python scripts/calibrate_intake.py
```

Because the fixtures are drawn from known geometry, the true product mask is
known, so the flood-fill threshold is set from the segmentation's measured IoU
rather than from how plausible its output looks.

| Threshold | Value | Measured separation |
|---|---|---|
| `MIN_EDGE_PX` (blocking) | 256 | Pixel count needs no calibration to interpret |
| `MIN_BORDER_UNIFORMITY` | 0.35 | Backdrops that flood correctly 0.498–1.000 → IoU 0.985; ones that do not 0.208–0.217 → IoU 0.25 |
| `FLOOD_TOLERANCE_DE` | 12.0 | ΔE 8 leaves a gradient sweep at IoU 0.419; 12 lifts it to 0.985; 24 adds nothing |
| `MIN_SHARPNESS` | 0.35 | **Advisory, not blocking** — see below |

Two findings from this run:

- **Global sharpness punishes shallow depth of field**, and a product shot with a
  beautifully blurred background is the most common kind of good product
  photograph. Sharpness is now measured on the sharpest few of 16 tiles, so the
  question answered is "is anything in focus?" Bokeh fixtures went from failing to
  scoring 0.961 while a genuinely blurred frame still scores 0.026.
- **The focus threshold could not be honestly set from these fixtures.** Vector
  drawings have hard edges and so carry far more Laplacian energy than photographic
  detail: the usable class scored 0.961–1.000, a range real photographs do not
  occupy. A threshold drawn from that separation looked safe and was not — a
  flat-shaded synthetic portrait measures 0.108 and would have been refused. So
  `focus` is measured and reported on every job and blocks nothing, and recording
  it everywhere is what will make the week-6 calibration possible. Related known
  weakness: Laplacian variance reads sensor grain as detail, so it measures
  high-frequency energy rather than focus as such.

---

## Environment constraints

Verified on this machine, and they shaped the architecture:

| Resource | Reality | Consequence |
|---|---|---|
| Apple M1, 8 GB RAM | No local diffusion | Generation via hosted APIs or Colab |
| ~23 GB free disk | Won't fit torch + node_modules + media | **Free ≥40 GB before week 9.** Media in object storage; store embeddings, not files |
| OpenCV 5.0 ships an empty `cv2/data/` | The bundled Haar cascade XMLs were dropped, and face detection is the only thing OpenCV is here for | Pinned to `>=4.10,<5`. Verified: 4.14.0 bundles 17 cascades, 5.0.0 bundles none |
| No GPU | No local training | Colab free tier |
| No ffmpeg | Needed for chaining, smart crop, audio mix | `brew install ffmpeg` before week 13. Mock video is animated GIF meanwhile, with real frame timings so duration checks hold |

---

## Open items

- [x] **Verify a provider does native ≥10 s image-to-video** before committing
      budget (week 5). **Settled: yes, natively — chaining is not required.**
      Kling 2.5 Turbo Pro does 10 s at $0.07/s. See
      [docs/provider-spike.md](docs/provider-spike.md).
- [ ] **Run the two contract smoke tests ($0.39 total) before any bulk run.** The
      live adapters are written against fal's published schemas and tested
      against a stub transport, which proves they implement the documentation and
      cannot prove the documentation is right. `scripts/smoke_live.py --image
      --confirm-spend` then `--video --confirm-spend`, and play the clip to check
      it really is the length the API claims.
- [x] **The annotation tool and the label pipeline** (week 7–8): pair design
      with a guaranteed-connected comparison graph, a shareable 2AFC tool,
      Bradley-Terry fitting, and the quality-control machinery. See
      [docs/annotation-protocol.md](docs/annotation-protocol.md).
- [ ] **Start collecting real pairwise annotations by week 8.** The tool is
      built and verified against a simulated session; no human has used it. This
      is the critical path — if it slips, the trained predictor has no labels and
      the research half of the project collapses. Recruit early: the design costs
      ~18 minutes each across six annotators.
- [ ] **Re-derive the annotator-quality thresholds from the first real session.**
      `MIN_REPEAT_CONSISTENCY`, `MIN_PLAUSIBLE_LATENCY_MS` and `MAX_SIDE_BIAS_Z`
      are 2AFC-literature starting points, not measurements of this task. Section
      3 of `annotation_report.py` prints what to replace them with. Do it before
      excluding anyone's work.
- [x] Intake preprocessing: product cutout, palette extraction, face detection,
      reference validation (week 3–4). Cutout via `rembg` when installed, with a
      flood-fill fallback that is only trusted when the backdrop measures flat
      enough for it. Palette read from inside the product mask. Face detection via
      OpenCV Haar, advisory only.
- [ ] `pip install '.[intake]'` to enable `rembg` matting. Everything works
      without it — the flood-fill fallback covers plain backdrops and declines
      busy ones — but a trained matte handles lifestyle product photos, which the
      fallback correctly refuses to touch.
- [ ] Promote `focus` from measurement to blocking check once real uploads have
      supplied an honest distribution (week 6).
- [x] Live image and video adapters (`adproviders.fal`), registered so
      `AD_PROVIDER_MODE=live` resolves them and refusing to build without a key.
- [ ] Gate recalibration on real generations. `THRESH_PALETTE`,
      `THRESH_SAFE_AREA` and `THRESH_FOCUS` were set against the mock renderer
      and have no standing until they have seen real output — do this before the
      300-image corpus, not after.
- [ ] Colab: embedding extractors, Stage A pretrain, Stage B Bradley-Terry
      calibration, ablation tables.
- [ ] Next.js product UI, smart crop, ffmpeg audio mix, platform previews.
