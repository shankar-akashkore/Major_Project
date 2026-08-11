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

Once judgements exist, measure the corpus and train the predictor:

```bash
.venv/bin/python scripts/extract_features.py && .venv/bin/python scripts/train_predictor.py
```

That prints the split accounting, the annotator noise ceiling, every baseline as a
*paired* comparison, and the feature-group ablation —
see [docs/prediction-protocol.md](docs/prediction-protocol.md). The embedding
features need a GPU and come from `notebooks/colab_embeddings.ipynb`; without them
the run is a smaller experiment rather than a broken one.

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

Scores carry `model_version="heuristic-0"` until a trained model is on disk. The
heuristic is a documented linear blend over real numpy features — which is also one
of the baselines the trained model must beat, so it is the control condition rather
than throwaway scaffolding.

**A trained model is still a stub until it is shown to work.** `ModelCard.is_stub`
is True in three cases: features that came from stand-ins, a held-out accuracy that
was never measured, and — the one worth stating — an accuracy whose 95% interval
does not clear 0.5. Evaluated and found not to work is a stronger reason to label
than never evaluated at all. The first model this project saved sat at 0.496 and is
marked accordingly, in the training output and on every ranking it serves.

**A model that no longer fits its features is refused, not reinterpreted.** Saved
feature names are compared as an ordered list at serving time, so a reordering is
caught as surely as a missing column — a reordering is the more dangerous case,
because every weight still finds a number to multiply and the output stays in range.
A model trained with the Colab embedding blocks cannot be served on this laptop
(there is no torch to compute them), so the pipeline falls back to the heuristic
with a `warning` event rather than failing the job or pretending.

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

**Stand-in features cannot masquerade as real ones.** `adml.embeddings` can produce
deterministic pseudo-embeddings so the training path runs with no downloads, but
`is_real=False` travels with them into the feature table and onto every printed
result line, and `train_predictor.py` opens with a banner saying the run is not a
result. Because those vectors carry no visual information, a model trained on them
must score at chance — which makes them a leakage test as well as a placeholder.

**Every accuracy is printed against the ceiling, never against 1.0.** Human
annotators disagree with themselves on repeats, so a perfect predictor still cannot
match a single judgement all the time. At 0.60 test-retest agreement the ceiling is
0.72; a model at 0.68 is most of the way to achievable rather than mediocre. The
inversion is verified against a simulation, not asserted. Two ways to express "most
of the way" appear in this repo and they are different quantities, not a discrepancy:
`ModelCard.ceiling_fraction` is chance-corrected (`(0.68-0.5)/(0.72-0.5)` = 82%),
while the scale study in `docs/prediction-protocol.md` quotes the raw ratio
(`0.68/0.72` = 94%). The raw ratio awards a model at chance 69% of the ceiling, so
the card uses the corrected form.

**Reframing reports what it cost.** Delivering a 9:16 ad as 16:9 throws away most
of the frame. `adml.crop` places the window by saliency rather than centring it, and
every render says how much attention-weighted content survived *and* what a centre
crop would have kept — measured at 84% vs 75% for 4:5 and 56% vs 34% for 16:9. Below
70% retention it letterboxes instead of cropping and says why. A lossy reframe emits
a `warning` event rather than shipping quietly.

**Audio without a recorded licence cannot be constructed.** `adml.audio.AudioBed`
raises `LicenceMissing` without a title, source and licence. This project ships no
audio content: no bed is the normal state and the report says so, and the synthesised
tone used to exercise the mixing path is flagged `is_test_signal` so it can never be
described as a soundtrack.

**Video is decoded, not assumed.** Every clip measurement goes through
`adml.video`, which reads duration, frame rate and geometry out of the file with
ffprobe rather than trusting the provider's response. Delivered clips are checked
against the 8-10 s commitment from their bytes, and provider-versus-file drift is
emitted as a job event. Concatenation seams are detected from the pixels, so a
chained clip is identified whether or not the provider admits to one.

---

## Layout

```
apps/web/            Next.js product UI — not built yet (week 13)
services/api/        FastAPI: jobs, SSE progress, budget, media, annotation
  adapi/dev.html     single-file inspection harness (not the product UI)
  adapi/annotate.html  the 2AFC comparison tool, shareable, no build step
services/worker/     the pipeline: sampler, briefs, gate, scoring, delivery,
                     orchestrator
packages/schema/     the job contract — single source of truth, no heavy deps
packages/providers/  provider ABCs, MockProvider, CostGovernor, ledger, storage
packages/ml/         numpy features, pair design, Bradley-Terry + metrics,
                     set-wise splits, the pairwise head, the evaluation harness,
                     ffmpeg video I/O, model persistence and serving,
                     saliency-aware reframing, the audio mix
scripts/             calibrate_gate.py, calibrate_intake.py, smoke_live.py,
                     generate_corpus.py, build_corpus.py, annotation_report.py,
                     extract_features.py, train_predictor.py
notebooks/           colab_embeddings.ipynb, colab_video.ipynb — the only parts
                     that need torch or a GPU
fixtures/            uploads, generations, annotation corpus, golden demo set
tests/               298 tests
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
| ~~No ffmpeg~~ | Installed. Mock video is now H.264 MP4, so free runs exercise the same decode path as paid Kling clips | Falls back to animated GIF where ffmpeg is absent, with real frame timings either way so duration checks hold |

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
      ~20 minutes each across six annotators.
- [x] **The predictor and the evaluation harness** (week 9–11): grouped feature
      vectors sized to the label count, set-wise splitting, the numpy pairwise
      head, baselines compared with paired tests, the annotator noise ceiling, and
      the ablation runner. See
      [docs/prediction-protocol.md](docs/prediction-protocol.md).
- [ ] **The corpus needs 100 sets, and 24 will not do.** Measured: at 24 sets the
      pooled 95% margin is ±0.074, wider than every model difference that matters,
      so nothing about the predictor is resolvable. At 100 sets it is ±0.040 and
      the trained model reaches 95% of an oracle's accuracy. This is why the $12
      generation run is on the critical path rather than optional.
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
- [ ] **Run `notebooks/colab_embeddings.ipynb` on a GPU.** SigLIP and DINOv2 are
      written and the write format is asserted against the loader, but no GPU has
      executed it, so the `embedding`, `aesthetic` and `identity` rows of the
      ablation table are pending rather than zero.
- [ ] **Stage A pretraining is blocked on data access, not on code.** SMPD-Video
      and Pitt Ads both need registration. Stage B — the pairwise calibration — does
      not depend on it and runs today.
- [ ] The `identity` feature group is unbuilt: product-DINO and face-ArcFace
      similarity need the reference images alongside the candidates in Colab, which
      the manifest does not yet carry.
- [x] **The video half of the pipeline could not read a real video** (week 12).
      Every clip measurement decoded through PIL, which cannot open an MP4, and the
      mock provider wrote GIFs — so the first paid Kling clip would have raised
      `UnidentifiedImageError` in stage 7, after being billed. `adml.video` is now
      the one place a clip becomes frames, mock video is H.264 MP4, and delivered
      durations are verified from the bytes.
- [x] **The trained model can now reach the product** (week 12). `adml.serving`
      persists the fitted transform and the head together; `train_predictor.py
      --save` writes it; the pipeline loads it and falls back loudly when it does
      not fit. Before this the evaluation and the pipeline scored with different
      things and only one of them was in the report.
- [x] **The research tier exists** (week 12). `adproviders.PrerenderedVideoProvider`
      serves clips generated free on a Colab GPU from a manifest, at $0 and without
      requiring live mode. It refuses a miss rather than substituting a mock clip,
      which is what keeps the tier comparison meaningful.
- [ ] **Run `notebooks/colab_video.ipynb` on a GPU.** The provider, the manifest
      contract and the fingerprint round-trip are all tested, but no clip has been
      generated: the research-tier video corpus is empty, so the headline
      image→video rank-agreement number still has no data behind it.
- [ ] Video-stage prediction is untested end to end. The motion feature group and
      `score_video` now run against real MP4s, but only against mock ones — no
      generated video exists yet, from either tier.
- [x] **Delivery** (week 13): saliency-aware reframing with the loss measured
      against a centre crop, letterboxing when a crop would discard too much,
      platform mockup previews drawn from the same safe area the score uses, an
      ffmpeg audio mix gated on a recorded licence, report cards and a download
      bundle. See [docs/delivery-protocol.md](docs/delivery-protocol.md).
- [ ] **The API has no HTTP-level tests.** `/api/jobs/{id}/delivery` and
      `/api/jobs/{id}/bundle` were verified by hand against the real ASGI app and the
      pipeline behind them is covered, but the routes are not in the suite:
      `adapi.main` builds its store, storage and ledger at module scope, so a test
      cannot isolate them without a small refactor. Worth doing before the viva.
- [ ] **Supply a licensed music bed.** The mix is built and verified, and the project
      ships no audio, so a delivered clip is silent until a bed with recorded
      provenance is passed to the pipeline. The tone used in tests is flagged as a
      test signal and must not be shipped.
- [ ] Next.js product UI (week 13 carried forward). The FastAPI dev harness and the
      annotation tool cover inspection and label collection; the wizard, job board
      and ranked-results UI are still unbuilt.
