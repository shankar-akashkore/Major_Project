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

---

# The smoke test ran — 20 August 2026

**Both adapters called for real. $0.39 spent, exactly as forecast.** Everything
above this line was vendor documentation; everything below is measured.

**Confirmed as documented:** authentication (`Authorization: Key <id>:<secret>`),
the Seedream parameter names, **the seed is echoed back** (requested 7, returned 7
— the image-stage ablations really are reproducible), Kling's `duration` string
enum, both prices, and `seed_honoured=False` on video.

## Four things the documentation did not say

### 1. Seedream returns no dimensions at all

The `images[0]` object has no `width` and no `height`. `images[0].get("width")`
was therefore recording `None` on every generation, into an asset record the
delivery manifest treats as fact.

### 2. It returns JPEG for a `.png` output key

The recorded mime type was a guess keyed off a filename the *caller* chose. The
file on disk was JPEG.

Both are fixed the same way, by `adproviders.fal._measure_image`: read dimensions
and encoding from the bytes. The video path already did this with
`adml.video.probe`; the image path now matches it.

### 3. Requested pixel dimensions are treated as a shape, not a size

A request for 864x1536 came back **1920x3416** — the 9:16 ratio honoured to within
0.07%, the resolution Seedream's own choice. The ratio is the half that matters,
since delivery crops to platform safe areas from whatever it is handed. Do not
treat the returned size as predictable.

### 4. Kling delivers one frame long, and that broke the duration gate

A `"5"` request returned **5.041667 s** — 121 frames at 24 fps, because the
provider counts the closing frame. A `"10"` request therefore lands at ~10.0417 s.

`DurationCheck.in_window` tested `measured <= 10.0`, so **every premium clip this
project generates would have been flagged `OUTSIDE the 8-10s window`** — a warning
on every candidate, of the kind that teaches you to stop reading warnings.

Nothing in the suite could have caught it: the mock provider emits exactly 10.0 s.
Fixed with `DURATION_TOLERANCE_S = 0.1`, applied **to the ceiling only** — an
overshoot is an encoder counting frames, an undershoot is the project failing the
8 s output it promises. `VideoPrice.snap_duration` already resolves that same
asymmetry the same way.

**Also worth knowing:** the video payload sends no aspect ratio, so Kling takes it
from the start image (the smoke test's square product photo produced a 1440x1440
clip). In a real job the start image is the generated 9:16 frame, so this is
correct — but it means video geometry is inherited, never requested. The clip has
no audio track, as expected.

The raw responses are saved to `fixtures/smoke/raw_*.json`. The image call
predates that, which is why finding 1 was nearly lost to a terminal scrollback.

## Still to do before spending in bulk

1. ~~Recalibrate the quality gate against real generations.~~ Partly done — see
   "Measuring product scale" below. Every threshold has now been measured against
   eleven real frames; `THRESH_SAFE_AREA` is the one that fires, and it fired
   correctly. `THRESH_FOCUS` turns out to have far less headroom on real output
   (0.064) than the mock corpus suggested (0.33), which is worth knowing before it
   is trusted.
2. Reconcile the ledger against fal's own usage page. The ledger records the
   *estimate*; only fal knows the charge.
3. Only then bulk-generate the 300-image annotation corpus.

## Sources

- [Best image-to-video APIs 2026 — durations and pricing](https://fal.ai/learn/tools/best-image-to-video-apis-2026)
- [Kling Video V2.5 Turbo Pro image-to-video API](https://fal.ai/models/fal-ai/kling-video/v2.5-turbo/pro/image-to-video/api)
- [Kling Video V2.5 Turbo Pro API reference](https://fal.ai/docs/model-api-reference/video-generation-api/kling-video-v2.5-turbo-pro)
- [Seedream V4 edit API](https://fal.ai/models/fal-ai/bytedance/seedream/v4/edit)
- [Seedream V4.5 API reference](https://fal.ai/docs/model-api-reference/image-generation-api/bytedance-seedream-v4.5)
- [AI video API pricing 2026](https://www.buildmvpfast.com/api-costs/ai-video)
- [Seedream 4.5 pricing](https://openrouter.ai/bytedance-seed/seedream-4.5)

---

# The first full live job ran — 20 August 2026

**One premium job, $2.22.** Image quality was not the problem: Seedream's composed
frames were the best output this project has produced. **All three videos were
zooms.** A slow push across a still photograph, three times, which is a moving
poster and not an advertisement.

Kling was not at fault. It did exactly what the payload asked.

## What the payload actually asked for

`MotionIntent` had six levels and **five of them named the camera and nothing
else** — `slow_dolly_in`, `slow_dolly_out`, `orbit_left`, `handheld_drift`, and
`static_subtle`, which asked in as many words for "near-static frame with only
subtle natural movement". Only `product_present` described a human doing
anything.

The mood priors then closed the trap. The sampler draws preferred levels first,
and the default mood `warm_lifestyle` preferred
`[slow_dolly_in, product_present, static_subtle]`. Over three candidates that is
not a preference, it is a guarantee: two of every three clips camera-only, and
one explicitly asked to hold still. The frozen golden set had recorded the same
three levels since week 14 and nobody read them as a warning.

Three further things were wrong in the same direction:

1. **The user's own direction never reached the video stage.** `_build_motion_prompt`
   read the motion phrase, the mood and the duration. It never read
   `additional_prompt`. Someone who typed "the model picks the shoes up off the
   table" got that in their still and never once in their clip — the most specific
   creative information the job had, withheld from the stage that needed it most.
2. **The closing line argued against motion.** "Preserve the model's identity and
   the product's exact appearance across every frame" is a sentence a video model
   can only read as *change nothing*.
3. **The video stage sent the image negatives.** A list about extra fingers and
   warped labels that never mentions motion, so nothing in the payload ever
   discouraged panning across the input.

## What changed

The axis now names **what the subject does**; camera movement survives as a
subordinate clause on every level and as a level on none. `product_reveal`,
`hero_turn`, `in_use`, `offer_to_camera`, `pick_up`, `walk_in` — same cardinality,
so the sampler's minimum pairwise distance of 4 is unaffected.

The still and the clip were also introduced to each other. The image prompt is now
told the frame it is composing is frame 1 of a specific action, and asks for the
model caught mid-movement rather than posed. A settled composition has nowhere to
go, which leaves a video model no option but to move the camera.

`cfg_scale` went 0.5 → 0.7 (`KLING_CFG_SCALE`). Unlike the rest of this, that one
is a judgement call rather than a correction — fal documents the parameter as
prompt adherence and the failure was the prompt not being followed, but the effect
on motion amplitude is untested. It is one named constant so it can be A/B'd on a
single clip for $0.70.

## The lesson worth keeping

The vocabulary was written before there was anything to check it against, and it
was reviewed for coverage — six levels, well separated, sampler proves the
diversity — rather than for whether any of them described an advertisement. The
sampler's diversity guarantee was real and measured the whole time. It was
guaranteeing variety across six ways of not moving the subject.

**A well-tested axis can still be the wrong axis.** No unit test catches this,
because every component was behaving exactly as specified. Only the $2.22 did.

## Known follow-up: the axes can still contradict each other

The four axes are drawn independently, which is what guarantees the pairwise
distance — and nothing stops the draw pairing `close_up_product` (a tight frame on
the product) with `walk_in` (the model striding toward camera), or `profile` with
`offer_to_camera`. Those prompts argue with themselves, and a prompt that argues
with itself is a predictable way to waste $0.74.

Not fixed here, deliberately. A hard exclusion table would break the independence
the diversity guarantee rests on, and the size of the problem is unknown until a
real job draws one of these pairs and the result is looked at. Decide it with
output, the way the motion axis itself should have been decided.

## What the rewrite cost elsewhere

Renaming an enum that is also a feature column and a stored value has a blast
radius, and all of it surfaced immediately rather than silently:

* **The trained ranker went stale.** Its one-hot columns named the retired levels,
  so it refused to score and the pipeline fell back to the heuristic baseline
  *and said so*. Retrained with `scripts/train_predictor.py --save`. It is still
  marked a stub — held-out accuracy does not resolve above chance on synthetic
  mock data, which is the known state and not a consequence of this change.
* **The golden bundle was re-frozen** (free — it is a mock bundle). Order matters:
  freeze *after* retraining, or the manifest records heuristic scores and the next
  replay reports drift against the model.
* **84 corpus items and 42 job records** held retired values and could no longer
  be parsed. Migrated by nearest *rendered geometry* in the mock renderer rather
  than by name, a bijection, so the one-hot distribution is preserved exactly. All
  720 judgements and 371 pairs survived; `adgen.pre-motion-rewrite.db` is the
  backup.
* **The paid job's record is archived unaltered** at
  `fixtures/evidence/job-2f3f23c2963049c6-pre-motion-rewrite.json`. Only the
  design-point coordinate was migrated in the live database; `motion_prompt` was
  left exactly as sent everywhere, because the literal text handed to the provider
  is the part that is evidence. That file is why the prompts quoted above can be
  quoted at all.
* **The report was inferring the tier from the spend total.** `spent_usd > 0`
  cannot tell $0.39 of contract smoke tests from a $2.22 job that ran the whole
  pipeline, so it kept printing "no full job has been generated" after one had.
  It now reads `Harvest.n_premium_sets`. Same failure as the hardcoded `$0.0000`
  fixed on 20 August: a claim about the world, asserted instead of measured.

---

# Five images, two videos — 20 August 2026

**The stages were decoupled.** `candidate_count` is 5 and `video_count` is 2, and
they are separate request fields because they are separate decisions.

The economics forced it. An image is **$0.04** and a 10 s clip is **$0.70** —
seventeen and a half times the price — so the video stage was 94% of a job while
producing exactly as many options as the cheap stage did.

| | images | videos | cost |
|---|---|---|---|
| Old 3 → 3 | 3 × $0.04 | 3 × $0.70 | **$2.22** |
| New 5 → 2 | 5 × $0.04 | 2 × $0.70 | **$1.60** |

More creative range, 28% cheaper. What pays for it is the image-stage predictor,
which stops being a diagnostic and becomes the thing that decides where the money
goes.

## What that costs the research claim

Worth stating plainly, because it is the one real price of this change.

The headline number is the agreement between the image-stage ranking and the
video-stage ranking. A 3 → 3 job produced a **complete** paired observation: every
image rank had a video rank to be compared against. A 5 → 2 job produces a
**truncated** one — the three candidates the predictor cut have no video-stage
outcome and never will, so any correlation computed over what remains is
range-restricted by construction. Worse, a Spearman ρ over two items can only be
+1 or −1, and the top-1 hit rate now runs against a 50% baseline rather than 33%.

The pipeline already tolerated this — `_orders` and `stage_agreement` compare the
shared subset and skip sets with fewer than two in common — but tolerating it is
not the same as it being sound to quote.

**The fix is a request field, not a rewrite.** Set `video_count == candidate_count`
and the job animates everything, restoring the complete observation the evaluation
needs. `AdJobRequest.animates_everything` names the distinction so the report can
separate the two populations rather than pooling them. Product jobs run 5 → 2;
evaluation jobs animate everything and cost more, which is a budget line rather
than a design problem.

One practical consequence: a 5 → 5 evaluation job is **$3.70**, which is above the
`AD_BUDGET_PER_JOB_USD` cap of $2.50, so the governor will refuse it — correctly,
and loudly. That cap is a deliberate safety rail and is not being raised as a side
effect of this change. Either raise it consciously when an evaluation run is
wanted, or run evaluation at `candidate_count=3, video_count=3` for $2.22, which
fits under the existing cap and is the same complete observation on a smaller set.

## The diversity guarantee moved, and is now computed

Five candidates put the composition axis under pressure. On Reels only four
compositions are viable — `negative_space_top` puts the subject under the
platform's bottom chrome — so two of five must share one and the best achievable
minimum pairwise distance is **3, not 4**. On Feed, where all five are viable, five
candidates still separate on every axis.

`sampler.max_achievable_distance(n, platform=…, locked_angle=…)` computes that
bound by counting the axes with at least `n` levels, and the tests assert against
it rather than against a remembered number. The guarantee was never "4" — it was
"as separated as the axes allow", and that is now something the code can state.
Verified against the sampler in all eight combinations of platform, count, and
locked angle.

## Two things that would have gone wrong quietly

**The pre-flight would have over-reserved.** `estimate_job_cost` derived the video
count from the candidate count, so a 5 → 2 job would have reserved $3.70 against a
real cost of $1.60. A governor that over-reserves refuses jobs the budget could
have afforded — quieter than overspending and just as wrong. `video_count` is now
its own argument, defaulting to the candidate count so existing callers keep
meaning the 1:1 cascade.

**The cut would have read as a failure.** Three frames with no clip beside them
look like three things that broke. The image cards carry an explicit
`animated` / `not animated` badge and the stage header says "top 2 animated, 3
not", because a decision the system made has to be labelled as one.

---

# Eight minutes and no sound — 20 August 2026

The second live job produced two complaints, and neither was about the model.
The images took eight to ten minutes, and the delivered videos were silent.
Both turned out to be ours, and both had been true since the first line of the
pipeline was written.

## The eight minutes were spent waiting on purpose

The first live job's own record settles it. Three Seedream edits:

| slot | latency | created |
|---|---|---|
| 0 | 54.1 s | 09:02:27 |
| 1 | 46.9 s | 09:03:14 |
| 2 | 49.9 s | 09:04:04 |

Sum of latencies 150.9 s; wall clock from job start to the last frame, 152 s.
Nothing overlapped with anything. The gate — numpy over a full-resolution frame
— accounted for about 1.2 s across all three, so it was never the cost. The
stage loop was `for brief in briefs:` and each slot waited for its predecessor
to come back before it was even submitted.

fal is a *queue*. The time is spent waiting, and waiting is the one thing that
parallelises for nothing. At the new five candidates the sequential version cost
four minutes of a progress bar for zero additional work.

Both generation stages now fan out, bounded by `AD_MAX_CONCURRENT_GENERATIONS`
(default 5, matching the default candidate count — a limit of 4 would put the
fifth image in a wave of its own).

Expected wall clock for a 5→2 job, from the measured per-call latencies:

| stage | before | after |
|---|---|---|
| images (5 × ~50 s) | ~250 s | ~55 s |
| videos (2 × ~120 s) | ~240 s | ~130 s |
| **total generation** | **~8 min** | **~3 min** |

## Two things the fan-out broke, caught before they cost anything

**The budget check stopped being a check.** `CostGovernor.guarded_call` reads
the ledger, decides the call fits, then reserves. `status()` awaits the
database, and an await is a yield — four concurrent callers could each read the
same total, each conclude they fit under the cap, and each reserve. Nobody
exceeded the budget as they understood it and the budget was exceeded anyway.
Overshoot is bounded by (concurrency − 1) × the estimate, which for video is
$2.10 against a $2.50 cap: the exact failure the governor exists to prevent.
Check and reserve are now one critical section under `_reserve_lock`; the
provider call stays outside it, so the concurrency this protects is not the
concurrency it would destroy.

The test for this was written twice. The first version used `InMemoryLedger`
and passed with the lock deliberately removed — `total_spent` is `async def`
but awaits nothing, and a coroutine that never awaits never yields, so the
critical section was atomic by accident. The second version models a read that
*observes the value and then suspends*, which is what a real query does, and it
fails without the lock and passes with it. A concurrency test that has not been
watched to fail is not evidence.

**A 429 would have cost a candidate.** A rate-limited submission lands on a slot
that has already cleared the budget check. `FalClient` now waits one out —
honouring `Retry-After` when fal sends a numeric one, doubling from 2 s
otherwise, bounded by the retry count *and* by the caller's deadline. Retrying
is safe here and only here: the request was refused, so nothing was generated
and nothing was charged.

## The silence was three bugs stacked

Every layer was built. None of them was connected.

1. **Nothing ever supplied a bed.** `Pipeline(audio_bed=None)` was the default
   and no caller passed one. That was a deliberate decision — this project ships
   no music, and `AudioBed` refuses audio whose licence is unrecorded, because
   bundling music of unknown provenance into a deliverable is the one delivery
   mistake that cannot be corrected after publication. The reasoning holds. The
   conclusion did not: "we own no music" is answered by writing some, not by
   delivering an advertisement with no sound.
2. **Only the winner was mixed.** Delivery reframes the winner alone, which is
   right — nine re-encodes for output nobody asked for is not a saving worth
   making. Audio is the exception, because the video track is *copied*: mixing a
   runner-up costs a stream copy, not a generation. A runner-up that plays silent
   next to a winner that does not reads as a broken clip.
3. **The player could not have played it anyway.** `VideoStage.tsx` pointed at
   `video.asset` — the provider's own render, which has no audio track at all —
   and carried a hard-coded `muted`. `muted` is what autoplay policies require
   and nothing here autoplays. Two independent reasons the browser was silent,
   either of which would have survived fixing the other.

`adml.audio.choose_bed` now answers the question in one place: a licensed track
from the library if there is one, a track suited to the mood first, and a
synthesised bed otherwise. It never returns nothing.

## The synthesised bed

Not `tone_bed` — that is a 220 Hz sine for testing the mix path and nobody would
ship it. `generated_bed` is a four- or eight-bar chord progression scored to the
job's `Mood`: which chords, how often they change, how bright the result is, and
how much low end sits under it. One oscillator per note with its own attack and
release, rather than one oscillator whose frequency steps — stepping a running
oscillator's frequency is a phase discontinuity, which is a click, and a click
every two and a half seconds is worse than no music.

| mood | progression | harmonic rhythm | top |
|---|---|---|---|
| calm_premium | Am – F – G – Em | 4 bars | 5 kHz |
| warm_lifestyle | C – F – Am – G | 4 bars | 6.5 kHz |
| bold_confident | Dm – B♭sus4 – F – Csus4 | 4 bars | 8 kHz |
| high_energy | Am F C G Am F G Gsus4 | 8 bars | 9 kHz |

It is modest music. What it is is free, unencumbered, deterministic — the same
mood and duration produce identical bytes, so a golden replay is not disturbed
by its soundtrack — and *there*. A licensed track always wins.

## To use real music instead

Drop the file and a JSON sidecar of the same stem into
`<storage_root>/audio/` (override with `AD_AUDIO_LIBRARY`):

```
bright-morning.m4a
bright-morning.json   {"title": "Bright Morning", "source": "Pixabay",
                       "licence": "Pixabay Content Licence",
                       "url": "https://...", "moods": ["warm_lifestyle"]}
```

A file **without** a sidecar is skipped, not loaded with the provenance left
blank — the same rule `AudioBed` enforces at construction, applied one step
earlier. The failure being prevented is a track reaching a published deliverable
because a directory scan was permissive and nobody was ever asked. A track that
lists no moods is eligible for all of them, because the common case is one piece
of music and a user who wants it used.

## Still unproven

The concurrency has never run against fal. 500+ passing tests prove the pipeline
submits five at once and that a 429 is waited out; they cannot prove fal accepts
five at once from this account. If it refuses, the symptom is slowness rather
than failure, and the fix is one number in `.env`.

---

# A phone the size of a person — 21 August 2026

A live job on a photograph of a man and a photograph of an iPhone returned five
frames in which the handset ranged from implausibly large to taller than the
person holding it. The user compared the same two references and the same brief
against another hosted model and got correctly-scaled output. The gap was not
the model.

## Nothing in the prompt said how big the product was

`_build_image_prompt` emitted four blocks — REFERENCES, SHOT, LOOK, CONSTRAINTS
— running to roughly 300 tokens. It insisted on the face, the skin tone, the
label text, the brand palette, the platform safe area and the mid-movement pose.
It never stated the product's physical size, or its size relative to the model,
anywhere.

A reference-composition model receives two images: a full-frame photograph of a
person and a full-frame cutout of a phone. Nothing in that pair carries scale.
Absent an instruction it composites them at comparable *apparent* size, which is
the only inference available to it.

## And three phrases asked for the product to be bigger

| where | what it said |
|---|---|
| `CameraAngle.CLOSE_UP_PRODUCT` | "tight close-up centred on the product" |
| `Composition.PRODUCT_FOREGROUND` | "product **prominent** in the foreground" |
| every CONSTRAINTS block | "The product must be fully visible and **unobstructed**" |

The third is the interesting one. A phone held in a hand is *by definition*
partly obstructed — fingers are what holding is. Given an unsatisfiable
constraint the model found the loophole: draw the product too large for a hand
to cover. All three now name camera proximity, perspective and lighting as the
route to prominence, and the constraint asks for the product to be *readable*
while saying outright that a grip is expected.

Scale terms were added to the negative prompt as well, listed apart from the
artefact terms because they are not artefacts. Every frame in the failing job
was clean, well-lit and anatomically sound, and passed all five implemented
gate checks. It was simply the wrong size.

## Where the size comes from

`AdJobRequest.product_scale`, an optional five-level enum, resolving through
`effective_scale` to a per-vertical default. Levels name a body relation rather
than a measurement — a generator cannot act on "147 mm" any more precisely than
on "fits in one hand", and the hand relation is the part it can actually draw.

`Vertical.OTHER` maps to `None` rather than to a guess. Everything from lipstick
to a sofa is tagged `other`, and defaulting it to "one hand" would trade one
silent scale error for another. An unknown size still gets the half of the
instruction that needs no measurement — render at true size, prominence never
comes from enlargement — which is the half that fixes the bug.

## The ranker was selecting for the defect

The worst frame in the set won, and that was not luck.

`product_salience` carried **0.25** of the image-stage blend and was computed by
`focal_concentration` — the share of the saliency map held by its most salient
tenth. For a composite of a person and a product that rises with the size of the
product. It is not a product measurement at all; attributing attention to the
product specifically needs the product located in the frame, which is the
pending `product_identity` work.

The job's own numbers:

| candidate | scale | `product_salience` | rank |
|---|---|---|---|
| E | phone taller than the model | 0.45 | **1** |
| C | phone wider than his torso | 0.40 | 2 |
| B | phone in a hand, near-correct | 0.32 | 3 |

Ordered exactly by how oversized the handset was. The spread was 0.0325 of the
final score — enough to reorder a five-candidate set, and it did. The system
then promoted the two worst frames and paid $1.40 to animate them.

The feature is banded now rather than deleted, because "has a clear focal
subject at all" is still worth something and is the only thing it can honestly
claim to measure. Above a floor of 0.30 there is no further credit, so
satisfying it is no longer the same as maximising it. Weight dropped to 0.10,
redistributed to aesthetic (0.30), composition (0.25) and safe-area compliance
(0.20).

## A concentration ceiling cannot be the scale check

The obvious next move — reject a frame whose attention is too concentrated —
does not work, and the reason is worth recording so nobody tries it again.

| corpus | `focal_concentration` |
|---|---|
| real Seedream frames | 0.33 – 0.45 |
| mock generator | 0.58 – 0.73 |

The mock draws a synthetic hero blob and is *more* concentrated than any real
photograph. Any ceiling loose enough to pass the mock corpus sits above 0.73 and
therefore passes card E at 0.45 — the phone taller than the model — while a
ceiling tight enough to catch card E fails every mock frame and every test in
the suite. The two distributions do not overlap in the direction the check needs.

Measuring scale needs the product *located* in the frame. The honest route is
the product mask from `rembg` matched into the generated frame, giving a
bounding box whose height can be compared against the detected face box — an
adult face is a reliable human ruler. That check should be calibrated against
real frames from the re-test rather than against the mock distribution, which is
known to be wrong in exactly the relevant direction.

## Still unproven

The prompt fix has not run against Seedream. The tests prove the instruction is
emitted for every vertical and that the phrases which asked for enlargement are
gone; they cannot prove the model obeys it. That costs $0.20 — five images, no
video — and is the next thing worth spending money on.

## The golden bundle is now stale, and must not be accepted

`replay_golden.py aurora-reels` reports drift on all five slots: the image prompt
changed, so the frozen frames no longer correspond to the prompt recorded beside
them. That is the check working. `--accept` is the wrong response — it rewrites
the expectation to the new prompt while leaving frames generated from the old
one, producing a bundle that asserts a correspondence that does not exist.

The bundle needs a genuine re-freeze at $1.60, and not before the $0.20 image
test shows the new prompt is worth freezing.

## What a competing implementation does differently — 21 August 2026

A working system built on the same two-reference premise, producing visibly
better composition from the same photographs. Its source is ~150 lines of Express
controller. Worth reading closely, because three things separate it from ours and
they are not the three you would guess.

### Its entire image prompt is 37 words

```
Combine the person and product into a realistic photo.
Make the person naturally hold or use the product.
Match lighting, shadows, scale and perspective.
Make the person stand in professional studio lighting.
Output ecommerce-quality photo realistic imagery.
```

Two of those five lines — *"naturally hold or use the product"* and *"match
lighting, shadows, **scale** and perspective"* — do in eleven words what our
343-word template was not doing at all. A person naturally holding an object
constrains its size implicitly; naming scale constrains it explicitly. We wrote
nine times as much and said neither.

Verbosity is now a switch (`AD_PROMPT_STYLE`, `PromptStyle`) rather than a
conviction. The compact template runs at **137 words against 329** and keeps
every design axis, which is the floor for a prompt that has to express a
*sampled* design point. Theirs reaches 37 because it carries no design point at
all: one fixed brief, one image, no candidates, no ranking. A comparison against
that would not be measuring verbosity.

The style is recorded on the `BriefSet`, so the ablation is attributable after
the fact rather than reconstructed from timestamps.

### Its model is Gemini 3 Pro Image, not Seedream

| | ours | theirs |
|---|---|---|
| image | `seedream-4.5-edit` $0.040 | `gemini-3-pro-image-preview` @1K **$0.039** |
| video | `kling-2.5-turbo-pro` $0.07/s | `veo-3.1-generate-preview` (native audio) |

The image swap is **cost-neutral**, and the original plan named Gemini
multi-reference composition the best fit for this exact task before we chose
Seedream for its fal adapter. `IMAGE_PRICES` already carries a `gemini-flash-image`
entry at $0.039; there is no adapter behind it. This is the largest untested
variable in the output quality, and it costs the same per image either way.

Note: Gemini 3 Pro Image has **no free API tier** — billing must be enabled.
Gemini 2.5 Flash Image does have one. Verify both against Google's own pricing
page before committing; the figures here come from third-party summaries.

### It generates one candidate, not five

Its prompt hard-codes one safe composition — studio lighting, person holding
product. Ours samples the design space by construction, including corners like
`close_up_product` × `pick_up`, which is what produced the frame of a man lunging
at a floor-standing phone.

That is not a bug to fix by narrowing the space; the sampling *is* the research
contribution. But it has a consequence worth stating plainly: our worst candidate
is necessarily worse than their only candidate, and the component that is supposed
to make that irrelevant — a ranker that reliably promotes the good ones — is still
a heuristic placeholder awaiting pairwise human labels. Until Stage B calibration
lands, the multi-candidate design pays the cost of exploring bad corners without
yet collecting the benefit.

## Measuring product scale — 21 August 2026

Both live jobs returned a product drawn far larger than the person holding it, and
nothing in the quality gate could express that. Fixing it meant finding a
measurement, and three of the four candidates failed. The failures are recorded
because each rules out an approach that looks obvious from the outside.

### What does not work

**`focal_concentration` is not a product measurement.** It was carrying 25% of the
image score as `product_salience`. Measured across eleven real Seedream frames it
spans **0.31–0.50 with correct and oversized frames interleaved through the whole
range** — nike/img_1 (correct scale) sits at 0.430 while nike/img_2 (a trainer
larger than the model's torso) sits at 0.314. No cut through it separates the
classes, so it cannot be a gate check in either direction, and as a ranking
feature it was noise wearing a product's name. It is now banded and down to 0.10.

**Masked multi-scale template matching does not localise the product.** Sliding
the rembg cutout over the frame with `cv2.matchTemplate(..., TM_CCORR_NORMED,
mask=alpha)` across scales 0.06–0.95 returns, for every one of the five Nike
frames, *the smallest template in the search range* at a near-identical score
(0.955–0.963). That is the known degenerate behaviour of normalised correlation
with a mask — the score is maximised by the smallest window — and the identical
scores across visibly different frames confirm it is measuring nothing.

**Face detection would give a physical ruler, and must not be used for this.**
An adult head is a known size, so a face box would calibrate everything else in
the frame, and the Haar cascade does fire on 9 of the 11 frames. It is barred
anyway, by a commitment this project already made in writing: `FaceReport` records
that Haar cascades miss unevenly across skin tones and poses, which is why a miss
there produces an advisory and never a refusal. A gate built on one would reject
some people's photographs more often than others. The two frames where no face was
found are both frames where the composition crowded the model out — so the signal
is real, and taking it would still be wrong.

### What does work

The product's own colours, which the cutout hands over for free.
`adml.features.product_colour_signature` takes the chromatic dominant colours of
the cutout (dropping near-neutrals, which match any studio backdrop), and
`product_area_share` reports the share of the frame within ΔE76 12 of one of them.

Measured over both live jobs, labelled by looking at the frames:

| class | n | range |
|---|---|---|
| visibly correct scale | 5 | 0.011 – 0.094 |
| unlabelled | 3 | 0.084 – 0.113 |
| visibly oversized | 3 | **0.172 – 0.294** |

The classes separate with a 0.059 gap. `THRESH_PRODUCT_AREA = 0.16` sits inside
it, nearer the oversized end for the same reason every floor in `gate.py` sits
below its class minimum: a false reject costs a paid retry on a usable frame,
a miss costs a frame that is merely ranked.

It fires on exactly the three frames it should — including iphone/img_4, the card
the ranker previously promoted to #1 and paid to animate.

### What it does not claim

It is an **upper bound on product extent, not a measurement of it**: anything else
in the frame wearing the product's colours is counted too. That asymmetry is why
it is used only as a ceiling — over-reporting can make an innocent frame look
large, which one retry answers, while under-reporting cannot happen, so a
genuinely oversized product cannot hide from it.

It is **undefined for achromatic products**. A white trainer, a black phone and a
steel bottle have no signature to be found by, and for those the check reports
`implemented=False` with the reason rather than passing silently. That is a large
fraction of real products, and the honest reading is that this buys time until
`product_identity` (DINOv2) lands rather than replacing it.

It is **undefined without a real cutout**, which was nearly shipped as a bug.
`product_reference` falls back to the original upload when rembg produces nothing
usable, and the dominant colours of a product *photograph* are its backdrop's as
much as the product's — measured, such a signature matched 59% of a frame whose
product covered 3% of it. `product_colour_signature` now refuses a fully opaque
image outright. Real rembg cutouts here are 60–80% transparent, so the two
separate cleanly, and the test is a fact about the pixels rather than a label.

It is **undefined on mock output, for a reason worth recording**: intake extracts
the brand palette *from the product cutout*, and the mock renderer then paints its
backdrop, wardrobe and product from that same palette. The product's colours
genuinely cover the frame, so mock frames measure 0.48–0.57 while the product
block itself covers about 3%. That is a closed loop no real generator has —
Seedream put the trainer's tan on the trainer and painted the backdrop white — but
it makes the reading meaningless across its whole range, not merely high, and
which end a given seed lands on is arbitrary. Mock frames are therefore excluded
by tier. **The cost is that this check is exercised only against paid
generations**, so its calibration cannot be regression-tested by the default dev
path — the one real weakness in the arrangement.

Above `UNDISCRIMINATIVE_SHARE = 0.40` the check declines rather than rejects, on
the same reasoning: past that point the number is measuring a backdrop, and a
product covering more than 40% of a 9:16 frame alongside a human model is a pack
shot rather than a mis-scaled ad. It sits in the gap between the worst real
failure (0.294) and the lowest mock reading (0.478).

**Eleven frames is a calibration set, not a validation set.** The threshold is the
first thing due for revision when the corpus grows.
