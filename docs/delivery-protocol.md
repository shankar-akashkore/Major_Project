# Delivery protocol

Stage 8 turns one ranked clip into something publishable: the same creative at three
aspect ratios, mockup previews showing what each platform will cover, an audio mix,
and a bundle. Every part of it is local — no provider is called, nothing leaves the
machine, and a delivery run moves the ledger by $0.0000 (asserted in
`tests/test_delivery.py::test_delivery_costs_nothing`).

That makes it the cheapest stage in the project and the easiest to treat as
packaging. It is not packaging. Reframing an ad is **lossy**, and the size of the loss
is a fact about the deliverable that the person publishing it needs to be told.

---

## 1. Why not a centre crop

A job generates at one aspect ratio, the one its target platform derives. Delivery
needs the others. The one-line answer is a centre crop, and it is wrong for a
specific structural reason: **the design-space sampler deliberately puts the subject
off-centre.** Three of its five composition settings do —
`rule_of_thirds_left`, `rule_of_thirds_right` and `product_foreground` — so a centre
crop mutilates the majority of candidates rather than an unlucky few.

Measured on a real mock job reframed from its native 9:16:

| target | mode | smart crop | centre crop | difference |
|---|---|---|---|---|
| 9:16 | none | 100% | — | — |
| 4:5 | crop | **84%** | 75% | +9 pts |
| 16:9 | pad | **56%** | 34% | +22 pts |

"Retains 84%" on its own is not a claim about anything, so `ReframeReport` carries
`centre_crop_salience` alongside it and the delivery event stream quotes both.

The figures are *attention-weighted* saliency mass, not pixel area. Saliency inside
the target platform's safe area counts at full weight and saliency outside it at
0.35 — a window that frames the product perfectly but leaves it under Instagram's
caption bar has not solved the problem. The discount is not zero on purpose: content
under a caption bar is compromised rather than absent, and zeroing it would make the
objective indifferent between clipping the product and losing it entirely.

---

## 2. When cropping is the wrong operation

A 9:16 source reframed to 16:9 keeps a horizontal band about 32% of the original
height. On an ad featuring a standing model, that band is her waist.

So `adml.crop.plan` measures what a crop would retain and **switches to letterboxing
below 70%**, reporting why in prose: *"a crop would discard 44% of the salient
content, so the frame is letterboxed instead."* The fill is a blurred, over-scaled
copy of the frame rather than a flat bar — which is what social platforms and every
editing tool do, because a flat bar reads as a mistake and a blurred extension reads
as a deliberate frame. It also keeps the palette on-brand for free, being made of the
ad's own pixels.

The 0.70 threshold is a judgement, not a measurement. What justifies it is that it
separates the cases cleanly on real geometry: 9:16→4:5 lands at 84% and 9:16→16:9 at
56%, so the decision is not sitting on a knife edge.

---

## 3. The crop window is static, and that was measured

A per-frame window tracks a moving subject better and **jitters**. A jittering frame
is worse to watch than a slightly mis-framed steady one, so the window is fixed for
the whole clip — placed at the *median* of the per-frame ideal offsets, so one frame
where the subject leaves the shot cannot drag the window off the other forty-seven.

The cost of that decision is measured rather than assumed. `CropPlan` carries both
what the static window retains and what a perfect per-frame tracker would have
retained. Across every mock motion intent, reframed to 9:16:

| motion intent | subject travel | tracker would gain |
|---|---|---|
| `offer_to_camera` | 0.0% | **0.000** |
| `walk_in` | 4.0% | **0.000** |
| `in_use` | 5.9% | **0.000** |
| `product_reveal` | 6.0% | **0.000** |
| `hero_turn` | 8.0% | **0.002** |
| `pick_up` | 9.1% | **0.000** |

Two thousandths of a point. A subject in an 8–10 s ad works within the frame; it
does not cross it.

**Re-measured 20 August 2026**, after the motion axis was rewritten from camera
moves to performances — the retired levels (`static_subtle`, `handheld_drift`,
`orbit_left`) measured 3.3% / 4.6% / 12.5% travel for 0.000 / 0.000 / 0.002 gain,
so the conclusion survived the rewrite unchanged. The test is now parametrised over
the whole axis rather than a chosen three, so a level added later cannot quietly
break the assumption the crop window rests on.

The metric is not merely insensitive: driven with a synthetic subject crossing half
the frame it reports gains above 0.08, rising to 0.20 at 70% travel. So the small
number is a fact about the content, not about the measurement — which is what
`test_the_tracking_gain_metric_responds_to_a_subject_that_really_travels` exists to
establish.

### A caveat that cost a debugging session

`adml.features.saliency_map` is a spectral residual, and it **rings on synthetic
imagery with hard edges and no texture**. Measured on a single bright disc over a
perfectly flat field, the saliency profile came out periodic with a period of about a
sixteenth of the frame — and its column-wise *minimum sat exactly on the disc*. The
FFT of a disc rings and the residual amplifies the ringing, so the crop confidently
went to the empty half of the frame.

On the mock renderer's gradient-and-grain frames the same measurement puts 79% of its
mass in the correct half. So a flat vector fixture produces a nonsense crop and it is
the fixture that is wrong — which is why every crop test in this repo renders its
frames instead of drawing them. The identical limitation already applied to
`adml.features.sharpness`, documented in `mock_portrait_asset`.

---

## 4. Audio: mixed, not generated

Veo 3.1 produces native audio at **$0.40 per second** — one 9 s clip would be $3.60,
a tenth of the entire project budget, for a soundtrack. A royalty-free bed mixed with
ffmpeg is free and gives full control over the balance.

**A bed without a recorded licence cannot be constructed.** `adml.audio.AudioBed`
requires a title, a source and a licence string, and raises `LicenceMissing`
otherwise. This is not paperwork. An ad is a commercial artefact, a submitted project
is a published one, and "I found the mp3 in a folder" is how a music rights claim
happens. The licence travels into `AudioReport` and onto the bundle's `README.txt`,
so a file handed to someone else still says where its audio came from.

**This project ships no licensed audio content, and delivers a soundtrack anyway.**

Those two facts sat in contradiction for longer than they should have. The rule
above is right and unchanged; what was wrong was the conclusion drawn from it.
No bed was the normal state, no caller ever passed one, and the report said so —
politely, in a field nobody reads, under a video with no sound. An advertisement
with no sound is not a deliverable, and "we own no music" is answered by writing
some rather than by shipping silence and describing it.

`adml.audio.choose_bed` decides, in this order:

1. a **licensed track** from `<storage_root>/audio` (override: `AD_AUDIO_LIBRARY`)
   whose sidecar lists the job's mood;
2. any licensed track, if none lists that mood;
3. a **synthesised bed** scored to the mood — `generated_bed()`.

It never returns nothing. A library file needs a JSON sidecar of the same stem
recording title, source and licence; one without is skipped rather than loaded
with the provenance left blank, which is the `AudioBed` rule applied one step
earlier, where a directory scan would otherwise be the thing that decides.

`generated_bed()` is a four- or eight-bar chord progression, not a tone: one
oscillator per note with its own attack and release, chosen per `Mood` along with
the harmonic rhythm and how bright the result is. It is flagged `generated=True`
and credited as *"synthesised for this ad — no third-party rights"*, because a
manifest that leaves a reader to assume the soundtrack was licensed is the same
provenance failure as an unrecorded licence, pointing the other way. It is
deterministic, so a golden replay is not disturbed by its music. A licensed track
always wins.

`tone_bed()` remains, and remains a 220 Hz sine flagged `is_test_signal=True`, so
the mixing path can be exercised without either a music file or a synthesis run.
It is a test signal and can never be described as a soundtrack in a manifest.

**Audio goes on every clip, not only the winner.** This is the one place stage 8
departs from "the winner only". Reframing a runner-up is nine re-encodes for
output nobody asked for; mixing one is a stream copy, because the video track is
copied rather than re-encoded. The UI lets the user play every candidate, and a
runner-up that plays silent beside a winner that does not reads as a broken clip
rather than as a deliberate economy.

Two mix decisions worth recording:

**Loudness is normalised to −14 LUFS and then measured.** That is what Spotify,
YouTube and Instagram normalise playback to; a bed mastered louder simply gets turned
down on upload, changing the balance the ad was approved with. `loudnorm` runs
single-pass and corrects *toward* a target rather than landing exactly on it, so
`measure_loudness` reads the delivered figure back — measured at −14.0 on the bed-only
mix and −14.1 with a voiceover.

**Ducking is side-chained, not a fixed cut.** `sidechaincompress` keyed off the voice
lifts the music back between phrases; a fixed volume reduction leaves it flat through
every pause. The difference is audible and it is the whole point of mixing.

A caption too long to read inside the clip is caught before any mixing, from a
150 wpm estimate — advertising voiceover over music is read slower than conversation.
A 23-word caption estimates at 9.2 s and is refused for a 9 s clip.

---

## 5. Previews answer a question a score cannot

`safe_area_compliance` is a number. A frame with Instagram's caption bar drawn over it
is the same fact in the form the person publishing will act on.

The bands come from `adschema.Platform.safe_area` — the same rectangle the score is
computed against — rather than from a hand-placed mock. A drifting mock would be worse
than no preview at all: it would look authoritative while disagreeing with the number
printed beside it.

One preview per *distinct* safe area, since a second platform with identical geometry
produces an identical picture. Reels and Story share `(0.14, 0.20, 0, 0.14)`; Feed and
Facebook share `(0.05, 0.05)`.

The middle frame is previewed, matching the video scorer. The first frame would be the
generated still the image stage already showed the user.

---

## 6. The bundle, and the bug that shipped an empty one

`build_bundle` originally read the report off `result.delivery` — which the pipeline
assigns **after** `deliver()` returns. So the archive silently contained
`report.json` and `README.txt` and nothing else: 2 kB where 127 kB of media should
have been, with no error, no warning, and no empty-file indication. The download
looked like it worked.

The report is now an explicit argument, which cannot be got wrong in that direction,
and `test_the_bundle_actually_contains_the_media` probes every clip inside the zip
rather than counting filenames.

The bundle carries a plain-text `README.txt`, because **it outlives the conversation
that produced it**. Someone opening it in week 16 needs to be told that a 0.74 is a
within-set position from a possibly-stub model and not a click-through rate. It states
that, names the model, and says when the model is a stub.

Only the winner is reframed. Nine re-encodes for output nobody asked for is the
alternative, and a runner-up's value is its *score* — evidence for the ranking — not
its 16:9 variant. Every candidate keeps its native render and appears in the report
cards.

---

## Known limitations

- **No real generation has been delivered.** Every figure here comes from mock clips.
  The salience percentages will move on real generations, and the 0.70 pad threshold
  should be re-read against them.
- **`WARN_RETAINED_SALIENCE = 0.80` and `MIN_RETAINED_SALIENCE = 0.70` are
  judgements.** They separate the real cases cleanly (84% vs 56%) but neither is
  derived from anything about how much loss a viewer actually notices.
- **No TTS exists.** `voice_duration_estimate` warns when a caption cannot fit, and
  `mix` accepts a voice track, but nothing generates one. The estimate is words over a
  fixed rate and knows nothing about the words.
- **The API has no HTTP-level tests.** `/api/jobs/{id}/delivery` and
  `/api/jobs/{id}/bundle` were verified by hand against the real ASGI app, and the
  pipeline they call is covered, but the routes themselves are not in the suite —
  `adapi.main` builds its store, storage and ledger at module scope, so a test cannot
  isolate them without a refactor.
- **Chained-clip delivery is untested end to end.** `adml.video.detect_cuts` finds a
  seam from the pixels and the pipeline records it, but no chained clip has been
  generated, so a reframe across a seam has never been rendered.
