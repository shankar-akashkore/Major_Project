# Prediction protocol

How the performance predictor is trained and evaluated, and why the evaluation is
built the way it is. Weeks 9–11 of the plan.

Every number below was measured with the code in this repository, not quoted from
elsewhere. Reproduce them with `scripts/train_predictor.py`; the two synthetic
studies are described where they appear.

---

## 1. What is being predicted

One scalar per candidate image, used to order the three candidates of a job. The
target is human pairwise preference, collected by the annotation tool
(`docs/annotation-protocol.md`) and modelled with a Bradley-Terry likelihood.

The model is **linear over standardised features, fitted in numpy**. Three reasons,
in order of importance:

1. **The label count sets the model size.** With ~1,200 pairwise training
   comparisons the parameter budget is ~80 (`adml.featureset.parameter_budget`, at
   15 observations per parameter). A single 8-unit hidden layer over 30 inputs needs
   250. `--hidden` exists so the report can show what exceeding the budget costs; it
   is not the headline model.
2. **No train/serve gap.** The evaluation runs in `adml.predictor` and so does the
   API. The model in the ablation table is the same object the product scores with.
   torch stays behind a file boundary (§2) and never touches the laptop.
3. **The coefficients are readable.** With z-scored inputs the weights are
   comparable to each other, which makes "what did it learn" a question with an
   answer. They are still *directions*, not effect sizes — correlated features share
   credit arbitrarily.

---

## 2. Where the features come from

| group | source | runs on |
|---|---|---|
| `photometric` | luminance, RMS contrast, colourfulness, sharpness | laptop |
| `composition` | thirds alignment, subject scale, border uniformity | laptop |
| `salience` | spectral-residual focal concentration, safe-area share | laptop |
| `palette` | brand-gamut ΔE adherence | laptop |
| `context` | one-hot design axes, vertical, platform | laptop |
| `embedding` | SigLIP, DINOv2 → PCA to 8 dims per block, per fold | **Colab** |
| `aesthetic` | LAION aesthetic head over CLIP ViT-L/14 | **Colab** |
| `identity` | product-DINO, face-ArcFace similarity | **Colab**, not yet built |

The Colab half is `notebooks/colab_embeddings.ipynb`. It writes one `.npz` per
encoder, keyed by `item_id`, with a `__spec__` entry recording the model that
produced it. A file without `__spec__` is **refused on load**: an embedding block
whose provenance is unknown cannot appear in a reported result.

Two rules about missing data, both in `adml.embeddings`:

- **No zero-fill.** An item with no embedding is dropped from the table. After
  standardisation a zero vector *is the mean*, so a failed encoder run would appear
  as a mediocre candidate rather than as missing data.
- **Stand-ins are flagged and the flag propagates.** `hash_embeddings` produces
  deterministic vectors carrying no visual information, so the whole path can be
  exercised with no downloads. `is_real=False` travels into the feature table and
  into every printed line. A model trained on them *must* land at chance — that
  makes them a leakage test, not only a placeholder.

---

## 3. The split, and the 19 points it prevents

### The measurement

Features that are **uninformative by construction**: 200-dimensional random vectors,
with true item quality drawn independently of them. Any held-out accuracy above
0.500 is leakage. 60 sets, 180 items, ~700 comparisons.

| split | held-out accuracy | training accuracy |
|---|---|---|
| random over comparisons | **0.659** | 0.802 |
| set-wise (whole sets held out) | **0.465** | — |

**19.4 points of a result that does not exist.** 0.659 would look publishable.

The mechanism: with degree-10 comparisons per item, an item held out in one pair is
present in the training set through nine others. With 200 dimensions and 180 items
the features can identify individual items, so the model memorises item strengths
and reports them as generalisation. Asserted in
`tests/test_split.py::test_a_random_split_over_comparisons_reports_accuracy_that_is_not_there`.

### What the honest split costs

The unit is the **generation set** — three candidates from one pair of reference
photographs. A set goes entirely to train or entirely to test, because that is the
deployment question: given references never seen before, rank the three candidates.

Measured on the planned corpus (100 sets, 1,500 designed comparisons, 5 folds):

```
trainable (pooled)   5032
testable  (pooled)    532   [within-set 300, cross-set 232]
discarded (straddle) 1936   (26%)
per fold: ~106 test comparisons
pooled 95% margin at p=0.70: ±0.040
```

A comparison whose two items land on opposite sides is unusable in either half. The
discard rate follows `2p(1-p)` of the *cross-set* comparisons — 32% at a 20% holdout,
confirmed to within 5 points in `test_the_discard_fraction_matches_the_predicted_2pq`.
Within-set comparisons never straddle: both items share a set by construction.

Two consequences the evaluation is designed around:

- **Pool across folds.** 106 comparisons carries a ±9 point interval, which cannot
  separate a good model from a mediocre one. Folds are disjoint, so every held-out
  comparison is predicted exactly once and the pooled 532 is a valid sample at ±4.
- **Within-set test observations are capped at `3 × n_sets`.** With three candidates
  there are only three possible within-set pairs per set, and every one of them is
  testable in exactly one fold. 100 sets means **300 deployment-task test
  observations, no matter how many judgements are collected.** More statistical power
  on the headline claim requires more *sets*, not more comparisons per set.

---

## 4. The ceiling: why 0.68 can be near-perfect

Labels are single human judgements, and humans disagree with themselves on repeats.
Model a judgement as expressing the underlying preference with probability `q`. Two
independent judgements of the same pair agree with probability `q² + (1−q)²`, so an
observed test-retest agreement `a` implies

```
q = (1 + √(2a − 1)) / 2
```

and a predictor with perfect knowledge of the preference still matches a single
judgement only `q` of the time.

| test-retest agreement | ceiling on pairwise accuracy |
|---|---|
| 0.60 | 0.724 |
| 0.667 *(measured, §7)* | 0.789 |
| 0.75 | 0.854 |
| 0.85 | 0.919 |

A model at 0.68 against a ceiling of 0.72 is at 94% of what is achievable. The same
number against an implicit ceiling of 1.0 reads as mediocre. Which sentence the
report contains depends entirely on whether the ceiling is computed.

**The derivation is verified, not asserted.** `test_the_ceiling_formula_matches_a_simulation_end_to_end`
simulates judges at `q ∈ {0.65, 0.75, 0.85, 0.95}`, measures the agreement between
repeated judgements, inverts it, and checks the result against both `q` and the
accuracy an oracle actually achieves. All three agree to ±0.01.

**Two assumptions, both stated in the code:**

- *Errors are independent across showings.* A repeat is partly remembered, which
  inflates agreement and therefore the ceiling. That direction is conservative — the
  model looks further from ceiling than it is.
- *There is one underlying preference.* Consistent idiosyncratic taste is not noise.
  Where annotators genuinely differ, this counts real disagreement as error, which is
  why the inter-annotator ceiling is lower and both are reported.

---

## 5. Comparing models: paired, never marginal

Two models sharing features agree on most comparisons, so their accuracies are
strongly correlated and overlapping marginal intervals say almost nothing. What
carries information is the difference on the comparisons where they *disagree*.

`adml.evaluate.paired_difference` bootstraps the difference on the shared held-out
comparisons and reports an exact two-sided McNemar test on the discordant ones.
Exact rather than chi-squared: the discordant count is often under thirty here.

`test_pairing_resolves_a_difference_that_independent_intervals_would_not` constructs
the case explicitly — two models with overlapping marginal intervals, one strictly
better wherever they differ, resolved only by the paired test.

Every ablation row is a paired comparison against the same full model. A row without
a `*` is **not** evidence that the group does not matter; at this corpus size most
single groups cannot be resolved either way, and saying so is the result.

---

## 6. How many sets are enough

Synthetic corpora at increasing size, using the **real measured feature
distribution** (rows resampled from the actual corpus, so the correlation structure
is realistic) and a known preference function: quality is a fixed linear combination
of four features plus a per-item idiosyncratic component the features cannot explain.
The **oracle** row scores by the latent quality itself and is therefore the ceiling
for that simulation.

| sets | pairs | testable | trained | BT-ridge | salience-only | **oracle** | trained / oracle | cycles |
|---|---|---|---|---|---|---|---|---|
| 25 | 371 | 124 | 0.629 | 0.653 | 0.661 | 0.702 | 90% | 16% |
| 50 | 750 | 266 | 0.669 | 0.729 | 0.680 | 0.759 | 88% | 20% |
| 100 | 1,500 | 530 | 0.681 | 0.677 | 0.715 | 0.719 | 95% | 15% |
| 200 | 3,000 | 1,073 | 0.686 | 0.676 | 0.644 | 0.726 | 94% | 14% |

Three readings:

1. **Absolute accuracy near 0.68 is the task's noise floor, not the model's
   failure.** The oracle — which knows the latent quality exactly — reaches only
   0.72. The trained model gets 94–95% of that from 100 sets onward. This is the
   ceiling argument of §4 reproduced from the other direction.
2. **100 sets is the right corpus size.** Below it the model is measurably further
   from the oracle; above it nothing improves. The budget line ($12 for 300 images)
   and the statistical requirement agree, which was not guaranteed.
3. **The two-step baseline is not beaten** — see §8.

---

## 7. Recommendation: judge each within-set pair twice

Intransitive triples are the reason. Three candidates judged once per pair can cycle
— A beats B, B beats C, C beats A — and such a set has no true best. It is excluded
from top-1 retention rather than resolved, because breaking the cycle by score or by
id manufactures a ground truth to be graded against.

Measured on 100 sets, doubling **only** the within-set pairs (300 of the 1,500):

| | judgements | testable | intransitive sets | 95% margin |
|---|---|---|---|---|
| 1 judgement per pair | 1,500 | 532 | 15 / 100 | ±0.039 |
| within-set pairs twice | 1,800 | 832 | **5 / 100** | ±0.031 |

Cost: **+20% effort — 17 minutes each becomes 20 minutes each** across six
annotators. It buys three things: two thirds of the intransitive sets disappear, the
testable pool grows by 56%, and multiply-judged pairs finally yield an
inter-annotator agreement, which is currently `n/a` and therefore gives no ceiling at
all.

Doubling *every* pair (3,000 judgements, 33 minutes each) cut cycles to 4% and moved
accuracy 0.681 → 0.701 — marginally better than the within-set-only change, for
almost double the human time. Not worth it.

**Do the within-set doubling.** `scripts/build_corpus.py --judgements-per-pair` and
the store's coverage-first policy control this.

---

## 8. What the measurements contradicted

Two claims that seemed obviously true and were not.

**The pairwise objective does not beat the two-step baseline here.** The plan and the
first draft of `adml.predictor` both argued that fitting Bradley-Terry strengths and
then regressing features onto them throws away the information that matters: a
strength from three comparisons and one from twelve arrive equally confident, and
squared error chases the least-observed items hardest. Measured, the two approaches
stayed within a point of each other at every corpus size, and the only resolved
difference went the *wrong* way (50 sets, −0.060 [−0.102, −0.019]).

The likely reason is the previous phase. `adml.pairs.design_pairs` gives every item
the same number of comparisons by construction — degree exactly 10, every item — so
the variance imbalance the objection depends on has largely been designed out. The
degree-balanced sampling built to guarantee graph connectivity also rescued the
baseline it was supposed to beat.

The pairwise objective remains the primary method for reasons that survive the
measurement: no intermediate fit, ties handled as observations rather than as an
interpolation problem, and no dependence on degree balance — which holds for the
annotation corpus by design but will **not** hold for judgements accumulating
unevenly on live product jobs. What it is not, on this corpus, is demonstrably more
accurate than the simpler thing. Both are reported.

**Z-scoring one-hot columns gives rare levels spurious leverage.** Dividing an
indicator by its own standard deviation scales it by how *rare* the level is.
Measured on the real corpus:

| level set for | z-scored maximum |
|---|---|
| 3 of 87 items | **+5.29** |
| 9 of 87 | +2.94 |
| 30 of 87 | +1.38 |
| 44 of 87 | +1.01 |

A design axis the sampler rarely picked would arrive with five times the leverage of
a continuous feature and absorb five times as much of the L2 penalty, purely because
few items have it. Rarity is not importance. `adml.featureset.Standardiser` now
leaves indicator columns at 0/1; centring is skipped too and would not matter anyway,
since the pairwise loss sees only score differences, in which per-column constants
cancel exactly.

---

## 9. Two other things the measurements forced

**A single inner validation split cannot select a hyper-parameter.** The first
version chose the L2 penalty on one 20% inner holdout of the training sets — about
five sets, roughly twenty comparisons. It picked a different penalty on nearly every
outer fold, ranging across four orders of magnitude (1e-4 to 1). `fit_with_selection`
now averages validation loss over **inner k-fold cross-validation** (k=3 by default),
which uses every training set for validation exactly once. The selection stabilised
and pooled accuracy moved 0.574 → 0.601 on the real corpus.

**Constant columns must be dropped and named.** The corpus is generated on a single
platform deliberately — mixing aspect ratios would make a comparison partly about
framing shape. So `platform` is constant, and so is `palette` when no brand palette
is supplied. Thirteen of 49 columns are dropped per fold on the current corpus. This
is why the ablation table reports *fitted* column counts rather than table widths: a
group whose every column is constant produces a row identical to the full model, and
printing the table width there would imply a comparison that never happened.

---

## 10. Current state, on 24 mock sets

Not a result — mock images, simulated annotators — but the pipeline end to end.

```
features : 87 items x 49 columns (photometric 4, composition 3, salience 2,
                                  palette 2, context 38)
431 observations over 29 sets, 5 folds
testable (pooled) 149    discarded 564 (26%)    margin ±0.074
ceiling: test-retest 0.667 (n=39) -> 0.789     inter-annotator n/a (n=0)

pairwise (trained)  0.601 [0.520, 0.678]   (76% of ceiling)
BT-target ridge     0.641 [0.564, 0.715]   (81% of ceiling)
salience-only       0.601 [0.520, 0.681]
random              0.500 [0.419, 0.584]
sharpness-only      0.460 [0.383, 0.540]

only resolved difference: trained beats sharpness-only, +0.141 [+0.027, +0.248], p=0.022
28% of sets intransitive
```

The honest reading: **at 24 sets nothing about the model is resolvable.** The margin
(±0.074) is wider than every difference that matters. That is the finding the §6
scaling study exists to put a number on, and it is why the 100-set corpus is on the
critical path rather than optional.

Also visible: the `context` group *hurts* (+0.060 when removed, unresolved), which is
what the parameter budget predicts — 36 fitted features against a 19-parameter
budget, most of them one-hots with nothing to say.

---

## 11. Serving: the model the report describes is the model the product uses

Until week 12 the trained head could not be used by the pipeline at all. Training
ran in `scripts/train_predictor.py`, the fitted objects lived for the length of that
process, and `adworker.scoring` went on computing a hand-weighted blend stamped
`heuristic-0`. The evaluation and the product were scoring with two different
things, and only one of them was in this document.

`adml.serving` closes that gap. Four decisions inside it are worth recording.

**The fitted transform is part of the model.** A `FeaturePipeline` holds PCA
components and standardisation statistics fitted on training rows. Saving weights
alone and re-fitting the transform at serving time would standardise against
whatever candidates happened to be in front of it — *three frames from one job* — so
a feature's value would depend on its peers rather than on itself. Both are written
to one npz, or neither is.

**Feature names are compared as an ordered list, and a mismatch raises.** The
dangerous case is not a missing column, it is a *reordered* one: every weight still
finds a number to multiply, the output stays inside 0-1, and nothing looks wrong.

**A model can be unservable for a permanent, legitimate reason.** A model fitted
with the Colab embedding blocks needs SigLIP and DINOv2 columns, and the serving path
has no torch to compute them. That is not a bug to fix — it is the consequence of the
file boundary this project is built on. So `score_image_set` falls back to the
heuristic and the pipeline emits a `warning` event naming the mismatch. The
alternative failure, a ranking silently produced by the baseline while a model was
configured, is exactly what goes unnoticed until a write-up quotes the wrong thing.
`train_predictor.py --save` says which groups a saved model needs, so the choice
between "the model for the evaluation" and "the model the product can serve" is made
deliberately rather than discovered per job.

**The 0-1 score is a within-set position, not a calibrated prediction.** The
pairwise objective has no output bias, so a constant added to every score cancels in
every comparison and is not identifiable. `ServedScores.relative()` maps the set's
worst to 0.0 and its best to 1.0 and says so; a single candidate maps to 0.5, because
one item carries no ranking information. The component breakdown shown beside a
served ranking stays heuristic — the trained head is one number over z-scored
features, and decomposing it back into named components would be invention.

### A model is a stub until it is shown to work

`ModelCard.is_stub` is true in three cases, and the third is the one that needed
adding:

| case | why it is a stub |
|---|---|
| stand-in features | plausible rankings from vectors carrying no visual information |
| accuracy never measured | nothing establishes it beats guessing |
| **accuracy did not resolve above 0.5** | it *was* measured, and it does not work |

The bar is the 95% interval's lower bound, not the point estimate — the same
standard applied to every other difference in this document. The first model this
project ever saved scored 0.496 with a CI low of 0.441 and would have shipped as a
validated predictor without that third clause.

Note that `ceiling_fraction` is **chance-corrected**: 0.68 against a 0.72 ceiling is
`(0.68-0.5)/(0.72-0.5)` = 82%, not the 94% the raw ratio in §6 quotes. Both appear in
this repo and they are different quantities. The raw ratio awards a model sitting at
chance 69% of the ceiling, which is why the card uses the corrected form.

---

## 12. Video features, and the decoder that was never exercised

The motion feature group existed before week 12 and could not have worked on a real
clip. Every video measurement decoded through `adml.features.load_frames`, which is
PIL; PIL cannot open an MP4. Mock video was written as GIF — the mock provider even
rewrote `.mp4` output keys to `.gif` — so the whole video half of the pipeline had
never seen the format every real provider returns. The first paid Kling clip would
have raised `UnidentifiedImageError` in stage 7, *after* being generated and billed.

`adml.video` is now the only place a clip becomes frames, and the mock provider
writes H.264 MP4 where ffmpeg is available, so free runs travel the same path as paid
ones. Three measurements came out of building it.

### Cut detection had to be rebuilt on the wrong statistic's failure

The first implementation scored transitions by frame correlation. It failed on
exactly the case it existed for: a concatenation of two flat-coloured segments has no
spatial structure to correlate, so **every transition including the seam scored
1.000**. The underlying bug was in `temporal_consistency`, whose guard treated "both
frames are constant" as "the frames are identical" — navy and orange are both
constant and differ by 0.62 in mean intensity.

`frame_similarity` now falls back to intensity agreement when either frame is flat.
This is not a contrivance: ad creative routinely ends on a solid-colour card behind a
logo, and any fade to white or black passes through frames with no structure at all.

Cut detection moved onto motion energy, where the separation is enormous:

| content | largest per-transition mean absolute difference |
|---|---|
| mock clips, every motion intent | **0.0093** |
| high-motion `testsrc` pattern | 0.0093 |
| navy→orange concatenation seam | **0.6201** |

67x. Detection needs *both* an absolute floor (0.15) and a robust z-score, and each
covers the other's blind spot: a `static_subtle` mock clip put its largest transition
30 robust deviations above its own median at an absolute difference of 0.0021 — that
is film grain, and the relative test alone calls it a cut. Conversely a uniformly
flickering clip exceeds any absolute floor everywhere, and then no transition is a
distinguishable cut; the clip is simply broken.

The floor is **provisional and derived from synthetic clips only**. What has not been
tested is a real generation with fast camera movement, or a cut between two genuinely
similar shots — both narrow that 67x gap. Re-derive it before quoting a cut count.

The payoff is that seam cost is measurable on *any* clip. `detect_cuts` found the
seam of a 4.5+4.5 s concatenation at transition 23 of 47 without being told it
existed, and scored it at 0.38. So a provider that chained internally without saying
so is detected from the pixels, which no `was_chained` flag would reveal.

### Two things the motion group deliberately does not carry

`seam_consistency` is NaN for every natively generated clip. A column missing for
most rows cannot be learned from, and filling it with 1.0 would assert a perfect seam
where there is no seam. It stays a diagnostic.

`duration_seconds` **would be a tier label in disguise.** The premium provider
delivers 10 s natively and the research tier reaches the window by chaining two ~5 s
clips, so in this corpus duration separates the tiers almost perfectly. A ranker given
that column could score "premium" rather than "good", and the tier comparison the
report wants would be circular. This is the same family of error as the split leak in
§3: a feature that predicts the label through provenance rather than content.

---

## Known limitations

- **No real annotations exist yet.** Every accuracy above is either simulated or
  computed from simulated judges. The ceiling in §10 is the simulation's noise, not
  a measurement of human annotators.
- **No embeddings have been computed.** `notebooks/colab_embeddings.ipynb` is written
  and its write format is asserted against the loader
  (`test_the_notebook_write_format_loads_without_the_repo`), but no GPU has run it.
  The `embedding`, `aesthetic` and `identity` rows of the ablation table are pending,
  not zero.
- **Stage A pretraining is blocked on data access**, not on code. SMPD-Video and
  Pitt Ads both need registration. Stage B — the pairwise calibration — does not
  depend on it and is what runs today.
- **The `identity` group is unbuilt.** Product-DINO and face-ArcFace similarity need
  the reference images alongside the candidates in Colab, which the manifest does not
  yet carry.
- **`OBSERVATIONS_PER_PARAMETER = 15` is a rule of thumb**, not a measurement of this
  task. It is deliberately stricter than the usual 10 because human preference labels
  are noisier than the outcomes that rule was written for.
- **Video-stage prediction is untested on generated video.** The motion group and
  `score_video` now decode real MP4s, but only mock ones: no clip has been generated
  by either tier, so the headline image→video rank-agreement number
  (`adml.evaluate.stage_agreement`) has never been computed on real data.
- **The research-tier video corpus is empty.** `notebooks/colab_video.ipynb` is
  written and the provider, manifest contract and fingerprint round-trip are all
  tested (`tests/test_prerendered.py`), but no GPU has run it. This is what stands
  between the project and a video-stage evaluation — the paid tier buys about 28
  clips in total, which is a demo, not a validation set.
- **The cut-detection floor is calibrated on synthetic clips only** (§12). Real fast
  camera movement, and cuts between similar shots, both narrow the measured gap.
- **No model has beaten chance yet.** The only model saved so far scored 0.496 on 24
  mock sets with simulated judges. That is the expected outcome at this corpus size
  and with these labels (§6, §10), and it is why the model is marked a stub — but it
  means the serving path has been exercised, not validated.
