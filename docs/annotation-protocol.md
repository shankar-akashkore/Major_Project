# The annotation protocol

How the project's preference labels are collected, and why in this particular way.
This is the week 7–8 deliverable and the critical path: annotation feeds Stage B
calibration in week 11, which feeds the evaluation in week 15. Nothing downstream
can be better than the labels.

---

## What is being asked

Two generated advertisement images, side by side, one question: **which would you
be more likely to stop and click on?** Two-alternative forced choice, with an
explicit "too close to call" that is recorded as a tie rather than forced into a
preference. Keyboard-driven (← → ↓), median response around 2 s in simulation.

Not a rating scale. Absolute ratings drift within a session and are not comparable
across annotators; pairwise comparisons are stable, and Bradley-Terry converts them
into a single latent strength per image.

The tool is one page with no build step: start the API and share
`http://<your-ip>:8000/api/annotate/ui`.

---

## Design of the comparison set

Two kinds of pair, for two different reasons.

**Within-set** — the 3 candidates from one generation set, all pairs. Product,
model and brand are held constant, so the comparison isolates the design point.
This is the deployment task (rank 3 candidates for one job) and what the sampler
ablation needs.

**Cross-set** — pairs across different sets. Individually less informative, and
collectively indispensable:

> Bradley-Terry infers one latent strength per item from who beat whom. If the
> comparison graph is disconnected, strengths *within* each component are
> estimable and strengths *across* components are not comparable at all, because
> no data links the scales.

The obvious design is the broken one. Compare only within each set and 100 sets of
3 give **100 disconnected triangles** — 100 incomparable scales, and a fit that
returns confident numbers anyway. `adml.pairs.design_pairs` therefore adds
cross-set pairs, then *verifies* connectivity with union-find and stitches any
remaining components with pairs explicitly labelled `BRIDGE`, so the report can say
how much of the connectivity was engineered rather than sampled. A test asserts the
exact arithmetic: within-set-only over 100 sets needs 99 bridges.

Cross-set pairs are generated as a sequence of random near-perfect matchings, so
every item gains one comparison per round *by construction*. Sampling pairs
uniformly at random instead leaves some items with one comparison and others with
fifteen, and an item nobody compared has no estimable strength at all.

### Sizing

Recovering a reliable ranking over `n` items by pairwise comparison needs on the
order of `n log n` comparisons. Measured, for the planned 300-image corpus at 8
cross-set rounds:

| | |
|---|---|
| items | 300 (100 sets of 3) |
| within-set pairs | 300 |
| cross-set pairs | 1,200 |
| bridges needed | 0 |
| comparisons per item | **exactly 10 for every item** |
| total pairs | 1,500 |
| human time, single-judged | 110 min |
| **per annotator, 6 annotators** | **~18 min** |

Eighteen minutes each is a realistic ask. That number is why the design is worth
checking *before* recruiting: a design nobody finishes leaves a partially observed
graph, which is the disconnection problem arriving by the back door.

---

## Quality control

Volunteers clicking through a task as a favour are the single biggest threat to
label quality, so five mechanisms measure it. None of them excludes anyone
automatically — `scripts/annotation_report.py` prints the numbers and the flags,
and dropping someone's work is a decision made deliberately and written down.

**1. Position is randomised server-side.** Which image appears on the left comes
from a hash of `(annotator, pair, showing)`, computed identically by the serving
and recording endpoints. The client is never asked which side it drew, so a stale
or edited client cannot corrupt the analysis, and no server state is needed.
A judgement records *the side clicked* plus which item was on that side; the
winner is derived. Storing only the winner would be smaller and would silently
destroy the ability to measure left-side preference, a well-known 2AFC artefact.

**2. Side bias is tested as a z-score**, not a fraction. Picking left 13 times out
of 20 is unremarkable; 325 out of 500 is not looking at the images. Flag at
|z| > 3.

**3. Repeats measure test-retest consistency.** 8% of servings re-show a pair the
annotator has already judged, and **the sides are mirrored** so the repeat tests a
preference rather than a remembered keypress. Catch pairs are excluded from repeat
candidates — re-asking a question with a right answer measures nothing.

**4. Catch trials are the direct evidence of attention.** A real candidate against
a heavily blurred, desaturated, contrast-crushed copy of itself. Decoys are built
by `adml.degrade`, which *measures* the result and refuses a decoy whose sharpness
did not drop by 0.30 — a subtly-worse decoy tests eyesight and would flag careful
people as careless. Measured on real corpus images: sharpness 0.999 → 0.010.

**5. Latency is measured from paint, not from request.** The client prefetches the
next pair's images and starts the clock when both have rendered, so the number is
decision time rather than network jitter.

### Two thresholds that had to be derived rather than picked

A simulated session with four attentive annotators, one always-left clicker and
one random clicker exposed both.

**Catch accuracy cannot be thresholded as a fraction.** At `MIN_CATCH_ACCURACY =
0.80` with 6 trials, 5/6 passes and 4/6 fails — so an attentive annotator who
mis-clicks twice is excluded while a lucky guesser is kept. In the first run this
flagged one of the four *careful* annotators. Six trials genuinely cannot separate
83% accuracy from 95%, and no choice of threshold changes that. Replaced with a
one-sided binomial tail against pure guessing, `P(X ≥ correct | p = 0.5)`, flagged
above 0.10; and `MIN_CATCH_TRIALS` set to **10** from the false-positive
arithmetic, not by feel:

| trials | flag fires at | false-positive rate for a 95%-accurate annotator |
|---|---|---|
| 6 | anything below 6/6 | **27%** |
| 10 | 7/10 or worse | **1.1%** |

**A rate cannot deliver enough catch trials.** At 6%, a 45-judgement session
expects 2.7 — below the minimum, so the one mechanism that directly detects
clicking-through could not fire at all for a short session, which is the common
case for a volunteer. Worse, a screen first reached after 50 trials has not
screened anything: someone who is going to click through does it from the start.
Replaced with unconditional checkpoints at judgement 3, 8, 14, 21, 29, 38, 48, 59,
71 and 84 — three inside the opening 15, reaching the minimum by 84 — with the 6%
rate on top. This implies a floor on the corpus: an annotator sees each catch pair
once, so **the corpus needs at least 10 decoys** for the screen to work.

After both fixes, on the same simulated session:

| annotator | n | retest | catch | p(guess) | left% | z | median ms | flagged |
|---|---|---|---|---|---|---|---|---|
| Ana (attentive) | 120 | 80% | 11/12 | 0.00 | 50% | +0.0 | 2598 | — |
| Ben (attentive) | 120 | 60% | 11/12 | 0.00 | 47% | −0.7 | 2568 | — |
| Cara (attentive) | 110 | 80% | 12/12 | 0.00 | 50% | −0.1 | 2548 | — |
| Dev (attentive) | 110 | 67% | 9/12 | 0.07 | 43% | −1.5 | 2472 | — |
| Eli (clicks left) | 100 | 0% | 7/12 | 0.39 | 100% | +10.0 | 294 | catch, repeats, side, speed |
| Fay (random) | 100 | 60% | 4/12 | 0.93 | 33% | −3.4 | 1056 | catch, side |

Four attentive annotators, no false positives; both bad actors caught. Note Fay was
**invisible before the catch-trial fix** — random clicking with plausible latency
looks like a preference.

### Every remaining threshold is provisional

`MIN_REPEAT_CONSISTENCY`, `MIN_PLAUSIBLE_LATENCY_MS` and `MAX_SIDE_BIAS_Z` are
starting points from the usual 2AFC ranges, not measurements of *this* task. The
intake focus threshold already had to be walked back after being set against
unrepresentative fixtures, so section 3 of `annotation_report.py` prints the
observed distribution of each one and says what to replace it with. **Do that after
the first real session, before excluding anyone's work.**

---

## Reading the collected labels

`scripts/annotation_report.py` answers, in order:

1. **Coverage** — and specifically whether the *observed* graph is connected. A
   connected design says nothing if half the pairs are unjudged; Bradley-Terry is
   fit on what was collected. Coverage is measured over the design excluding catch
   pairs, which are served on a schedule and would flatter the number.
2. **Per-annotator reliability** — the table above.
3. **Observed distributions**, for replacing the provisional thresholds.
4. **Inter-annotator agreement** — Krippendorff's α over pairs judged more than
   once. α rather than raw agreement because on a two-alternative task 50% raw
   agreement is what coin-flipping produces. α rather than Cohen's κ because there
   are more than two annotators and each judges a different overlapping subset,
   which κ cannot express. Only first showings count — including a repeat would
   inflate agreement with an annotator's agreement with themselves.
5. **Split-half reliability** — fit Bradley-Terry on half the *annotators*, fit on
   the other half, correlate the strengths. The strongest single check that the
   labels carry signal, and it needs no ground truth. The split is on people, not
   comparisons: splitting comparisons puts the same person on both sides and
   measures the fitter's stability instead of the annotators' agreement.
6. **The fit** — convergence, connectivity, and the extremes so the ranking can be
   eyeballed against the images.

### The fit itself

Hunter's minorization-maximization, in `adml.ranking`. No learning rate, no
autograd, no GPU; converges monotonically in ~100 iterations on the laptop.

One real trap: the unregularised MLE gives an item that wins every comparison an
**unbounded** strength, and the iteration walks toward infinity while appearing to
make progress. `prior_strength` gives every item a small number of games against a
phantom opponent, half won and half lost, which keeps estimates finite and pulls
sparsely-observed items toward the middle — also the honest thing to report for an
item nobody looked at twice. A side effect worth naming: with a prior the fit is
computable even on a disconnected graph, so `is_connected` is reported separately.
Computable is not comparable.

Ties are counted as half a win in each direction rather than dropped. A forced
choice between two equivalent candidates is noise; a recorded tie is data.

Verified against synthetic data drawn from the model: order recovered exactly,
Spearman ρ = 1.0, and the *spacing* recovered to within 0.2 on a true gap of 4.0.

---

## Running it

```bash
# 1. Generate the corpus. Free in mock mode; $12 for 300 real images.
.venv/bin/python scripts/generate_corpus.py --sets 100 --price   # cost, no calls
.venv/bin/python scripts/generate_corpus.py --sets 100           # mock, free

# 2. Design the comparisons. Free, no provider involved. --plan writes nothing.
.venv/bin/python scripts/build_corpus.py --plan --annotators 6
.venv/bin/python scripts/build_corpus.py --commit --decoys 12

# 3. Serve the tool and share the URL.
.venv/bin/python -m uvicorn adapi.main:app --host 0.0.0.0 --app-dir services/api
#    → http://<your-ip>:8000/api/annotate/ui

# 4. Read the labels.
.venv/bin/python scripts/annotation_report.py
```

Steps 1 and 2 are re-runnable and additive: items and pairs are de-duplicated, and
the design is recomputed over the *whole* corpus so a new batch is joined to the
old rather than forming its own component.

---

## Ethics

Enrolment requires an affirmative `agreed_to_research_use` and is refused without
it — 422, the same stance the pipeline takes on model releases. Annotators are
identified by a nickname they choose, judgements are reported in aggregate, and the
nickname is never published. This is the report's ethics-section evidence, not a
formality: these are classmates' judgements going into a paper.

---

## Known limitations, for the write-up

- **Corpus diversity is bounded by the reference photographs supplied.** With four
  reference pairs and 100 sets, each product recurs 25 times under different design
  points. That is what the within-set comparisons want, and it means a cross-set
  comparison partly reflects which product was photographed better rather than
  which creative decision was made. Record how many reference pairs the corpus was
  built from.
- **Gate-failing candidates are excluded** from the corpus on purpose. Ranking a
  mangled face teaches the predictor to detect mangled faces, which the gate
  already does for free.
- **Agreement needs deliberate overlap.** The coverage-first serving policy spreads
  judgements thin by design, so pairs judged by 2+ annotators accumulate slowly.
  Run `build_corpus --judgements-per-pair 2` when planning if α matters early.
- **The simulated session is not evidence about humans.** It validated the
  machinery — that the flags fire on bad actors and not on good ones — and nothing
  about real annotator behaviour. Every number in the table above is synthetic.
- **Two problems only a browser found**, both fixed, both worth knowing about
  because neither would have failed a test:
  - The mock renderer burns the design point and the seed into the corner of every
    frame, which is what makes the dev harness readable and which *told the
    annotator the condition they were meant to judge blind*. `generate_corpus.py`
    now turns that caption off; a real generation has no such text, so labels
    collected against captioned frames would not have transferred either.
  - Rotating platforms across sets put 9:16 and 4:5 images into the same
    comparison. A letterboxed 4:5 frame beside a full-height 9:16 one is not an
    equal presentation, and the judgement would partly be about framing shape.
    The corpus is now single-platform by default (`--platform`).
