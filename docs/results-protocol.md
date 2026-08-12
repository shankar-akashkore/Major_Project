# The results protocol

How the write-up's numbers get from the code to the page, and the four ways that
journey can produce something false while every individual step looks right.

`scripts/report.py` writes `docs/results/`. `scripts/stage_agreement.py` measures the
headline claim. Neither is a formatting convenience: each exists because of a specific
way this project could have overclaimed.

---

## 1. Transcription is the unrecorded step

Every number in a report normally makes one journey nobody logs — out of a terminal,
through a person, into a document. Two things happen on that journey.

A number goes **stale**: the model is retrained, the table is not, and there is no way
to tell by looking. And a number **loses its caveat**: an accuracy measured on
simulated labels is copied without the "simulated" beside it, and from then on it is
just an accuracy.

So the tables and the figures are built from the same objects the evaluation harness
produced, by one command, and the document records the commit that produced it. The
generated files say `Do not edit — regenerate` because a generated document that looks
editable will be edited, and the edit will vanish on the next run without anyone
noticing which numbers went back.

What is deliberately **not** generated is the argument. Interpretation, related work
and the discussion are the author's. This produces the evidence they refer to.

---

## 2. A cell is a value or a stated absence

There are no blanks. A blank cell is indistinguishable from a zero, from a rounding
artefact, and from a row somebody deleted.

`adml.report.Pending` is a cell type carrying a reason, and it renders as a visible
`pending — <reason>` in markdown, LaTeX and plain text alike. A row that cannot be
computed at all is added with `pending_row`, which keeps it **named**:

> A reader who cannot see that LLM-as-judge is one of the planned baselines cannot
> tell whether it was tried and lost or was never run.

Rows blocked on the same cause collapse into one line. Six baselines waiting on the
same missing dataset is one problem, not six; printed as six it buries every other
outstanding item.

Two separate gates decide whether a number may be printed at all, and they are
independent:

| Gate | Question | Fails when |
|:---|:---|:---|
| quotable | is the sample big enough to carry an interval? | fewer than `MIN_QUOTABLE_SETS` sets |
| is-a-result | did the numbers come from real labels and real features? | stub scores, simulated labels, hash embeddings |

A stub agreement over fifty sets is quotable and not a result. A trained agreement
over one set is a result and not quotable. Both have to pass.

---

## 3. The claim is not what is easy to compute

The project's claim is that ranking three candidates **as images** anticipates how the
finished **videos** are preferred. "Preferred" means by a person. That is the number
the project stands on.

What is computable today is different: the image-stage ranking against the
*video-stage ranking*. Both come from this codebase, over overlapping feature groups,
which guarantees some agreement by construction. It measures whether the video stage
reorders anything — internal consistency — not whether either stage is right.

`adml.stages.Harvest` keeps them apart by naming:

- `model_vs_model()` is computable and is reported as a **diagnostic**. It has no
  `headline` alias, so a caller that wants to print it has to print what it is.
- `human_stage_agreement()` returns a `HumanStageAgreement` whose `summary()` is
  `PENDING — <reason>` until pairwise judgements over generated videos exist.

Passing the first off as the second would be the single most likely way this project
overclaims, so the two live in different functions with different return types rather
than behind one flag.

The blocker is not code. It is: run `notebooks/colab_video.ipynb` on a GPU to fill the
clip corpus, build the video corpus, and collect judgements at `/annotate/ui`.

---

## 4. A replayed job is not a second observation

This one was a live trap, set by week 14's own machinery.

Every golden replay writes a fresh `JobRecord` — new job id, new storage prefix,
**bit-identical candidates**. Measured per job id, running `replay_golden.py --all`
twenty times before the viva would report twenty generation sets in perfect agreement,
with a confidence interval of zero width. A fabricated headline result, produced
entirely by honest-looking code, with no bug anywhere in it.

Verified in the store rather than assumed:

```
6565a3cf35  img0 sha=9b419a2b8da22966  img1 sha=c99e3c5a76671106  img2 sha=232ba3e2c5640d51
78b27ed254  img0 sha=9b419a2b8da22966  img1 sha=c99e3c5a76671106  img2 sha=232ba3e2c5640d51
```

Two job ids, two storage prefixes, one set of bytes.

So `adml.stages.set_key` hashes the candidates' own content digests, sorted by
candidate slot. Storage keys are excluded deliberately — they carry the job id, which
is exactly what differs between a job and its replay. Three consequences:

1. **Replays collapse** onto the job they replay, and the count of collapsed records
   is reported rather than hidden.
2. **The kept job is the original.** Records are processed oldest first, so the job id
   in the output is the one that produced the set, not the last replay of it.
3. **Disagreement is a defect.** Two jobs sharing a set key ran the same ranker over
   the same bytes, so a different ordering is nondeterminism in the scorer, not new
   evidence. Those become `Harvest.conflicts` and the script exits non-zero.

When a candidate has no `sha256` there is nothing to key on, so the job id is used and
`keyed_by_content` is `False` — the report then says n is an upper bound, rather than
claiming a deduplication that did not happen.

Slot matters: the same three frames in different slots is a different set, because the
design point attached to each slot differs. Same reasoning as `clip_fingerprint` keying
on motion intent rather than prompt text.

---

## 5. Figures without matplotlib

`adml.figures` writes SVG from the standard library. This is a disk-budget
consequence, not a preference: matplotlib and its dependencies are a few hundred
megabytes and this machine has under 10 GB free, which already went on the parts of
the project with no alternative. Four bar-and-whisker plots is a few hundred lines of
coordinate arithmetic.

SVG because it is text: it diffs, it scales into a printed report without resampling,
and it needs no image library to write.

Two properties are load-bearing and are what any future port to matplotlib would have
to preserve:

**Every interval is drawn.** A point estimate with no interval is drawn *hollow*, in
the warning colour, labelled `no interval` — the shape is the warning, so it survives
being cropped away from its caption. A bar chart of point estimates is the most
misleading artefact this project could produce, because at this corpus size the
intervals are wider than every difference between models.

**Every figure carries its own provenance line**, in the warning colour when the data
is simulated, inside the SVG. An image pasted into a slide deck arrives with its
caveats rather than losing them in transit.

A missing measurement is drawn as missing: `not measured` text and no mark at all. A
NaN must never become a dot at zero.

### The four figures

| Figure | Answers |
|:---|:---|
| `accuracy_forest` | do the models' intervals overlap, and where is the noise ceiling? |
| `ablation_deltas` | which feature groups carry weight, and which cannot be resolved? |
| `stage_scatter` | did candidates hold their rank between the stages? |
| `sample_size_curve` | how many sets before the differences become resolvable? |

A forest plot rather than bars, because the comparison that matters is interval
overlap and bars invite reading the tops of two rectangles as a difference. The
accuracy is shown against the measured ceiling, never against 1.0 — with single noisy
human judgements, 1.0 is the wrong denominator and using it silently is the
difference between "near-perfect" and "mediocre" for the same model.

---

## 6. LaTeX has to be ASCII

The headings this project produces contain `ρ`, `τ`, `Δ`, `§` and `%`; the feature
group names contain underscores. Under pdflatex a bare Greek letter is not a
typographic nit but a hard error — *Unicode character ρ not set up for use with
LaTeX* — in a generated file nobody thinks to open until the build fails, a week
before the deadline.

So `_tex` escapes the LaTeX special characters and then maps the symbols to macros
(`ρ` → `$\rho$`, `§` → `\S{}`, `—` → `---`). The order matters: symbol replacement
runs *after* character escaping, because the replacements are themselves LaTeX and
must not be escaped in turn. A test asserts the whole emitted document
`isascii()`, which is the property that makes the file safe to `\input` under any
engine.

The caveat goes in the `\caption`, not a footnote: a caption travels with the float
when LaTeX moves it to another page, and a footnote does not.

---

## 7. What is committed

`docs/results/` is committed, generated files and all — unlike generated media, which
never is. The reason is the same one that makes the golden manifest committed: it is
text, it diffs, and a change in the numbers therefore shows up in review. "The
ablation moved" is worth seeing in a diff.

Each file records the commit that produced it, so a stale artefact is detectable
rather than merely suspected.

---

## Known limitations

- **The `--check` gate is all-or-nothing.** It fails while *any* table rests on
  simulated labels, which today is all of them. It becomes useful as a per-table gate
  once some tables are real and others are not.
- **`CHAR_WIDTH` is an approximation.** Deciding whether a value label fits before the
  right edge uses a fixed advance width rather than a real font metric, which would
  need a font library. Being wrong only risks flipping a label that would have fitted.
- **The sample-size curve is a simulation, not an observation** — it comes from §6 of
  [prediction-protocol.md](prediction-protocol.md) and is recorded as a constant so
  the figure that justifies the generation spend does not need a training run to exist
  first. It is stamped as simulated.
- **`human_stage_agreement` has no caller that can succeed yet.** It is written and
  tested against constructed orderings; wiring it to real video judgements needs a
  mapping from judged video items back to generation sets, which the corpus manifest
  does not yet carry.
- **The models and ablation tables have no figure** in the current output. They are
  pending, and an empty plot would be worse than none.
- **Rank shift is reported over placements, not sets.** A distribution of zeros over
  one set says almost nothing, and the text says so, but the count is placements so it
  looks larger than the evidence is.
