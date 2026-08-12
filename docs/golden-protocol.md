# Golden demo protocol

Budget rule 6 says: *freeze a golden demo set by week 14 and demo from fixtures;
never generate live during the viva.* This is how that works, and why it is not
simply a screen recording.

The financial argument is the obvious one — a premium job costs $2.22, roughly eight
exist in the whole budget, and a demo that regenerates spends one every time it is
shown. The stronger argument is that **the video stage is not reproducible at any
price.** Kling's image-to-video endpoint takes no seed (`honours_seed=False` in
`pricing.py`), so a job that produced three good candidates cannot be asked to
produce them again. Freezing is not a cost optimisation. It is the only way to be
able to show a specific result twice.

```bash
.venv/bin/python scripts/freeze_golden.py --slug aurora-reels
```

```bash
.venv/bin/python scripts/replay_golden.py --all
```

---

## 1. What is frozen, and what is recomputed

This is the load-bearing decision, and it is not the obvious one.

The obvious freeze is the whole `JobRecord`: replay means deserialising it and
rendering the UI. That would be a video recording of a demo rather than a demo. The
gate, both ranking stages and the whole of delivery would never run, so a regression
in any of them would first be noticed in front of an examiner.

So only what cannot be recomputed is frozen:

| Frozen | Why it cannot be recomputed |
|---|---|
| Upload bytes | The user's photographs. Not derivable from anything. |
| LLM brief text | A sampled completion; a re-run rewords it. |
| Generated frames | Provider-side sampling. A seed narrows it, nothing fixes it. |
| Generated clips | Kling has no seed at all, so nothing narrows it. |

Everything else runs for real on every replay: intake, the product cutout, palette
extraction, the quality gate, the image-stage ranking, ffmpeg decoding and clip
measurement, the final ranking, the reframes, the previews, the zip.

That makes a golden bundle two things at once — the demo that gets shown, and a
**regression test over almost every stage the project has.** The bundle carries the
ranking its frozen job produced, and `compare_replay` says whether today's code still
produces it. `scripts/replay_golden.py --all` exits non-zero when it does not.

Measured on the bundle in this repo: a replay of a 3-candidate job reproduces the
image-stage order, the video-stage order, the winner and every score component
**bit-for-bit**, and moves the ledger by $0.0000.

---

## 2. The replay that read nothing

The first replay written here reported a clean pass: same ranking, same scores, no
spend. It had opened zero frozen files.

The cost governor's generation cache is consulted **before** the provider is called.
A replay hands the pipeline the same uploads, which hash to the same digests, which
produce the same fingerprints as the freeze — so every lookup hit, and the golden
providers were never reached. The output was correct entirely by accident of what
happened to be in the local `adgen.db`. On a fresh machine with an unpacked bundle
and an empty database, the replay would have taken a code path that had never been
executed.

Two changes, because one of them is a fix and the other is a detector:

- **`CostGovernor(..., use_cache=False)`.** In a replay the bundle *is* the cache, and
  which bytes get served must not depend on database state. Writing is disabled with
  reading, not merely for symmetry: a replay's assets live under a throwaway job id,
  and caching them would point a later real run at a directory that gets cleaned up.
- **`GoldenSession.audit()`.** Counts what was actually served and reports zero as a
  problem. The flag prevents the bug; the audit is what would notice it coming back.

Both directions are tested —
`test_the_generation_cache_would_otherwise_hide_the_bundle` reproduces the failure,
and `test_a_replay_serves_the_frozen_bytes` asserts 3 frames and 3 clips read from
the bundle with a clean audit.

## 3. The freeze that captured nothing

The same boundary, from the other side. The freeze wraps each provider in a
`Recording*` adapter — and the first bundle it produced held three ranked candidates,
a delivery report, and **zero frames**. The generations had come from the cache, so no
provider was called, so the wrapper saw nothing.

Nothing about the cache is wrong here: serving a repeat request for free is the point
of it, and in a live freeze it is the difference between $0 and $2.22. So the bundle
is completed from the job record instead (`GoldenRecorder.backfill`), whose asset
references point at the same bytes, and the manifest records which generations came
that way:

```json
"backfilled": ["frame on slot 0 came from the generation cache", ...]
```

That matters for reading the bundle later: a backfilled frame was produced by an
earlier run, so `provenance.cost_usd` is that run's cost and not this one's.

**One thing is lost.** The job record keeps only the candidate that was promoted per
slot, so a slot that needed a gate retry is frozen as a single attempt. That is the
right shape for a replay — the frozen frame is the one that passed, so the gate will
pass it first time and never ask for a retry — but a bundle built by backfill cannot
demonstrate the retry path. A bundle whose generations were recorded live keeps every
attempt.

## 4. The request has to be the submitted one

Stage 1 writes the extracted palette back into `request.theme.palette` and sets
`palette_auto_extracted=True`. Freeze the request the pipeline returned and a replay
starts with a palette it never had to derive — so the intake extraction path the
bundle exists to exercise is skipped, and skipped silently.

`build_golden_bundle` therefore takes the request **as submitted**, and
`scripts/freeze_golden.py` snapshots it with `model_copy(deep=True)` before the run.
`test_the_frozen_request_is_the_one_that_was_submitted` asserts both halves: the
frozen palette is empty, and the record's is not.

---

## 5. Drift is reported, not prevented

A frame is served on its candidate **slot**, because that is what is reproducible: the
sampler is seeded, so slot 1 is slot 1 across runs. The design point and a hash of the
image prompt are recorded alongside it, and compared on serve.

If they no longer match, the frame is still served. A demo that refuses to run because
a lighting enum was renamed is worse than one that runs and says what changed. But the
mismatch is collected, and the script exits non-zero on it.

`compare_replay` keeps four categories apart, because they mean different things:

| Category | Meaning | What to do |
|---|---|---|
| **spend** | A replay charged money | A bug in `golden.py` — fix it |
| **result** | Gate verdict, ranking or winner changed | The demo shows something else now |
| **serving** | Frozen media no longer matches what asked for it | Re-freeze, or accept the relabelling |
| **scores** | Numbers moved without reordering | A code change to explain |

The one drift that invalidates rather than relabels is a clip whose `start_sha256` no
longer matches the frame the image stage just produced. That candidate's image/video
pair is no longer the pair that was frozen, which is precisely the thing the headline
rank-agreement measurement is about.

### Accepting drift without paying again

Some drift is correct. When the ranker is retrained the scores *should* move, and
re-freezing to record that would mean paying for the generations again — while the
media, the expensive part, has not changed at all.

```bash
.venv/bin/python scripts/replay_golden.py aurora-reels --accept
```

rewrites only the expectation block. Every asset is left alone, so adopting a new
outcome costs nothing and does not change a single pixel of what the demo shows. It
refuses to accept a bundle whose replay spent money, because that is never the
correct new expectation.

---

## 6. Where a bundle lives, and what is committed

```
fixtures/golden/<slug>/golden.json     the manifest — committed
fixtures/golden/<slug>/assets/*        the frozen bytes — not committed
```

The manifest is source: it records what was generated, what it cost, what ranking it
produced, and every asset's SHA-256. A few kB of JSON. The media is not: eight bundles
of three 10 s clips is tens of megabytes, and "media lives in object storage, never in
git" is the one rule this repo has held to throughout.

The consequence is stated rather than hidden: **a fresh clone can read the record but
cannot replay the media.** `verify()` checks every asset by content hash, so a missing
or truncated file is reported by name instead of quietly changing the demo:

```bash
.venv/bin/python scripts/replay_golden.py --verify --all
```

and the copy that survives a laptop reinstall is a zip:

```bash
.venv/bin/python scripts/freeze_golden.py --slug aurora-reels --pack
```

That is the artefact that goes on a USB stick the week before the viva.

---

## 7. Replay is a third provider mode

`ProviderMode.REPLAY` rather than a variant of `MOCK`, because the two make opposite
promises about the pixels: mock output is synthetic and says so, replay output is what
a paid provider really returned. Both cost nothing, and `Settings.is_live` — the only
thing any budget check reads — is False for both.

The governor's zero-estimate assertion is written against *not live* rather than *is
mock* for that reason. A golden bundle of a premium job carries model names priced in
dollars, so if a replay provider forgot to override `estimate_cost` the assertion is
what catches it, rather than a silent charge.

The three golden providers **share one `GoldenSession`**: it holds the per-slot attempt
counter and collects the drift notes. So the single-provider getters refuse replay mode
and point at `get_providers()`, rather than returning something that works until the
first gate retry.

The API route takes a slug explicitly and works whatever mode the app is in:

```
GET  /api/golden                  what is frozen, and whether it is replayable
POST /api/jobs/golden/{slug}      replay it as a real job
```

Requiring a restart into replay mode would mean the only way to show a frozen job is
to reconfigure the service — and flipping to live mode to get there would arm the paid
image provider at the same time. The drift verdict is appended to the replayed job's
own event log, so a regression is visible in the run that found it rather than in a
script's stdout nobody reads during a demo.

---

## Known limitations

- **The bundle in this repo is frozen from synthetic references.** It demonstrates the
  machinery, not the output. The real golden set needs the premium budget released and
  real photographs; `freeze_golden.py` prints this caveat on every synthetic freeze.
- **A backfilled bundle cannot demonstrate the gate retry path** (§3).
- **`--accept` adopts scores wholesale.** It does not distinguish a retrained ranker
  from a scoring bug; that judgement is the reader's, which is why the drift is printed
  before anything is written.
- **Only the promoted candidates are frozen.** A candidate the gate rejected is not in
  the bundle, so a replay cannot show the rejection — it will re-derive it only if the
  rejected frame happened to be recorded live.
- **The `expectation` block compares a fixed set of quantities.** Something the
  pipeline starts producing that is not in `GoldenExpectation` can change without any
  replay noticing.
