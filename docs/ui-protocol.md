# The UI protocol

What the web app is allowed to say, and the three places a screen can quietly undo
what the pipeline was careful about.

`apps/web` is Next.js 15 with the App Router, Tailwind 4 and no component library.
It talks to FastAPI over the routes in `adapi.main`. The interesting parts are not
the components; they are the boundary with the Python and the rules about rendering.

---

## 1. The contract is generated, not transcribed

The web app needs the same types the API returns. Hand-writing them is correct on
the day it is done — and then a field is renamed in Python, the TypeScript keeps
compiling because it describes a shape nothing produces any more, and the mismatch
surfaces as an empty panel weeks later.

This is the same unrecorded transcription step
[results-protocol.md](results-protocol.md) §1 exists to remove, and it fails the
same way.

So `scripts/export_types.py` writes `apps/web/lib/contract.ts` from the Pydantic
models, and `--check` fails when the committed copy has drifted:

```bash
.venv/bin/python scripts/export_types.py --check --diff
```

`tests/test_typescript.py::test_the_committed_contract_still_describes_the_models`
runs the same comparison in the suite, so drift is a test failure rather than
something to remember.

Four decisions inside the generator are worth knowing:

**Every field is required, and nullable where Python says so.** A field with a
default is optional to the *constructor* and always present on the *wire*, because
`model_dump(mode="json")` emits it. Marking it `?:` would send every reader off
writing existence checks against values that are always there, and would hide the
genuinely absent fields among them.

**Enums emit a values array as well as a union type.** A `<select>` has to iterate
its options; listing nine verticals by hand in a component is the same drift, one
indirection later. Every dropdown in the wizard reads a generated `*_VALUES` array.

**Properties are named, not emitted.** This is the trap worth stating plainly:

| Python | What arrives |
|:---|:---|
| `JobRecord.job_id` | nothing — it is `record.request.job_id` |
| `RankedCandidate.rank_shift` | nothing — recomputed in `presentation.ts` |
| `JobResult.image_stage_order` | nothing |
| `GateResult.verified_identity`, `.reason` | nothing |
| `BudgetStatus.remaining_usd` | nothing |

Pydantic serialises fields, not properties. A client reading `record.job_id` gets
`undefined`, and a template that interpolates it renders the string "undefined". So
each generated interface carries a `// derived (not on the wire):` comment listing
its own, and everything in that column is recomputed once in `lib/presentation.ts`
beside a test that pins the arithmetic to the Python.

**Field descriptions become JSDoc.** Several are load-bearing caveats —
`ScoreBreakdown.is_stub` explains that the number is not a prediction,
`GateCheck.implemented` that a pending check always passes. Carrying them across
puts the caveat in the tooltip of the person about to render the number.

### The envelopes are generated too

`GET /api/config`, `GET /api/golden`, the job-board row, the ledger and the delivery
envelope used to return ad-hoc dicts. There was nothing to generate from, so
`lib/api.ts` hand-wrote five types and they were the only shapes in the app that
could drift silently — this section used to say so and call fixing it the obvious
next change.

They are models now (`adschema.api`), added to the generator's roots because
nothing in the pipeline returns them: without being named as roots the routes the
UI actually calls would have stayed the only untyped ones. `lib/api.ts` declares no
shapes at all any more.

Modelling the config found a field that had never crossed the wire. `SafeArea` is a
four-tuple in Python — top, bottom, left, right — and the dict built by hand
emitted two of them. Instagram Reels covers **14% of the right edge** with its
action rail, which is exactly where a CTA ends up underneath the share button, and
the wizard had been reporting `9:16 · safe area 14% top, 20% bottom` since the day
it was written. The renderer had the real numbers the whole time; only the screen
was missing them. That is the failure mode of a hand-assembled response, and it is
not one a type checker can see.

The tuple became `SafeAreaBox`, a model, for a related reason: a tuple crosses the
wire as a positional array, and `safe_area[2]` on the client is the kind of reader
that survives a reordering without complaining.

---

## 2. A stub is not a prediction, and an absence is not a zero

Everything the pipeline records about its own uncertainty can be thrown away by one
component. `{score.overall.toFixed(3)}` turns a placeholder into `0.496` on a
projector. `lib/presentation.ts` is the counterpart to the report generator: the
report stops it happening on paper, this stops it happening on screen.

Four rules, each with a test in `lib/presentation.test.ts`:

1. **A stub score is never plain text.** `scoreDisplay()` returns a tagged union —
   `value` | `stub` | `absent` — so the stub case has to be handled rather than
   skipped. A returned boolean beside a string would get dropped at the call site
   and nothing would complain. On screen a placeholder renders in the warning
   colour, struck through, naming the model version that produced it: the
   *ordering* is real, the *value* is not comparable to anything.
2. **A missing number says so.** `null` and `NaN` render as `—`, never `0%`. A zero
   is a measurement and an absence is not, and at a glance they are identical.
3. **An unimplemented check is not a pass.** The quality gate passes checks whose
   model has not landed, so `gateSummary()` splits `passed` from `pending` and only
   reports `verifiedIdentity` when both identity checks genuinely ran. Without the
   split the screen would claim identity verification that never happened.
4. **Money has four decimal places.** At two, a real $0.0004 charge and a genuinely
   free run both print `$0.00` — the difference between the cache working and having
   paid twice.

The warning colour is the same amber the report's SVG figures use, so a screenshot
of this UI and a figure in the write-up agree about what a caveat looks like.

### The mode is always on screen

Mock, live and replay produce output that looks identical. A mock frame is drawn
procedurally, a replayed frame came out of a frozen bundle, and only a live frame
was generated by a model just now. A screenshot with no mode on it cannot be told
apart, so the mode and the settings banner are in the header of every page rather
than on a settings screen nobody opens.

The budget is there for the same reason `adapi.main` gives: with $35 total, "how
much is left" belongs on screen at all times.

---

## 3. One job is not the measurement

The results page shows the image-stage order, the video-stage order, and each
candidate's movement between them. That is the project's claim, made visible on a
single job — and a single job is an illustration, not evidence.

Three candidates agreeing is an anecdote. The headline is a rank correlation over
many generation sets, which `scripts/stage_agreement.py` withholds until there are
enough of them to carry an interval. So the strip is followed by a sentence saying
exactly that, and pointing at the script. A screenshot of one lucky job must not be
readable as the result.

The distinction `presentation.ts` keeps: **"held" is a finding, not a missing
value.** A candidate that did not move means the image stage predicted it exactly,
which is what the project is trying to measure. A candidate with no image-stage rank
at all is a different thing and says `no image-stage rank`.

### The gate and the score are never merged

The pipeline separates hard pass/fail quality control from the learned performance
prediction, specifically so a quality defect cannot be laundered into a slightly
lower score. They are rendered as separate panels for the same reason — merging them
here would undo that on the last hop.

---

## 4. Tests without a test framework

`pnpm test` is `node --test`, over `lib/*.test.ts` and `components/*.test.tsx`. Node
22 strips TypeScript types natively, so the honesty layer needs no framework, no
transpiler and no config — and the files under test are the same files Next builds,
rather than a second module graph that can disagree with the first.

The cost is that imports carry their extension (`./contract.ts`), which Node's ESM
resolver requires; `allowImportingTsExtensions` in `tsconfig.json` makes TypeScript
accept it, and is safe because the app never emits.

### The components are tested, and JSX needed a loader

Testing the honesty layer proved that `scoreDisplay` tags a placeholder. It proved
nothing about anything on screen *reading* the tag, which is the whole risk: this
document used to list "a component that stops calling `scoreDisplay` and formats a
number directly would pass everything here" as a known limitation.

The obstacle was that Node strips *types*, and `<div/>` is not a type — it is a
syntax error. The options were a test framework carrying its own transform, or
`tsx-loader.mjs`: two module hooks that resolve the `@/` alias and hand `.tsx` to
the TypeScript compiler already installed. Thirty lines, no new dependency, same
trade as `adml.figures` drawing SVG rather than installing matplotlib.

The tests render with `react-dom/server` and assert on **sentences**, not structure,
so moving a caveat into a different element passes and dropping it does not.
Effects do not run, which costs nothing: none of these components fetch. `Shell` and
the pages do, and are verified in a browser instead.

One assertion is on the source rather than the output: **no component formats
money.** `usd()` uses four decimal places for a recorded reason, and a component
doing its own `toFixed(2)` would undo it somewhere no render test happened to look.
It is not a ban on `toFixed` — `DeliveryPanel` formats LUFS with it, which is a
loudness measurement and not money.

Two things surfaced while wiring this up, both worth recording because neither was
visible before:

- `ApiError` declared `readonly status: number` as a constructor parameter — which
  *emits code* rather than being a type to erase, and Node's strip-only loader
  refuses it outright. Nothing had imported `lib/api.ts` under `node --test` before,
  so the whole test file failed with a syntax error a long way from its cause.
- A fixture used `verdict: "fail"`. The vocabulary is `GateVerdict` —
  `pass | retry | reject` — and the generated union caught it. A hand-written type
  would have accepted the string and the test would have passed while asserting on
  a badge that can never render.

Same trade as `adml.figures` drawing SVG rather than installing matplotlib, and the
same reason: this machine has under 20 GB free and the project has already spent its
disk on the parts with no alternative. No component library either — shadcn/ui is a
copy-paste catalogue rather than a dependency, so what it would have added here is a
CLI and a config file for eight components that are a few hundred lines of Tailwind.

---

## 5. Talking to the API

The browser calls FastAPI directly at `NEXT_PUBLIC_API_BASE` (default
`http://localhost:8000`) rather than through a Next rewrite. A rewrite would avoid
CORS and would also sit in the middle of the SSE progress stream, where a proxy that
buffers turns a live progress bar into one that jumps to 100% at the end.
`CORSMiddleware` already allows `http://localhost:3000` explicitly.

**Media URLs are rewritten.** `LocalStorage.url_for` returns a root-relative
`/media/<key>`, which is right for the FastAPI dev harness on the same origin and
wrong from port 3000 — it resolves against Next, which has no such route, and every
frame renders broken. `resolveMediaUrl()` puts the API's origin back in front of a
root-relative URL and leaves an absolute one alone, because that is the deployment
case where object storage hands out a signed URL that must not be touched.

**The record is the source of truth, not the event stream.** The events say what
happened; the record is what was persisted. Rebuilding the result client-side from
progress messages would put a second, subtly different account of the outcome on
screen, and the two would diverge exactly when something went wrong. So the stream
drives the progress bar and the record drives everything else, re-fetched when the
stream closes.

**Errors carry the server's message.** The two refusals this app can provoke are
worth reading verbatim: the consent gate's 422 names which attestation is missing,
and the governor's refusal names the cap that was hit.

### The consent gate is a gate

`ConsentAttestation.is_valid` requires both attestations and the API returns 422
without them. The submit button stays disabled and lists what is missing, because a
form that lets you submit and then explains the refusal has taught you nothing about
why the refusal exists. Verified both ways: with both boxes false the API returns
422 with its own text, and with both true the job runs.

---

## 6. What was verified in a browser

Against `uvicorn` in mock mode, with the ledger at `$0.0000` before and after:

- **The wizard's derivations are live, and now complete.** `instagram_reels` shows
  `9:16 · safe area 14% top, 20% bottom, 14% right` — the right edge appearing for
  the first time, from the modelled config. Switching platform re-derives it:
  `instagram_feed` gives `4:5 · 5% top, 5% bottom`, `youtube_instream` gives
  `16:9 · 5% top, 12% bottom`, `tiktok` gives `9:16 · 12% top, 22% bottom, 14% right`.
  The zero edges are omitted rather than printed as `0% left`. Every dropdown is
  populated from a generated `*_VALUES` array.
- **A golden bundle replays in one click** and the replay's own drift check lands in
  the job's event log: *replay of 'aurora-reels' matches what was frozen*.
- **The SSE stream is incremental**, not batched: 34 events at 34 distinct arrival
  times spanning 2 ms to 2910 ms, with video generation and delivery taking about
  1.2 s each.
- **The multipart upload path works** with real uploaded files, and produced the
  more interesting case — image order `B › A › C` became video order `A › B › C`,
  rendered as `▲ up 1 (was #2)`, `▼ down 1 (was #1)`, `= held (was #3)`.
- **Every stub score renders marked**, with `pairwise-linear-20260811` at the image
  stage and `heuristic-0` at the video stage named on screen.
- **The gate reports `5 check(s) passed, 3 not yet implemented`** and states that
  identity is *not* verified.
- **A padded reframe is not reported as a loss**: 16:9 says a crop *would* have kept
  56%, which is why it was letterboxed instead. The 4:5 crop reports `84%`, `+9%
  against a centre crop`.
- **Intake advisories are real measurements** of the uploads: *looks out of focus or
  heavily upscaled (focus 0.34 of 1.0, below 0.35)*.

Re-checked after the API became injectable, because a refactor of the wiring is
exactly the change that breaks one route and no test:

- **A job started from the wizard still runs end to end.** Image order `B › A › C`,
  video order `B › A › C`, all three badges `= held (was #n)` with titles reading
  `Image stage #n → video stage #n` — and the "one job is an illustration" sentence
  underneath, which is the whole reason that screenshot is safe to show.
- **The golden replay route still replays.** `POST /api/jobs/golden/aurora-reels`
  returned its summary, the job completed, and the drift check landed in the job's own
  event log: *replay of 'aurora-reels' matches what was frozen*. The budget read
  `$0.0000` before and after.
- **The reframes still describe themselves honestly**: `A crop would have kept 55%`
  under the padded 16:9, `Kept 84% of the salient content — +10% against a centre
  crop` under the 4:5.
- **Money is still at four decimals** everywhere on the page: `$35.0000`, `$0.0000`,
  `$2.5000`. Interactions were driven programmatically through the DOM, because
  synthetic mouse clicks from the automation pane do not reach React's handlers in
  this environment.

---

## 7. The routes have their own tests now

The UI was tested and the pipeline was tested; the HTTP layer between them was not,
which is exactly where a renamed field or a swallowed refusal survives a green
suite. `adapi.main` built its settings, storage, ledger, job store and event bus at
module scope, so importing it opened the real database and pointed at the real
fixtures root — nothing could be isolated. `create_app(Services)` takes them as an
argument instead, and `tests/test_api.py` runs the whole surface against a temporary
directory and an in-memory ledger.

Four of those tests are about claims rather than status codes:

- **The consent gate refuses before a record exists.** A version that created the
  job and then failed it would look almost identical in a browser and would leave a
  row describing a person whose likeness there was no permission to process.
- **A mock job has an empty ledger, not one that sums to zero.**
  `CostGovernor.charge` returns before it reserves anything in any non-live mode, so
  a free run *cannot* leave a row behind. That makes zero a structural fact rather
  than an arithmetic coincidence — a row there would mean a live provider had leaked
  into a free run.
- **`/api/jobs/{id}/ledger` is a 404 for an unknown id**, which it was not before.
  An empty ledger and a missing job are both zero dollars and mean opposite things:
  the first says the run was free, the second says the question was about nothing.
- **Two apps do not share state.** That is what the factory bought, and it is the
  test that stops the module-level singletons coming back.

---

## Known limitations

- **The pages are not rendered in tests.** `Shell` and the three pages fetch on
  mount, so they are covered by having been driven in a browser rather than by a
  regression test. The components they compose are rendered and asserted.
- **The SSE stream is tested for history and closure, not for latency.** That a
  client connecting after a job finishes still receives the whole progress log is
  asserted; that events arrive incrementally rather than in one batch was measured in
  a browser (34 events at 34 distinct arrival times) and is not in the suite.
- **The job board does not paginate.** It reads the 25 most recent and stops.
- **No authentication.** Every job is visible to whoever opens the page, which is
  correct for a single-developer dev loop and nothing else. Supabase Auth is in the
  plan and not built.
- **The budget polls on a timer** (10 s) rather than being pushed. Spend only
  changes during a job, so this is chatter while idle.
- **Uploads are not validated client-side beyond `accept`.** Dimension and focus
  problems are reported by intake after the job starts, not before it.
- **The wizard cannot lock the aspect ratio.** `aspect_ratio_override` exists in the
  schema and is deliberately not exposed; platform-derived geometry is the whole
  point of that field being derived.
- **Rank movement is shown per job only.** The aggregate is
  `scripts/stage_agreement.py`, and nothing in the UI reads it yet — a page that
  showed the harvest across jobs would be the natural next addition, and would have
  to carry the same withholding rule the script does.
