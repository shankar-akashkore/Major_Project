"""The annotation endpoints: enrol, fetch a comparison, record a judgement.

Two rules shape this router.

**The client is told the minimum.**  ``/next`` returns two image URLs and nothing
else — not the pair kind, not which item is which, not whether this is a catch
trial or a repeat.  An annotator who can see that a trial is a screening test can
pass the screen while clicking through everything else, and an annotator who can
see they are repeating a pair will answer from memory.

**The client is trusted with the minimum.**  It reports the side clicked and how
long the decision took.  Everything else — which item was on that side, which
showing this is — is recomputed server-side by
:mod:`adapi.annotation_store`, so a stale page or an edited request cannot write
a judgement that misdescribes what was on screen.

Enrolment refuses without ``agreed_to_research_use``.  These are classmates'
judgements going into a report, and the opt-in is the ethics section's evidence.

**The store arrives per request, not per process.**  This router used to keep its
store and its RNG in module globals set by ``bind()``, so the last application
created in a process owned them — two apps in one test run shared an annotation
database whatever their own settings said, and a test that seeded the RNG changed
the serving order for every test after it.  :class:`Annotating` is now attached to
``app.state`` by :func:`attach` and injected like any other dependency.  It is
deliberately *not* ``adapi.main.Services``: this router must stay importable
without pulling in the pipeline, and ``main`` imports this module.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Annotated

from adschema import AnnotatorProfile, Choice, CorpusStats
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .annotation_store import AnnotationStore

router = APIRouter(prefix="/api/annotate", tags=["annotation"])


@dataclasses.dataclass(slots=True)
class Annotating:
    """What the annotation routes need: somewhere to write, and serving randomness.

    The RNG is held rather than taken fresh per request because the repeat and
    catch-trial injection is a property of a *session* — a test seeds it once and
    gets a deterministic sequence of trials, which is the only way to assert that
    a repeat comes back at all.
    """

    store: AnnotationStore
    rng: random.Random = dataclasses.field(default_factory=random.Random)


def attach(app: FastAPI, store: AnnotationStore, *, rng: random.Random | None = None) -> Annotating:
    """Give one application its annotation state. Returns it for tests to hold."""
    state = Annotating(store=store, rng=rng or random.Random())
    app.state.annotating = state
    return state


def _annotating(request: Request) -> Annotating:
    state = getattr(request.app.state, "annotating", None)
    if state is None:  # pragma: no cover - wiring error, not a runtime path
        raise HTTPException(503, "annotation store is not configured")
    return state


Ann = Annotated[Annotating, Depends(_annotating)]


class EnrolRequest(BaseModel):
    label: str = Field(default="", max_length=64)
    cohort: str = Field(default="", max_length=32)
    agreed_to_research_use: bool = False


class JudgeRequest(BaseModel):
    annotator_id: str
    pair_id: str
    choice: Choice
    latency_ms: int = Field(
        default=0,
        ge=0,
        description="Time from both images being painted to the keypress. The client "
        "measures it because only the client knows when the pixels appeared.",
    )


class SideView(BaseModel):
    """One image as the annotator sees it. Carries no provenance by design."""

    item_id: str
    url: str


class NextPair(BaseModel):
    pair_id: str
    left: SideView
    right: SideView
    judged_by_you: int = Field(description="How many comparisons you have completed.")


@router.post("/enrol", response_model=AnnotatorProfile)
async def enrol(ann: Ann, body: EnrolRequest) -> AnnotatorProfile:
    if not body.agreed_to_research_use:
        raise HTTPException(
            422,
            "Consent is required: judgements are anonymised and used only as research "
            "data for this project's evaluation.",
        )
    return await ann.store.enrol(
        AnnotatorProfile(
            label=body.label.strip(),
            cohort=body.cohort.strip(),
            agreed_to_research_use=True,
        )
    )


@router.get("/next/{annotator_id}", response_model=NextPair | None)
async def next_pair(ann: Ann, annotator_id: str) -> NextPair | None:
    store = ann.store
    profile = await store.get_annotator(annotator_id)
    if profile is None:
        raise HTTPException(404, f"no annotator {annotator_id!r}")
    if not profile.may_annotate:  # pragma: no cover - enrolment refuses these
        raise HTTPException(403, "this annotator has not consented to research use")

    chosen = await store.next_pair(annotator_id, rng=ann.rng)
    if chosen is None:
        return None
    pair, left_item, _showing = chosen
    right_item = pair.other(left_item)
    items = await store.get_items([left_item, right_item])
    if len(items) != 2:
        raise HTTPException(500, f"pair {pair.pair_id!r} references a missing item")
    return NextPair(
        pair_id=pair.pair_id,
        left=SideView(item_id=left_item, url=f"/media/{items[left_item].asset.key}"),
        right=SideView(item_id=right_item, url=f"/media/{items[right_item].asset.key}"),
        judged_by_you=await store.count_judgements(annotator_id),
    )


@router.post("/judge")
async def judge(ann: Ann, body: JudgeRequest) -> dict:
    store = ann.store
    profile = await store.get_annotator(body.annotator_id)
    if profile is None:
        raise HTTPException(404, f"no annotator {body.annotator_id!r}")
    try:
        judgement = await store.record(
            annotator_id=body.annotator_id,
            pair_id=body.pair_id,
            choice=body.choice,
            latency_ms=body.latency_ms,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    # The showing index is deliberately not returned: telling the client this was
    # a repeat would let a future version of the UI reveal it to the annotator.
    return {"recorded": judgement.judgement_id}


@router.get("/progress", response_model=CorpusStats)
async def progress(ann: Ann) -> CorpusStats:
    return await ann.store.stats()


@router.get("/quality")
async def quality(ann: Ann) -> list[dict]:
    """Per-annotator reliability. The numbers and the flags, never an exclusion.

    Whose work gets dropped is a decision to make deliberately and write down —
    `scripts/annotation_report.py` is where that argument belongs.
    """
    return [
        {
            **q.model_dump(),
            "side_bias": q.side_bias,
            "side_bias_z": q.side_bias_z,
            "repeat_consistency": q.repeat_consistency,
            "catch_accuracy": q.catch_accuracy,
            "flags": q.flags,
            "is_trustworthy": q.is_trustworthy,
        }
        for q in await ann.store.annotator_quality()
    ]


@router.get("/ui", response_class=HTMLResponse)
async def annotation_ui() -> str:
    """The comparison tool itself — one shareable page, no build step."""
    return (Path(__file__).parent / "annotate.html").read_text(encoding="utf-8")
