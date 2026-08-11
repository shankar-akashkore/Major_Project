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
"""

from __future__ import annotations

import random
from pathlib import Path

from adschema import AnnotatorProfile, Choice, CorpusStats
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .annotation_store import AnnotationStore

router = APIRouter(prefix="/api/annotate", tags=["annotation"])

#: Set by the app at startup. A module-level handle keeps the router importable
#: without a database, which the tests rely on.
_store: AnnotationStore | None = None

#: Serving randomness. Seeded per process rather than per request so a test can
#: make the repeat and catch injection deterministic.
_rng = random.Random()


def bind(store: AnnotationStore, *, rng: random.Random | None = None) -> None:
    global _store, _rng
    _store = store
    if rng is not None:
        _rng = rng


def _require_store() -> AnnotationStore:
    if _store is None:  # pragma: no cover - wiring error, not a runtime path
        raise HTTPException(503, "annotation store is not configured")
    return _store


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
async def enrol(body: EnrolRequest) -> AnnotatorProfile:
    if not body.agreed_to_research_use:
        raise HTTPException(
            422,
            "Consent is required: judgements are anonymised and used only as research "
            "data for this project's evaluation.",
        )
    store = _require_store()
    return await store.enrol(
        AnnotatorProfile(
            label=body.label.strip(),
            cohort=body.cohort.strip(),
            agreed_to_research_use=True,
        )
    )


@router.get("/next/{annotator_id}", response_model=NextPair | None)
async def next_pair(annotator_id: str) -> NextPair | None:
    store = _require_store()
    profile = await store.get_annotator(annotator_id)
    if profile is None:
        raise HTTPException(404, f"no annotator {annotator_id!r}")
    if not profile.may_annotate:  # pragma: no cover - enrolment refuses these
        raise HTTPException(403, "this annotator has not consented to research use")

    chosen = await store.next_pair(annotator_id, rng=_rng)
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
async def judge(body: JudgeRequest) -> dict:
    store = _require_store()
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
async def progress() -> CorpusStats:
    return await _require_store().stats()


@router.get("/quality")
async def quality() -> list[dict]:
    """Per-annotator reliability. The numbers and the flags, never an exclusion.

    Whose work gets dropped is a decision to make deliberately and write down —
    `scripts/annotation_report.py` is where that argument belongs.
    """
    store = _require_store()
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
        for q in await store.annotator_quality()
    ]


@router.get("/ui", response_class=HTMLResponse)
async def annotation_ui() -> str:
    """The comparison tool itself — one shareable page, no build step."""
    return (Path(__file__).parent / "annotate.html").read_text(encoding="utf-8")
