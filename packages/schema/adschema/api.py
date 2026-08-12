"""The API's envelope shapes.

Every other module here describes something the *pipeline* produces.  This one
describes what the *HTTP layer* wraps it in: the config the UI boots from, a row
on the job board, the acknowledgement of a launched job.

They were dicts until now, and dicts were the last place a field could be renamed
without anything noticing.  ``docs/ui-protocol.md`` §1 said so plainly and listed
these two routes as the exception to the rule the rest of the app follows —
``lib/api.ts`` hand-wrote their types, so a key changed here and the TypeScript
kept compiling against a shape nothing produced.  Modelling them removes the
exception: they now go through :mod:`adschema.typescript` like everything else and
``scripts/export_types.py --check`` fails on drift.

Two notes on what is *not* here.

**No provider imports.** :class:`AppConfig` mirrors fields off ``Settings``, and
the direction of the dependency matters: ``adproviders`` imports ``adschema``, so
this module cannot import ``Settings`` to build itself.  ``adapi.main`` does the
mapping, which is one place and is covered by ``tests/test_api.py``.

**The annotation router still returns dicts** for ``/judge`` and ``/quality``.
That is deliberate rather than missed: its consumer is the inline JavaScript in
``annotate.html``, which is untyped, so a model there would buy documentation and
not a drift check.  The routes the Next app reads are the ones modelled.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .delivery import DeliveryReport
from .enums import AspectRatio, JobState, Platform, ProviderMode, SafeArea
from .job import SpendEntry


class SafeAreaBox(BaseModel):
    """Fractions of each edge that platform chrome may cover.

    A model rather than the four-tuple :class:`SafeArea` because a tuple crosses
    the wire as a positional array, and ``safe_area[2]`` on the client is exactly
    the kind of reader that survives a reordering without complaining.
    """

    top: float = Field(ge=0.0, le=1.0)
    bottom: float = Field(ge=0.0, le=1.0)
    left: float = Field(ge=0.0, le=1.0)
    right: float = Field(ge=0.0, le=1.0)

    @classmethod
    def of(cls, area: SafeArea) -> SafeAreaBox:
        return cls(top=area.top, bottom=area.bottom, left=area.left, right=area.right)

    def describe(self) -> str:
        """The non-zero edges as prose, e.g. ``14% top, 20% bottom, 14% right``.

        Only the non-zero ones: ``0% left`` is a true statement that makes the
        three that matter harder to read.
        """
        parts = [
            f"{value:.0%} {name}"
            for name, value in (
                ("top", self.top),
                ("bottom", self.bottom),
                ("left", self.left),
                ("right", self.right),
            )
            if value > 0
        ]
        return ", ".join(parts) if parts else "none"


class PlatformProfile(BaseModel):
    """Geometry the UI derives from the chosen platform rather than asking for."""

    aspect_ratio: AspectRatio
    safe_area: SafeAreaBox

    @classmethod
    def of(cls, platform: Platform) -> PlatformProfile:
        return cls(
            aspect_ratio=platform.aspect_ratio,
            safe_area=SafeAreaBox.of(platform.safe_area),
        )


class AppConfig(BaseModel):
    """What the UI needs to render honestly: the mode, and what it costs.

    ``provider_mode`` is the load-bearing field.  Mock, replay and live produce
    output that looks identical on screen, so a page that does not state the mode
    cannot be told apart from one that spent money.
    """

    provider_mode: ProviderMode
    is_live: bool = Field(description="True only when real provider calls are armed.")
    is_replay: bool = Field(description="True when output comes from a frozen bundle.")
    image_provider: str
    video_provider: str
    golden_set: str | None = Field(description="Slug replayed in replay mode, else null.")
    banner: str = Field(
        description="The settings banner verbatim. It names the providers actually "
        "wired up, which is the difference between 'live mode' and 'live mode, but "
        "the video provider fell back to a mock'."
    )
    platforms: dict[Platform, PlatformProfile] = Field(
        default_factory=dict,
        description="Every platform's derived geometry, so the wizard reads it rather "
        "than hardcoding a ratio that then disagrees with the renderer.",
    )


class GoldenSummary(BaseModel):
    """One frozen demo bundle, as the job board lists it.

    ``replayable`` is the field to read before offering the button: a bundle whose
    media has gone missing or whose hashes no longer match is listed with its
    ``problems`` rather than hidden, because a demo set that quietly shrinks is
    worse than one that says what broke.
    """

    slug: str
    title: str
    created_at: str = Field(description="ISO-8601, as recorded in the bundle manifest.")
    summary: str
    frames: int = Field(ge=0)
    clips: int = Field(ge=0)
    image_model: str
    video_model: str
    original_cost_usd: float = Field(
        ge=0.0, description="What the bundle cost when it was frozen. A replay costs zero."
    )
    replayable: bool
    problems: list[str] = Field(default_factory=list)
    winner_slot: int | None = Field(
        default=None,
        description="Candidate slot the frozen run ranked first, when the bundle "
        "records an expectation to check the replay against.",
    )


class JobSummary(BaseModel):
    """A row on the job board, which is deliberately not the whole record.

    The board lists jobs; a record carries every brief, score breakdown and event.
    Fetching all of that to render a table would make the list page the slowest in
    the app for information it does not show.
    """

    job_id: str
    state: JobState
    product_name: str
    platform: Platform
    created_at: str = Field(description="ISO-8601 of the request, not of the result.")
    cost_usd: float = Field(ge=0.0)
    winner: int | None = Field(
        default=None, description="Source image index of the top-ranked video, if ranked."
    )


class Launched(BaseModel):
    """Acknowledgement of a job that has been queued but has not run yet.

    Returned by all three launch routes rather than one shape each: the client's
    next move is identical in every case — open the event stream for this id — and
    three near-identical types would invite reading the wrong one.
    """

    job_id: str
    state: JobState
    golden_set: str | None = Field(
        default=None, description="Set when this job is a replay of a frozen bundle."
    )
    replaying: str | None = Field(
        default=None, description="Summary of the bundle being replayed, if any."
    )


class LedgerResponse(BaseModel):
    """Every charge attributed to one job.

    Entries are typed rather than passed through as dicts.  A ledger the client
    cannot read field by field is a ledger nobody looks at, and this is the record
    that answers "did that job spend anything?".
    """

    job_id: str
    total_usd: float = Field(ge=0.0)
    entries: list[SpendEntry] = Field(default_factory=list)


class DeliveryResponse(BaseModel):
    """Stage 8's output on its own, without re-fetching the whole record.

    Separate route because a client polling for the finished renders should not
    have to pull every candidate's brief and score breakdown to find out whether
    the zip exists yet.
    """

    job_id: str
    summary: str
    delivery: DeliveryReport
    download_url: str | None = Field(
        default=None, description="Null until the bundle has been written."
    )


__all__ = [
    "AppConfig",
    "DeliveryResponse",
    "GoldenSummary",
    "JobSummary",
    "Launched",
    "LedgerResponse",
    "PlatformProfile",
    "SafeAreaBox",
]
