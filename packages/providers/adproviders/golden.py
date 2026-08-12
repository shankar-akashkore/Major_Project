"""The golden demo set: jobs frozen to disk so the demo never generates live.

The budget rule this exists to serve is rule 6: *freeze a golden demo set by week
14 and demo from fixtures; never generate live during the viva.*  The reasoning is
not only financial.  A live demo of this system depends on two hosted APIs being
up, a network being usable in an exam room, and $2.22 of budget still being
unspent — and the video stage is not reproducible even when all three hold, because
Kling's image-to-video endpoint has no seed parameter.  A job that produced good
candidates once cannot be asked to produce them again.  So the only way to be able
to show a specific result is to have kept it.

**What is frozen, and what is recomputed.**  This is the load-bearing decision, and
it is not the obvious one.  The obvious freeze is the whole :class:`JobRecord`:
replay would then mean deserialising it and rendering the UI.  That would be a
video recording of a demo rather than a demo — the gate, the ranker and the whole
delivery stage would never run, and a regression in any of them would first be
noticed in front of an examiner.

So only the parts that cannot be recomputed are frozen:

===================  ==========================================================
Frozen               Why it cannot be recomputed
===================  ==========================================================
Upload bytes         The user's photographs. Not derivable from anything.
LLM brief text       A sampled completion; a re-run rewords it.
Generated frames     Provider-side sampling; seeds narrow it, nothing fixes it.
Generated clips      Kling has no seed at all, so nothing narrows it.
===================  ==========================================================

Everything else — intake, the product cutout, the palette extraction, the quality
gate, the image-stage ranking, video decoding and measurement, the final ranking,
the reframes, the previews, the bundle — runs for real against the frozen bytes,
every replay.  A golden bundle is therefore a regression test as well as a demo: it
carries the ranking the frozen job produced, and :func:`compare_replay` says whether
today's code still produces it.

**Drift is reported rather than prevented.**  A frame is served on its candidate
slot, and the design point and prompt that produced it are recorded alongside.  If
the sampler or the prompt template has changed since the freeze, the frame is still
served — a demo that refuses to run because a lighting enum was renamed is worse
than one that runs and says so — but the mismatch is collected as a drift note, and
:mod:`scripts.replay_golden` exits non-zero on it.  Ordering changes and numeric
score changes are reported separately, because they mean different things: a
changed score is a code change to explain, a changed *order* is a changed result.

**A replay cannot spend.**  The replay providers estimate every call at zero, so
they satisfy the governor's non-live assertion that nothing in a free run may cost
anything.  The original cost is not thrown away — it is recorded in the bundle's
provenance, because "these three clips cost $2.10 to make" is part of what the
bundle is evidence of.

**A miss is an error, not a substitute**, for the same reason it is in
:mod:`adproviders.prerendered`: quietly serving a mock frame inside a run labelled
as a premium golden demo would put a rounded rectangle on screen where a paid
generation belongs.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adschema import AdJobRequest, AssetRef, JobRecord, Tier

from .base import (
    ImageGenRequest,
    ImageGenResult,
    ImageProvider,
    LLMProvider,
    ProviderError,
    ProviderUnavailable,
    VideoGenRequest,
    VideoGenResult,
    VideoProvider,
)
from .storage import Storage

#: Bundle schema version. An older bundle fails with a version message rather
#: than a KeyError halfway through a replay.
GOLDEN_VERSION = 1
#: The committed part of a bundle.
MANIFEST_NAME = "golden.json"
#: Storage-key prefix every bundle lives under, relative to the storage root.
GOLDEN_PREFIX = "golden"
#: Subdirectory holding the frozen bytes. Deliberately *not* committed — see
#: docs/golden-protocol.md and the .gitignore rule.
ASSET_SUBDIR = "assets"

#: Two scores are the same number if they agree to this. Float arithmetic over the
#: same numpy on the same machine is bit-identical, so this is not a fudge factor
#: for noise — it is the boundary above which a difference means the code changed.
SCORE_TOLERANCE = 1e-6

#: Request field per upload role, so a replay can rebind the uploads to a new job.
_UPLOAD_FIELDS = {
    "human": "human_model_image",
    "product": "product_image",
    "logo": "logo_image",
}


class GoldenError(RuntimeError):
    """A bundle is malformed, incomplete, or from an incompatible version."""


class GoldenMiss(ProviderError):
    """The bundle holds no recording for a call the pipeline made."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --- Bundle parts -----------------------------------------------------------


@dataclass(frozen=True)
class FrozenAsset:
    """One recorded byte-blob, addressed by content.

    ``sha256`` is what makes a bundle checkable: assets are not committed to git,
    so the manifest is the only record of what the media *was*, and a truncated
    download or a half-extracted zip has to be caught by name rather than showing
    up as a strange demo.
    """

    key: str
    sha256: str
    size_bytes: int
    mime_type: str
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "key": self.key,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mime_type": self.mime_type,
        }
        if self.width is not None:
            out["width"] = self.width
        if self.height is not None:
            out["height"] = self.height
        return out

    @classmethod
    def from_dict(cls, raw: dict) -> FrozenAsset:
        missing = {"key", "sha256", "size_bytes"} - set(raw)
        if missing:
            raise GoldenError(f"frozen asset is missing {sorted(missing)}: {raw}")
        return cls(
            key=str(raw["key"]),
            sha256=str(raw["sha256"]),
            size_bytes=int(raw["size_bytes"]),
            mime_type=str(raw.get("mime_type", "application/octet-stream")),
            width=raw.get("width"),
            height=raw.get("height"),
        )

    def as_ref(self, url: str | None = None) -> AssetRef:
        return AssetRef(
            key=self.key,
            url=url,
            mime_type=self.mime_type,
            sha256=self.sha256,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True)
class FrozenFrame:
    """One recorded image generation, on its candidate slot.

    ``design_point`` and ``prompt_sha256`` are the drift detectors.  The frame is
    served on ``slot``/``attempt`` because those are reproducible from a seeded
    sampler; the other two say whether the thing the sampler now asks for is still
    the thing this frame was made for.
    """

    slot: int
    attempt: int
    asset: FrozenAsset
    model: str
    tier: str
    seed: int
    design_point: dict = field(default_factory=dict)
    prompt_sha256: str = ""
    latency_ms: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "attempt": self.attempt,
            "asset": self.asset.to_dict(),
            "model": self.model,
            "tier": self.tier,
            "seed": self.seed,
            "design_point": self.design_point,
            "prompt_sha256": self.prompt_sha256,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> FrozenFrame:
        try:
            return cls(
                slot=int(raw["slot"]),
                attempt=int(raw.get("attempt", 1)),
                asset=FrozenAsset.from_dict(raw["asset"]),
                model=str(raw["model"]),
                tier=str(raw.get("tier", Tier.MOCK.value)),
                seed=int(raw.get("seed", 0)),
                design_point=dict(raw.get("design_point") or {}),
                prompt_sha256=str(raw.get("prompt_sha256", "")),
                latency_ms=int(raw.get("latency_ms", 0)),
                cost_usd=float(raw.get("cost_usd", 0.0)),
            )
        except KeyError as exc:
            raise GoldenError(f"frozen frame is missing {exc}: {raw}") from exc


@dataclass(frozen=True)
class FrozenClip:
    """One recorded video generation, on the image slot it animated.

    ``start_sha256`` is the frame this clip was made from.  It is checked on
    serve: a clip animating a frame that has since changed is the one drift that
    invalidates the pair rather than merely relabelling it.
    """

    slot: int
    asset: FrozenAsset
    model: str
    tier: str
    duration_seconds: float
    fps: int = 24
    seed: int = 0
    seed_honoured: bool = True
    was_chained: bool = False
    seam_consistency: float | None = None
    start_sha256: str = ""
    motion: str = ""
    latency_ms: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "asset": self.asset.to_dict(),
            "model": self.model,
            "tier": self.tier,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "seed": self.seed,
            "seed_honoured": self.seed_honoured,
            "was_chained": self.was_chained,
            "seam_consistency": self.seam_consistency,
            "start_sha256": self.start_sha256,
            "motion": self.motion,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> FrozenClip:
        try:
            return cls(
                slot=int(raw["slot"]),
                asset=FrozenAsset.from_dict(raw["asset"]),
                model=str(raw["model"]),
                tier=str(raw.get("tier", Tier.MOCK.value)),
                duration_seconds=float(raw["duration_seconds"]),
                fps=int(raw.get("fps", 24)),
                seed=int(raw.get("seed", 0)),
                seed_honoured=bool(raw.get("seed_honoured", True)),
                was_chained=bool(raw.get("was_chained", False)),
                seam_consistency=raw.get("seam_consistency"),
                start_sha256=str(raw.get("start_sha256", "")),
                motion=str(raw.get("motion", "")),
                latency_ms=int(raw.get("latency_ms", 0)),
                cost_usd=float(raw.get("cost_usd", 0.0)),
            )
        except KeyError as exc:
            raise GoldenError(f"frozen clip is missing {exc}: {raw}") from exc


@dataclass(frozen=True)
class GoldenExpectation:
    """The outcome the frozen job produced, to compare a replay against.

    ``original_cost_usd`` is what the freeze cost, and it is kept separate from
    everything else here on purpose: a replay's own cost must be exactly zero, so
    the two numbers are checked against different things rather than one being
    quietly reinterpreted as the other.
    """

    state: str
    image_order: list[int]
    video_order: list[int]
    winner_slot: int | None
    image_scores: dict[int, float]
    video_scores: dict[int, float]
    gate_verdicts: dict[int, str]
    scored_by: str
    delivery_summary: str
    original_cost_usd: float

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "image_order": self.image_order,
            "video_order": self.video_order,
            "winner_slot": self.winner_slot,
            "image_scores": {str(k): v for k, v in self.image_scores.items()},
            "video_scores": {str(k): v for k, v in self.video_scores.items()},
            "gate_verdicts": {str(k): v for k, v in self.gate_verdicts.items()},
            "scored_by": self.scored_by,
            "delivery_summary": self.delivery_summary,
            "original_cost_usd": self.original_cost_usd,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> GoldenExpectation:
        return cls(
            state=str(raw.get("state", "")),
            image_order=[int(i) for i in raw.get("image_order", [])],
            video_order=[int(i) for i in raw.get("video_order", [])],
            winner_slot=(None if raw.get("winner_slot") is None else int(raw["winner_slot"])),
            image_scores={int(k): float(v) for k, v in (raw.get("image_scores") or {}).items()},
            video_scores={int(k): float(v) for k, v in (raw.get("video_scores") or {}).items()},
            gate_verdicts={int(k): str(v) for k, v in (raw.get("gate_verdicts") or {}).items()},
            scored_by=str(raw.get("scored_by", "")),
            delivery_summary=str(raw.get("delivery_summary", "")),
            original_cost_usd=float(raw.get("original_cost_usd", 0.0)),
        )

    @classmethod
    def from_record(cls, record: JobRecord) -> GoldenExpectation:
        result = record.result
        if result is None:
            return cls(
                state=record.state.value,
                image_order=[],
                video_order=[],
                winner_slot=None,
                image_scores={},
                video_scores={},
                gate_verdicts={},
                scored_by="",
                delivery_summary="",
                original_cost_usd=0.0,
            )
        scored_by = ""
        for candidate in result.images:
            if candidate.score is not None:
                scored_by = candidate.score.model_version
                break
        return cls(
            state=record.state.value,
            image_order=list(result.image_stage_order),
            video_order=list(result.video_stage_order),
            winner_slot=(
                result.winner.video.source_image_index if result.winner is not None else None
            ),
            image_scores={c.index: c.score.overall for c in result.images if c.score is not None},
            video_scores={
                v.source_image_index: v.score.overall for v in result.videos if v.score is not None
            },
            gate_verdicts={
                c.index: c.gate.verdict.value for c in result.images if c.gate is not None
            },
            scored_by=scored_by,
            delivery_summary=(result.delivery.summary() if result.delivery else ""),
            original_cost_usd=round(result.total_cost_usd, 6),
        )


# --- The bundle -------------------------------------------------------------


@dataclass
class GoldenBundle:
    """One frozen job: its request, its recorded generations, and its outcome."""

    slug: str
    request: dict
    title: str = ""
    created_at: str = ""
    uploads: dict[str, FrozenAsset] = field(default_factory=dict)
    frames: list[FrozenFrame] = field(default_factory=list)
    clips: list[FrozenClip] = field(default_factory=list)
    completions: dict[str, dict] = field(default_factory=dict)
    expectation: GoldenExpectation | None = None
    provenance: dict = field(default_factory=dict)

    # --- Lookups ---

    def frame(self, slot: int, attempt: int) -> FrozenFrame | None:
        for frozen in self.frames:
            if frozen.slot == slot and frozen.attempt == attempt:
                return frozen
        return None

    def clip(self, slot: int) -> FrozenClip | None:
        for frozen in self.clips:
            if frozen.slot == slot:
                return frozen
        return None

    def assets(self) -> list[FrozenAsset]:
        return [
            *self.uploads.values(),
            *(f.asset for f in self.frames),
            *(c.asset for c in self.clips),
        ]

    @property
    def image_model(self) -> str:
        recorded = self.frames[0].model if self.frames else "mock"
        return str(self.provenance.get("image_model") or recorded)

    @property
    def video_model(self) -> str:
        recorded = self.clips[0].model if self.clips else "mock"
        return str(self.provenance.get("video_model") or recorded)

    @property
    def llm_model(self) -> str:
        return str(self.provenance.get("llm_model") or "mock")

    def summary(self) -> str:
        cost = float(self.provenance.get("cost_usd", 0.0))
        size_mb = sum(a.size_bytes for a in self.assets()) / 1e6
        return (
            f"{self.slug}: {len(self.frames)} frame(s), {len(self.clips)} clip(s), "
            f"{size_mb:.1f} MB, {self.image_model} + {self.video_model}, "
            f"cost ${cost:.4f} when frozen"
        )

    # --- Persistence ---
    #
    # The manifest goes through the filesystem and the assets go through
    # ``Storage``, and that split is the point rather than an inconsistency: the
    # manifest is source and is committed, the assets are media and are not.

    def to_dict(self) -> dict:
        return {
            "version": GOLDEN_VERSION,
            "slug": self.slug,
            "title": self.title,
            "created_at": self.created_at,
            "provenance": self.provenance,
            "request": self.request,
            "uploads": {role: asset.to_dict() for role, asset in self.uploads.items()},
            "frames": [f.to_dict() for f in self.frames],
            "clips": [c.to_dict() for c in self.clips],
            "completions": self.completions,
            "expectation": (self.expectation.to_dict() if self.expectation else None),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> GoldenBundle:
        version = raw.get("version")
        if version != GOLDEN_VERSION:
            raise GoldenError(
                f"golden bundle is version {version!r}, this code reads version "
                f"{GOLDEN_VERSION}. Re-freeze it with scripts/freeze_golden.py."
            )
        if not raw.get("request"):
            raise GoldenError("golden bundle has no request; it cannot be replayed")
        expectation = raw.get("expectation")
        return cls(
            slug=str(raw.get("slug", "")),
            request=dict(raw["request"]),
            title=str(raw.get("title", "")),
            created_at=str(raw.get("created_at", "")),
            uploads={
                role: FrozenAsset.from_dict(item)
                for role, item in (raw.get("uploads") or {}).items()
            },
            frames=[FrozenFrame.from_dict(item) for item in raw.get("frames", [])],
            clips=[FrozenClip.from_dict(item) for item in raw.get("clips", [])],
            completions=dict(raw.get("completions") or {}),
            expectation=(GoldenExpectation.from_dict(expectation) if expectation else None),
            provenance=dict(raw.get("provenance") or {}),
        )

    def manifest_path(self, root: Path | str) -> Path:
        return Path(root) / GOLDEN_PREFIX / self.slug / MANIFEST_NAME

    def save(self, root: Path | str) -> Path:
        path = self.manifest_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> GoldenBundle:
        path = Path(path)
        if path.is_dir():
            path = path / MANIFEST_NAME
        if not path.is_file():
            raise ProviderUnavailable(
                f"no golden bundle manifest at {path}. Freeze one with "
                "`scripts/freeze_golden.py --slug <name>`, or list what exists with "
                "`scripts/replay_golden.py --list`."
            )
        try:
            raw = json.loads(path.read_text())
        except ValueError as exc:
            raise GoldenError(f"golden manifest at {path} is not valid JSON: {exc}") from exc
        bundle = cls.from_dict(raw)
        if not bundle.slug:
            bundle.slug = path.parent.name
        return bundle

    # --- Integrity ---

    def verify(self, storage: Storage) -> list[str]:
        """Problems with this bundle's media.  Empty means it will replay.

        Checked by content hash rather than by existence: the failure mode worth
        catching is a partially-extracted archive or a truncated transfer, and
        both leave a file that is present and wrong.
        """
        problems: list[str] = []
        for asset in self.assets():
            try:
                data = storage.get_bytes(asset.key)
            except (FileNotFoundError, ValueError):
                problems.append(f"missing: {asset.key}")
                continue
            if len(data) != asset.size_bytes:
                problems.append(
                    f"wrong size: {asset.key} is {len(data)} bytes, "
                    f"manifest says {asset.size_bytes}"
                )
            elif _sha256(data) != asset.sha256:
                problems.append(f"content changed: {asset.key} does not match its recorded hash")
        if not self.frames:
            problems.append("no frames recorded; there is nothing to replay")
        return problems

    # --- Replay input ---

    def materialise_request(self, job_id: str, storage: Storage) -> AdJobRequest:
        """Rebind the frozen request onto a new job id, with its uploads copied in.

        A fresh job id rather than the frozen one, because reusing it makes the
        second replay collide with the first in the job store — the same trap the
        demo endpoint hit when it derived ids from seeds.

        The uploads are copied into the new job's own upload keys so intake writes
        its derivatives beside them exactly as it would for a real submission.
        """
        raw = deepcopy(self.request)
        raw["job_id"] = job_id
        raw["created_at"] = datetime.now(UTC).isoformat()
        for role, frozen in self.uploads.items():
            field_name = _UPLOAD_FIELDS.get(role)
            if field_name is None:
                raise GoldenError(f"unknown upload role {role!r} in bundle {self.slug!r}")
            data = storage.get_bytes(frozen.key)
            suffix = Path(frozen.key).suffix or ".png"
            asset = storage.put_bytes(f"uploads/{job_id}/{role}{suffix}", data, frozen.mime_type)
            asset.width, asset.height = frozen.width, frozen.height
            raw[field_name] = asset.model_dump(mode="json")
        try:
            return AdJobRequest.model_validate(raw)
        except ValueError as exc:
            raise GoldenError(
                f"the frozen request in bundle {self.slug!r} no longer validates against "
                f"the current schema: {exc}"
            ) from exc

    # --- Transfer ---

    def pack(self, dest: Path | str, storage: Storage) -> Path:
        """Write the whole bundle — manifest and media — as one zip.

        This is the artefact that goes on a USB stick.  The media is not in git, so
        a laptop reinstall the week before the viva would otherwise take the demo
        with it.
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{self.slug}/{MANIFEST_NAME}", json.dumps(self.to_dict(), indent=2))
            for asset in self.assets():
                zf.writestr(asset.key, storage.get_bytes(asset.key))
        return dest


def build_golden_bundle(
    slug: str,
    *,
    request: AdJobRequest,
    record: JobRecord,
    recorder: GoldenRecorder,
    provider_mode: str,
    image_model: str,
    video_model: str,
    llm_model: str,
    title: str = "",
) -> GoldenBundle:
    """Assemble a bundle from a finished run.

    ``request`` must be the request **as submitted**, not the one the pipeline
    returned.  Stage 1 writes the extracted palette back into ``theme.palette`` and
    sets ``palette_auto_extracted``, so freezing the mutated request would hand a
    replay a palette it never had to extract — the intake path the bundle is
    supposed to exercise would be skipped, and quietly.
    """
    # Called here rather than left to the caller: a freeze that silently omitted
    # every cached generation is exactly the failure this guards, and a step a
    # caller has to remember is a step a caller will forget.
    backfilled = recorder.backfill(record, image_model=image_model, video_model=video_model)
    uploads = recorder.freeze_uploads(request)
    raw = request.model_dump(mode="json")
    for role, frozen in uploads.items():
        raw[_UPLOAD_FIELDS[role]] = frozen.as_ref().model_dump(mode="json")

    result = record.result
    return GoldenBundle(
        slug=slug,
        title=title or request.product_name,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        request=raw,
        uploads=uploads,
        frames=list(recorder.frames),
        clips=list(recorder.clips),
        completions=dict(recorder.completions),
        expectation=GoldenExpectation.from_record(record),
        provenance={
            "provider_mode": provider_mode,
            # The provider names, not the recorded per-generation ones. These are
            # what the price table is asked about on replay — `estimate_job_cost`
            # looks all three up before the first stage runs — so they have to be
            # real model keys rather than descriptions of the recording.
            "image_model": image_model,
            "video_model": video_model,
            "llm_model": llm_model,
            "cost_usd": round(result.total_cost_usd, 6) if result else 0.0,
            "frozen_by": "scripts/freeze_golden.py",
            # Which generations were served from the governor's cache rather than
            # called at freeze time. Recorded because it changes what the bundle is
            # evidence of: a cached frame was produced by an earlier run, and
            # `cost_usd` above is that run's cost, not this one's.
            "backfilled": backfilled,
        },
    )


def list_bundles(root: Path | str) -> list[GoldenBundle]:
    """Every readable bundle under the storage root, newest first."""
    base = Path(root) / GOLDEN_PREFIX
    if not base.is_dir():
        return []
    bundles = []
    for manifest in sorted(base.glob(f"*/{MANIFEST_NAME}")):
        bundles.append(GoldenBundle.load(manifest))
    return sorted(bundles, key=lambda b: b.created_at, reverse=True)


# --- Recording (freeze side) ------------------------------------------------


class GoldenRecorder:
    """Collects what the real providers returned, on the way past.

    Wrapping rather than re-reading the job record afterwards: the record keeps
    asset *references*, and by the time a job finishes its generation keys may
    already have been overwritten by a retry or cleaned up. The bytes are copied
    into the bundle at the moment they are produced.
    """

    def __init__(self, slug: str, storage: Storage):
        self.slug = slug
        self.storage = storage
        self.frames: list[FrozenFrame] = []
        self.clips: list[FrozenClip] = []
        self.completions: dict[str, dict] = {}
        self._attempts: dict[int, int] = {}

    def asset_key(self, name: str, suffix: str) -> str:
        return f"{GOLDEN_PREFIX}/{self.slug}/{ASSET_SUBDIR}/{name}{suffix}"

    def freeze_asset(self, name: str, source: AssetRef) -> FrozenAsset:
        """Copy an asset's bytes into the bundle and describe them."""
        data = self.storage.get_bytes(source.key)
        suffix = Path(source.key).suffix or ".bin"
        key = self.asset_key(name, suffix)
        stored = self.storage.put_bytes(key, data, source.mime_type)
        return FrozenAsset(
            key=stored.key,
            sha256=_sha256(data),
            size_bytes=len(data),
            mime_type=source.mime_type,
            width=source.width,
            height=source.height,
        )

    def record_image(self, request: ImageGenRequest, result: ImageGenResult) -> None:
        slot = request.brief.index
        attempt = self._attempts.get(slot, 0) + 1
        self._attempts[slot] = attempt
        name = f"img_{slot}" if attempt == 1 else f"img_{slot}_r{attempt}"
        self.frames.append(
            FrozenFrame(
                slot=slot,
                attempt=attempt,
                asset=self.freeze_asset(name, result.asset),
                model=result.model,
                tier=result.tier.value,
                seed=result.seed,
                design_point=request.brief.design_point.model_dump(mode="json"),
                prompt_sha256=_digest_of(request.brief.image_prompt),
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
            )
        )

    def record_video(self, request: VideoGenRequest, result: VideoGenResult) -> None:
        slot = request.brief.index
        self.clips.append(
            FrozenClip(
                slot=slot,
                asset=self.freeze_asset(f"vid_{slot}", result.asset),
                model=result.model,
                tier=result.tier.value,
                duration_seconds=result.duration_seconds,
                fps=result.fps,
                seed=result.seed,
                seed_honoured=result.seed_honoured,
                was_chained=result.was_chained,
                seam_consistency=result.seam_consistency,
                start_sha256=request.start_image.sha256 or "",
                motion=request.brief.design_point.motion.value,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
            )
        )

    def record_completion(self, system: str, user: str, response: dict) -> None:
        # Empty responses are not recorded. The mock LLM returns ``{}`` to signal
        # "no LLM here", and storing that would make a mock freeze look like it had
        # a frozen completion that happened to be empty.
        if response:
            self.completions[completion_key(system, user)] = response

    def backfill(self, record: JobRecord, *, image_model: str, video_model: str) -> list[str]:
        """Freeze generations no provider was asked for, and say which they were.

        This exists because of a boundary that is easy to miss: the cost governor's
        content-addressed cache is checked *before* the provider is called, so a job
        whose generations are already cached completes without any provider being
        invoked — and a wrapper around the provider records nothing.  The first
        freeze written here produced a bundle with three ranked candidates, a
        delivery report, and zero frames.

        Nothing about the cache is wrong; serving a repeat request for free is the
        point of it, and in a live freeze it is the difference between $0 and $2.22.
        So the bundle is completed from the job record instead, whose asset
        references point at the same bytes.

        One thing is lost: the record keeps only the candidate that was promoted per
        slot, so a slot that needed a gate retry is frozen as a single attempt.  That
        is the right shape for a replay — the frozen frame is the one that passed, so
        the gate will pass it first time — but a bundle built this way cannot
        demonstrate the retry path.
        """
        notes: list[str] = []
        result = record.result
        if result is None:
            return notes

        for candidate in result.images:
            if any(f.slot == candidate.index for f in self.frames):
                continue
            self.frames.append(
                FrozenFrame(
                    slot=candidate.index,
                    attempt=1,
                    asset=self.freeze_asset(f"img_{candidate.index}", candidate.asset),
                    model=image_model,
                    tier=candidate.tier.value,
                    seed=candidate.seed,
                    design_point=candidate.brief.design_point.model_dump(mode="json"),
                    prompt_sha256=_digest_of(candidate.brief.image_prompt),
                    latency_ms=candidate.latency_ms,
                    cost_usd=candidate.cost_usd,
                )
            )
            notes.append(f"frame on slot {candidate.index} came from the generation cache")

        starts = {c.index: (c.asset.sha256 or "") for c in result.images}
        for video in result.videos:
            slot = video.source_image_index
            if any(c.slot == slot for c in self.clips):
                continue
            self.clips.append(
                FrozenClip(
                    slot=slot,
                    asset=self.freeze_asset(f"vid_{slot}", video.asset),
                    model=video_model,
                    tier=video.tier.value,
                    duration_seconds=video.duration_seconds,
                    fps=video.fps,
                    seed=video.seed,
                    seed_honoured=video.seed_honoured,
                    was_chained=video.was_chained,
                    seam_consistency=video.seam_consistency,
                    start_sha256=starts.get(slot, ""),
                    motion=video.brief.design_point.motion.value,
                    latency_ms=video.latency_ms,
                    cost_usd=video.cost_usd,
                )
            )
            notes.append(f"clip on slot {slot} came from the generation cache")
        # Deterministic order, so two freezes of the same job produce the same
        # manifest byte-for-byte and a diff of two bundles is readable.
        self.frames.sort(key=lambda f: (f.slot, f.attempt))
        self.clips.sort(key=lambda c: c.slot)
        return notes

    def freeze_uploads(self, request: AdJobRequest) -> dict[str, FrozenAsset]:
        uploads: dict[str, FrozenAsset] = {}
        for role, field_name in _UPLOAD_FIELDS.items():
            asset = getattr(request, field_name, None)
            if asset is not None:
                uploads[role] = self.freeze_asset(role, asset)
        return uploads


def completion_key(system: str, user: str) -> str:
    """Cache key for one LLM call.

    Built from both halves of the prompt because the system prompt carries the
    role instruction and the user prompt carries the request; a change to either
    changes what the completion means.
    """
    return _digest_of(system + "\x00" + user)


class RecordingImageProvider(ImageProvider):
    """Delegates to a real provider and keeps a copy of what came back."""

    name = "recording"

    def __init__(self, inner: ImageProvider, recorder: GoldenRecorder):
        self.inner = inner
        self.recorder = recorder

    @property
    def model(self) -> str:  # type: ignore[override]
        return self.inner.model

    def estimate_cost(self, count: int = 1) -> float:
        return self.inner.estimate_cost(count)

    async def generate(self, request: ImageGenRequest) -> ImageGenResult:
        result = await self.inner.generate(request)
        self.recorder.record_image(request, result)
        return result


class RecordingVideoProvider(VideoProvider):
    name = "recording"

    def __init__(self, inner: VideoProvider, recorder: GoldenRecorder):
        self.inner = inner
        self.recorder = recorder

    @property
    def model(self) -> str:  # type: ignore[override]
        return self.inner.model

    def estimate_cost(self, seconds: float, count: int = 1) -> float:
        return self.inner.estimate_cost(seconds, count)

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        result = await self.inner.generate(request)
        self.recorder.record_video(request, result)
        return result


class RecordingLLMProvider(LLMProvider):
    name = "recording"

    def __init__(self, inner: LLMProvider, recorder: GoldenRecorder):
        self.inner = inner
        self.recorder = recorder

    @property
    def model(self) -> str:  # type: ignore[override]
        return self.inner.model

    async def complete_json(self, system: str, user: str, schema_hint: str) -> dict:
        response = await self.inner.complete_json(system, user, schema_hint)
        self.recorder.record_completion(system, user, response)
        return response


# --- Replay ----------------------------------------------------------------


class GoldenSession:
    """One replay of one bundle: the providers, and what serving them noticed.

    The three providers share this object so drift notes land in one place the
    caller can read after the job finishes — the pipeline drops a provider's
    ``raw`` payload, so there is nowhere else for them to go.
    """

    def __init__(self, bundle: GoldenBundle, storage: Storage):
        self.bundle = bundle
        self.storage = storage
        self.notes: list[str] = []
        self.served_frames = 0
        self.served_clips = 0
        self._attempts: dict[int, int] = {}

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def next_attempt(self, slot: int) -> int:
        attempt = self._attempts.get(slot, 0) + 1
        self._attempts[slot] = attempt
        return attempt

    def read(self, asset: FrozenAsset) -> bytes:
        try:
            data = self.storage.get_bytes(asset.key)
        except (FileNotFoundError, ValueError) as exc:
            raise GoldenMiss(
                f"bundle {self.bundle.slug!r} lists {asset.key} but it is not in storage. "
                "The bundle's media is not committed to git — unpack its zip into the "
                "storage root, or re-freeze it."
            ) from exc
        if _sha256(data) != asset.sha256:
            raise GoldenMiss(
                f"{asset.key} does not match the hash recorded in bundle "
                f"{self.bundle.slug!r}; refusing to replay altered media"
            )
        return data

    def audit(self) -> list[str]:
        """Whether this replay actually read the bundle.

        A guard against the failure that is invisible in the output: if something
        upstream of the provider satisfies the pipeline — the governor's generation
        cache is the one that does it — the replay completes, produces the right
        ranking, and never opens a single frozen file.  Nothing in the result would
        say so, which is why it is checked here rather than inferred.
        """
        problems: list[str] = []
        if self.bundle.frames and self.served_frames == 0:
            problems.append(
                "no frozen frame was served: the pipeline obtained its generations "
                "elsewhere (the governor's cache, unless it was disabled), so this "
                "replay did not exercise the bundle"
            )
        if self.bundle.clips and self.served_clips == 0:
            problems.append(
                "no frozen clip was served: as above, the video stage did not come from the bundle"
            )
        return problems

    def image_provider(self) -> GoldenImageProvider:
        return GoldenImageProvider(self)

    def video_provider(self) -> GoldenVideoProvider:
        return GoldenVideoProvider(self)

    def llm_provider(self) -> GoldenLLMProvider:
        return GoldenLLMProvider(self)


class GoldenImageProvider(ImageProvider):
    """Serves frozen frames on candidate slots.

    ``model`` is the model that produced them, not a placeholder, so the price
    table still answers questions about the frozen run correctly and the job
    record says which endpoint the pixels came from.  Only :meth:`estimate_cost`
    is overridden — the generation happened, and was charged, once.
    """

    name = "golden"

    def __init__(self, session: GoldenSession):
        self.session = session

    @property
    def model(self) -> str:  # type: ignore[override]
        return self.session.bundle.image_model

    def estimate_cost(self, count: int = 1) -> float:
        return 0.0

    async def generate(self, request: ImageGenRequest) -> ImageGenResult:
        bundle = self.session.bundle
        slot = request.brief.index
        attempt = self.session.next_attempt(slot)
        frozen = bundle.frame(slot, attempt)
        if frozen is None:
            held = sorted({(f.slot, f.attempt) for f in bundle.frames})
            raise GoldenMiss(
                f"bundle {bundle.slug!r} has no frame for slot {slot} attempt {attempt}; "
                f"it holds {held}. The quality gate is passing or failing differently than "
                "it did at freeze time, so a different number of attempts is being asked "
                "for. Refusing to substitute a generated frame."
            )

        now = request.brief.design_point.model_dump(mode="json")
        if frozen.design_point and now != frozen.design_point:
            changed = sorted(
                k
                for k in set(now) | set(frozen.design_point)
                if now.get(k) != frozen.design_point.get(k)
            )
            self.session.note(
                f"slot {slot}: the sampler now asks for a different design point "
                f"({', '.join(changed)}); the frozen frame was made for the old one"
            )
        prompt_now = _digest_of(request.brief.image_prompt)
        if frozen.prompt_sha256 and prompt_now != frozen.prompt_sha256:
            self.session.note(
                f"slot {slot}: the image prompt has changed since the freeze "
                "(template edit or a different LLM scene), so the frozen frame no "
                "longer corresponds to the prompt shown alongside it"
            )

        data = self.session.read(frozen.asset)
        asset = self.session.storage.put_bytes(request.output_key, data, frozen.asset.mime_type)
        asset.width, asset.height = frozen.asset.width, frozen.asset.height
        self.session.served_frames += 1
        return ImageGenResult(
            asset=asset,
            model=frozen.model,
            tier=Tier(frozen.tier),
            # Zero, because this run did not pay for it. What the freeze paid is in
            # the bundle's provenance, where it describes the artefact rather than
            # this execution of it.
            cost_usd=0.0,
            latency_ms=frozen.latency_ms,
            seed=frozen.seed,
            cache_hit=True,
            raw={"golden": bundle.slug, "source_key": frozen.asset.key},
        )


class GoldenVideoProvider(VideoProvider):
    """Serves frozen clips on the image slot each one animated."""

    name = "golden"

    def __init__(self, session: GoldenSession):
        self.session = session

    @property
    def model(self) -> str:  # type: ignore[override]
        return self.session.bundle.video_model

    def estimate_cost(self, seconds: float, count: int = 1) -> float:
        return 0.0

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        bundle = self.session.bundle
        slot = request.brief.index
        frozen = bundle.clip(slot)
        if frozen is None:
            raise GoldenMiss(
                f"bundle {bundle.slug!r} has no clip for slot {slot}; it holds "
                f"{sorted(c.slot for c in bundle.clips)}. At freeze time this candidate "
                "did not reach the video stage — most likely it failed the gate then and "
                "passes now. Refusing to substitute a mock clip."
            )

        # The one drift that invalidates rather than relabels: a clip animating a
        # frame that is no longer the frame it started from is not evidence about
        # that candidate any more.
        start = request.start_image.sha256 or ""
        if frozen.start_sha256 and start and start != frozen.start_sha256:
            self.session.note(
                f"slot {slot}: this clip was animated from a different start frame than "
                "the one the image stage just produced — the image/video pair for this "
                "candidate is no longer the pair that was frozen"
            )

        data = self.session.read(frozen.asset)
        asset = self.session.storage.put_bytes(request.output_key, data, frozen.asset.mime_type)
        self.session.served_clips += 1
        return VideoGenResult(
            asset=asset,
            model=frozen.model,
            tier=Tier(frozen.tier),
            cost_usd=0.0,
            latency_ms=frozen.latency_ms,
            seed=frozen.seed,
            duration_seconds=frozen.duration_seconds,
            fps=frozen.fps,
            was_chained=frozen.was_chained,
            seam_consistency=frozen.seam_consistency,
            seed_honoured=frozen.seed_honoured,
            cache_hit=True,
            raw={"golden": bundle.slug, "source_key": frozen.asset.key},
        )


class GoldenLLMProvider(LLMProvider):
    """Replays the brief enrichment, so the frozen prompts are reproducible.

    A miss returns ``{}`` rather than raising.  That is not leniency: the brief
    compiler already degrades to its deterministic template on any LLM failure, so
    raising here would produce the same briefs by a noisier route.  What matters is
    that the miss is *recorded*, because the template's scene text differs from the
    frozen one and the image prompt therefore will not match its frame.
    """

    name = "golden"

    def __init__(self, session: GoldenSession):
        self.session = session

    @property
    def model(self) -> str:  # type: ignore[override]
        return self.session.bundle.llm_model

    async def complete_json(self, system: str, user: str, schema_hint: str) -> dict:
        bundle = self.session.bundle
        if not bundle.completions:
            # Frozen without an LLM at all (the mock returns nothing), so the
            # template path is the frozen path. Nothing to report.
            return {}
        key = completion_key(system, user)
        response = bundle.completions.get(key)
        if response is None:
            self.session.note(
                "the brief compiler is sending a different prompt than it did at freeze "
                "time, so the frozen scene text could not be replayed and the briefs fell "
                "back to the template"
            )
            return {}
        return dict(response)


# --- Drift -----------------------------------------------------------------


@dataclass(frozen=True)
class GoldenDrift:
    """What a replay did differently from the job that was frozen."""

    slug: str
    ordering: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)
    serving: list[str] = field(default_factory=list)
    spend: list[str] = field(default_factory=list)

    @property
    def lines(self) -> list[str]:
        return [*self.spend, *self.ordering, *self.serving, *self.numeric]

    @property
    def ok(self) -> bool:
        return not self.lines

    def summary(self) -> str:
        if self.ok:
            return f"{self.slug}: replayed clean — same ranking, same scores, no spend"
        parts = []
        if self.spend:
            parts.append(f"{len(self.spend)} spend")
        if self.ordering:
            parts.append(f"{len(self.ordering)} ordering")
        if self.serving:
            parts.append(f"{len(self.serving)} serving")
        if self.numeric:
            parts.append(f"{len(self.numeric)} numeric")
        return f"{self.slug}: drift — " + ", ".join(parts)


def compare_replay(
    bundle: GoldenBundle,
    record: JobRecord,
    *,
    spent_usd: float = 0.0,
    notes: list[str] | None = None,
) -> GoldenDrift:
    """Diff a replayed job against the bundle's frozen expectation.

    Four categories, kept apart because they mean different things.  *Spend* is a
    bug in this module.  *Ordering* is a changed result — the thing the demo is
    supposed to show.  *Serving* is the frozen media no longer matching what asked
    for it.  *Numeric* is a scoring change that did not (yet) reorder anything,
    which is a code change to explain rather than a broken demo.
    """
    expected = bundle.expectation
    ordering: list[str] = []
    numeric: list[str] = []
    spend: list[str] = []
    serving = list(notes or [])

    if abs(spent_usd) > 1e-9:
        spend.append(f"the replay spent ${spent_usd:.4f}; a replay must cost exactly nothing")

    if expected is None:
        serving.append("bundle carries no frozen expectation, so nothing was compared")
        return GoldenDrift(bundle.slug, ordering, numeric, serving, spend)

    result = record.result
    if record.state.value != expected.state:
        ordering.append(
            f"state: replay ended {record.state.value}, frozen job ended {expected.state}"
            + (f" — {record.error}" if record.error else "")
        )
    if result is None:
        return GoldenDrift(bundle.slug, ordering, numeric, serving, spend)

    verdicts = {c.index: c.gate.verdict.value for c in result.images if c.gate is not None}
    for slot in sorted(set(verdicts) | set(expected.gate_verdicts)):
        was, now = expected.gate_verdicts.get(slot), verdicts.get(slot)
        if was != now:
            ordering.append(f"gate on slot {slot}: was {was}, now {now}")

    if list(result.image_stage_order) != expected.image_order:
        ordering.append(
            f"image-stage order: was {expected.image_order}, now {list(result.image_stage_order)}"
        )
    if list(result.video_stage_order) != expected.video_order:
        ordering.append(
            f"video-stage order: was {expected.video_order}, now {list(result.video_stage_order)}"
        )
    winner = result.winner.video.source_image_index if result.winner else None
    if winner != expected.winner_slot:
        ordering.append(f"winner: was slot {expected.winner_slot}, now slot {winner}")

    scored_by = next((c.score.model_version for c in result.images if c.score), "")
    if expected.scored_by and scored_by != expected.scored_by:
        numeric.append(f"scored by {scored_by!r}, frozen job was scored by {expected.scored_by!r}")

    for label, now_scores, was_scores in (
        (
            "image",
            {c.index: c.score.overall for c in result.images if c.score is not None},
            expected.image_scores,
        ),
        (
            "video",
            {v.source_image_index: v.score.overall for v in result.videos if v.score is not None},
            expected.video_scores,
        ),
    ):
        for slot in sorted(set(now_scores) | set(was_scores)):
            was, now = was_scores.get(slot), now_scores.get(slot)
            if was is None or now is None:
                numeric.append(f"{label} score on slot {slot}: was {was}, now {now}")
            elif abs(was - now) > SCORE_TOLERANCE:
                numeric.append(
                    f"{label} score on slot {slot}: {was:.6f} → {now:.6f} ({now - was:+.6f})"
                )

    now_summary = result.delivery.summary() if result.delivery else ""
    if expected.delivery_summary and now_summary != expected.delivery_summary:
        numeric.append(f"delivery: was {expected.delivery_summary!r}, now {now_summary!r}")

    return GoldenDrift(bundle.slug, ordering, numeric, serving, spend)
