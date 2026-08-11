"""The research tier: clips generated free on a Colab GPU, served from a manifest.

This is the $0 line in the budget, and it is what makes a video-stage evaluation
possible at all.  The premium tier costs $0.70 for a 10 s clip, so the $20 premium
allocation buys about 28 videos — enough for golden demos and nowhere near enough
to validate a ranker.  Open weights (LTX-Video, Wan 2.1) on Colab's free GPU cost
nothing per clip, which is where the volume comes from.

**The design mirrors** :mod:`adml.embeddings`: torch stays on the far side of a
file boundary.  A notebook generates clips from corpus stills and writes them
alongside a manifest; this provider reads the manifest and serves the bytes.
Nothing here imports torch, needs a GPU, or touches the network, so the same
pipeline code paths run on the laptop.

Two refusals are load-bearing.

**A miss is an error, not a substitute.**  Asked for a clip it does not have, this
provider raises.  It does not fall back to the mock renderer.  A silent fallback
would put synthetic placeholder frames into a run labelled "research tier", and
the tier comparison in the write-up would be comparing Kling against a rounded
rectangle.

**Fingerprints are content-derived.**  A clip is matched on the start frame's
hash and the motion prompt, not on a filename.  Renaming a file cannot silently
re-point a candidate at a different clip, and a start frame that has changed since
generation will miss rather than serve a stale animation of the old frame.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from adschema import Tier

from .base import (
    ProviderError,
    ProviderUnavailable,
    VideoGenRequest,
    VideoGenResult,
    VideoProvider,
)
from .storage import Storage

#: Manifest filename, written by the Colab notebook next to the clips.
MANIFEST_NAME = "clips.json"
#: Manifest schema version. Bumped if the entry shape changes, so an old manifest
#: fails with a version message rather than a KeyError.
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ClipEntry:
    """One pre-rendered clip.

    ``model`` is recorded per clip rather than per manifest: a Colab session that
    runs out of VRAM on LTX-Video and finishes on Wan produces a mixed corpus, and
    the report needs to know which clip came from which generator.
    """

    fingerprint: str
    key: str
    model: str
    duration_seconds: float
    fps: int
    was_chained: bool = False
    #: Seed the notebook used, and whether the generator honoured it. Open-weights
    #: pipelines usually do, which is the research tier's one advantage over Kling
    #: for ablation work.
    seed: int = 0
    seed_honoured: bool = True
    note: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> ClipEntry:
        missing = {"fingerprint", "key", "model", "duration_seconds"} - set(raw)
        if missing:
            raise ProviderError(f"clip entry is missing {sorted(missing)}: {raw}")
        return cls(
            fingerprint=str(raw["fingerprint"]),
            key=str(raw["key"]),
            model=str(raw["model"]),
            duration_seconds=float(raw["duration_seconds"]),
            fps=int(raw.get("fps", 24)),
            was_chained=bool(raw.get("was_chained", False)),
            seed=int(raw.get("seed", 0)),
            seed_honoured=bool(raw.get("seed_honoured", True)),
            note=str(raw.get("note", "")),
        )


def clip_fingerprint(start_image_digest: str, motion_intent: str, duration_seconds: float) -> str:
    """Content address for a clip request.

    Three inputs, and the choice of the second one is the load-bearing decision.

    **The motion *intent*, not the motion prompt.**  The obvious key would be the
    prompt text the video model actually receives, since that is what determines
    the animation.  It cannot be: :func:`adworker.briefs.compile_briefs` expands
    briefs through an LLM, so the prompt text is not reproducible between runs, and
    a fingerprint built on it would miss for every clip the moment the compiler
    produced a differently-worded sentence for the same design point.
    :class:`adschema.MotionIntent` is a short stable enum that both sides have —
    the notebook reads it from the corpus manifest, the pipeline reads it off
    ``brief.design_point.motion``.

    The aspect ratio and the seed are deliberately absent.  The notebook derives
    the aspect ratio from the start frame it was handed, and open-weights seeds are
    not comparable across generators, so neither would agree between the two sides.
    """
    payload = json.dumps(
        {
            "start": start_image_digest,
            "motion": str(motion_intent).strip(),
            # Rounded: a request for 9.0 s and one for 9.0000001 s are the same
            # request, and float formatting should not split them into two.
            "duration": round(float(duration_seconds), 2),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass
class ClipManifest:
    """The notebook's output index."""

    entries: dict[str, ClipEntry]
    generated_at: str = ""
    notebook: str = ""

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def models(self) -> dict[str, int]:
        """Clip count per generator, for the tier comparison in the write-up."""
        counts: dict[str, int] = {}
        for entry in self.entries.values():
            counts[entry.model] = counts.get(entry.model, 0) + 1
        return counts

    @classmethod
    def load(cls, path: Path | str) -> ClipManifest:
        path = Path(path)
        if path.is_dir():
            path = path / MANIFEST_NAME
        if not path.is_file():
            raise ProviderUnavailable(
                f"no research-tier clip manifest at {path}. Generate one by running "
                "notebooks/colab_video.ipynb on a Colab GPU, then unzip its output "
                "into the storage root. Until then keep AD_VIDEO_PROVIDER=mock."
            )
        try:
            raw = json.loads(path.read_text())
        except ValueError as exc:
            raise ProviderError(f"clip manifest at {path} is not valid JSON: {exc}") from exc

        version = raw.get("version")
        if version != MANIFEST_VERSION:
            raise ProviderError(
                f"clip manifest at {path} is version {version!r}, this code reads "
                f"version {MANIFEST_VERSION}. Re-run the notebook to regenerate it."
            )
        entries = [ClipEntry.from_dict(item) for item in raw.get("clips", [])]
        return cls(
            entries={entry.fingerprint: entry for entry in entries},
            generated_at=str(raw.get("generated_at", "")),
            notebook=str(raw.get("notebook", "")),
        )

    def summary(self) -> str:
        models = ", ".join(f"{name} x{count}" for name, count in sorted(self.models.items()))
        when = f" generated {self.generated_at}" if self.generated_at else ""
        return f"{len(self.entries)} clips ({models or 'none'}){when}"


class PrerenderedVideoProvider(VideoProvider):
    """Serves clips a Colab notebook already generated.  Always free.

    ``model`` is a placeholder here and the real generator is read back per clip
    from the manifest, because one manifest can legitimately mix LTX-Video and Wan
    output.  :attr:`VideoProvider.tier` and the price table therefore describe the
    tier rather than a single endpoint.
    """

    name = "prerendered"
    model = "ltx-video"

    def __init__(
        self,
        storage: Storage,
        manifest_path: Path | str | None = None,
        *,
        manifest: ClipManifest | None = None,
    ):
        self.storage = storage
        self._manifest = manifest
        self._manifest_path = manifest_path

    @property
    def manifest(self) -> ClipManifest:
        if self._manifest is None:
            if self._manifest_path is None:
                raise ProviderUnavailable(
                    "PrerenderedVideoProvider needs a manifest path. Set "
                    "AD_CLIP_MANIFEST to the directory the notebook's output was "
                    "unzipped into."
                )
            self._manifest = ClipManifest.load(self._manifest_path)
        return self._manifest

    def fingerprint_for(self, request: VideoGenRequest) -> str:
        digest = request.start_image.sha256
        if not digest:
            # Hash the bytes rather than fall back to the key: the key is
            # job-scoped, so two jobs animating an identical frame would look like
            # different requests and the second would miss.
            digest = hashlib.sha256(self.storage.get_bytes(request.start_image.key)).hexdigest()
        return clip_fingerprint(
            digest,
            request.brief.design_point.motion.value,
            request.duration_seconds,
        )

    async def generate(self, request: VideoGenRequest) -> VideoGenResult:
        started = time.perf_counter()
        fingerprint = self.fingerprint_for(request)
        entry = self.manifest.entries.get(fingerprint)
        if entry is None:
            raise ProviderError(
                f"no pre-rendered clip for fingerprint {fingerprint} "
                f"(start frame {request.start_image.key}, {request.duration_seconds:.1f}s). "
                f"The manifest holds {len(self.manifest)} clips. Either this frame was not "
                "in the batch the notebook animated, or it has been regenerated since — "
                "re-run the notebook over the current corpus. Refusing to substitute a "
                "mock clip, which would put placeholder frames in a research-tier run."
            )

        if not self.storage.exists(entry.key):
            raise ProviderError(
                f"the manifest lists {entry.key} for fingerprint {fingerprint}, but that "
                "asset is not in storage. The notebook's output zip was probably not "
                "fully extracted into the storage root."
            )

        # Copied to the job's own key so a research-tier run leaves the same asset
        # layout as a premium one, and so the corpus clip is not mutated by
        # anything downstream that writes back to a candidate's key.
        data = self.storage.get_bytes(entry.key)
        asset = self.storage.put_bytes(request.output_key, data, "video/mp4")

        return VideoGenResult(
            asset=asset,
            model=entry.model,
            tier=Tier.RESEARCH,
            cost_usd=0.0,
            latency_ms=round((time.perf_counter() - started) * 1000),
            seed=entry.seed,
            duration_seconds=entry.duration_seconds,
            fps=entry.fps,
            was_chained=entry.was_chained,
            seed_honoured=entry.seed_honoured,
            # A cache hit in every meaningful sense: the generation already
            # happened, off this machine, and cost nothing to serve.
            cache_hit=True,
            raw={"fingerprint": fingerprint, "source_key": entry.key, "note": entry.note},
        )
