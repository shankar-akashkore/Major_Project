"""Media storage.

Media never lives on the laptop's filesystem in a real run — the dev machine has
under 10 GB free, and generated video would fill it in an afternoon.  Everything
moves as a storage key plus a short-lived signed URL.

``LocalStorage`` exists so the mock pipeline runs with zero external setup; the
Supabase backend swaps in behind the same interface.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

from adschema import AssetRef


@runtime_checkable
class Storage(Protocol):
    """Minimal object-storage surface the pipeline depends on."""

    def put_bytes(self, key: str, data: bytes, mime_type: str = "image/png") -> AssetRef: ...

    def get_bytes(self, key: str) -> bytes: ...

    def url_for(self, key: str) -> str: ...

    def exists(self, key: str) -> bool: ...


class LocalStorage:
    """Filesystem-backed storage under a root directory.

    Keys are relative POSIX paths (``generations/<job>/img_0.png``).  Traversal
    outside the root is refused — keys can originate from request data, so this
    is a real boundary rather than a formality.
    """

    def __init__(self, root: Path | str = "./fixtures", url_prefix: str = "/media"):
        self.root = Path(root).resolve()
        self.url_prefix = url_prefix.rstrip("/")
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"storage key {key!r} escapes the storage root")
        return candidate

    def put_bytes(self, key: str, data: bytes, mime_type: str = "image/png") -> AssetRef:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return AssetRef(
            key=key,
            url=self.url_for(key),
            mime_type=mime_type,
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def get_bytes(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def url_for(self, key: str) -> str:
        return f"{self.url_prefix}/{key}"

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()


def get_storage(backend: str = "local", root: Path | str = "./fixtures") -> Storage:
    if backend == "local":
        return LocalStorage(root)
    if backend == "supabase":  # pragma: no cover - lands with the deployment step
        raise NotImplementedError(
            "Supabase storage backend not wired yet; keep AD_STORAGE_BACKEND=local"
        )
    raise ValueError(f"unknown storage backend {backend!r}")
