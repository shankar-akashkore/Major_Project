"""Export the identity references ``notebooks/colab_embeddings.ipynb`` needs.

    .venv/bin/python scripts/export_references.py
    .venv/bin/python scripts/export_references.py --job 7dec4335b3994214

Collects the human-model photo and the product reference (the rembg cutout when
intake kept one, matching exactly what the generator and the gate were actually
given — see ``adschema.IntakeReport.product_reference``) from real job records,
deduplicates them by content, and writes ``references.zip`` plus
``references.json`` in the shape the notebook's identity-reference upload cell
expects.

Runs on the laptop, no GPU, no torch: this only copies bytes and hashes keys. The
encoders that turn these into vectors are the ArcFace and DINOv2 cells the
notebook now has — this script is what gives them something to embed.

**Deduplicated by ``sha256 or key``, not by job.** The same uploaded photo is
reused across many jobs, and ``adworker.gate._identity_key`` looks vectors up by
that same value — embedding it once is both cheaper and the only way a candidate
from job B finds the vector a reference from job A already produced.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("schema", "providers", "ml"):
    sys.path.insert(0, str(ROOT / "packages" / package))
sys.path.insert(0, str(ROOT / "services" / "api"))

import adproviders as P  # noqa: E402
from adapi.store import JobStore  # noqa: E402
from adschema import AssetRef  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "fixtures" / "references"


def _identity_key(asset: AssetRef) -> str:
    """Matches ``adworker.gate._identity_key`` exactly — the two must agree, or a
    vector written here is one the gate can never find."""
    return asset.sha256 or asset.key


def _product_reference(request, result) -> AssetRef:
    """Mirrors ``adworker.pipeline._product_reference``: the cutout when intake
    kept a trustworthy one, the original upload otherwise. Duplicated rather than
    imported because that function is private to the pipeline module and this
    script has no business depending on worker internals to read two fields."""
    if result is not None and result.intake is not None:
        return result.intake.product_reference(request.product_image)
    return request.product_image


async def _run(args: argparse.Namespace) -> int:
    settings = P.Settings()
    storage = P.get_storage(settings.storage_backend, settings.storage_root)
    jobs = JobStore.from_url(settings.database_url)

    if args.job:
        record = await jobs.get(args.job)
        records = [record] if record else []
        if not records:
            print(f"no job found with id {args.job}")
            return 1
    else:
        records, _total = await jobs.page(limit=args.limit, offset=0)

    if not records:
        print("no jobs found — generate at least one before exporting references")
        return 1

    # identity_key -> (role, key, bytes). A dict, so a photo reused across many
    # jobs is read and zipped exactly once.
    collected: dict[str, tuple[str, str, bytes]] = {}
    skipped = 0
    for record in records:
        request = record.request
        for role, asset in (
            ("human", request.human_model_image),
            ("product", _product_reference(request, record.result)),
        ):
            if asset is None:
                continue
            key = _identity_key(asset)
            if key in collected:
                continue
            try:
                blob = storage.get_bytes(asset.key)
            except (KeyError, OSError):
                skipped += 1
                continue
            collected[key] = (role, asset.key, blob)

    if not collected:
        print("no reference images could be read from any job")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "references.zip"
    manifest_path = out_dir / "references.json"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for _role, key, blob in collected.values():
            archive.writestr(key, blob)

    manifest = [
        {"identity_key": identity_key, "key": key, "role": role}
        for identity_key, (role, key, _blob) in sorted(collected.items())
    ]
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=1)

    by_role = {"human": 0, "product": 0}
    for role, _key, _blob in collected.values():
        by_role[role] += 1
    print(
        f"wrote    : {zip_path.relative_to(ROOT)}, {manifest_path.relative_to(ROOT)}  "
        f"({by_role['human']} human, {by_role['product']} product, from {len(records)} job(s))"
    )
    if skipped:
        print(f"skipped  : {skipped} reference(s) whose bytes could not be read")
    print("\nNext, in notebooks/colab_embeddings.ipynb:")
    print("  section '2b. Upload identity references' — upload both files above")
    print("  the ArcFace and (extended) DINOv2 sections then embed them")
    print("  automatically, keyed by identity_key — the same key")
    print("  adworker.gate._identity_key computes from a candidate's own asset,")
    print("  so evaluate_image finds them without any further wiring.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default=None, help="export references from one job only")
    parser.add_argument("--limit", type=int, default=500, help="jobs to scan when --job is omitted")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
