"""Compute the numpy features for every corpus item, and cache them.

    .venv/bin/python scripts/extract_features.py
    .venv/bin/python scripts/extract_features.py --out fixtures/corpus/features.json

Runs on the laptop, no GPU, no torch.  Saliency, palette and sharpness are all
FFT and box filters, so 300 images take a couple of minutes and cost nothing.

The cache exists because these measurements are deterministic but not free, and
the training script is run dozens of times while the features change never.  It is
keyed by item id and merged on write, so re-running after generating more sets only
measures the new items.

**The manifest is the Colab handoff.**  ``--manifest`` writes the item ids and
their storage keys so ``notebooks/colab_embeddings.ipynb`` knows what to encode and
under what id to write it back.  Without it the npz keys would have to be guessed
from filenames, and a mismatch there would silently drop items from the training
set rather than raise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("schema", "providers", "ml"):
    sys.path.insert(0, str(ROOT / "packages" / package))
sys.path.insert(0, str(ROOT / "services" / "api"))

import adproviders as P  # noqa: E402
from adapi.annotation_store import AnnotationStore  # noqa: E402
from adml import featureset as FS  # noqa: E402
from adschema import ItemKind  # noqa: E402

DEFAULT_OUT = ROOT / "fixtures" / "corpus" / "features.json"
DEFAULT_MANIFEST = ROOT / "fixtures" / "corpus" / "manifest.json"


def _load_cache(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    with path.open() as handle:
        data = json.load(handle)
    return data.get("features", {})


async def _run(args: argparse.Namespace) -> int:
    settings = P.Settings()
    storage = P.get_storage(settings.storage_backend, settings.storage_root)
    store = AnnotationStore.from_url(settings.database_url)
    await store.create_all()

    kind = ItemKind(args.kind)
    # Decoys are included on purpose: a decoy is never a training item, but the
    # catch-trial report benefits from knowing how far its measurements moved.
    items = await store.list_items(kind=kind, include_decoys=True)
    if not items:
        print("No corpus items. Generate some first:")
        print("  .venv/bin/python scripts/generate_corpus.py --sets 12")
        print("  .venv/bin/python scripts/build_corpus.py --commit")
        await store.engine.dispose()
        return 1

    out = Path(args.out)
    cache = {} if args.rebuild else _load_cache(out)
    todo = [it for it in items if it.item_id not in cache]
    print(f"corpus   : {len(items)} {kind.value} items ({len(items) - len(todo)} already cached)")

    started = time.monotonic()
    failed: list[tuple[str, str]] = []
    for index, item in enumerate(todo, start=1):
        try:
            data = storage.get_bytes(item.asset.key)
            if kind is ItemKind.IMAGE:
                cache[item.item_id] = FS.image_features(item, data)
            else:
                cache[item.item_id] = FS.video_features(item, data)
        except Exception as exc:  # noqa: BLE001 - one bad asset must not lose the rest
            # Recorded and skipped rather than raised: the item is simply absent
            # from the feature table, which `build_table` already treats as
            # missing data rather than filling with a plausible zero.
            failed.append((item.item_id, f"{type(exc).__name__}: {exc}"))
        if index % 25 == 0 or index == len(todo):
            rate = index / max(time.monotonic() - started, 1e-6)
            print(f"  {index:>4}/{len(todo)}  {rate:.1f} items/s")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        json.dump(
            {
                "kind": kind.value,
                "n_items": len(cache),
                "feature_names": sorted({k for v in cache.values() for k in v}),
                "features": cache,
            },
            handle,
            indent=1,
            sort_keys=True,
        )
    print(f"\nwrote    : {out.relative_to(ROOT)}  ({len(cache)} items)")

    if failed:
        print(f"FAILED   : {len(failed)} items could not be measured")
        for item_id, reason in failed[:5]:
            print(f"  {item_id}  {reason}")

    if args.manifest:
        manifest = Path(args.manifest)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w") as handle:
            json.dump(
                {
                    "kind": kind.value,
                    "storage_root": str(settings.storage_root),
                    "items": [
                        {
                            "item_id": it.item_id,
                            "key": it.asset.key,
                            "set_id": it.set_id,
                            "is_decoy": it.is_decoy,
                            # For notebooks/colab_video.ipynb. `motion` is the design
                            # axis, not the prompt text: the clip fingerprint is built
                            # on the intent enum because LLM-expanded prompt text is
                            # not reproducible between runs and would miss on every
                            # lookup. See adproviders.prerendered.clip_fingerprint.
                            "motion": it.motion.value if it.motion else None,
                            "seed": it.seed,
                        }
                        for it in items
                    ],
                },
                handle,
                indent=1,
            )
        print(f"manifest : {manifest.relative_to(ROOT)}  ({len(items)} items)")
        print("\nNext, for the embedding features (needs a GPU):")
        print("  1. zip the corpus images:")
        print(f"       cd {settings.storage_root} && zip -qr /tmp/corpus.zip corpus/")
        print("  2. open notebooks/colab_embeddings.ipynb in Colab, upload the zip")
        print("     and the manifest, run it, download the npz files")
        print("  3. drop them in fixtures/corpus/embeddings/ and train:")
        print("       .venv/bin/python scripts/train_predictor.py")
        print("\nThe same zip and manifest drive the free video corpus:")
        print("     notebooks/colab_video.ipynb  ->  fixtures/research/  ($0 on Colab)")

    budget_note = FS.parameter_budget(1200)
    print(
        f"\nnote     : ~1,200 training comparisons supports ~{budget_note} free parameters; "
        f"{len(cache)} items x {len(next(iter(cache.values()), {}))} measured features"
    )
    await store.engine.dispose()
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--manifest",
        nargs="?",
        const=str(DEFAULT_MANIFEST),
        default=str(DEFAULT_MANIFEST),
        help="write the Colab handoff manifest (default on)",
    )
    parser.add_argument("--kind", default=ItemKind.IMAGE.value, choices=[k.value for k in ItemKind])
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="ignore the cache and re-measure everything",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
