"""Write the web app's TypeScript contract from the Pydantic schema.

    # regenerate
    .venv/bin/python scripts/export_types.py

    # fail if the committed file no longer matches the models
    .venv/bin/python scripts/export_types.py --check

The check is the point.  Generating types is a convenience; noticing that the
committed types have stopped describing the API is the thing that prevents a
renamed field from surfacing as an empty panel during a demo.  Run it in the same
breath as `ruff check`.

`--diff` prints what moved, because "the contract drifted" is not actionable and
"`RankedCandidate.image_stage_rank` is gone" is.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "schema"))

from adschema import typescript  # noqa: E402

DEFAULT_OUT = ROOT / "apps" / "web" / "lib" / "contract.ts"
COMMAND = "scripts/export_types.py"


def _diff(committed: str, generated: str, path: Path) -> list[str]:
    return list(
        difflib.unified_diff(
            committed.splitlines(keepends=True),
            generated.splitlines(keepends=True),
            fromfile=f"{path} (committed)",
            tofile="generated from adschema",
            n=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the contract")
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit non-zero if the committed file has drifted",
    )
    parser.add_argument("--diff", action="store_true", help="print the drift")
    args = parser.parse_args()

    out = Path(args.out)
    generated = typescript.generate()
    committed = out.read_text(encoding="utf-8") if out.exists() else ""

    if args.check:
        if not out.exists():
            print(f"{out} does not exist — run {COMMAND}")
            return 1
        if committed == generated:
            lines = generated.count("\n")
            print(f"{out.relative_to(ROOT)} matches adschema ({lines} lines)")
            return 0
        print(f"{out.relative_to(ROOT)} has drifted from adschema — run {COMMAND}")
        if args.diff:
            sys.stdout.writelines(_diff(committed, generated, out))
        return 1

    if committed == generated:
        print(f"{out.relative_to(ROOT)} already current")
        return 0
    if args.diff and committed:
        sys.stdout.writelines(_diff(committed, generated, out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generated, encoding="utf-8")
    verb = "updated" if committed else "wrote"
    print(f"{verb} {out.relative_to(ROOT)} ({generated.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
