"""Tests for the generated TypeScript contract.

The properties worth asserting are the ones that decide whether a UI author can
trust the file:

* **The committed file still describes the models.** This is the drift test. Every
  other property here would hold on a file that stopped matching the API weeks ago.
* **A defaulted field is not optional.** Pydantic emits it, so `?:` would send
  every reader off writing existence checks against a value that is always there.
* **A property is named, not emitted.** ``JobRecord.job_id`` is a Python property,
  so the JSON has no ``job_id`` at the top level — a UI that reads one gets
  ``undefined``, and the comment is what stops that being discovered live.
* **Output is byte-stable.** A generator whose output reorders between runs makes
  ``--check`` cry drift on every commit, and a check that always fails is a check
  nobody reads.
"""

from __future__ import annotations

import enum
import re
from pathlib import Path

import pytest
from adschema import (
    DEFAULT_CANDIDATE_COUNT,
    MAX_DURATION_S,
    MIN_DURATION_S,
    AspectRatio,
    JobRecord,
    Platform,
    RankedCandidate,
    Vertical,
)
from adschema import typescript as TS
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "apps" / "web" / "lib" / "contract.ts"


@pytest.fixture(scope="module")
def contract() -> str:
    return TS.generate()


# --- The drift test -------------------------------------------------------


def test_the_committed_contract_still_describes_the_models(contract: str) -> None:
    """The one test here that can fail for a reason outside this file.

    When it does, the fix is `scripts/export_types.py`, and the diff is worth
    reading: a field that vanished from the Python is a field the UI is still
    rendering.
    """
    assert COMMITTED.exists(), f"{COMMITTED} is missing — run scripts/export_types.py"
    assert COMMITTED.read_text(encoding="utf-8") == contract, (
        "apps/web/lib/contract.ts has drifted from adschema. "
        "Run: .venv/bin/python scripts/export_types.py --diff"
    )


def test_generating_twice_gives_the_same_bytes() -> None:
    assert TS.generate() == TS.generate()


# --- Field emission ------------------------------------------------------


def test_no_field_is_optional(contract: str) -> None:
    """No declaration uses `?:`, because model_dump emits defaulted fields.

    Checked over declarations rather than the whole file: the header explains the
    decision in prose and so contains the very token being ruled out.
    """
    declarations = _declarations(contract)
    # Guard the guard: `optional == []` also holds when the helper matches nothing.
    assert len(declarations) > 100, "the declaration matcher found almost nothing"
    assert [line for line in declarations if "?:" in line] == []


def test_a_nullable_field_is_nullable_with_null_last(contract: str) -> None:
    assert "logo_image: AssetRef | null;" in contract
    assert "| null" in contract
    assert not re.search(r":\s*null \| ", contract), "null should sort last in a union"


def test_a_datetime_becomes_a_named_alias_not_a_bare_string(contract: str) -> None:
    """`Timestamp` tells a reader to parse it; `string` invites concatenation."""
    assert "export type Timestamp = string;" in contract
    assert "created_at: Timestamp;" in contract


def test_a_list_of_models_becomes_an_array(contract: str) -> None:
    assert "images: ImageCandidate[];" in contract
    assert "ranking: RankedCandidate[];" in contract


def test_a_dict_keyed_by_string_becomes_a_record(contract: str) -> None:
    assert "platform_renders: Record<string, AssetRef>;" in contract
    assert "previews: Record<string, AssetRef>;" in contract


def test_a_number_stays_a_number_whether_int_or_float(contract: str) -> None:
    assert "duration_seconds: number;" in contract
    assert "candidate_count: number;" in contract


# --- Properties ----------------------------------------------------------


def test_job_id_is_flagged_as_derived_rather_than_emitted(contract: str) -> None:
    """The specific trap: `JobRecord.job_id` is a property, so the JSON has none.

    A client reading `record.job_id` gets undefined and has to reach through to
    `record.request.job_id`. Nothing in the JSON says so, which is why the comment
    does.
    """
    assert "job_id" in TS._properties(JobRecord)
    block = _interface(contract, "JobRecord")
    assert "derived (not on the wire): job_id" in block
    assert not re.search(r"^  job_id:", block, re.MULTILINE)


def test_the_headline_derivation_is_flagged(contract: str) -> None:
    """`rank_shift` is the project's headline made visible, and it is a property."""
    assert "rank_shift" in TS._properties(RankedCandidate)
    assert "derived (not on the wire): rank_shift" in contract
    # ...but the two ranks it is computed from are genuinely on the wire.
    block = _interface(contract, "RankedCandidate")
    assert "rank: number;" in block
    assert "image_stage_rank: number | null;" in block


def test_a_model_without_properties_gets_no_derived_comment(contract: str) -> None:
    block = _interface(contract, "AudienceSpec")
    assert "derived" not in block


# --- Enumerations --------------------------------------------------------


@pytest.mark.parametrize("member", [AspectRatio, Platform, Vertical])
def test_every_enum_member_reaches_the_values_array(contract: str, member: type[enum.Enum]) -> None:
    name = TS._screaming_snake(member.__name__)
    block = _const(contract, f"{name}_VALUES")
    entries = [
        line.strip().rstrip(",").strip('"') for line in block.splitlines() if _is_entry(line)
    ]
    assert entries == [item.value for item in member]


def test_the_values_array_name_breaks_at_word_boundaries() -> None:
    assert TS._screaming_snake("AspectRatio") == "ASPECT_RATIO"
    assert TS._screaming_snake("MotionIntent") == "MOTION_INTENT"
    assert TS._screaming_snake("Platform") == "PLATFORM"


def test_a_short_enum_stays_on_one_line_and_a_long_one_wraps(contract: str) -> None:
    assert 'export type AspectRatio = "9:16" | "4:5" | "1:1" | "16:9";' in contract
    assert "export type Vertical =\n  |" in contract


# --- Constants -----------------------------------------------------------


def test_the_duration_window_crosses_as_numbers(contract: str) -> None:
    """The form has to enforce 8-10 s, and a hardcoded 8 in a component is a
    second source of truth that will not move when this one does."""
    assert f"export const MIN_DURATION_S = {int(MIN_DURATION_S)};" in contract
    assert f"export const MAX_DURATION_S = {int(MAX_DURATION_S)};" in contract
    assert f"export const DEFAULT_CANDIDATE_COUNT = {DEFAULT_CANDIDATE_COUNT};" in contract


def test_an_integral_float_does_not_print_a_decimal_point() -> None:
    assert TS._ts_number(8.0) == "8"
    assert TS._ts_number(9.5) == "9.5"
    assert TS._ts_number(3) == "3"


# --- Documentation -------------------------------------------------------


def test_a_load_bearing_caveat_reaches_the_tooltip(contract: str) -> None:
    """`is_stub` is the field that stops a placeholder being read as a prediction.

    It is only useful if the person rendering the number sees it, so it has to
    survive into the JSDoc rather than staying in the Python.
    """
    block = _interface(contract, "ScoreBreakdown")
    assert "is_stub: boolean;" in block
    assert "trained head" in block


def test_constraints_are_stated_for_whoever_builds_the_input(contract: str) -> None:
    """A form that lets someone ask for a 4-second video has moved a constraint
    the schema already knows about into a 422 the user has to read."""
    block = _interface(contract, "AdJobRequest")
    # Appended in parentheses where the field also has a description.
    assert "range 8.0–10.0" in block
    assert "range 1–6" in block, "the candidate and video counts are bounded"
    assert "at least 0" in block

    # Standing alone, sentence-cased, where the field has no description. This
    # used to be asserted on `candidate_count`, which has since gained one — so it
    # is checked where the form still actually occurs rather than dropped, since
    # what is under test is the generator's two renderings, not this one field.
    assert "/** Range 0.0–1.0. */" in _interface(contract, "ScoreBreakdown")


def test_docstring_markup_does_not_leak_into_jsdoc(contract: str) -> None:
    """Sphinx roles are correct in Python and noise in an editor tooltip."""
    assert ":class:" not in contract
    assert "``" not in contract
    # ...and the referenced name survives, shortened, as a Markdown code span.
    assert "`Platform`" in contract


def test_only_the_summary_line_of_a_docstring_is_carried(contract: str) -> None:
    """Twenty lines of design rationale belong in the Python file, not a tooltip."""
    block = _interface(contract, "AdJobRequest")
    assert "Everything needed to run one ad job." in block
    # The rest of the docstring explains why the candidate and video counts are
    # separate fields. It is three paragraphs of rationale that belong in the
    # Python source and nowhere near a TypeScript consumer.
    assert "separate decisions" not in block
    assert "Aspect ratio is derived" not in block


# --- Unmapped types ------------------------------------------------------


def test_an_unmappable_annotation_becomes_unknown_rather_than_a_guess() -> None:
    """A wrong type that compiles is worse than one that forces a decision."""

    class Odd(BaseModel):
        weird: complex = Field(default=0j)

    emitted = TS.Emitter().render_model("Odd", Odd)
    assert "weird: unknown;" in emitted


def test_a_nested_model_is_discovered_without_being_named_as_a_root() -> None:
    emitter = TS.Emitter()
    emitter.register(JobRecord)
    # Reached only through JobRecord -> JobResult -> ranking -> video -> brief.
    assert "DesignPoint" in emitter.models
    assert "MotionIntent" in emitter.enums


# --- Helpers -------------------------------------------------------------


def _chunk(contract: str, opener: str) -> str:
    """The blank-line-separated block that starts with `opener`.

    Split rather than matched with a regex: the obvious pattern here wants both
    ``re.DOTALL`` (to span the body) and a repeated group (to pick up the comments
    above the declaration), and those two together backtrack until the test run
    has to be killed. The emitter already joins its blocks with a blank line, so
    the structure is there to be split on.
    """
    for block in contract.split("\n\n"):
        for line in block.splitlines():
            if line.startswith(opener):
                return block
    raise AssertionError(f"no block starting {opener!r} in the contract")


def _declarations(contract: str) -> list[str]:
    """Every field declaration line, without the comments around them."""
    return [
        line
        for line in contract.splitlines()
        if re.match(r"^  [a-z_]+[?]?: ", line) and line.rstrip().endswith(";")
    ]


def _is_entry(line: str) -> bool:
    """A line inside a `_VALUES` array, as opposed to its JSDoc or its braces."""
    return line.startswith("  ") and line.rstrip().endswith(",") and '"' in line


def _interface(contract: str, name: str) -> str:
    """The interface block, including the comments the emitter put above it."""
    return _chunk(contract, f"export interface {name} {{")


def _const(contract: str, name: str) -> str:
    return _chunk(contract, f"export const {name}: readonly ")
