"""Tests for the results tables.

The properties here are all about what a reader would conclude from the rendered
output, because that is the only thing a table is for:

* **A number never appears without its interval** and a value that has no interval
  says so.
* **A missing measurement is stated, never blank.** A blank cell is
  indistinguishable from a zero, from a rounding artefact, and from a deleted row.
* **A table discloses its own label source**, in every format, so a float copied
  into a document keeps its caveat.
* **The LaTeX is ASCII and escaped.** A Greek letter in a heading is a hard
  pdflatex error, in a generated file nobody hand-checks until the build fails.
"""

from __future__ import annotations

from adml import report as RP

NAN = float("nan")


class FakeEvaluation:
    """Stands in for adml.evaluate.Evaluation, which needs the training stack."""

    def __init__(self, name: str, accuracy: float, ci: tuple[float, float], n: int = 100):
        self.name = name
        self.accuracy = accuracy
        self._ci = ci
        self.n = n

    def ci(self, *, seed: int = 0) -> tuple[float, float]:
        return self._ci


class FakeDifference:
    def __init__(self, name_b: str, delta: float, ci: tuple[float, float], resolved: bool):
        self.name_a = "pairwise (trained)"
        self.name_b = name_b
        self.delta = delta
        self.ci = ci
        self.resolved = resolved


class FakeAblationRow:
    def __init__(self, label, n_features, accuracy, ci, delta_vs_full=None):
        self.label = label
        self.n_features = n_features
        self.accuracy = accuracy
        self.ci = ci
        self.delta_vs_full = delta_vs_full


REAL = RP.Stamp(n=400, unit="comparisons")
SIMULATED = RP.Stamp(n=400, unit="comparisons", labels_are_real=False)


def simple_table(stamp: RP.Stamp = REAL) -> RP.Table:
    table = RP.Table(
        key="t",
        caption="A caption",
        columns=[RP.Column("a", "A"), RP.Column("b", "B", "right")],
        stamp=stamp,
    )
    table.add(a="row", b=RP.Value(0.5, (0.4, 0.6)))
    return table


# --- Cells -------------------------------------------------------------------


def test_a_value_prints_with_its_interval():
    assert str(RP.Value(0.612, (0.563, 0.661))) == "0.612 [0.563, 0.661]"


def test_a_value_without_an_interval_prints_the_point_alone():
    assert str(RP.Value(0.612)) == "0.612"


def test_a_non_finite_interval_is_dropped_rather_than_printed_as_nan():
    assert str(RP.Value(0.612, (NAN, NAN))) == "0.612"


def test_a_non_finite_value_is_not_a_number():
    assert str(RP.Value(NAN, (0.1, 0.2))) == "n/a"


def test_a_signed_value_keeps_its_sign_in_the_interval_too():
    assert str(RP.Value(-0.048, (-0.081, -0.015), signed=True)) == "-0.048 [-0.081, -0.015]"
    assert str(RP.Value(0.048, (0.015, 0.081), signed=True)) == "+0.048 [+0.015, +0.081]"


def test_a_percentage_value_rounds_to_whole_percent():
    assert str(RP.Value(0.845, percent=True)) == "84%"


def test_a_pending_cell_renders_as_pending():
    assert str(RP.Pending("needs a GPU run")) == "pending"


def test_an_unset_cell_looks_like_a_mistake_rather_than_an_empty_column():
    """A blank would read as a legitimate empty value. An em dash reads as a gap."""
    assert RP.cell_text(None) == "—"


def test_a_bare_nan_float_becomes_not_available():
    assert RP.cell_text(NAN) == "n/a"


# --- Provenance --------------------------------------------------------------


def test_a_stamp_names_simulated_labels():
    assert "labels are simulated" in SIMULATED.line()
    assert "NOT A RESULT" in SIMULATED.line()
    assert not SIMULATED.is_a_result


def test_a_stamp_names_stand_in_features():
    stamp = RP.Stamp(n=10, features_are_real=False)

    assert "stand-ins" in stamp.line()
    assert not stamp.is_a_result


def test_a_real_stamp_has_no_caveats():
    assert REAL.caveats() == []
    assert REAL.is_a_result


def test_a_stamp_does_not_repeat_the_set_count_it_already_gave():
    stamp = RP.Stamp(n=3, unit="generation sets", n_sets=3)

    assert stamp.line().count("generation sets") == 1


def test_a_stamp_reports_sets_when_they_differ_from_the_unit():
    stamp = RP.Stamp(n=400, unit="comparisons", n_sets=24)

    assert "24 generation sets" in stamp.line()


# --- Markdown ----------------------------------------------------------------


def test_markdown_puts_the_caveat_above_the_header():
    body = simple_table(SIMULATED).to_markdown()

    caveat_at = body.index("NOT A RESULT")
    header_at = body.index("| A | B |")
    assert caveat_at < header_at


def test_a_real_table_has_no_warning_marker():
    body = simple_table(REAL).to_markdown()

    assert "NOT A RESULT" not in body
    assert "⚠️" not in body


def test_markdown_renders_every_column_of_every_row():
    body = simple_table().to_markdown()

    assert "| row | 0.500 [0.400, 0.600] |" in body


# --- Absent rows -------------------------------------------------------------


def test_rows_blocked_on_the_same_reason_collapse_into_one_line():
    """Six baselines blocked on one missing dataset is one problem, not six."""
    table = simple_table()
    for name in ("trained", "ridge", "random"):
        table.pending_row(name, "no labels yet")

    grouped = table.grouped_absences()

    assert grouped == [("trained, ridge, random", "no labels yet")]
    assert table.to_markdown().count("no labels yet") == 1


def test_rows_blocked_on_different_reasons_stay_apart():
    table = simple_table()
    table.pending_row("trained", "no labels yet")
    table.pending_row("embedding", "needs a GPU")

    assert len(table.grouped_absences()) == 2


def test_an_absent_row_is_named_in_the_table_it_belongs_to():
    table = simple_table()
    table.pending_row("LLM-as-judge", "never run")

    body = table.to_markdown()
    assert "LLM-as-judge" in body
    assert "never run" in body


# --- LaTeX -------------------------------------------------------------------


def test_latex_escapes_the_characters_this_project_produces():
    table = RP.Table(
        key="t",
        caption="Ablation of focal_concentration & 50% of cases",
        columns=[RP.Column("a", "Group")],
        stamp=REAL,
    )
    table.add(a="minus product_identity")

    tex = table.to_latex()

    assert r"focal\_concentration" in tex
    assert r"\&" in tex
    assert r"50\%" in tex
    assert r"product\_identity" in tex


def test_latex_turns_greek_into_maths_rather_than_a_build_error():
    table = RP.Table(
        key="t",
        caption="Spearman ρ and Kendall τ",
        columns=[RP.Column("a", "Δ vs full")],
        stamp=REAL,
    )
    table.add(a="0.5")

    tex = table.to_latex()

    assert r"$\rho$" in tex
    assert r"$\tau$" in tex
    assert r"$\Delta$" in tex
    assert "ρ" not in tex


def test_the_whole_latex_document_is_ascii():
    """The one property that makes the file safe to \\input under any engine."""
    report = RP.Report(title="R")
    section = report.section("S", "body")
    table = simple_table(SIMULATED)
    table.pending_row("a row", "because ρ needs 100 sets — see §6")
    section.tables.append(table)

    tex = report.to_latex_tables()

    assert tex.isascii(), [c for c in tex if not c.isascii()]


def test_a_non_result_table_says_so_in_its_latex_caption():
    """The caption travels with the float when it moves page; a footnote does not."""
    tex = simple_table(SIMULATED).to_latex()

    assert r"\textbf{[NOT A RESULT]}" in tex


def test_latex_column_count_matches_the_row_cells():
    table = RP.Table(
        key="t",
        caption="c",
        columns=[RP.Column("a", "A"), RP.Column("b", "B"), RP.Column("c", "C")],
        stamp=REAL,
    )
    table.add(a="1", b="2", c="3")
    table.pending_row("x", "y")

    for line in table.to_latex().splitlines():
        if line.strip().endswith(r"\\"):
            assert line.count("&") == 2, line


# --- The report --------------------------------------------------------------


def test_the_report_lists_every_outstanding_reason_once():
    report = RP.Report(title="R")
    first = simple_table()
    first.pending_row("a", "no labels yet")
    second = simple_table()
    second.pending_row("b", "no labels yet")
    section = report.section("S")
    section.tables += [first, second]

    outstanding = report.outstanding

    assert len(outstanding) == 1
    assert "a; b" in outstanding[0]


def test_a_report_with_a_simulated_table_is_not_a_result():
    report = RP.Report(title="R")
    report.section("S").tables.append(simple_table(SIMULATED))

    assert not report.is_a_result
    assert "not a result yet" in report.front_matter()


def test_a_report_of_real_tables_is_a_result():
    report = RP.Report(title="R")
    report.section("S").tables.append(simple_table(REAL))

    assert report.is_a_result
    assert "not a result yet" not in report.front_matter()


def test_an_empty_report_is_not_a_result():
    """Nothing measured must not pass as everything measured."""
    assert not RP.Report(title="R").is_a_result


def test_the_report_says_it_is_generated_and_must_not_be_edited():
    report = RP.Report(title="R", command="scripts/report.py")

    assert "Do not edit" in report.front_matter()
    assert "scripts/report.py" in report.front_matter()


# --- Builders ----------------------------------------------------------------


def test_the_models_table_is_ordered_best_first():
    table = RP.models_table(
        [
            ("random", FakeEvaluation("random", 0.50, (0.45, 0.55))),
            ("pairwise (trained)", FakeEvaluation("pairwise (trained)", 0.61, (0.56, 0.66))),
        ],
        stamp=REAL,
        ceiling=0.72,
    )

    assert [r["model"] for r in table.rows] == ["pairwise (trained)", "random"]


def test_the_models_table_reports_accuracy_as_a_fraction_of_the_ceiling():
    table = RP.models_table(
        [("pairwise (trained)", FakeEvaluation("pairwise (trained)", 0.61, (0.56, 0.66)))],
        stamp=REAL,
        ceiling=0.72,
    )

    assert str(table.rows[0]["of_ceiling"]) == "85%"


def test_without_a_ceiling_the_models_table_says_pending_rather_than_using_one():
    table = RP.models_table(
        [("pairwise (trained)", FakeEvaluation("pairwise (trained)", 0.61, (0.56, 0.66)))],
        stamp=REAL,
        ceiling=None,
    )

    cell = table.rows[0]["of_ceiling"]
    assert isinstance(cell, RP.Pending)
    assert "no ceiling" in cell.reason


def test_an_unresolved_model_difference_is_marked_in_the_table():
    table = RP.models_table(
        [
            ("pairwise (trained)", FakeEvaluation("pairwise (trained)", 0.61, (0.56, 0.66))),
            ("random", FakeEvaluation("random", 0.50, (0.45, 0.55))),
        ],
        stamp=REAL,
        ceiling=0.72,
        differences=[FakeDifference("random", 0.11, (-0.01, 0.23), resolved=False)],
    )

    row = next(r for r in table.rows if r["model"] == "random")
    assert "unresolved" in str(row["delta"])


def test_a_nan_accuracy_sorts_last_rather_than_unpredictably():
    table = RP.models_table(
        [
            ("broken", FakeEvaluation("broken", NAN, (NAN, NAN))),
            ("random", FakeEvaluation("random", 0.50, (0.45, 0.55))),
        ],
        stamp=REAL,
    )

    assert [r["model"] for r in table.rows] == ["random", "broken"]


def test_the_ablation_table_marks_the_reference_row():
    table = RP.ablation_table(
        [
            FakeAblationRow("full", 20, 0.61, (0.56, 0.66)),
            FakeAblationRow(
                "minus composition",
                14,
                0.56,
                (0.51, 0.61),
                FakeDifference("full", -0.048, (-0.081, -0.015), resolved=True),
            ),
        ],
        stamp=REAL,
    )

    assert table.rows[0]["verdict"] == "reference"
    assert table.rows[1]["verdict"] == "resolved"
    assert str(table.rows[1]["delta"]) == "-0.048 [-0.081, -0.015]"


def test_every_format_carries_the_caption_the_caveat_and_the_absence():
    """Three renderers, one set of facts. A format that drops the caveat is the
    format somebody will paste into the report."""
    table = simple_table(SIMULATED)
    table.pending_row("a row", "a reason")

    markdown, latex, text = table.to_markdown(), table.to_latex(), table.to_text()

    for body in (markdown, latex, text):
        assert "A caption" in body
        assert "NOT A RESULT" in body
        assert "a row" in body
        assert "a reason" in body
        assert "0.500" in body
