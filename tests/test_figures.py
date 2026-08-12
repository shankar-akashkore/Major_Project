"""Tests for the SVG figures.

Pixel positions are not asserted — they are a drawing decision and would make the
suite fight every layout tweak.  What is asserted is the part a reader's conclusion
depends on: that a figure admits what it is, that an interval is never silently
dropped, and that a missing measurement is drawn as missing rather than as zero.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from adml import figures as F

NAN = float("nan")


def parse(svg: str) -> ET.Element:
    """Well-formedness is the baseline property: an SVG that does not parse is a
    figure that silently fails to render in a report."""
    return ET.fromstring(svg)


def texts(svg: str) -> list[str]:
    return [(el.text or "") for el in parse(svg).iter("{http://www.w3.org/2000/svg}text")]


def count(svg: str, tag: str) -> int:
    return len(list(parse(svg).iter(f"{{http://www.w3.org/2000/svg}}{tag}")))


REAL = F.Provenance(n=400, unit="comparisons")
SIMULATED = F.Provenance(n=400, unit="comparisons", labels_are_real=False)
STAND_IN = F.Provenance(n=400, unit="comparisons", features_are_real=False)

MODELS = [
    F.ModelPoint("trained", 0.61, (0.56, 0.66), is_trained=True),
    F.ModelPoint("random", 0.50, (0.45, 0.55)),
]


# --- Every figure is a valid document ---------------------------------------


def test_every_figure_parses_as_xml():
    svgs = [
        F.accuracy_forest(MODELS, provenance=REAL, ceiling=0.72),
        F.ablation_deltas(
            [F.AblationPoint("minus a", -0.04, (-0.07, -0.01), resolved=True)], provenance=REAL
        ),
        F.stage_scatter([(0, 0), (1, 1)], provenance=REAL, rho=0.5),
        F.sample_size_curve([(12, 0.10), (100, 0.04)], provenance=REAL, target=0.045),
    ]
    for svg in svgs:
        assert parse(svg).tag == "{http://www.w3.org/2000/svg}svg"


def test_a_label_with_markup_in_it_cannot_break_the_document():
    """Labels come from feature-group names and model names, which are code-supplied
    today — but an unescaped one would corrupt the whole figure rather than one row."""
    svg = F.accuracy_forest(
        [F.ModelPoint('<script>alert("x")</script> & more', 0.6, (0.5, 0.7))], provenance=REAL
    )

    parse(svg)  # would raise if the label escaped its element
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


def test_the_viewbox_matches_the_declared_size():
    svg = F.accuracy_forest(MODELS, provenance=REAL)
    root = parse(svg)

    width = float(root.get("width"))
    height = float(root.get("height"))
    assert root.get("viewBox") == f"0 0 {width:g} {height:g}".replace(".0 ", " ")


# --- Provenance travels with the picture ------------------------------------


def test_simulated_labels_are_named_in_the_figure_itself():
    svg = F.accuracy_forest(MODELS, provenance=SIMULATED, ceiling=0.72)

    assert any("SIMULATED LABELS" in t and "NOT A RESULT" in t for t in texts(svg))


def test_stand_in_features_are_named_in_the_figure_itself():
    svg = F.accuracy_forest(MODELS, provenance=STAND_IN)

    assert any("STAND-IN FEATURES" in t for t in texts(svg))


def test_a_real_figure_carries_its_sample_size_without_a_warning():
    svg = F.accuracy_forest(MODELS, provenance=REAL)

    joined = " ".join(texts(svg))
    assert "n = 400 comparisons" in joined
    assert "NOT A RESULT" not in joined


def caption_fill(svg: str) -> str:
    """The fill of the caveat line, found by its content rather than its position."""
    for el in parse(svg).iter("{http://www.w3.org/2000/svg}text"):
        if (el.text or "").startswith("n = "):
            return el.get("fill", "")
    raise AssertionError("no caption element found")


def test_the_caveat_is_drawn_in_the_warning_colour_only_when_warranted():
    """A caveat set in body ink is a caveat nobody reads — and body text set in the
    warning colour is a warning that stops meaning anything."""
    assert caption_fill(F.accuracy_forest(MODELS, provenance=SIMULATED)) == F.WARN
    assert caption_fill(F.accuracy_forest(MODELS, provenance=STAND_IN)) == F.WARN
    assert caption_fill(F.accuracy_forest(MODELS, provenance=REAL)) == F.INK


def test_a_result_figure_has_no_warning_text():
    svg = F.accuracy_forest(MODELS, provenance=REAL, ceiling=0.72)

    assert "NOT A RESULT" not in svg
    assert "not measured" not in svg


def test_provenance_reports_whether_it_is_a_result():
    assert REAL.is_a_result
    assert not SIMULATED.is_a_result
    assert not STAND_IN.is_a_result


# --- Intervals are never dropped --------------------------------------------


def test_every_measured_model_gets_a_whisker():
    """Three line segments per interval — the span and two caps — so the count is
    the cheapest way to assert no interval was skipped."""
    svg = F.accuracy_forest(MODELS, provenance=REAL)

    # 2 frame edges + 2 rows x 3 whisker segments + gridlines + the chance rule.
    assert count(svg, "line") >= 2 + 6


def test_a_point_without_an_interval_is_drawn_hollow_and_labelled():
    svg = F.accuracy_forest([F.ModelPoint("one fold only", 0.61, (NAN, NAN))], provenance=REAL)

    assert "no interval" in " ".join(texts(svg))
    # Hollow means filled with the paper colour and stroked in the warning colour.
    assert re.search(rf'<circle[^>]*fill="{F.PAPER}"[^>]*stroke="{F.WARN}"', svg)


def test_an_unmeasured_model_says_so_instead_of_plotting_zero():
    svg = F.accuracy_forest(
        [
            F.ModelPoint("trained", 0.61, (0.56, 0.66), is_trained=True),
            F.ModelPoint("LLM-as-judge", NAN, (NAN, NAN)),
        ],
        provenance=REAL,
    )

    assert "not measured" in " ".join(texts(svg))
    # One dot for the measured model. A NaN accuracy must not become a mark.
    assert count(svg, "circle") == 1


def test_the_ceiling_is_drawn_and_labelled_when_known():
    svg = F.accuracy_forest(MODELS, provenance=REAL, ceiling=0.724)

    assert any("ceiling 0.724" in t for t in texts(svg))
    assert any("of ceiling" in t for t in texts(svg))


def test_no_ceiling_means_no_ceiling_claim():
    """Without a measured ceiling, accuracy must not be reported as a fraction of
    one — that would silently use 1.0 as the denominator."""
    svg = F.accuracy_forest(MODELS, provenance=REAL, ceiling=None)

    assert not any("of ceiling" in t for t in texts(svg))


def test_chance_is_always_drawn():
    svg = F.accuracy_forest(MODELS, provenance=REAL)

    assert any("chance 0.50" in t for t in texts(svg))


# --- Ablation ---------------------------------------------------------------


def test_an_unresolved_ablation_row_is_marked_unresolved():
    svg = F.ablation_deltas(
        [F.AblationPoint("minus context", -0.004, (-0.039, 0.031), resolved=False)],
        provenance=REAL,
    )

    assert "unresolved" in " ".join(texts(svg))


def test_a_resolved_ablation_row_is_not_marked_unresolved():
    svg = F.ablation_deltas(
        [F.AblationPoint("minus composition", -0.048, (-0.081, -0.015), resolved=True)],
        provenance=REAL,
    )

    assert "unresolved" not in " ".join(texts(svg))


def test_the_ablation_delta_is_printed_with_its_sign():
    svg = F.ablation_deltas(
        [F.AblationPoint("minus composition", -0.048, (-0.081, -0.015), resolved=True)],
        provenance=REAL,
    )

    assert "-0.048" in " ".join(texts(svg))


def test_an_ablation_row_with_no_measurement_is_not_drawn_at_zero():
    svg = F.ablation_deltas([F.AblationPoint("minus embedding", NAN, (NAN, NAN))], provenance=REAL)

    assert "not measured" in " ".join(texts(svg))
    assert count(svg, "circle") == 0


def test_an_empty_ablation_still_renders():
    """Nothing to ablate is the state before the first training run, and the report
    generator must not crash on it."""
    parse(F.ablation_deltas([], provenance=REAL))


# --- Stage agreement --------------------------------------------------------


def test_the_scatter_counts_repeated_cells():
    svg = F.stage_scatter([(0, 0), (0, 0), (0, 0), (1, 1)], provenance=REAL, rho=1.0)

    assert "3" in texts(svg)
    assert "1" in texts(svg)


def test_a_withheld_rho_is_stated_rather_than_left_blank():
    svg = F.stage_scatter([(0, 0)], provenance=REAL, rho=None)

    assert any("withheld" in t for t in texts(svg))


def test_a_known_rho_is_printed():
    svg = F.stage_scatter([(0, 0), (1, 1)], provenance=REAL, rho=-0.25)

    assert any("-0.250" in t for t in texts(svg))


def test_an_empty_scatter_says_there_are_no_observations():
    svg = F.stage_scatter([], provenance=REAL, rho=None)

    assert any("no paired observations" in t for t in texts(svg))


def test_out_of_range_ranks_are_ignored_rather_than_drawn_off_grid():
    svg = F.stage_scatter([(0, 0), (7, 7)], provenance=REAL, rho=None, n_candidates=3)

    # One counted cell, so one mark plus its count label.
    assert count(svg, "circle") == 1


# --- Sample size ------------------------------------------------------------


def test_the_sample_size_curve_marks_where_the_corpus_is_and_needs_to_be():
    svg = F.sample_size_curve(
        [(24, 0.074), (100, 0.040)],
        provenance=REAL,
        marks=[(24, "now"), (100, "needed")],
        target=0.045,
    )

    joined = " ".join(texts(svg))
    assert "now" in joined
    assert "needed" in joined
    assert "resolvable below 0.045" in joined


def test_a_curve_with_nothing_measured_says_so():
    svg = F.sample_size_curve([], provenance=REAL)

    assert any("no measurements" in t for t in texts(svg))


def test_a_curve_ignores_non_finite_widths():
    svg = F.sample_size_curve([(24, NAN), (100, 0.04), (200, 0.03)], provenance=REAL)

    assert count(svg, "path") == 1
    # Two usable points, so two markers on the line.
    assert count(svg, "circle") == 2
