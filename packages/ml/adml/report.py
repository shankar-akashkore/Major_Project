"""Results tables that carry their own provenance, in markdown and in LaTeX.

The problem this solves is transcription.  Every number in a report normally makes
one unrecorded journey — from a terminal, through a person, into a document — and
that journey is where a figure measured on simulated labels quietly becomes a
figure presented as measured.  It is also where a number goes stale: the model is
retrained, the table is not, and nobody can tell.

So a :class:`Table` here is built from the evaluation objects directly and rendered
to whichever format the write-up needs.  Three rules follow from that.

**A cell is a number, or it is a stated absence.**  :class:`Pending` is a cell type
with a reason attached, and it renders as a visible "pending" in every format.  A
blank cell is indistinguishable from a zero, from a rounding artefact, and from a
row somebody deleted, so blanks are not allowed.

**A table states what its numbers rest on.**  :class:`Stamp` records the label
source, the feature source and the sample size, and every renderer prints it. A
table built from stub scores says so above its own header rather than in a caption
the reader may not have.

**Rendering never rounds silently.**  A value is formatted once, by the column that
owns it, so the markdown and the LaTeX cannot disagree about what the number was.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

# --- Cells -------------------------------------------------------------------


@dataclass(frozen=True)
class Pending:
    """A measurement that has not been made, and why.

    Distinct from a missing value on purpose: the report has many rows that cannot
    be filled yet, and the difference between "we measured this and it was zero"
    and "this needs a GPU run first" is the difference between a result and a plan.
    """

    reason: str
    label: str = "pending"

    def __str__(self) -> str:
        return self.label


@dataclass(frozen=True)
class Value:
    """A number, its interval, and how to print it.

    The interval is part of the cell rather than a neighbouring column because the
    two must never be separated by a layout change — an accuracy that loses its
    interval on the way into a table reads as far more certain than it is.
    """

    value: float
    ci: tuple[float, float] | None = None
    digits: int = 3
    signed: bool = False
    percent: bool = False

    def __str__(self) -> str:
        if not _finite(self.value):
            return "n/a"
        if self.percent:
            body = f"{self.value:.0%}"
        else:
            sign = "+" if self.signed else ""
            body = f"{self.value:{sign}.{self.digits}f}"
        if self.ci and all(_finite(v) for v in self.ci):
            sign = "+" if self.signed else ""
            lo, hi = self.ci
            body += f" [{lo:{sign}.{self.digits}f}, {hi:{sign}.{self.digits}f}]"
        return body


Cell = Value | Pending | str | int | float | None


def _finite(value: float | None) -> bool:
    return value is not None and not math.isnan(value) and not math.isinf(value)


def cell_text(cell: Cell) -> str:
    """One place where a cell becomes characters, so no two formats disagree."""
    if cell is None:
        # Not a blank: an unset cell is a bug in the caller, and it should look like
        # one rather than like a legitimately empty column.
        return "—"
    if isinstance(cell, float):
        return "n/a" if not _finite(cell) else f"{cell:.3f}"
    return str(cell)


# --- Provenance --------------------------------------------------------------


@dataclass
class Stamp:
    """What every table has to disclose about itself."""

    n: int = 0
    unit: str = "comparisons"
    n_sets: int | None = None
    labels_are_real: bool = True
    features_are_real: bool = True
    scorer: str = ""
    note: str = ""

    @property
    def is_a_result(self) -> bool:
        return self.labels_are_real and self.features_are_real

    def caveats(self) -> list[str]:
        out: list[str] = []
        if not self.labels_are_real:
            out.append("labels are simulated — NOT A RESULT")
        if not self.features_are_real:
            out.append("features are stand-ins, not encoder output — NOT A RESULT")
        return out

    def line(self) -> str:
        parts = [f"n = {self.n} {self.unit}"]
        # Only when it says something the count above does not: a table whose unit is
        # already sets would otherwise read "n = 1 generation sets · 1 generation sets".
        redundant = self.unit == "generation sets" and self.n_sets == self.n
        if self.n_sets is not None and not redundant:
            parts.append(f"{self.n_sets} generation sets")
        if self.scorer:
            parts.append(f"scored by {self.scorer}")
        parts += self.caveats()
        if self.note:
            parts.append(self.note)
        return " · ".join(parts)


# --- Tables ------------------------------------------------------------------


@dataclass
class Column:
    key: str
    heading: str
    align: str = "left"


@dataclass
class Table:
    """A results table, renderable without losing its caveats."""

    key: str
    caption: str
    columns: list[Column]
    rows: list[dict[str, Cell]] = field(default_factory=list)
    stamp: Stamp = field(default_factory=Stamp)
    #: Rows that could not be computed at all, with the reason. Kept apart from
    #: ``rows`` so a renderer can list them after the table without inventing cells
    #: for every column.
    absent: list[tuple[str, str]] = field(default_factory=list)

    def add(self, **cells: Cell) -> None:
        self.rows.append(dict(cells))

    def pending_row(self, label: str, reason: str) -> None:
        """A row named in the report that has no data yet.

        Named rather than omitted: a reader who cannot see that LLM-as-judge is one
        of the planned baselines cannot tell whether it was tried and lost or never
        run.
        """
        self.absent.append((label, reason))

    def grouped_absences(self) -> list[tuple[str, str]]:
        """Absent rows collapsed to one entry per distinct reason.

        Six baselines blocked on the same missing labels is one problem, not six.
        Printed as six it buries every other outstanding item; printed as one, with
        the six names, it says exactly as much in a line.
        """
        by_reason: dict[str, list[str]] = {}
        for label, reason in self.absent:
            by_reason.setdefault(reason, []).append(label)
        return [(", ".join(labels), reason) for reason, labels in by_reason.items()]

    @property
    def pending_reasons(self) -> list[tuple[str, str]]:
        """Every ``(what, why)`` a cell or row is unfilled, for the summary."""
        out = list(self.grouped_absences())
        for row in self.rows:
            for cell in row.values():
                if isinstance(cell, Pending):
                    out.append((self.caption, cell.reason))
        return out

    # --- Renderers ---

    def to_markdown(self) -> str:
        lines = [f"**{self.caption}**", ""]
        # The caveat goes above the header rather than below the table: a reader who
        # skims the numbers and stops has still seen it.
        if self.stamp.is_a_result:
            lines += [f"*{self.stamp.line()}*", ""]
        else:
            lines += [f"> ⚠️ **{self.stamp.line()}**", ""]
        header = "| " + " | ".join(c.heading for c in self.columns) + " |"
        rule = "| " + " | ".join(_md_rule(c.align) for c in self.columns) + " |"
        lines += [header, rule]
        for row in self.rows:
            lines.append("| " + " | ".join(cell_text(row.get(c.key)) for c in self.columns) + " |")
        for label, reason in self.grouped_absences():
            cells = [f"**{label}**", f"*pending — {reason}*"]
            cells += [""] * max(0, len(self.columns) - 2)
            lines.append("| " + " | ".join(cells[: len(self.columns)]) + " |")
        return "\n".join(lines) + "\n"

    def to_latex(self) -> str:
        """A ``table`` float, ready to \\input into the report.

        The caveat goes into the caption rather than a footnote: a caption travels
        with the float when it is moved to another page, and a footnote does not.
        """
        spec = "".join({"left": "l", "right": "r", "center": "c"}[c.align] for c in self.columns)
        caption = _tex(self.caption)
        stamp = _tex(self.stamp.line())
        if not self.stamp.is_a_result:
            caption = f"\\textbf{{[NOT A RESULT]}} {caption}"

        lines = [
            "\\begin{table}[htbp]",
            "  \\centering",
            f"  \\caption{{{caption}}}",
            f"  \\label{{tab:{self.key}}}",
            f"  \\begin{{tabular}}{{{spec}}}",
            "    \\hline",
            "    " + " & ".join(f"\\textbf{{{_tex(c.heading)}}}" for c in self.columns) + " \\\\",
            "    \\hline",
        ]
        for row in self.rows:
            lines.append(
                "    " + " & ".join(_tex(cell_text(row.get(c.key))) for c in self.columns) + " \\\\"
            )
        for label, reason in self.grouped_absences():
            cells = [f"\\textit{{{_tex(label)}}}", f"\\textit{{pending: {_tex(reason)}}}"]
            cells += [""] * max(0, len(self.columns) - 2)
            lines.append("    " + " & ".join(cells[: len(self.columns)]) + " \\\\")
        lines += [
            "    \\hline",
            "  \\end{tabular}",
            f"  \\par\\smallskip\\footnotesize {stamp}",
            "\\end{table}",
        ]
        return "\n".join(lines) + "\n"

    def to_text(self) -> str:
        """Fixed-width, for a terminal and for pasting into a plain-text note."""
        widths = [
            max(len(c.heading), *(len(cell_text(r.get(c.key))) for r in self.rows or [{}]))
            for c in self.columns
        ]
        out = [self.caption, self.stamp.line(), ""]
        out.append("  ".join(h.ljust(w) for h, w in zip(_headings(self), widths, strict=True)))
        out.append("  ".join("-" * w for w in widths))
        for row in self.rows:
            out.append(
                "  ".join(
                    cell_text(row.get(c.key)).ljust(w)
                    for c, w in zip(self.columns, widths, strict=True)
                )
            )
        for label, reason in self.grouped_absences():
            out.append(f"{label}  pending — {reason}")
        return "\n".join(out) + "\n"


def _headings(table: Table) -> list[str]:
    return [c.heading for c in table.columns]


def _md_rule(align: str) -> str:
    return {"left": ":---", "right": "---:", "center": ":---:"}[align]


_TEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


#: Applied *after* character escaping, because the replacements are themselves LaTeX
#: and must not be escaped in turn.  These are the symbols this project's own headings
#: and stamps contain: under pdflatex a bare Greek letter is not a typographic nit but
#: a hard error ("Unicode character rho not set up for use with LaTeX"), in a generated
#: file nobody thinks to check until the build fails.
_TEX_SYMBOLS = {
    "ρ": r"$\rho$",
    "τ": r"$\tau$",
    "σ": r"$\sigma$",
    "μ": r"$\mu$",
    "Δ": r"$\Delta$",
    "±": r"$\pm$",
    "≤": r"$\leq$",
    "≥": r"$\geq$",
    "×": r"$\times$",
    "→": r"$\rightarrow$",
    "§": r"\S{}",
    "·": r"\textperiodcentered{}",
    "—": "---",
    "–": "--",
    "’": "'",
    "“": "``",
    "”": "''",
    # Decorative only, and there is no sensible LaTeX for it. The words beside it
    # carry the warning.
    "⚠️": "",
    "⚠": "",
}


def _tex(text: str) -> str:
    """Escape for LaTeX, including the characters this project actually produces.

    Feature-group names carry underscores and the stamps carry per-cent signs, both
    of which are LaTeX syntax; an unescaped one fails the build a week before the
    deadline, in a file nobody edited by hand.
    """
    out: list[str] = []
    for char in str(text):
        out.append(_TEX_ESCAPES.get(char, char))
    joined = "".join(out)
    for symbol, macro in _TEX_SYMBOLS.items():
        joined = joined.replace(symbol, macro)
    return joined


# --- The document ------------------------------------------------------------


@dataclass
class Section:
    """One part of the results write-up: prose the generator owns, plus tables."""

    heading: str
    body: str = ""
    tables: list[Table] = field(default_factory=list)
    figures: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Report:
    """The whole results artefact.

    Prose lives here rather than in a hand-edited document because the sentences
    that have to change when the data changes are exactly the sentences that carry
    the numbers. What a human should write — interpretation, related work, the
    argument — is deliberately *not* generated; see :meth:`front_matter`.
    """

    title: str
    sections: list[Section] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    command: str = ""

    def section(self, heading: str, body: str = "") -> Section:
        found = Section(heading=heading, body=body)
        self.sections.append(found)
        return found

    @property
    def tables(self) -> list[Table]:
        return [t for s in self.sections for t in s.tables]

    @property
    def outstanding(self) -> list[str]:
        """Every stated absence in the report, one entry per distinct reason.

        Grouped across tables as well as within them: the missing annotations block
        the models table, the ablation and the headline claim, and listing that
        blocker three times makes the list longer without making it more useful.
        """
        by_reason: dict[str, list[str]] = {}
        for table in self.tables:
            for what, reason in table.pending_reasons:
                names = by_reason.setdefault(reason, [])
                if what not in names:
                    names.append(what)
        return [f"**{'; '.join(names)}** — {reason}" for reason, names in by_reason.items()]

    @property
    def is_a_result(self) -> bool:
        return all(t.stamp.is_a_result for t in self.tables) and bool(self.tables)

    def front_matter(self) -> str:
        """The banner, and the standing warning against editing this file.

        A generated document that looks editable will be edited, and the edit will
        be lost on the next run without anyone noticing which numbers went back.
        """
        stamp = self.generated_at.strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"# {self.title}",
            "",
            f"*Generated {stamp}"
            + (f" by `{self.command}`" if self.command else "")
            + ". Do not edit — regenerate.*",
            "",
        ]
        if not self.is_a_result:
            lines += [
                "> **This document is not a result yet.** At least one table below rests on",
                "> simulated labels or stand-in features. Each such table says so above its",
                "> own header, and the outstanding list below names what has to happen for",
                "> the numbers to mean something.",
                "",
            ]
        if self.outstanding:
            lines.append("## Outstanding before these numbers stand")
            lines.append("")
            for reason in self.outstanding:
                lines.append(f"- {reason}")
            lines.append("")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        parts = [self.front_matter()]
        for section in self.sections:
            parts.append(f"## {section.heading}\n")
            if section.body:
                parts.append(section.body.strip() + "\n")
            for path, caption in section.figures:
                parts.append(f"![{caption}]({path})\n")
            for table in section.tables:
                parts.append(table.to_markdown())
        return "\n".join(parts)

    def to_latex_tables(self) -> str:
        """Only the floats. The prose belongs in the author's own document."""
        # Kept ASCII: the whole file is asserted to be ASCII so a stray symbol cannot
        # reach a pdflatex build, and a comment is part of the file.
        header = (
            "% Generated by scripts/report.py. Do not edit - regenerate.\n"
            "% \\input this file, or copy individual table environments out of it.\n"
        )
        return header + "\n".join(t.to_latex() for t in self.tables)


# --- Builders over the evaluation objects ------------------------------------


def models_table(
    evaluations: Sequence[tuple[str, object]],
    *,
    stamp: Stamp,
    ceiling: float | None = None,
    trained_name: str = "pairwise (trained)",
    differences: Sequence[object] = (),
) -> Table:
    """The model comparison, best first, each against the trained model.

    ``evaluations`` is ``(label, adml.evaluate.Evaluation)``. Duck-typed rather than
    imported so this module stays free of the training stack and can be unit-tested
    on plain objects.
    """
    table = Table(
        key="models",
        caption="Held-out pairwise accuracy against baselines",
        columns=[
            Column("model", "Model"),
            Column("accuracy", "Accuracy [95% CI]", "right"),
            Column("of_ceiling", "% of ceiling", "right"),
            Column("n", "n", "right"),
            Column("delta", "Δ vs trained [95% CI]", "right"),
        ],
        stamp=stamp,
    )

    by_name = {label: ev for label, ev in evaluations}
    deltas = {getattr(d, "name_b", ""): d for d in differences}
    order = sorted(by_name, key=lambda k: -_safe(getattr(by_name[k], "accuracy", float("nan"))))

    for label in order:
        ev = by_name[label]
        accuracy = getattr(ev, "accuracy", float("nan"))
        ci = getattr(ev, "ci", lambda: (float("nan"),) * 2)()
        of_ceiling: Cell
        if _finite(ceiling) and ceiling and _finite(accuracy):
            of_ceiling = Value(accuracy / ceiling, percent=True)
        else:
            of_ceiling = Pending("no ceiling measured: needs repeat and multiply-judged pairs")

        delta_cell: Cell = "—" if label == trained_name else Pending("not compared")
        difference = deltas.get(label)
        if difference is not None:
            marker = "" if getattr(difference, "resolved", False) else " (unresolved)"
            delta_cell = (
                str(
                    Value(
                        getattr(difference, "delta", float("nan")),
                        getattr(difference, "ci", None),
                        signed=True,
                    )
                )
                + marker
            )

        table.add(
            model=label,
            accuracy=Value(accuracy, ci),
            of_ceiling=of_ceiling,
            n=getattr(ev, "n", 0),
            delta=delta_cell,
        )
    return table


def ablation_table(rows: Sequence[object], *, stamp: Stamp) -> Table:
    """One row per feature group, from :func:`adml.evaluate.ablate`."""
    table = Table(
        key="ablation",
        caption="Feature-group ablation, each row paired against the full model",
        columns=[
            Column("variant", "Variant"),
            Column("cols", "Columns", "right"),
            Column("accuracy", "Accuracy [95% CI]", "right"),
            Column("delta", "Δ vs full [95% CI]", "right"),
            Column("verdict", "Verdict"),
        ],
        stamp=stamp,
    )
    for row in rows:
        difference = getattr(row, "delta_vs_full", None)
        if difference is None:
            delta_cell: Cell = "—"
            verdict: Cell = "reference"
        else:
            delta_cell = Value(
                getattr(difference, "delta", float("nan")),
                getattr(difference, "ci", None),
                signed=True,
            )
            verdict = "resolved" if getattr(difference, "resolved", False) else "unresolved"
        table.add(
            variant=getattr(row, "label", "?"),
            cols=getattr(row, "n_features", 0),
            accuracy=Value(getattr(row, "accuracy", float("nan")), getattr(row, "ci", None)),
            delta=delta_cell,
            verdict=verdict,
        )
    return table


def _safe(value: float) -> float:
    """NaN sorts last rather than unpredictably."""
    return value if _finite(value) else -math.inf
