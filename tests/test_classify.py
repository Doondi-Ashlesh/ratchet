"""Tests for the triage layer.

The load-bearing assertion is that DEFECT never reaches `actionable()`. Those
errors are editable, so a naive classifier would hand them to an agent — which
would silence them with a cast, drop the count, and bless a real bug.
"""
from __future__ import annotations

from ratchet.classify import (
    Category,
    actionable,
    classify,
    from_result,
    summary,
)
from ratchet.tools import Diagnostic, ToolResult


def _d(code: str, file: str = "a.py", line: int = 1) -> Diagnostic:
    return Diagnostic(file=file, line=line, code=code, message=f"msg for {code}")


def test_missing_annotations_go_to_the_agent() -> None:
    out = classify([_d("type-arg"), _d("no-untyped-def"), _d("var-annotated")])

    assert [c.category for c in out] == [Category.ANNOTATION] * 3


def test_suspected_bugs_go_to_a_human() -> None:
    out = classify([_d("attr-defined"), _d("func-returns-value"), _d("assignment")])

    assert [c.category for c in out] == [Category.DEFECT] * 3


def test_unresolved_imports_are_config_not_code() -> None:
    out = classify([_d("import-not-found"), _d("import-untyped")])

    assert [c.category for c in out] == [Category.CONFIG] * 2


def test_a_downstream_error_is_cascading_when_the_file_has_a_config_error() -> None:
    out = classify([_d("import-not-found", "x.py"), _d("misc", "x.py")])

    assert out[1].category is Category.CASCADING
    assert "re-measure" in out[1].reason


def test_the_same_code_is_unknown_without_a_config_error_in_that_file() -> None:
    """The negative case, and the reason the rule is per-file rather than global.
    A `misc` on its own is not evidence of anything — it gets escalated."""
    out = classify([_d("import-not-found", "x.py"), _d("misc", "other.py")])

    assert out[1].category is Category.UNKNOWN


def test_an_unrecognized_code_is_never_guessed_at() -> None:
    out = classify([_d("some-future-mypy-code")])

    assert out[0].category is Category.UNKNOWN
    assert "some-future-mypy-code" in out[0].reason


def test_actionable_excludes_defects_even_though_they_are_editable() -> None:
    """The whole point of splitting FIXABLE. See failure-log 004."""
    out = classify([_d("type-arg"), _d("attr-defined"), _d("import-not-found")])

    work = actionable(out)

    assert len(work) == 1
    assert work[0].code == "type-arg"


def test_summary_reports_every_category_including_empty_ones() -> None:
    """A missing key would read as 'no defects' rather than 'zero defects'."""
    out = summary(classify([_d("type-arg")]))

    assert out == {
        "annotation": 1,
        "defect": 0,
        "config": 0,
        "cascading": 0,
        "unknown": 0,
    }


def test_from_result_reads_a_run_mypy_payload() -> None:
    result = ToolResult(
        True,
        {
            "diagnostics": [
                {"file": "a.py", "line": 3, "code": "type-arg", "message": "m", "severity": "error"}
            ]
        },
    )

    out = from_result(result)

    assert len(out) == 1
    assert out[0].category is Category.ANNOTATION
    assert out[0].line == 3


def test_from_result_on_a_failed_run_yields_nothing() -> None:
    """run_mypy failing is not the same as finding no errors."""
    assert from_result(ToolResult(False, error_type="mypy_not_installed")) == []
