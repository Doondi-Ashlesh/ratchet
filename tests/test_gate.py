"""Tests for the ratchet.

The load-bearing case is the CONFIG reveal: a session that fixes imports may
legitimately RAISE the total, because mypy under-reports what it cannot analyze.
A naive "accept if the count fell" gate rejects correct work, and this suite is
what stops someone simplifying `judge()` back into that.
"""
from __future__ import annotations

from ratchet.classify import Category, Classified
from ratchet.gate import Measurement, judge


def _m(
    annotation: int = 0,
    defect: int = 0,
    config: int = 0,
    cascading: int = 0,
    unknown: int = 0,
) -> Measurement:
    return Measurement(
        {
            "annotation": annotation,
            "defect": defect,
            "config": config,
            "cascading": cascading,
            "unknown": unknown,
        }
    )


def _c(category: Category) -> Classified:
    return Classified(
        file="a.py", line=1, code="x", message="m", category=category, reason="r"
    )


# ── measurement ──────────────────────────────────────────────────────────────

def test_measurement_is_built_from_classified_diagnostics() -> None:
    m = Measurement.of([_c(Category.ANNOTATION), _c(Category.ANNOTATION), _c(Category.DEFECT)])

    assert m.get(Category.ANNOTATION) == 2
    assert m.get(Category.DEFECT) == 1
    assert m.total == 3


# ── the session must actually do its job ─────────────────────────────────────

def test_a_session_that_changed_nothing_is_rejected() -> None:
    v = judge(_m(annotation=10), _m(annotation=10), Category.ANNOTATION)

    assert not v.accepted
    assert "no progress" in v.reason


def test_a_session_that_made_its_own_category_worse_is_rejected() -> None:
    v = judge(_m(annotation=10), _m(annotation=12), Category.ANNOTATION)

    assert not v.accepted
    assert "no progress" in v.reason


# ── the reveal case: why judge() is not a subtraction ────────────────────────

def test_a_config_fix_is_accepted_even_when_the_total_rises() -> None:
    """Fixing imports lets mypy see code it could not analyze before. Errors that
    appear were always there. Rejecting this would reject correct work."""
    before = _m(annotation=87, config=17, cascading=7)          # total 111
    after = _m(annotation=95, config=13, cascading=3, unknown=6)  # total 117

    v = judge(before, after, Category.CONFIG)

    assert v.accepted
    assert "reveals allowed" in v.reason


def test_the_observed_pydantic_run_is_accepted() -> None:
    """The real numbers from installing pydantic stubs against SdkAgent."""
    before = _m(annotation=87, config=17, cascading=7)              # 111
    after = _m(annotation=84, config=13, cascading=3, unknown=2)    # 102

    assert judge(before, after, Category.CONFIG).accepted


def test_a_config_session_that_fixed_no_imports_is_still_rejected() -> None:
    """The reveal allowance is not a blanket exemption — config must still fall."""
    v = judge(_m(config=17), _m(config=17, annotation=3), Category.CONFIG)

    assert not v.accepted


# ── source edits get no such allowance ───────────────────────────────────────

def test_an_annotation_session_that_introduced_a_defect_is_rejected() -> None:
    """Five fixed and one bug introduced nets -4. The total improved. Reject anyway."""
    v = judge(_m(annotation=10), _m(annotation=5, defect=1), Category.ANNOTATION)

    assert not v.accepted
    assert "defect" in v.reason


def test_an_annotation_session_that_moved_errors_sideways_is_rejected() -> None:
    """annotation fell but unknown rose by the same amount — nothing improved."""
    v = judge(_m(annotation=10), _m(annotation=7, unknown=3), Category.ANNOTATION)

    assert not v.accepted
    assert "the total did not" in v.reason


def test_real_progress_is_accepted() -> None:
    v = judge(_m(annotation=10, config=2), _m(annotation=6, config=2), Category.ANNOTATION)

    assert v.accepted
    assert v.deltas["annotation"] == -4


def test_the_verdict_reports_every_delta() -> None:
    v = judge(_m(annotation=10), _m(annotation=8), Category.ANNOTATION)

    assert set(v.deltas) == {"annotation", "defect", "config", "cascading", "unknown"}
