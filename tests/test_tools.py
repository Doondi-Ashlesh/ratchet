"""Tests for the tool layer.

What's under test is the contract, not mypy: every tool returns a ToolResult,
no tool raises, and each distinct failure arrives as a branchable error_type.

Fixtures are written to tmp_path so the suite is hermetic — it depends on no
other repo existing on this machine.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ratchet import tools

CLEAN = "def f(x: int) -> int:\n    return x\n"
UNTYPED = "def f(x):\n    return x\n"


def test_read_file_returns_content(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text(CLEAN, encoding="utf-8")

    r = tools.read_file(str(p))

    assert r.ok
    assert r.data["content"] == CLEAN


def test_read_missing_file_is_data_not_an_exception(tmp_path: Path) -> None:
    """The whole point of the layer: the agent gets something to branch on."""
    r = tools.read_file(str(tmp_path / "nope.py"))

    assert r.ok is False
    assert r.error_type == "file_not_found"
    assert "nope.py" in r.error_message


def test_write_file_updates_an_existing_file(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text(UNTYPED, encoding="utf-8")

    r = tools.write_file(str(p), CLEAN)

    assert r.ok
    assert p.read_text(encoding="utf-8") == CLEAN


def test_write_file_refuses_to_create(tmp_path: Path) -> None:
    """A deliberate constraint: this agent edits what exists. Creating new
    modules is a much larger permission than fixing a type."""
    p = tmp_path / "new.py"

    r = tools.write_file(str(p), CLEAN)

    assert r.ok is False
    assert r.error_type == "file_not_found"
    assert not p.exists()


def test_run_mypy_reports_zero_on_clean_source(tmp_path: Path) -> None:
    (tmp_path / "clean.py").write_text(CLEAN, encoding="utf-8")

    r = tools.run_mypy(str(tmp_path))

    assert r.ok
    assert r.data["error_count"] == 0
    assert r.data["by_code"] == {}


def test_run_mypy_surfaces_the_error_code_and_message(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text(UNTYPED, encoding="utf-8")

    r = tools.run_mypy(str(tmp_path))

    assert r.data["error_count"] >= 1
    assert "no-untyped-def" in r.data["by_code"]

    d = r.data["diagnostics"][0]
    assert d["line"] == 1
    assert d["message"]                      # kept intact — it's what the model reads
    assert d["severity"] == "error"


def test_ok_means_the_tool_ran_not_that_the_code_is_clean(tmp_path: Path) -> None:
    """Pinning the semantic. Finding 111 errors is a successful run of the tool.
    `ok=False` is reserved for the tool failing to produce an answer at all."""
    (tmp_path / "bad.py").write_text(UNTYPED, encoding="utf-8")

    r = tools.run_mypy(str(tmp_path))

    assert r.ok is True
    assert r.data["error_count"] > 0


def test_a_missing_mypy_binary_is_a_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent executable is an environment problem, not a code problem, and
    the agent must be able to tell those apart."""

    def boom(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("mypy")

    monkeypatch.setattr(tools.subprocess, "run", boom)

    r = tools.run_mypy(str(tmp_path))

    assert r.ok is False
    assert r.error_type == "mypy_not_installed"


def test_tally_counts_and_orders_by_frequency() -> None:
    assert tools._tally(["a", "b", "a", "c", "a", "b"]) == {"a": 3, "b": 2, "c": 1}
    assert tools._tally([]) == {}
