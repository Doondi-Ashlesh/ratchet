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


def _write(p: Path, text: str) -> None:
    """Write exactly these bytes. write_text() translates \\n to \\r\\n on Windows,
    so the fixture would not contain what it claims to contain."""
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def test_read_file_returns_content(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    _write(p, CLEAN)

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
    _write(p, UNTYPED)

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
    _write((tmp_path / "clean.py"), CLEAN)

    r = tools.run_mypy(str(tmp_path))

    assert r.ok
    assert r.data["error_count"] == 0
    assert r.data["by_code"] == {}


def test_run_mypy_surfaces_the_error_code_and_message(tmp_path: Path) -> None:
    _write((tmp_path / "bad.py"), UNTYPED)

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
    _write((tmp_path / "bad.py"), UNTYPED)

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


def test_run_mypy_groups_errors_by_file_worst_first(tmp_path: Path) -> None:
    """by_file is how a session picks which file to work on next, so the
    ordering is part of the contract, not a presentation detail."""
    _write((tmp_path / "one.py"), UNTYPED)
    _write((tmp_path / "three.py"), "def a(x):\n    return x\ndef b(x):\n    return x\ndef c(x):\n    return x\n")

    by_file = tools.run_mypy(str(tmp_path)).data["by_file"]

    assert len(by_file) == 2
    assert list(by_file.values()) == sorted(by_file.values(), reverse=True)
    assert "three.py" in next(iter(by_file))


def test_read_then_write_is_byte_exact(tmp_path: Path) -> None:
    """A revert must restore the file, not a line-ending-normalised lookalike.
    Without newline="" this rewrites every line on Windows, and the session's
    safety property becomes conditional on which OS wrote the repo."""
    for original in (b"a = 1\nb = 2\n", b"a = 1\r\nb = 2\r\n"):
        p = tmp_path / "f.py"
        p.write_bytes(original)

        result = tools.write_file(str(p), str(tools.read_file(str(p)).data["content"]))

        assert result.ok
        assert p.read_bytes() == original
