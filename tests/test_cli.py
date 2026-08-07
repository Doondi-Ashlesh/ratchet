"""Tests for the CLI, which double as the end-to-end test.

Every other suite tests one module in isolation. This one drives the real chain —
subprocess out to mypy, parse, classify, report — so the seams between modules are
covered. Exit codes get their own tests because they are the contract with CI, and
a wrong one is invisible until a pipeline silently passes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ratchet import tools
from ratchet.cli import EXIT_FOUND, EXIT_OK, EXIT_TOOL_FAILED, main

CLEAN = "def f(x: int) -> int:\n    return x\n"
UNTYPED = "def f(x):\n    return x\n"


def test_clean_code_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "clean.py").write_text(CLEAN, encoding="utf-8")

    assert main(["check", str(tmp_path)]) == EXIT_OK
    assert "next: clean" in capsys.readouterr().out


def test_code_with_errors_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Non-zero on findings, same as mypy and ruff, so this composes into CI."""
    (tmp_path / "bad.py").write_text(UNTYPED, encoding="utf-8")

    assert main(["check", str(tmp_path)]) == EXIT_FOUND


def test_the_whole_chain_runs_and_routes_correctly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The integration test: real mypy, real parse, real triage, real report."""
    (tmp_path / "bad.py").write_text(UNTYPED, encoding="utf-8")

    main(["check", str(tmp_path)])
    out = capsys.readouterr().out

    assert "annotation" in out and "agent may attempt" in out
    assert "ready for an agent" in out          # untyped def is annotation work
    assert "total" in out


def test_a_tool_failure_is_not_a_finding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 2, not 1. A pipeline must be able to tell 'mypy is missing' from
    'your code is clean', and both would be silence otherwise."""

    def boom(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("mypy")

    monkeypatch.setattr(tools.subprocess, "run", boom)

    assert main(["check", str(tmp_path)]) == EXIT_TOOL_FAILED
    assert "mypy_not_installed" in capsys.readouterr().err


def test_json_output_is_machine_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "bad.py").write_text(UNTYPED, encoding="utf-8")

    main(["check", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["total"] >= 1
    assert payload["counts"]["annotation"] >= 1
    assert set(payload["counts"]) == {
        "annotation",
        "defect",
        "config",
        "cascading",
        "unknown",
    }


def test_config_errors_are_reported_as_the_first_thing_to_fix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """failure-log 003: config first. An unresolved import both hides errors and
    produces cascading ones, so any other ordering wastes agent sessions."""
    (tmp_path / "bad.py").write_text(
        "import definitely_not_a_real_package\n\ndef f(x):\n    return x\n", encoding="utf-8"
    )

    main(["check", str(tmp_path)])
    out = capsys.readouterr().out

    assert "resolve" in out and "config error(s) first" in out


def test_check_requires_a_path(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["check"])


def test_an_unknown_subcommand_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_every_run_flag_reaches_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """--deadline was first wired into the checkpointed branch only, so the flag
    silently did nothing without --checkpoint. This pins that every flag the
    parser accepts is actually forwarded, on the path people use by default."""
    from ratchet import loop

    seen: dict[str, object] = {}
    monkeypatch.setattr(loop, "run_loop", lambda path, **kw: seen.update(kw, path=path) or {})

    main(["run", "pkg", "--deadline", "12", "--max-files", "3", "--order", "smallest"])

    assert seen["deadline_s"] == 12.0
    assert seen["max_files"] == 3
    assert seen["order"] == "smallest"
