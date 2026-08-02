"""Tests for the session — the loop that ties everything together.

These run the real chain including real mypy subprocess calls, because the thing
under test is whether measure -> propose -> apply -> re-measure -> judge produces
the right verdict on real diagnostics. Only the model is faked.

The load-bearing test is the byte-exact revert. Everything else in Ratchet is
read-only; this is the one component that writes to someone's source tree, and
"it reverted" is worth nothing if it reverted to a line-ending-normalised
lookalike.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from ratchet import model, session
from ratchet.session import run_session

UNTYPED = "def f(x):\n    return x\n"
GOOD_FIX = "def f(x: int) -> int:\n    return x\n"
CLEAN = "x: int = 1\n"

# Fixes the annotation error and introduces an undefined name in its place:
# annotation 1 -> 0, unknown 0 -> 1, total unchanged. The gate must refuse this.
SIDEWAYS_FIX = "def f(x: int) -> int:\n    return nope\n"


@pytest.fixture(autouse=True)
def _never_call_a_live_model() -> Iterator[None]:
    yield
    model.set_completer(None)


def _write(p: Path, text: str) -> None:
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def test_a_file_with_no_annotation_work_is_left_alone(tmp_path: Path) -> None:
    p = tmp_path / "clean.py"
    _write(p, CLEAN)
    before = p.read_bytes()

    result = run_session(str(tmp_path), str(p))

    assert result.kept is False
    assert "no annotation work" in result.reason
    assert p.read_bytes() == before


def test_a_good_fix_is_kept(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    _write(p, UNTYPED)
    model.set_completer(lambda _: GOOD_FIX)

    result = run_session(str(tmp_path), str(p))

    assert result.kept is True
    assert result.after.total < result.before.total
    assert "int" in p.read_text(encoding="utf-8")


def test_a_sideways_fix_is_reverted_byte_for_byte(tmp_path: Path) -> None:
    """The load-bearing test. The fix trades one error for another, the gate
    refuses it, and the file must come back EXACTLY as it was — not a
    line-ending-normalised lookalike.

    The exact rejection reason is pinned in test_gate.py; asserting it here too
    would make this brittle. `return nope` makes mypy infer Any, which is itself
    an annotation error, so the gate refuses at the no-progress check rather than
    the total check. Either way it refuses, and either way the bytes must return.
    """
    p = tmp_path / "a.py"
    p.write_bytes(b"def f(x):\r\n    return x\r\n")   # CRLF on purpose
    before = p.read_bytes()
    model.set_completer(lambda _: SIDEWAYS_FIX)

    result = run_session(str(tmp_path), str(p))

    assert result.kept is False
    assert p.read_bytes() == before


def test_a_model_that_proposes_nothing_touches_no_files(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    _write(p, UNTYPED)
    before = p.read_bytes()
    model.set_completer(lambda _: UNTYPED)

    result = run_session(str(tmp_path), str(p))

    assert result.kept is False
    assert "unchanged" in result.reason
    assert p.read_bytes() == before


def test_the_file_is_restored_if_measuring_blows_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between write and verdict must not leave a modified source tree."""
    p = tmp_path / "a.py"
    _write(p, UNTYPED)
    before = p.read_bytes()
    model.set_completer(lambda _: GOOD_FIX)

    calls = {"n": 0}
    real = session.run_mypy

    def flaky(target: str, strict: bool = True) -> object:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("measurement exploded")
        return real(target, strict)

    monkeypatch.setattr(session, "run_mypy", flaky)

    with pytest.raises(RuntimeError, match="exploded"):
        run_session(str(tmp_path), str(p))

    assert p.read_bytes() == before


def test_the_session_measures_the_whole_target_not_just_the_file(tmp_path: Path) -> None:
    """A fix in one file can create errors in another. A session that only checked
    its own file would export the mess to a neighbour and report success."""
    p = tmp_path / "a.py"
    _write(p, UNTYPED)
    _write(tmp_path / "other.py", "def g(y):\n    return y\n")
    model.set_completer(lambda _: GOOD_FIX)

    result = run_session(str(tmp_path), str(p))

    assert result.before.total >= 2      # both files counted, not just a.py
