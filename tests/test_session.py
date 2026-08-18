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
from fake_model import FunctionModel

from ratchet import llm, session
from ratchet.session import run_session

UNTYPED = "def f(x):\n    return x\n"
GOOD_FIX = "def f(x: int) -> int:\n    return x\n"
CLEAN = "x: int = 1\n"

# Fixes the annotation error and introduces an undefined name in its place:
# annotation 1 -> 0, unknown 0 -> 1, total unchanged. The gate must refuse this.
# Annotation-only on purpose. It has to reach the GATE, and rewriting a
# statement no longer does: the guard refuses anything that is not an annotation
# edit before a measurement is ever taken. `Nope` is undefined, so this trades an
# annotation error for a name error, which is exactly the trade the gate exists
# to refuse.
SIDEWAYS_FIX = "def f(x: Nope) -> Nope:\n    return x\n"


@pytest.fixture(autouse=True)
def _never_call_a_live_model() -> Iterator[None]:
    yield
    llm.set_model(None)


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
    llm.set_model(FunctionModel(fn=lambda _: GOOD_FIX))

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
    llm.set_model(FunctionModel(fn=lambda _: SIDEWAYS_FIX))

    result = run_session(str(tmp_path), str(p))

    assert result.kept is False
    assert p.read_bytes() == before
    # A rejected attempt is evidence. Discarding it leaves nobody able to say what
    # the model actually wrote, which is what the next session needs to be told.
    # Compared with endings normalised: the fixture is CRLF, so the proposal was
    # correctly converted to match it, which is the behaviour tested in test_agent.
    assert result.proposed.replace("\r\n", "\n").strip() == SIDEWAYS_FIX.strip()
    assert result.deltas != {c: 0 for c in result.deltas}


def test_a_model_that_proposes_nothing_touches_no_files(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    _write(p, UNTYPED)
    before = p.read_bytes()
    llm.set_model(FunctionModel(fn=lambda _: UNTYPED))

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
    llm.set_model(FunctionModel(fn=lambda _: GOOD_FIX))

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
    llm.set_model(FunctionModel(fn=lambda _: GOOD_FIX))

    result = run_session(str(tmp_path), str(p))

    assert result.before.total >= 2      # both files counted, not just a.py

def test_a_rejected_attempt_is_retried_with_feedback(tmp_path: Path) -> None:
    """The whole point: the model is 80% right and misses one consistent thing.
    Discarding the attempt throws away the only information that would fix it."""
    p = tmp_path / "a.py"
    _write(p, UNTYPED)
    prompts: list[str] = []

    def completer(prompt: str) -> str:
        prompts.append(prompt)
        return SIDEWAYS_FIX if len(prompts) == 1 else GOOD_FIX

    llm.set_model(FunctionModel(fn=completer))

    result = run_session(str(tmp_path), str(p))

    assert result.kept is True
    assert result.attempts == 2
    assert len(result.history) == 1                 # one rejection, recorded
    assert "REJECTED" in prompts[1]                 # attempt 2 was told it failed
    assert "Nope" in prompts[1]                     # and told the specific error


def test_each_attempt_starts_from_the_original_file(tmp_path: Path) -> None:
    """Attempt 2 must not be asked to patch attempt 1's broken output. Compounding
    attempts produce errors nobody can attribute, and the feedback would describe
    a file that no longer exists."""
    p = tmp_path / "a.py"
    _write(p, UNTYPED)
    prompts: list[str] = []

    def completer(prompt: str) -> str:
        prompts.append(prompt)
        return SIDEWAYS_FIX if len(prompts) == 1 else GOOD_FIX

    llm.set_model(FunctionModel(fn=completer))

    run_session(str(tmp_path), str(p))

    file_section = prompts[1].split("--- BEGIN")[1]
    assert "return x" in file_section               # the original
    assert "return nope" not in file_section        # not attempt 1's output


def test_repeated_failure_exhausts_and_escalates(tmp_path: Path) -> None:
    """An unbounded repair loop is an unbounded bill. Exhaustion is an escalation
    with a record of what was tried, not a silent give-up."""
    p = tmp_path / "a.py"
    _write(p, UNTYPED)
    before = p.read_bytes()

    def completer(prompt: str) -> str:
        return SIDEWAYS_FIX

    llm.set_model(FunctionModel(fn=completer))

    result = run_session(str(tmp_path), str(p), max_attempts=2)

    assert result.kept is False
    assert result.attempts == 2
    assert "exhausted" in result.reason
    assert len(result.history) == 2                 # every rejection kept
    assert p.read_bytes() == before                 # and the file is untouched
