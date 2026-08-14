"""Tests for how the agent is paced: verify between edits, and stop when spinning.

Both rules come from one measured trajectory. It made 16 edits against 2
`check_work` calls, looped to the recursion limit, and spent 389k tokens producing
no change at all. An agent that edits without verifying cannot discover it is off
course until its budget is gone, and one that keeps checking an unmoved number is
only paying to be told the same thing again.

The prompt already asked for both. It was ignored. These are enforced in the tool
and in the loop, because a request is not a constraint.
"""
from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from fake_model import ScriptedModel, calls, says

from ratchet import llm
from ratchet.agent_tools import EDITS_BEFORE_CHECK, Session, build_tools
from ratchet.gate import Measurement
from ratchet.session import run_agent_session

UNTYPED = "def f(a):\n    return a\n\n\ndef g(b):\n    return b\n\n\ndef h(c):\n    return c\n"
GOOD = "def f(a: int) -> int:\n    return a\n"


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    yield
    llm.set_model(None)


def _repo(tmp_path: Path, files: dict[str, str]) -> str:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            f.write(body)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
         "commit", "-qm", "base"],
        check=True,
    )
    return str(tmp_path)


def _tools(tmp_path: Path) -> tuple[Session, dict[str, object]]:
    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    session = Session(target, path, UNTYPED, Measurement.of([]))
    return session, {t.name: t for t in build_tools(session)}


def test_edits_are_refused_until_the_agent_verifies(tmp_path: Path) -> None:
    """The budget is spent, and the next edit is refused with a reason it can act
    on rather than silently ignored."""
    session, tools = _tools(tmp_path)

    for i, name in enumerate(["def f(a):", "def g(b):", "def h(c):"]):
        result = str(tools["edit_file"].invoke(
            {"old_string": name, "new_string": name.replace("):", ": int) -> int:")}
        ))
        assert '"ok": true' in result, f"edit {i} should have landed: {result}"

    assert session.must_check
    blocked = str(tools["edit_file"].invoke(
        {"old_string": "    return a", "new_string": "    return int(a)"}
    ))

    assert '"ok": false' in blocked
    assert "verify_first" in blocked
    assert "check_work" in blocked


def test_checking_restores_the_edit_budget(tmp_path: Path) -> None:
    """The rule is a pacing constraint, not a cap on how much work is allowed."""
    session, tools = _tools(tmp_path)
    session.edits_since_check = EDITS_BEFORE_CHECK
    assert session.must_check

    tools["check_work"].invoke({})

    assert not session.must_check
    landed = str(tools["edit_file"].invoke(
        {"old_string": "def f(a):", "new_string": "def f(a: int) -> int:"}
    ))
    assert '"ok": true' in landed


def test_a_refused_edit_does_not_reach_the_file(tmp_path: Path) -> None:
    """A refusal that still wrote would be worse than no rule at all."""
    session, tools = _tools(tmp_path)
    session.edits_since_check = EDITS_BEFORE_CHECK

    tools["edit_file"].invoke({"old_string": "def f(a):", "new_string": "def f(a: int) -> int:"})

    assert Path(session.path).read_text(encoding="utf-8") == UNTYPED


def test_an_unmoved_error_count_across_checks_is_stalled(tmp_path: Path) -> None:
    """Judged on the annotation count, not the verdict: a trajectory can be
    rejected for several different reasons while still making progress."""
    session, _ = _tools(tmp_path)

    session.checks = [5, 4, 3]
    assert not session.stalled, "a falling count is progress"

    session.checks = [3, 3, 3]
    assert session.stalled

    session.accepted = True
    assert not session.stalled, "acceptance is not stalling"


def test_a_stalled_trajectory_stops_instead_of_looping(tmp_path: Path) -> None:
    """The 389k-token file. Repeated checks on an unmoved number is the agent
    saying it has run out of ideas; continuing only pays to hear it again."""
    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")

    # Edit, check, forever. Nothing it writes changes the annotation count.
    script = []
    for _ in range(12):
        script.append(calls(("edit_file", {"old_string": "return a", "new_string": "return  a"})))
        script.append(calls(("check_work", {})))
    script.append(says("done"))

    model = ScriptedModel(replies=script)
    llm.set_model(model)

    result = run_agent_session(target, path)

    assert not result.kept
    assert model.seen < len(script), "it ran the whole script instead of stopping"
