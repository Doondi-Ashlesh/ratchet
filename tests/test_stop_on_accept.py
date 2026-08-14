"""The agent must stop the moment the gate accepts.

Measured before this existed: every file in one benchmark target ran to its step
ceiling AFTER already succeeding, at 669k tokens against 15.8k for a single prompt
producing the same result.

Cost is the smaller half of the argument. The session keeps the file as it stands
when the trajectory ends, not as it stood at its best moment, so every step after
acceptance is a fresh chance to lose it and no chance to gain anything.
"""
from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from fake_model import ScriptedModel, calls, says

from ratchet import llm
from ratchet.session import run_agent_session

UNTYPED = "def f(x):\n    return x\n"
GOOD = "def f(x: int) -> int:\n    return x\n"


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


def test_the_agent_stops_once_the_gate_accepts(tmp_path: Path) -> None:
    """The fourth reply would undo the work. It must never be reached."""
    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")

    model = ScriptedModel(replies=[
        calls(("write_whole_file", {"content": GOOD})),
        calls(("check_work", {})),
        calls(("write_whole_file", {"content": UNTYPED})),   # would throw it away
        says("done"),
    ])
    llm.set_model(model)

    result = run_agent_session(target, path)

    assert result.kept
    assert Path(path).read_text(encoding="utf-8") == GOOD, "the undo never ran"
    assert model.seen <= 2, f"kept calling the model after acceptance ({model.seen} calls)"


def test_a_rejected_trajectory_still_runs_on(tmp_path: Path) -> None:
    """The stop is conditional on acceptance, not on `check_work` being called.
    An agent told REJECTED has to keep working."""
    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    sideways = "import nope\n\n\ndef f(x: int) -> int:\n    return x\n"

    model = ScriptedModel(replies=[
        calls(("write_whole_file", {"content": sideways})),
        calls(("check_work", {})),
        calls(("write_whole_file", {"content": GOOD})),
        calls(("check_work", {})),
        says("done"),
    ])
    llm.set_model(model)

    result = run_agent_session(target, path)

    assert model.seen > 2, "a rejection stopped the agent that should have continued"
    assert result.kept
    assert Path(path).read_text(encoding="utf-8") == GOOD
