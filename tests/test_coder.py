"""Tests for the agent, now that the loop itself is LangGraph's.

What is worth testing changed with the swap. The turn counting, transcript
assembly and tool dispatch are the framework's problem and are covered upstream.
What is ours is the part around the loop: that a write cannot reach disk without
passing the guard, that the agent is confined to one file, that it sees the real
verdict, and that a rejected trajectory is undone completely.
"""
from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from fake_model import ScriptedModel, calls, says

from ratchet import llm
from ratchet.classify import Category, Classified
from ratchet.gate import Measurement
from ratchet.session import run_agent_session

UNTYPED = "def f(x):\n    return x\n"
GOOD = "def f(x: int) -> int:\n    return x\n"
SIDEWAYS = "import nope\n\n\ndef f(x: int) -> int:\n    return x\n"


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    yield
    llm.set_model(None)


def _repo(tmp_path: Path, files: dict[str, str]) -> str:
    """A git repository with a commit in it.

    The commit is not ceremony. Sessions run inside a worktree checked out from
    HEAD, and a repository with no commits has no HEAD to check out, so an
    uncommitted fixture would be testing a configuration the tool refuses.
    """
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


def _diag(path: str) -> list[Classified]:
    return [Classified(file=path, line=1, code="no-untyped-def", message="missing annotations",
                       category=Category.ANNOTATION, reason="types are missing")]


def _script(*replies: object) -> None:
    llm.set_model(ScriptedModel(replies=list(replies)))  # type: ignore[arg-type]


# ── the session contract ──────────────────────────────────────────────────────


def test_accepted_work_is_kept(tmp_path: Path) -> None:
    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    _script(calls(("write_whole_file", {"content": GOOD})), says())

    result = run_agent_session(target, path)

    assert result.kept
    assert Path(path).read_text(encoding="utf-8") == GOOD


def test_a_rejected_trajectory_is_undone_completely(tmp_path: Path) -> None:
    """The agent wrote real changes across several steps. A refused verdict has to
    undo all of them, not just the last one."""
    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    _script(
        calls(("write_whole_file", {"content": GOOD})),
        calls(("write_whole_file", {"content": SIDEWAYS})),
        says(),
    )

    result = run_agent_session(target, path)

    assert not result.kept
    assert Path(path).read_text(encoding="utf-8") == UNTYPED


def test_writing_nothing_is_not_the_same_as_writing_badly(tmp_path: Path) -> None:
    """Collapsing the two would make the accept rate unreadable."""
    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    _script(says("I looked and did nothing"))

    result = run_agent_session(target, path)

    assert not result.kept
    assert "no change written" in result.reason
    assert Path(path).read_text(encoding="utf-8") == UNTYPED


def test_a_model_failure_does_not_leave_unverified_edits(tmp_path: Path) -> None:
    """A crash mid-trajectory ends this file, not the whole run - but nothing the
    gate has not approved may survive it."""
    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    llm.set_model(ScriptedModel(
        replies=[calls(("write_whole_file", {"content": SIDEWAYS}))], explode_after=1,
    ))

    result = run_agent_session(target, path)

    assert not result.kept
    assert Path(path).read_text(encoding="utf-8") == UNTYPED


# ── what the tools refuse ─────────────────────────────────────────────────────


def test_the_guard_stops_a_write_before_it_reaches_disk(tmp_path: Path) -> None:
    """A model that can watch the error count fall will reach for `# type: ignore`.
    Refusing at the write costs a step; refusing afterwards costs the session."""
    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    cheat = "def f(x):  # type: ignore\n    return x\n"
    _script(calls(("write_whole_file", {"content": cheat})), says())

    result = run_agent_session(target, path)

    assert Path(path).read_text(encoding="utf-8") == UNTYPED, "the cheat never landed"
    assert not result.kept


def test_an_ambiguous_edit_is_refused(tmp_path: Path) -> None:
    """Uniqueness is the safety property: a patch matching twice lands in the wrong
    place about half the time, and usually still parses."""
    body = "def f(x):\n    return x\n\n\ndef g(y):\n    return x\n"
    target = _repo(tmp_path, {"a.py": body})
    path = str(tmp_path / "a.py")
    _script(
        calls(("edit_file", {"old_string": "    return x", "new_string": "    return 0"})),
        says(),
    )

    run_agent_session(target, path)

    assert Path(path).read_text(encoding="utf-8") == body


def test_an_anchored_edit_lands(tmp_path: Path) -> None:
    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    _script(
        calls(("edit_file", {"old_string": "def f(x):", "new_string": "def f(x: int) -> int:"})),
        says(),
    )

    result = run_agent_session(target, path)

    assert result.kept
    assert Path(path).read_text(encoding="utf-8") == GOOD


def test_check_work_reports_the_real_verdict(tmp_path: Path) -> None:
    """The bug this closes: the agent used to see only its own file's errors while
    being judged on the whole package."""
    from ratchet.agent_tools import Session, build_tools

    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    original = Path(path).read_text(encoding="utf-8")
    session = Session(target, path, original, Measurement.of(_diag(path)))
    tools = {t.name: t for t in build_tools(session)}

    tools["write_whole_file"].invoke({"content": SIDEWAYS})
    verdict = str(tools["check_work"].invoke({}))

    assert "REJECTED" in verdict
    assert "categories_that_rose" in verdict
    assert session.self_checked


def test_the_agent_cannot_name_another_file(tmp_path: Path) -> None:
    """The tools take no path at all, so there is no argument to point somewhere
    else. The session reverts one file; a write anywhere else would survive it."""
    from ratchet.agent_tools import Session, build_tools

    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    session = Session(target, path, UNTYPED, Measurement.of([]))

    for tool in build_tools(session):
        assert "path" not in tool.args, f"{tool.name} exposes a path argument"


def test_reasoning_is_off_by_default() -> None:
    """A reasoning model returns its deliberation as content with no tool call, and
    an agent loop reads that as 'finished'. Measured live: three files in a row
    stopped after two steps having written nothing."""
    from ratchet import llm

    assert llm.thinking_enabled() is False


def test_reasoning_can_be_turned_back_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whether deliberation helps on this task is measurable, so it stays a setting
    rather than becoming a hard-coded assumption."""
    from ratchet import llm

    monkeypatch.setenv("RATCHET_THINKING", "on")
    assert llm.thinking_enabled() is True


def test_an_agent_that_writes_nothing_is_told_to_act(tmp_path: Path) -> None:
    """The loop should not depend on a provider setting to notice no work was done.

    A model that narrates its plan instead of executing it produces a message with
    no tool call, which an agent loop reads as "finished". Measured live: three
    files in a row stopped after two steps having written nothing.
    """
    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    model = ScriptedModel(replies=[says("I would add annotations to f.")])
    llm.set_model(model)

    result = run_agent_session(target, path)

    assert model.seen > 1, "it was pushed to act rather than taken at its word"
    assert not result.kept
    assert Path(path).read_text(encoding="utf-8") == UNTYPED


def test_the_model_gets_a_workable_timeout_and_output_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both settings were lost in the library swap and cost 30 of 80 calls in one
    benchmark, which surfaced as sessions that ran ten minutes, reported zero
    tokens and wrote nothing.

    The timeout lives on a private client attribute because there is no
    constructor path to it. Asserted here so that a future version moving it fails
    the suite rather than quietly losing a third of the next run's calls.
    """
    from ratchet import llm

    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    llm.set_model(None)

    model = llm.chat_model()

    assert model._client.timeout == llm.timeout_s()  # type: ignore[attr-defined]
    assert model._client.timeout >= 300  # type: ignore[attr-defined]
    assert model.max_tokens == llm.max_tokens()  # type: ignore[attr-defined]
    assert model.max_tokens >= 8192  # type: ignore[attr-defined]
