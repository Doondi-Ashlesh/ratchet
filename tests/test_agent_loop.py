"""Tests for the tool-calling agent loop.

Every test here runs offline against an injected model, because the point of these
is the harness's behaviour when a model does something wrong, and a real model
cannot be asked to reliably misbehave on cue.

The important ones are the refusals. A loop that works when the model cooperates
is easy; what this phase adds is a model that can write to disk, and the tests
that matter are the ones proving what it cannot do.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ratchet import agent, model
from ratchet.classify import Category, Classified

UNTYPED = "def f(x):\n    return x\n"
GOOD = "def f(x: int) -> int:\n    return x\n"
IGNORED = "def f(x):  # type: ignore\n    return x\n"


@pytest.fixture(autouse=True)
def _offline() -> Iterator[None]:
    yield
    model.set_conversant(None)


def _diag(path: str) -> list[Classified]:
    return [Classified(file=path, line=1, code="no-untyped-def", message="missing annotations",
                       category=Category.ANNOTATION, reason="types are missing")]


def _call(name: str, **arguments: Any) -> model.ToolCall:
    return model.ToolCall(id=f"c{name}", name=name, arguments=arguments)


def _script(*replies: model.Reply) -> Any:
    """A model that says these things in order, then stops."""
    turns = iter(replies)

    def conversant(_messages: list[dict[str, Any]]) -> model.Reply:
        return next(turns, model.Reply(content="done"))

    return conversant


def _target(tmp_path: Path, body: str = UNTYPED) -> tuple[str, str]:
    f = tmp_path / "a.py"
    with f.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    return str(tmp_path), str(f)


def test_the_agent_reads_writes_and_stops(tmp_path: Path) -> None:
    target, path = _target(tmp_path)
    model.set_conversant(_script(
        model.Reply(tool_calls=(_call("read_file", path=path),)),
        model.Reply(tool_calls=(_call("write_file", path=path, content=GOOD),)),
        model.Reply(content="annotated"),
    ))

    t = agent.work(target, path, _diag(path))

    assert t.changed
    assert t.stopped == "model"
    assert Path(path).read_text(encoding="utf-8") == GOOD
    assert [s.tool for s in t.steps] == ["read_file", "write_file"]


def test_a_write_the_guard_rejects_never_reaches_disk(tmp_path: Path) -> None:
    """The whole reason the guard moved into the tool. A model that can watch the
    error count fall will reach for `# type: ignore`, and rejecting it after the
    fact costs a session; rejecting it at the write costs a turn."""
    target, path = _target(tmp_path)
    model.set_conversant(_script(
        model.Reply(tool_calls=(_call("write_file", path=path, content=IGNORED),)),
        model.Reply(content="gave up"),
    ))

    t = agent.work(target, path, _diag(path))

    assert Path(path).read_text(encoding="utf-8") == UNTYPED, "the cheat never landed"
    assert not t.changed
    assert t.guard_rejections == 1


def test_a_rejected_write_tells_the_model_why(tmp_path: Path) -> None:
    """A refusal the model cannot act on is just a failed turn. The violation text
    has to reach the transcript."""
    target, path = _target(tmp_path)
    seen: list[str] = []

    def conversant(messages: list[dict[str, Any]]) -> model.Reply:
        seen.extend(str(m.get("content", "")) for m in messages if m.get("role") == "tool")
        if len(seen) < 1:
            return model.Reply(tool_calls=(_call("write_file", path=path, content=IGNORED),))
        return model.Reply(content="understood")

    model.set_conversant(conversant)
    agent.work(target, path, _diag(path))

    assert any("type: ignore" in s for s in seen)


def test_the_agent_cannot_write_outside_the_file_it_was_given(tmp_path: Path) -> None:
    """The session reverts one file. A write anywhere else would survive a rejected
    trajectory."""
    target, path = _target(tmp_path)
    other = tmp_path / "b.py"
    with other.open("w", encoding="utf-8", newline="") as fh:
        fh.write(UNTYPED)

    model.set_conversant(_script(
        model.Reply(tool_calls=(_call("write_file", path=str(other), content=GOOD),)),
        model.Reply(content="stopped"),
    ))

    t = agent.work(target, path, _diag(path))

    assert other.read_text(encoding="utf-8") == UNTYPED
    assert t.steps[0].tool == "write_file" and not t.steps[0].ok
    assert "wrong_file" in t.steps[0].detail


def test_the_agent_cannot_read_outside_the_target(tmp_path: Path) -> None:
    target, path = _target(tmp_path)
    outside = tmp_path.parent / "secrets.py"
    outside.write_text("KEY = 'x'\n", encoding="utf-8")

    model.set_conversant(_script(
        model.Reply(tool_calls=(_call("read_file", path=str(outside)),)),
        model.Reply(content="stopped"),
    ))

    t = agent.work(target, path, _diag(path))

    assert "outside_target" in t.steps[0].detail


def test_malformed_arguments_are_reported_not_raised(tmp_path: Path) -> None:
    """A stray brace should cost a turn, not the trajectory."""
    target, path = _target(tmp_path)
    model.set_conversant(_script(
        model.Reply(tool_calls=(model.ToolCall(id="c1", name="write_file", malformed="bad json"),)),
        model.Reply(content="stopped"),
    ))

    t = agent.work(target, path, _diag(path))

    assert not t.steps[0].ok
    assert "bad_arguments" in t.steps[0].detail
    assert t.stopped == "model"


def test_an_unknown_tool_is_refused(tmp_path: Path) -> None:
    target, path = _target(tmp_path)
    model.set_conversant(_script(
        model.Reply(tool_calls=(_call("rm_rf", path="/"),)),
        model.Reply(content="stopped"),
    ))

    t = agent.work(target, path, _diag(path))

    assert "no_such_tool" in t.steps[0].detail


def test_turns_are_bounded(tmp_path: Path) -> None:
    """A model that never stops is the normal failure of an agent loop, not an
    exotic one."""
    target, path = _target(tmp_path)
    model.set_conversant(lambda _m: model.Reply(tool_calls=(_call("read_file", path=path),)))

    t = agent.work(target, path, _diag(path), max_turns=3)

    assert t.turns == 3
    assert t.stopped == "max_turns"


def test_tool_calls_are_bounded_separately_from_turns(tmp_path: Path) -> None:
    """One turn can request several tools, so a turn budget alone does not bound
    the work."""
    target, path = _target(tmp_path)
    model.set_conversant(lambda _m: model.Reply(
        tool_calls=tuple(_call("read_file", path=path) for _ in range(4))
    ))

    t = agent.work(target, path, _diag(path), max_turns=10, max_calls=5)

    assert t.stopped == "max_calls"
    assert len(t.steps) <= 8


def test_non_annotation_work_is_refused(tmp_path: Path) -> None:
    target, path = _target(tmp_path)
    wrong = [Classified(file=path, line=1, code="assignment", message="bad",
                        category=Category.DEFECT, reason="the code is wrong")]

    with pytest.raises(ValueError, match="ANNOTATION"):
        agent.work(target, path, wrong)


def test_the_trajectory_records_whether_it_checked_its_own_work(tmp_path: Path) -> None:
    """Asking for self-verification in the prompt makes it likely. Recording it is
    how you find out how likely."""
    target, path = _target(tmp_path)
    model.set_conversant(_script(
        model.Reply(tool_calls=(_call("write_file", path=path, content=GOOD),)),
        model.Reply(content="done"),
    ))

    assert not agent.work(target, path, _diag(path)).self_checked


def test_trimming_keeps_the_task_and_drops_whole_exchanges() -> None:
    """A tool result whose call has been dropped is a message the provider rejects,
    so trimming has to remove pairs rather than messages."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
    ]
    for i in range(6):
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}"}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "result"})

    trimmed = agent._trim(messages, keep=4)

    assert trimmed[0]["content"] == "rules"
    assert trimmed[1]["content"] == "task"
    assert trimmed[2]["role"] == "assistant", "never starts with an orphaned tool result"
    assert len(trimmed) < len(messages)


# ── the session wrapper ───────────────────────────────────────────────────────
# The agent writes the file itself, so the property that matters is that a
# trajectory the gate refuses leaves nothing behind.


def _repo(tmp_path: Path, files: dict[str, str]) -> str:
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name, body in files.items():
        p = tmp_path / name
        with p.open("w", encoding="utf-8", newline="") as fh:
            fh.write(body)
    return str(tmp_path)


def test_an_accepted_trajectory_keeps_the_agents_work(tmp_path: Path) -> None:
    from ratchet.session import run_agent_session

    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    model.set_conversant(_script(
        model.Reply(tool_calls=(_call("write_file", path=path, content=GOOD),)),
        model.Reply(content="done"),
    ))

    result = run_agent_session(target, path)

    assert result.kept
    assert Path(path).read_text(encoding="utf-8") == GOOD


def test_a_rejected_trajectory_restores_the_file(tmp_path: Path) -> None:
    """The agent wrote real changes to disk across several turns. A refused verdict
    has to undo all of them, not the last one."""
    from ratchet.session import run_agent_session

    sideways = "import nope\n\ndef f(x: int) -> int:\n    return x\n"
    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    model.set_conversant(_script(
        model.Reply(tool_calls=(_call("write_file", path=path, content=GOOD),)),
        model.Reply(tool_calls=(_call("write_file", path=path, content=sideways),)),
        model.Reply(content="done"),
    ))

    result = run_agent_session(target, path)

    assert not result.kept
    assert Path(path).read_text(encoding="utf-8") == UNTYPED


def test_a_trajectory_that_wrote_nothing_is_not_a_rejection(tmp_path: Path) -> None:
    """No change and a bad change are different outcomes, and collapsing them
    would make the accept rate unreadable."""
    from ratchet.session import run_agent_session

    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    model.set_conversant(_script(model.Reply(content="I looked and did nothing")))

    result = run_agent_session(target, path)

    assert not result.kept
    assert "no change written" in result.reason
    assert Path(path).read_text(encoding="utf-8") == UNTYPED


def test_a_crash_mid_trajectory_still_restores(tmp_path: Path) -> None:
    """The revert is unconditional. A measurement that blows up must not leave the
    agent's edits on disk."""
    from ratchet.session import run_agent_session

    target = _repo(tmp_path, {"a.py": UNTYPED})
    path = str(tmp_path / "a.py")
    calls = {"n": 0}

    def conversant(_m: list[dict[str, Any]]) -> model.Reply:
        calls["n"] += 1
        if calls["n"] == 1:
            return model.Reply(tool_calls=(_call("write_file", path=path, content=GOOD),))
        raise RuntimeError("boom")

    model.set_conversant(conversant)

    with pytest.raises(RuntimeError, match="boom"):
        run_agent_session(target, path)

    assert Path(path).read_text(encoding="utf-8") == UNTYPED
