"""Tests for the multi-session loop.

These drive the real graph with real mypy calls and a faked model, because what
is under test is the orchestration: does the queue drain, does history stop a
known-bad file being dispatched again, does a killed run resume.

The load-bearing test is `test_a_repeatedly_failing_file_is_not_dispatched_again`.
That behaviour is the entire reason this rung exists — one intractable file
consumed a dozen model calls across four runs before anything noticed.
"""
from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from ratchet import history, loop, model

UNTYPED = "def f(x):\n    return x\n"
GOOD = "def f(x: int) -> int:\n    return x\n"
SIDEWAYS = "def f(x: int) -> int:\n    return nope\n"


@pytest.fixture(autouse=True)
def _never_call_a_live_model() -> Iterator[None]:
    yield
    model.set_completer(None)


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            f.write(body)
    return tmp_path


def test_the_queue_drains_and_the_count_falls(tmp_path: Path) -> None:
    _repo(tmp_path, {"a.py": UNTYPED, "b.py": UNTYPED})
    model.set_completer(lambda _: GOOD)

    out = loop.run_loop(str(tmp_path))

    assert out["queue"] == []
    assert len(out["results"]) == 2
    assert all(r["kept"] for r in out["results"])
    assert sum(out["latest"].values()) < sum(out["baseline"].values())


def test_max_files_bounds_the_run(tmp_path: Path) -> None:
    _repo(tmp_path, {"a.py": UNTYPED, "b.py": UNTYPED, "c.py": UNTYPED})
    model.set_completer(lambda _: GOOD)

    out = loop.run_loop(str(tmp_path), max_files=1)

    assert len(out["results"]) == 1


def test_a_repeatedly_failing_file_is_not_dispatched_again(tmp_path: Path) -> None:
    """failure-log 017. Without this, every run rebuilds the same failure from
    zero and pays for it again."""
    _repo(tmp_path, {"a.py": UNTYPED})
    h = history.History()
    h.record("a.py", kept=False, attempts=3, reason="regressed: unknown +2")
    h.record("a.py", kept=False, attempts=3, reason="regressed: unknown +2")
    history.save(str(tmp_path), h)

    calls: list[str] = []
    model.set_completer(lambda p: calls.append(p) or GOOD)

    out = loop.run_loop(str(tmp_path), max_failures=2)

    assert calls == [], "a file with a failure streak was dispatched anyway"
    assert out["results"] == []
    assert out["skipped"][0]["file"] == "a.py"
    assert "failed 2x" in out["skipped"][0]["reason"]


def test_the_block_lifts_below_the_threshold(tmp_path: Path) -> None:
    _repo(tmp_path, {"a.py": UNTYPED})
    h = history.History()
    h.record("a.py", kept=False, attempts=3, reason="regressed: unknown +2")
    history.save(str(tmp_path), h)
    model.set_completer(lambda _: GOOD)

    out = loop.run_loop(str(tmp_path), max_failures=2)

    assert len(out["results"]) == 1


def test_a_run_writes_its_verdicts_to_committed_state(tmp_path: Path) -> None:
    _repo(tmp_path, {"a.py": UNTYPED})
    model.set_completer(lambda _: GOOD)

    loop.run_loop(str(tmp_path))
    h = history.load(str(tmp_path))

    assert h.files["a.py"].accepted == 1
    assert (tmp_path / ".ratchet" / "state.json").is_file()


def test_files_with_no_annotation_work_never_enter_the_queue(tmp_path: Path) -> None:
    _repo(tmp_path, {"clean.py": "x: int = 1\n"})
    calls: list[str] = []
    model.set_completer(lambda p: calls.append(p) or GOOD)

    out = loop.run_loop(str(tmp_path))

    assert calls == []
    assert out["results"] == []


def test_a_run_where_nothing_succeeds_escalates(tmp_path: Path) -> None:
    """Exhausting every file is a result that needs a person, not a silent exit."""
    _repo(tmp_path, {"a.py": UNTYPED})
    model.set_completer(lambda _: SIDEWAYS)

    out = loop.run_loop(str(tmp_path), max_attempts=1)

    assert not any(r["kept"] for r in out["results"])
    assert "could not be fixed automatically" in out["escalation"]


def test_the_worst_file_is_worked_first(tmp_path: Path) -> None:
    """Ordering is part of the contract: the file with the most work first, so a
    bounded run spends its budget where the count moves most."""
    _repo(tmp_path, {
        "one.py": UNTYPED,
        "three.py": "def a(x):\n    return x\ndef b(x):\n    return x\ndef c(x):\n    return x\n",
    })
    model.set_completer(lambda _: GOOD)

    out = loop.run_loop(str(tmp_path), max_files=1)

    assert out["results"][0]["file"] == "three.py"


def test_a_run_resumes_from_its_checkpoint(tmp_path: Path) -> None:
    """The reason this is a graph and not a while loop. State survives the
    process, so a killed run continues rather than starting over."""
    _repo(tmp_path, {"a.py": UNTYPED, "b.py": UNTYPED})
    model.set_completer(lambda _: GOOD)
    db = str(tmp_path / "ckpt.sqlite")

    with loop.make_checkpointer(db) as saver:
        graph = loop.build(checkpointer=saver)
        config = {"configurable": {"thread_id": "t1"}}
        graph.invoke({
            "target": str(tmp_path), "max_attempts": 3, "max_files": 0,
            "max_failures": 2, "queue": [], "results": [], "skipped": [],
            "baseline": {}, "latest": {}, "escalation": "",
        }, config=config)

    with loop.make_checkpointer(db) as saver:
        graph = loop.build(checkpointer=saver)
        snap = graph.get_state({"configurable": {"thread_id": "t1"}})

    assert len(snap.values["results"]) == 2      # the finished run was persisted


def test_history_keys_use_forward_slashes_on_every_platform(tmp_path: Path) -> None:
    """The state file is committed and shared. Keyed with the OS separator, a
    Windows run writes `pkg\a.py` and a Linux run looks up `pkg/a.py`, so the
    same file appears twice and every skip rule quietly stops working."""
    _repo(tmp_path, {"pkg/a.py": UNTYPED})
    model.set_completer(lambda _: GOOD)

    loop.run_loop(str(tmp_path))
    keys = list(history.load(str(tmp_path)).files)

    assert keys == ["pkg/a.py"]
    assert not any("\\" in k for k in keys)


def test_the_deadline_stops_dispatch_and_keeps_what_was_earned(tmp_path: Path) -> None:
    """The whole point of the bound: a run that runs out of time reports its
    results rather than losing them. An outer `timeout` killing the process is
    what this replaces, and that discarded everything."""
    _repo(tmp_path, {"a.py": UNTYPED, "b.py": UNTYPED, "c.py": UNTYPED})

    calls = {"n": 0}

    def slow(_: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            # The first session alone consumes the budget.
            loop.time.sleep(0.05)
        return GOOD

    model.set_completer(slow)

    out = loop.run_loop(str(tmp_path), deadline_s=0.01)

    assert len(out["results"]) == 1, "the in-flight session is never abandoned"
    assert out["results"][0]["kept"]
    assert [s["reason"] for s in out["skipped"]] == ["run deadline reached"] * 2
    assert out["queue"] == []


def test_no_deadline_means_no_bound(tmp_path: Path) -> None:
    """Zero is off, not 'expire immediately' — the default must not silently
    stop a run from doing any work at all."""
    _repo(tmp_path, {"a.py": UNTYPED, "b.py": UNTYPED})
    model.set_completer(lambda _: GOOD)

    out = loop.run_loop(str(tmp_path), deadline_s=0.0)

    assert len(out["results"]) == 2
    assert out["skipped"] == []
