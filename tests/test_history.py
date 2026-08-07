"""Tests for cross-run memory.

The store is what stops the loop rediscovering the same failure every run. Its
load path is deliberately forgiving — a missing or corrupt file is an empty
history, never an error — because a first run on a fresh repo is the normal case
and a harness that refuses to start over an unreadable cache is useless.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ratchet import history


def _git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def test_a_missing_store_is_an_empty_history(tmp_path: Path) -> None:
    h = history.load(str(tmp_path))

    assert h.version == history.SCHEMA_VERSION
    assert h.files == {}


def test_failures_accumulate_and_a_success_clears_the_streak() -> None:
    h = history.History()

    h.record("a.py", kept=False, attempts=3, reason="regressed: unknown +2")
    h.record("a.py", kept=False, attempts=3, reason="regressed: config +1")

    assert h.files["a.py"].failures == 2
    assert h.files["a.py"].attempts == 6

    h.record("a.py", kept=True, attempts=1, reason="annotation 5 -> 4")

    assert h.files["a.py"].failures == 0        # the streak, not the total
    assert h.files["a.py"].accepted == 1
    assert h.files["a.py"].last_status == "kept"


def test_a_file_is_blocked_only_after_enough_failures() -> None:
    h = history.History()
    h.record("a.py", kept=False, attempts=3, reason="regressed: unknown +2")

    assert h.blocked("a.py", max_failures=2) == ""

    h.record("a.py", kept=False, attempts=3, reason="regressed: unknown +2")

    why = h.blocked("a.py", max_failures=2)
    assert "failed 2x" in why
    assert "unknown" in why                      # the reason travels with the block


def test_an_unseen_file_is_never_blocked() -> None:
    assert history.History().blocked("never-seen.py", max_failures=1) == ""


def test_a_round_trip_survives_the_filesystem(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    h = history.History()
    h.record("pkg/a.py", kept=False, attempts=2, reason="regressed: defect +1")

    written = history.save(str(tmp_path), h)
    back = history.load(str(tmp_path))

    assert written.exists()
    assert back.files["pkg/a.py"].failures == 1
    assert back.files["pkg/a.py"].last_reason == "regressed: defect +1"


def test_state_lives_at_the_git_root_not_beside_the_target(tmp_path: Path) -> None:
    """It has to be committable and reviewable in a diff, which means it belongs
    with the repository rather than next to whatever subdirectory was measured."""
    _git_repo(tmp_path)
    pkg = tmp_path / "src" / "pkg"
    pkg.mkdir(parents=True)

    assert history.path_for(str(pkg)) == tmp_path / ".ratchet" / "state.json"


def test_an_unknown_schema_version_is_discarded_not_guessed_at(tmp_path: Path) -> None:
    """Fields can change meaning between versions. Reading them anyway is how a
    harness silently starts acting on state it does not understand."""
    d = tmp_path / ".ratchet"
    d.mkdir()
    (d / "state.json").write_text(
        json.dumps({"version": 999, "files": {"a.py": {"failures": 47}}}), encoding="utf-8"
    )

    assert history.load(str(tmp_path)).files == {}


def test_corrupt_json_does_not_stop_a_run(tmp_path: Path) -> None:
    d = tmp_path / ".ratchet"
    d.mkdir()
    (d / "state.json").write_text("{ this is not json", encoding="utf-8")

    assert history.load(str(tmp_path)).files == {}
