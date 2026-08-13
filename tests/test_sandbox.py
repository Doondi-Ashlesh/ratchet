"""Tests for the isolated workspace.

The claim being tested is narrow and load-bearing: while a session runs, the
caller's files do not change, and after a rejected session there is nothing to
undo. Everything else here is in service of that.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ratchet import sandbox

BODY = "def f(x):\n    return x\n"


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8", newline="")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
         "commit", "-qm", "base"],
        check=True,
    )
    return tmp_path


def test_the_sandbox_is_a_separate_copy(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"pkg/a.py": BODY})

    with sandbox.workspace(str(root / "pkg")) as box:
        inner = Path(box.inside(str(root / "pkg" / "a.py")))
        assert inner.exists()
        assert inner != root / "pkg" / "a.py"
        assert inner.read_text(encoding="utf-8") == BODY


def test_editing_the_sandbox_does_not_touch_the_original(tmp_path: Path) -> None:
    """The whole point. Before this existed the agent edited the caller's file and
    the session put it back, which is only safe if nothing goes wrong in between."""
    root = _repo(tmp_path, {"pkg/a.py": BODY})
    real = root / "pkg" / "a.py"

    with sandbox.workspace(str(root / "pkg")) as box:
        Path(box.inside(str(real))).write_text("wrecked\n", encoding="utf-8")
        assert real.read_text(encoding="utf-8") == BODY

    assert real.read_text(encoding="utf-8") == BODY


def test_promote_copies_the_work_back(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"pkg/a.py": BODY})
    real = root / "pkg" / "a.py"
    fixed = "def f(x: int) -> int:\n    return x\n"

    with sandbox.workspace(str(root / "pkg")) as box:
        Path(box.inside(str(real))).write_text(fixed, encoding="utf-8", newline="")
        box.promote(str(real))

    assert real.read_text(encoding="utf-8") == fixed


def test_promote_preserves_bytes_exactly(tmp_path: Path) -> None:
    """Copied in binary. Re-encoding on the way back would turn a three-line change
    into a whole-file diff on a repository that does not use this platform's
    line endings."""
    crlf = "def f(x):\r\n    return x\r\n"
    root = _repo(tmp_path, {"pkg/a.py": crlf})
    real = root / "pkg" / "a.py"

    with sandbox.workspace(str(root / "pkg")) as box:
        inner = Path(box.inside(str(real)))
        inner.write_bytes(inner.read_bytes().replace(b"f(x)", b"f(x: int)"))
        box.promote(str(real))

    assert real.read_bytes() == crlf.replace("f(x)", "f(x: int)").encode()


def test_uncommitted_work_is_carried_into_the_sandbox(tmp_path: Path) -> None:
    """The agent should see the code as it is on disk, not as it was at the last
    commit. Otherwise it fixes a version of the file nobody is looking at."""
    root = _repo(tmp_path, {"pkg/a.py": BODY})
    real = root / "pkg" / "a.py"
    real.write_text("def g(y):\n    return y\n", encoding="utf-8", newline="")

    with sandbox.workspace(str(root / "pkg")) as box:
        assert "def g(y)" in Path(box.inside(str(real))).read_text(encoding="utf-8")


def test_the_sandbox_is_removed_afterwards(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"pkg/a.py": BODY})

    with sandbox.workspace(str(root / "pkg")) as box:
        location = box.root
    assert not location.exists()


def test_the_sandbox_is_removed_even_when_the_body_raises(tmp_path: Path) -> None:
    """A sandbox that survives a crash is a mess in a temp directory that nobody
    will attribute to this."""
    root = _repo(tmp_path, {"pkg/a.py": BODY})
    location = None

    with pytest.raises(RuntimeError, match="boom"), sandbox.workspace(str(root / "pkg")) as box:
        location = box.root
        raise RuntimeError("boom")

    assert location is not None
    assert not location.exists()


def test_a_target_outside_a_repo_is_refused(tmp_path: Path) -> None:
    """Refused loudly rather than silently falling back to editing in place. A
    fallback that quietly removes the isolation is worse than no isolation, because
    nobody would know which one they got."""
    plain = tmp_path / "loose"
    plain.mkdir()
    (plain / "a.py").write_text(BODY, encoding="utf-8")

    with pytest.raises(sandbox.NotAGitRepo), sandbox.workspace(str(plain)):
        pass
