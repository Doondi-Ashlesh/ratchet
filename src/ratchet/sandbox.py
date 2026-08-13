"""An isolated copy of the target, so the agent never edits the real thing.

Until this existed, the agent wrote directly into the caller's repository and the
session put it back afterwards. That works when everything goes right, and the
failure modes are all bad: a crash between write and revert leaves the file
modified, a rejected session overwrites uncommitted work in that file, and a
process killed mid-run leaves no trace of what it was doing.

The stronger arrangement is that a rejected session has nothing to undo, because
nothing was ever written where it mattered. The agent works in a git worktree, the
oracle measures that worktree, and the file is copied back only after the gate
accepts. Rejection is a directory that gets deleted.

A worktree rather than a directory copy: git already knows which files belong to
the project, so it costs a checkout rather than a recursive copy of whatever
happens to be sitting in the tree, and there is no `.venv` or `node_modules` to
exclude by hand.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class NotAGitRepo(RuntimeError):
    """The sandbox is a git worktree, so there has to be a git."""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, check=False,
    )


def repo_root(target: str) -> Path:
    result = _git(Path(target).resolve(), "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        raise NotAGitRepo(f"{target} is not inside a git repository")
    return Path(result.stdout.strip()).resolve()


@dataclass(frozen=True)
class Sandbox:
    """A throwaway checkout. `target` and `inside()` are the sandbox's own paths."""

    root: Path
    origin: Path
    target: str

    def inside(self, path: str) -> str:
        """Map a real path to its counterpart in the sandbox."""
        return str(self.root / Path(path).resolve().relative_to(self.origin))

    def promote(self, path: str) -> None:
        """Copy the sandbox's version of `path` over the real one.

        Byte for byte, in binary. Reading and rewriting as text would re-encode
        line endings, which is how a "promotion" turns a three-line change into a
        whole-file diff on a repository that does not use this platform's endings.
        """
        source = Path(self.inside(path))
        destination = Path(path).resolve()
        destination.write_bytes(source.read_bytes())


@contextmanager
def workspace(target: str) -> Iterator[Sandbox]:
    """Check the target's repository out somewhere disposable.

    Uncommitted changes to tracked files are carried across, because the agent
    should see the code as it is on disk rather than as it was at the last commit.
    Untracked files are not: they are usually build output or scratch, and a
    sandbox that copies them is a directory copy wearing a worktree's clothes.

    The worktree is removed on the way out whatever happened, including on a crash.
    A sandbox that survives a failure is just a mess in a temp directory that
    nobody will attribute to this.
    """
    origin = repo_root(target)
    root = Path(tempfile.gettempdir()) / f"ratchet-{uuid.uuid4().hex[:12]}"

    created = _git(origin, "worktree", "add", "--detach", "--quiet", str(root), "HEAD")
    if created.returncode != 0:
        raise NotAGitRepo(f"could not create a worktree from {origin}: {created.stderr.strip()}")

    try:
        # Carry over uncommitted work. Applied as a patch rather than checked out,
        # so the sandbox starts from the same bytes the caller is looking at.
        diff = _git(origin, "diff", "HEAD")
        if diff.stdout.strip():
            subprocess.run(
                ["git", "-C", str(root), "apply", "--whitespace=nowarn", "-"],
                input=diff.stdout, text=True, capture_output=True, check=False,
            )

        relative = Path(target).resolve().relative_to(origin)
        yield Sandbox(root=root, origin=origin, target=str(root / relative))

    finally:
        _git(origin, "worktree", "remove", "--force", str(root))
        # `worktree remove` refuses in some states and leaves the directory behind.
        # The registration is pruned either way, so a stale entry cannot accumulate
        # and make the next `worktree add` fail.
        shutil.rmtree(root, ignore_errors=True)
        _git(origin, "worktree", "prune")
