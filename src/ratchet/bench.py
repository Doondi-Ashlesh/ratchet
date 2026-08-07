"""Measure the base rate before scaling it.

Orchestration is a multiplier. Dispatching an agent across twenty files without
knowing its per-file success rate produces an expensive way to fail nineteen
times, and that failure is indistinguishable from the harness being broken.

This runs one session per file and restores the file afterwards regardless of the
verdict, so every trial starts from the same baseline and files are measured
independently. Keeping an accepted fix would move the baseline for every
subsequent trial and make the numbers unreadable.

Requires the target to sit inside a clean git working tree. That is not
politeness: the restore is `git checkout`, so uncommitted work would be destroyed
by it.
"""
from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ratchet import model
from ratchet.classify import actionable, from_result
from ratchet.session import run_session
from ratchet.tools import run_mypy


class NotAGitRepo(RuntimeError):
    """The restore mechanism is `git checkout`, so there has to be a git."""


class DirtyTree(RuntimeError):
    """Uncommitted work would be destroyed by the per-trial restore."""


@dataclass(frozen=True)
class Trial:
    """One file, one session, then put back exactly as it was found."""

    file: str
    codes: tuple[str, ...]
    errors: int
    kept: bool
    attempts: int
    reason: str
    seconds: float
    tokens_in: int
    tokens_out: int


@dataclass(frozen=True)
class BenchReport:
    target: str
    trials: tuple[Trial, ...]
    seconds: float

    @property
    def accepted(self) -> int:
        return sum(1 for t in self.trials if t.kept)

    @property
    def accept_rate(self) -> float:
        return self.accepted / len(self.trials) if self.trials else 0.0

    def by_code(self) -> dict[str, tuple[int, int]]:
        """code -> (accepted, total). A file counts once per distinct code it contains.

        This is the number that decides whether the agent should be dispatched at
        all for a given class of error, and it cannot be read off an overall rate.
        """
        out: dict[str, tuple[int, int]] = {}
        for t in self.trials:
            for code in t.codes:
                a, n = out.get(code, (0, 0))
                out[code] = (a + (1 if t.kept else 0), n + 1)
        return dict(sorted(out.items(), key=lambda kv: (-kv[1][1], kv[0])))

    def rejections(self) -> dict[str, int]:
        """Rejection reasons coarsened to their leading phrase, most common first."""
        out: dict[str, int] = {}
        for t in self.trials:
            if t.kept:
                continue
            key = t.reason.split(":")[0].strip()[:40]
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    return proc.stdout.strip()


def _repo_root(target: str) -> Path:
    start = Path(target).resolve()
    root = _git(start if start.is_dir() else start.parent, "rev-parse", "--show-toplevel")
    if not root:
        raise NotAGitRepo(f"{target} is not inside a git repository")
    return Path(root)


def _require_clean(root: Path) -> None:
    dirty = _git(root, "status", "--porcelain")
    if dirty:
        raise DirtyTree(
            f"{root} has uncommitted changes; the per-trial restore would destroy "
            f"them:\n{dirty}"
        )


def _targets(target: str, max_files: int) -> list[tuple[str, tuple[str, ...], int]]:
    """Files with annotation work, fewest errors first.

    Smallest first on purpose. A rate dominated by the three hardest files in the
    repo is not a base rate, and cheap trials surface a broken setup before much
    money has been spent.
    """
    by_file: dict[str, list[str]] = {}
    for c in actionable(from_result(run_mypy(target))):
        by_file.setdefault(c.file, []).append(c.code)
    ranked = sorted(by_file.items(), key=lambda kv: (len(kv[1]), kv[0]))
    return [(f, tuple(sorted(set(codes))), len(codes)) for f, codes in ranked[:max_files]]


def run_bench(
    target: str,
    max_files: int = 20,
    max_attempts: int = 3,
    on_trial: Callable[[Trial], None] | None = None,
) -> BenchReport:
    """One session per file, restoring the file after each so trials are independent."""
    root = _repo_root(target)
    _require_clean(root)

    started = time.monotonic()
    trials: list[Trial] = []

    for path, codes, errors in _targets(target, max_files):
        model.reset_usage()
        t0 = time.monotonic()
        try:
            result = run_session(target, path, max_attempts=max_attempts)
        finally:
            # Restore regardless of outcome, including on a crash. Every trial has
            # to start from the same baseline or the numbers mean nothing.
            _git(root, "checkout", "--", path)

        used = model.usage()
        trial = Trial(
            file=str(Path(path).resolve().relative_to(root)),
            codes=codes,
            errors=errors,
            kept=result.kept,
            attempts=result.attempts,
            reason=result.reason,
            seconds=round(time.monotonic() - t0, 1),
            tokens_in=used["input"],
            tokens_out=used["output"],
        )
        trials.append(trial)
        if on_trial is not None:
            on_trial(trial)

    return BenchReport(
        target=target, trials=tuple(trials), seconds=round(time.monotonic() - started, 1)
    )


def format_report(r: BenchReport) -> str:
    lines = [
        f"ratchet bench {r.target}",
        f"  {len(r.trials)} files · {r.seconds}s",
        "",
        f"  accept rate      {r.accepted}/{len(r.trials)}   {r.accept_rate * 100:.0f}%",
    ]

    kept = [t for t in r.trials if t.kept]
    if kept:
        lines.append(
            f"  attempts (kept)  {sum(t.attempts for t in kept) / len(kept):.1f} avg"
        )

    tin = sum(t.tokens_in for t in r.trials)
    tout = sum(t.tokens_out for t in r.trials)
    if tin or tout:
        lines.append(f"  tokens           {tin:,} in / {tout:,} out")

    lines += ["", "  by error code"]
    for code, (a, n) in r.by_code().items():
        lines.append(f"    {code:<22} {a}/{n}   {a / n * 100:.0f}%")

    rejected = r.rejections()
    if rejected:
        lines += ["", "  rejections"]
        for reason, n in rejected.items():
            lines.append(f"    {reason:<40} {n}")

    return "\n".join(lines)
