"""One session: fix one file, or leave the repo exactly as it was found.

The loop is measure -> propose -> apply -> re-measure -> judge -> keep or revert.
Every step except `propose` is deterministic, which is what makes the verdict
mechanical rather than a matter of opinion.

Two decisions worth knowing about.

The whole target is re-measured, not just the edited file. A change in one file
can create errors in another, and a session that only checked its own file would
happily export its mess to a neighbour and report success.

The edit is applied in place and reverted on rejection, rather than being tested
in a scratch copy. mypy resolves imports differently for an isolated file than for
one inside its package, so a scratch measurement would answer a question nobody
asked. The cost is a window where the file on disk is modified: the revert runs in
a `finally`, and the target is a git repo, so `git checkout` is the backstop if
the process is killed outright.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ratchet.agent import propose
from ratchet.classify import Category, actionable, from_result
from ratchet.gate import Measurement, judge
from ratchet.tools import run_mypy, write_file

_NO_CHANGE: Mapping[str, int] = {c.value: 0 for c in Category}


@dataclass(frozen=True)
class SessionResult:
    """What one session did. `kept` is the only field the loop must branch on."""

    path: str
    kept: bool
    reason: str
    before: Measurement
    after: Measurement
    deltas: Mapping[str, int]


def _same_file(a: str, b: str) -> bool:
    """mypy reports absolute paths; callers pass whatever they typed."""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return a == b


def _unchanged(path: str, reason: str, before: Measurement) -> SessionResult:
    return SessionResult(path, False, reason, before, before, _NO_CHANGE)


def run_session(target: str, path: str) -> SessionResult:
    """Attempt one file. Returns without touching disk unless the gate accepts.

    `target` is what gets measured (usually the package root); `path` is the single
    file the agent may edit. Choosing which file is the loop's job, not this one's.
    """
    before_diags = from_result(run_mypy(target))
    before = Measurement.of(before_diags)

    work = [c for c in actionable(before_diags) if _same_file(c.file, path)]
    if not work:
        return _unchanged(path, "no annotation work in this file", before)

    proposal = propose(path, work)
    if not proposal.changed:
        return _unchanged(path, proposal.note or "no change proposed", before)

    written = write_file(path, proposal.proposed)
    if not written.ok:
        return _unchanged(path, f"{written.error_type}: {written.error_message}", before)

    try:
        after = Measurement.of(from_result(run_mypy(target)))
        verdict = judge(before, after, Category.ANNOTATION)

        if not verdict.accepted:
            write_file(path, proposal.original)
            return SessionResult(path, False, verdict.reason, before, before, _NO_CHANGE)

        return SessionResult(path, True, verdict.reason, before, after, verdict.deltas)

    except BaseException:
        # Never leave a modified file behind because measuring blew up. The revert
        # is unconditional here; the caller can retry from a known state.
        write_file(path, proposal.original)
        raise
