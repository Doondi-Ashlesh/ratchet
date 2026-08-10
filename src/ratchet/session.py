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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ratchet.agent import propose
from ratchet.classify import Category, Classified, actionable, from_result
from ratchet.coder import work as agent_work
from ratchet.gate import Measurement, Verdict, judge
from ratchet.guard import check as guard_check
from ratchet.tools import read_file, run_mypy, write_file

_NO_CHANGE: Mapping[str, int] = {c.value: 0 for c in Category}


@dataclass(frozen=True)
class SessionResult:
    """What one session did. `kept` is the only field the loop must branch on.

    On rejection the repo is left unchanged, but `after` and `deltas` still
    describe what the attempt WOULD have done, and `proposed` holds what it wrote.
    A rejected attempt is evidence, not garbage: the next session has to be told
    what it did wrong, and a human debugging a refusal cannot work from a verdict
    alone. Reporting zeroes here once produced a result whose reason cited a delta
    its own deltas did not show.
    """

    path: str
    kept: bool
    reason: str
    before: Measurement
    after: Measurement
    deltas: Mapping[str, int]
    proposed: str = ""
    attempts: int = 1
    history: tuple[str, ...] = ()


def _same_file(a: str, b: str) -> bool:
    """mypy reports absolute paths; callers pass whatever they typed."""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return a == b


def _unchanged(path: str, reason: str, before: Measurement) -> SessionResult:
    return SessionResult(path, False, reason, before, before, _NO_CHANGE)


def _regressions(before: Sequence[Classified], after: Sequence[Classified]) -> list[Classified]:
    """Errors present after an attempt that were not there before.

    Keyed on (file, code, message) rather than line: an edit shifts line numbers,
    so including the line would make every surviving error look new.
    """
    seen = {(c.file, c.code, c.message) for c in before}
    return [c for c in after if (c.file, c.code, c.message) not in seen]


def _feedback(verdict: Verdict, regressions: Sequence[Classified]) -> str:
    """What to tell the next attempt.

    The verdict alone ("regressed: unknown +3") names a category, not a mistake.
    The specific diagnostics name the mistake, which is the difference between a
    correction the model can act on and one it can only guess at.
    """
    lines = [f"Your previous attempt was REJECTED: {verdict.reason}.", ""]
    if regressions:
        lines.append("It introduced these errors, which did not exist before:")
        lines += [f"  line {c.line}  {c.code}  {c.message}" for c in regressions[:10]]
        lines.append("")
    lines.append("Fix the original errors WITHOUT introducing those. Start from the")
    lines.append("file below, which has been restored to its original state.")
    lines.append("")
    return "\n".join(lines)


def run_session(target: str, path: str, max_attempts: int = 3) -> SessionResult:
    """Attempt one file, retrying with feedback on rejection.

    `target` is what gets measured (usually the package root); `path` is the single
    file the agent may edit. Choosing which file is the loop's job, not this one's.

    Each attempt starts from the ORIGINAL file, not from the previous attempt's
    output. The model is told what went wrong rather than asked to patch its own
    broken edit — compounding attempts produce errors nobody can attribute, and the
    feedback would describe a file that no longer exists.

    Bounded because an unbounded repair loop is an unbounded bill. Exhausting the
    attempts is an escalation, not a failure: the history says what was tried.
    """
    before_diags = from_result(run_mypy(target))
    before = Measurement.of(before_diags)

    work = [c for c in actionable(before_diags) if _same_file(c.file, path)]
    if not work:
        return _unchanged(path, "no annotation work in this file", before)

    history: list[str] = []
    feedback = ""
    attempt = 0

    # `while True` rather than a bounded range so the function provably never falls
    # through: every exit below is an explicit return or raise.
    while True:
        attempt += 1

        proposal = propose(path, work, feedback=feedback)
        if not proposal.changed:
            return _unchanged(path, proposal.note or "no change proposed", before)

        # Structural check before anything touches disk. The gate judges one metric
        # and is blind to everything that metric does not cover — a deleted
        # docstring produces no mypy errors at all. Rejecting here also costs one
        # fewer mypy run than writing and measuring first.
        guard = guard_check(proposal.original, proposal.proposed)
        if not guard.ok:
            history.append(f"guard: {guard.reason}")
            if attempt >= max_attempts:
                return SessionResult(
                    path, False,
                    f"exhausted {max_attempts} attempts; escalating. Last: guard: {guard.reason}",
                    before, before, _NO_CHANGE, proposal.proposed, attempt, tuple(history),
                )
            # The guard already speaks in corrections, so it needs no translation.
            feedback = (
                f"Your previous attempt was REJECTED before it was even measured.\n\n"
                f"{guard.reason}.\n\n"
                f"Fix the original errors without doing that. Start from the file below.\n\n"
            )
            continue

        written = write_file(path, proposal.proposed)
        if not written.ok:
            return _unchanged(path, f"{written.error_type}: {written.error_message}", before)

        try:
            after_diags = from_result(run_mypy(target))
            after = Measurement.of(after_diags)
            verdict = judge(before, after, Category.ANNOTATION)

            if verdict.accepted:
                return SessionResult(
                    path, True, verdict.reason, before, after, verdict.deltas,
                    proposal.proposed, attempt, tuple(history),
                )

            write_file(path, proposal.original)
            history.append(verdict.reason)

            if attempt >= max_attempts:
                return SessionResult(
                    path, False,
                    f"exhausted {max_attempts} attempts; escalating. Last: {verdict.reason}",
                    before, after, verdict.deltas, proposal.proposed, attempt, tuple(history),
                )

            feedback = _feedback(verdict, _regressions(before_diags, after_diags))

        except BaseException:
            # Never leave a modified file behind because measuring blew up. The revert
            # is unconditional here; the caller can retry from a known state.
            write_file(path, proposal.original)
            raise


def run_agent_session(
    target: str, path: str, *, max_model_calls: int = 20, max_tool_calls: int = 40
) -> SessionResult:
    """One session where the model drives, for comparison against `run_session`.

    Same contract, different middle. The agent writes the file itself instead of
    returning one, so there is no separate apply step and no retry loop: a
    trajectory already contains its own retries, and the model saw the guard's
    objection at the moment it was raised rather than one attempt later.

    What does not change is the part that matters. The file is snapshotted before
    the agent runs, the same gate judges the same measurement afterwards, and
    anything short of an accepted verdict restores the snapshot byte for byte. The
    agent is a different way of producing a candidate, not a different standard for
    accepting one.
    """
    before_diags = from_result(run_mypy(target))
    before = Measurement.of(before_diags)

    todo = [c for c in actionable(before_diags) if _same_file(c.file, path)]
    if not todo:
        return _unchanged(path, "no annotation work in this file", before)

    read = read_file(path)
    if not read.ok:
        return _unchanged(path, f"{read.error_type}: {read.error_message}", before)
    original = str(read.data["content"])

    try:
        trajectory = agent_work(
            target, path, todo, before,
            max_model_calls=max_model_calls, max_tool_calls=max_tool_calls,
        )

        history = [f"{name}: {detail}" for name, ok, detail in trajectory.calls if not ok]
        if not trajectory.changed:
            return SessionResult(
                path, False,
                f"no change written ({trajectory.stopped}, {trajectory.steps} steps)",
                before, before, _NO_CHANGE, "", trajectory.steps, tuple(history),
            )

        after_diags = from_result(run_mypy(target))
        after = Measurement.of(after_diags)
        verdict = judge(before, after, Category.ANNOTATION)

        if verdict.accepted:
            return SessionResult(
                path, True, verdict.reason, before, after, verdict.deltas,
                trajectory.final, trajectory.steps, tuple(history),
            )

        write_file(path, original)
        return SessionResult(
            path, False, verdict.reason, before, after, verdict.deltas,
            trajectory.final, trajectory.steps, tuple(history),
        )

    except BaseException:
        write_file(path, original)
        raise
