"""The agent: propose a fix. It does not write, and it does not judge.

This is the only component whose output cannot be trusted, so it is given the
narrowest possible job. It receives a file and the annotation errors in it, and
returns proposed content. Applying that content, re-measuring, and deciding
whether to keep it all happen elsewhere — the model proposes, the harness
disposes.

The rules in the prompt are requests, not constraints. A model asked not to write
`# type: ignore` will mostly comply and will occasionally not, and there is no
prompt wording that changes "mostly" into "always". Enforcement is a separate
deterministic check; the prompt exists to make compliance likely, not certain.

Static instructions come before the file content so a prefix-caching endpoint has
something to hit — the rules are identical on every call, the file never is.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ratchet import model
from ratchet.classify import Category, Classified, from_result
from ratchet.gate import Measurement, judge
from ratchet.guard import check as guard_check
from ratchet.model import ToolCall
from ratchet.tools import (
    SCHEMAS,
    ToolResult,
    as_json,
    read_file,
    replace_once,
    run_mypy,
    write_file,
)

_RULES = """You are adding missing type annotations to a Python file so that
`mypy --strict` accepts it.

Rules:
- Do NOT add `# type: ignore` anywhere, for any reason.
- Do NOT use `Any` unless there is genuinely no more specific type.
- Do NOT change what the code does at runtime.
- Do NOT delete or rewrite code, comments, or docstrings.
- Every type you use must ALREADY be imported in this file, or you must add the
  import. Do not reference a name that is not in scope.
- Return the COMPLETE file, unchanged except for the annotations.
- Return only the file. No explanation, no markdown fences.
"""


@dataclass(frozen=True)
class Proposal:
    """What the agent came back with. `changed` is False when the model returned
    the file untouched, which is a real outcome — it could not fix these."""

    path: str
    original: str
    proposed: str
    changed: bool
    note: str = ""


def _format(diagnostics: Sequence[Classified]) -> str:
    return "\n".join(
        f"  line {d.line}  {d.code}  {d.message}" for d in sorted(diagnostics, key=lambda d: d.line)
    )


def _unfence(text: str) -> str:
    """Strip a markdown code fence if the model added one despite being asked not to.

    Not defensive coding for its own sake — models wrap code in fences habitually,
    and writing ```python as line 1 of a .py file is a syntax error that would
    read as the agent producing garbage.
    """
    body = text.strip()
    if not body.startswith("```"):
        return body
    lines = body.splitlines()
    lines = lines[1:]                                  # drop the opening fence
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize(text: str, like: str) -> str:
    """Match the original file's line endings and trailing newline.

    A model returns whatever endings it feels like. Writing those verbatim turns a
    three-line annotation fix into a whole-file diff, and drops the final newline
    that end-of-file-fixer exists to protect. Normalising here means `changed`
    compares content rather than formatting.
    """
    body = text.replace("\r\n", "\n").replace("\r", "\n")
    newline = "\r\n" if "\r\n" in like else "\n"
    if newline != "\n":
        body = body.replace("\n", newline)
    if like.endswith(("\n", "\r")) and not body.endswith(newline):
        body += newline
    return body


def propose(path: str, diagnostics: Sequence[Classified], feedback: str = "") -> Proposal:
    """Ask the model for a corrected version of `path`.

    Raises if handed anything but ANNOTATION work. That is a programming error in
    the caller, not a runtime condition — silently dropping a DEFECT would hide
    exactly the routing mistake the classifier exists to prevent.
    """
    wrong = {d.category for d in diagnostics} - {Category.ANNOTATION}
    if wrong:
        raise ValueError(f"agent may only be given ANNOTATION work, got {sorted(c.value for c in wrong)}")

    read = read_file(path)
    if not read.ok:
        return Proposal(path, "", "", changed=False, note=f"{read.error_type}: {read.error_message}")

    original = str(read.data["content"])
    prompt = (
        f"{_RULES}\n"
        f"Errors to fix:\n{_format(diagnostics)}\n\n"
        f"{feedback}"
        f"--- BEGIN {path} ---\n{original}\n--- END {path} ---\n"
    )

    try:
        raw = model.complete(prompt)
    except model.Truncated as e:
        # An outcome, not a crash. Surfacing it as a note names the real constraint
        # — this file may be too large to rewrite whole — instead of spending the
        # remaining attempts on syntax errors that were never the model's fault.
        return Proposal(path, original, "", changed=False, note=f"truncated: {e}")

    proposed = _normalize(_unfence(raw), like=original)
    if not proposed.strip():
        return Proposal(path, original, "", changed=False, note="model returned nothing")

    if proposed == original:
        # Exact comparison is meaningful now that both sides use the file's own
        # conventions. Before normalising, a byte-identical answer in different
        # line endings read as a change and cost a whole measure-and-reject cycle.
        return Proposal(
            path, original, proposed, changed=False, note="model returned the file unchanged"
        )

    return Proposal(path=path, original=original, proposed=proposed, changed=True)


# ── the agent loop ────────────────────────────────────────────────────────────
# `propose` above asks for a file and gets one back. Here the model chooses its own
# actions instead: read, write, type-check, in whatever order it decides, until it
# stops or hits a bound. `propose` is kept rather than replaced, because comparing
# the two modes on the same targets is the only way to find out whether an action
# space was worth its cost.

_SYSTEM = """You are fixing missing type annotations in one Python file so that
`mypy --strict` accepts it. You work by calling tools.

PROCEDURE - follow it every time:
1. The file's current contents are given to you below. You do not need to read it first.
2. Make one targeted change with `edit_file`. Prefer many small edits to one large one.
3. Call `check_work`.
4. If it says REJECTED, fix what it names and go back to step 2.
5. Only stop when `check_work` says ACCEPTED, or when it is clear you cannot get there.

Stopping before `check_work` accepts means the work is thrown away. A partial fix
scores exactly the same as no fix at all, so there is no reason to stop early.

HOW YOU ARE JUDGED - the annotation count must FALL, and no other error category
may RISE anywhere in the package. That second half is what usually fails:

- A type you name but never import becomes a NEW error, not a fixed one.
- Annotating a function makes its CALL SITES checkable, and errors can appear in
  OTHER files as a result. `check_work` shows you those. They are your problem.
- If a change you cannot avoid causes errors elsewhere, revert that one change and
  fix the rest. Fixing four of five errors and being accepted beats fixing five and
  being rejected.

RULES - each of these is enforced by a check, not a preference. A write that
breaks one is refused and never reaches the file:
- Never add `# type: ignore`. Silencing a diagnostic is not fixing it.
- Never use a bare `Any` or `object` as a whole annotation. `dict[str, Any]` is fine.
- Never delete or rewrite code, comments or docstrings.
- Never change what the code does at runtime.

You may only edit the one file named in the task.
"""


@dataclass(frozen=True)
class Step:
    """One executed tool call. The record of what the agent actually did, which is
    the thing a trajectory can be judged on and a summary line cannot."""

    tool: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class Trajectory:
    """The outcome of one agent loop.

    `stopped` distinguishes an agent that finished from one that was cut off, which
    the file contents alone cannot: both can leave a correctly edited file, and only
    one of them decided it was done.
    """

    path: str
    original: str
    final: str
    changed: bool
    steps: tuple[Step, ...] = ()
    turns: int = 0
    stopped: str = ""
    note: str = ""

    @property
    def guard_rejections(self) -> int:
        return sum(1 for s in self.steps if s.tool == "write_file" and not s.ok)

    @property
    def self_checked(self) -> bool:
        """Did it verify before finishing? Asking for this in the prompt makes it
        likely; recording it is how you find out how likely."""
        return any(s.tool == "check_work" for s in self.steps)


class _Workspace:
    """Executes tool calls on behalf of one session, and refuses the ones outside it.

    The policy lives here rather than in `tools.py` because it is per-session: the
    file tools are general, and what makes a write legal is which file this
    particular agent was dispatched to fix. Every refusal comes back as a normal
    tool result, so the agent can correct itself rather than being terminated.
    """

    def __init__(self, target: str, path: str, original: str, before: Measurement) -> None:
        self.target = Path(target).resolve()
        self.path = Path(path).resolve()
        self.original = original
        self.before = before
        self.steps: list[Step] = []
        self.accepted = False

    def _record(self, tool: str, result: ToolResult) -> str:
        detail = "" if result.ok else f"{result.error_type}: {result.error_message}"
        self.steps.append(Step(tool=tool, ok=result.ok, detail=detail))
        return as_json(result)

    def _inside_target(self, raw: str) -> bool:
        try:
            Path(raw).resolve().relative_to(self.target)
        except ValueError:
            return False
        return True

    def run(self, call: ToolCall) -> str:
        if call.malformed:
            return self._record(
                call.name, ToolResult(False, error_type="bad_arguments", error_message=call.malformed)
            )
        handler = {
            "read_file": self._read,
            "write_file": self._write,
            "edit_file": self._edit,
            "check_work": self._check,
        }.get(call.name)
        if handler is None:
            return self._record(
                call.name,
                ToolResult(False, error_type="no_such_tool", error_message=call.name),
            )
        return handler(call.arguments)

    def _read(self, args: dict[str, Any]) -> str:
        raw = str(args.get("path", ""))
        if not self._inside_target(raw):
            return self._record("read_file", ToolResult(
                False, error_type="outside_target",
                error_message=f"{raw} is outside {self.target}"))
        return self._record("read_file", read_file(raw))

    def _write(self, args: dict[str, Any]) -> str:
        raw = str(args.get("path", ""))
        if Path(raw).resolve() != self.path:
            # Not a safety afterthought. The session gate measures one file and
            # reverts one file, so a write anywhere else would survive a rejected
            # trajectory and leave the target modified by work nobody accepted.
            return self._record("write_file", ToolResult(
                False, error_type="wrong_file",
                error_message=f"you may only edit {self.path}, not {raw}"))

        content = _normalize(_unfence(str(args.get("content", ""))), like=self.original)
        return self._apply("write_file", content)

    def _apply(self, tool: str, candidate: str) -> str:
        """Guard, then write. The single path to disk, so no tool can bypass the check."""
        verdict = guard_check(self.original, candidate)
        if not verdict.ok:
            # The write never reaches disk. The model is told why, in the same terms
            # the gate would have used, while it can still act on it - a rejected
            # write inside the trajectory is feedback; the same rejection after it
            # would just be a lost session.
            return self._record(tool, ToolResult(
                False, error_type="rejected", error_message=verdict.reason))
        return self._record(tool, write_file(str(self.path), candidate))

    def _edit(self, args: dict[str, Any]) -> str:
        raw = str(args.get("path", "")) or str(self.path)
        if Path(raw).resolve() != self.path:
            return self._record("edit_file", ToolResult(
                False, error_type="wrong_file",
                error_message=f"you may only edit {self.path}, not {raw}"))

        current = read_file(str(self.path))
        if not current.ok:
            return self._record("edit_file", current)

        candidate, result = replace_once(
            str(current.data["content"]),
            str(args.get("old_string", "")),
            str(args.get("new_string", "")),
        )
        if not result.ok:
            return self._record("edit_file", result)
        return self._apply("edit_file", candidate)

    def _check(self, _args: dict[str, Any]) -> str:
        """Run the actual gate and report its actual verdict.

        This exists because the agent was previously shown a filtered view: the
        errors in its own file. The gate judges the whole package and rejects if any
        category rises anywhere, so an annotation that breaks a caller in a
        neighbouring file was invisible to the agent and fatal at the end. It was
        being graded on a measurement it could not see, which no amount of retrying
        fixes.
        """
        after_diags = from_result(run_mypy(str(self.target)))
        after = Measurement.of(after_diags)
        verdict = judge(self.before, after, Category.ANNOTATION)
        self.accepted = verdict.accepted

        mine = [d for d in after_diags if _same_path(d.file, self.path)]
        elsewhere = [
            {"file": Path(d.file).name, "line": d.line, "code": d.code, "message": d.message}
            for d in after_diags
            if not _same_path(d.file, self.path) and d.category is not Category.ANNOTATION
        ]
        rose = [k for k, v in verdict.deltas.items() if v > 0]

        return self._record("check_work", ToolResult(True, {
            "verdict": "ACCEPTED" if verdict.accepted else "REJECTED",
            "reason": verdict.reason,
            "categories_that_rose": rose,
            "errors_left_in_your_file": [
                {"line": d.line, "code": d.code, "message": d.message} for d in mine
            ],
            # Capped: this is the tail of a package-wide check and an agent that
            # receives 80 of someone else's errors will go and fix them.
            "new_or_other_errors_elsewhere": elsewhere[:15],
        }))


def _same_path(raw: str, path: Path) -> bool:
    try:
        return Path(raw).resolve() == path
    except OSError:
        return False


def _trim(messages: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """Bound the transcript, dropping the oldest exchanges first.

    Never drops the system prompt or the task: the agent can re-read a file it has
    forgotten, but it cannot re-derive what it was asked to do. Dropping is done in
    whole assistant/tool pairs, because a tool result whose call has been removed
    is a message the provider will reject.
    """
    if len(messages) <= keep + 2:
        return messages
    head, tail = messages[:2], messages[2:]
    while len(tail) > keep and tail:
        tail = tail[1:]
        while tail and tail[0].get("role") == "tool":
            tail = tail[1:]                # a result whose call just left
    return [*head, *tail]


def work(
    target: str,
    path: str,
    diagnostics: Sequence[Classified],
    before: Measurement | None = None,
    *,
    max_turns: int = 20,
    max_calls: int = 40,
    keep_messages: int = 16,
) -> Trajectory:
    """Let the model fix `path` by calling tools, and report what it did.

    Unlike `propose`, this writes. The session that calls it owns the snapshot and
    the revert; everything here can be undone by restoring one file, which is why
    `_Workspace` refuses a write to any other.

    Both bounds are the caller's, not the model's. `max_turns` bounds the
    conversation, `max_calls` bounds the work, and they are separate because a
    single turn can request several tools at once. They are set high because the
    first measured failure mode was a trajectory that ran out of turns with work
    still in flight, and an unfinished session scores the same as no session.
    """
    wrong = {d.category for d in diagnostics} - {Category.ANNOTATION}
    if wrong:
        raise ValueError(
            f"agent may only be given ANNOTATION work, got {sorted(c.value for c in wrong)}"
        )

    read = read_file(path)
    if not read.ok:
        return Trajectory(path, "", "", changed=False, stopped="error",
                          note=f"{read.error_type}: {read.error_message}")
    original = str(read.data["content"])

    baseline = before if before is not None else Measurement.of(from_result(run_mypy(target)))
    space = _Workspace(target, path, original, baseline)

    # The file is given rather than fetched. The first measured trajectory spent
    # five of its eight turns re-reading a file it had already been shown, and never
    # wrote anything. A tool call to obtain what the caller already holds is a round
    # trip bought with the budget that was meant to do the work.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"File to fix: {path}\n\n"
            f"mypy --strict reports these annotation errors in it:\n"
            f"{_format(diagnostics)}\n\n"
            f"--- BEGIN {Path(path).name} ---\n{original}\n--- END {Path(path).name} ---"
        )},
    ]

    stopped, turns = "max_turns", 0
    for turns in range(1, max_turns + 1):
        try:
            reply = model.converse(_trim(messages, keep_messages), SCHEMAS)
        except model.Truncated as e:
            stopped = "truncated"
            space.steps.append(Step("model", ok=False, detail=str(e)))
            break

        if reply.done:
            stopped = "model"
            break

        messages.append({
            "role": "assistant",
            "content": reply.content,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in reply.tool_calls
            ],
        })
        for call in reply.tool_calls:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": space.run(call)})

        # Stop the moment the real gate accepts. Continuing past acceptance can only
        # lose it: every further edit is another chance to break something, and the
        # session keeps the file as it stands at the end, not at its best moment.
        if space.accepted:
            stopped = "accepted"
            break

        if len(space.steps) >= max_calls:
            stopped = "max_calls"
            break

    after = read_file(path)
    final = str(after.data["content"]) if after.ok else original

    return Trajectory(
        path=path, original=original, final=final, changed=final != original,
        steps=tuple(space.steps), turns=turns, stopped=stopped,
    )
