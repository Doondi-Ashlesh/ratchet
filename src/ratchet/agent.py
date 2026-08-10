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
from ratchet.classify import Category, Classified
from ratchet.guard import check as guard_check
from ratchet.model import ToolCall
from ratchet.tools import SCHEMAS, ToolResult, as_json, read_file, run_mypy, write_file

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

The rules your work is judged against:
- Never add `# type: ignore`. Silencing a diagnostic is not fixing it.
- Never use a bare `Any` or `object` as a whole annotation. `dict[str, Any]` is fine.
- Never delete or rewrite code, comments or docstrings.
- Never change what the code does at runtime.
- Every type you name must already be imported, or you must add the import.

You may only edit the one file named in the task. Read it before you write it.
Check your work with run_mypy before you finish.

When you are finished, reply with a short summary and make no further tool call.
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
        return any(s.tool == "run_mypy" for s in self.steps)


class _Workspace:
    """Executes tool calls on behalf of one session, and refuses the ones outside it.

    The policy lives here rather than in `tools.py` because it is per-session: the
    file tools are general, and what makes a write legal is which file this
    particular agent was dispatched to fix. Every refusal comes back as a normal
    tool result, so the agent can correct itself rather than being terminated.
    """

    def __init__(self, target: str, path: str, original: str) -> None:
        self.target = Path(target).resolve()
        self.path = Path(path).resolve()
        self.original = original
        self.steps: list[Step] = []

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
            "run_mypy": self._mypy,
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

        verdict = guard_check(self.original, content)
        if not verdict.ok:
            # The write never reaches disk. The model is told why, in the same terms
            # the gate would have used, while it can still act on it — a rejected
            # write inside the trajectory is feedback; the same rejection after it
            # would just be a lost session.
            return self._record("write_file", ToolResult(
                False, error_type="rejected", error_message=verdict.reason))

        return self._record("write_file", write_file(str(self.path), content))

    def _mypy(self, args: dict[str, Any]) -> str:
        raw = str(args.get("target", "")) or str(self.path)
        if not self._inside_target(raw):
            return self._record("run_mypy", ToolResult(
                False, error_type="outside_target",
                error_message=f"{raw} is outside {self.target}"))

        result = run_mypy(raw)
        if not result.ok:
            return self._record("run_mypy", result)

        # Filtered to the file under work. The agent was dispatched one file; the
        # rest of the package's errors are not its business, and handing them over
        # invites it to go fix them in a file the gate is not watching.
        mine = [d for d in result.data.get("diagnostics", ()) if _same_file(d, self.path)]
        return self._record("run_mypy", ToolResult(True, {
            "file": str(self.path), "error_count": len(mine), "diagnostics": mine,
        }))


def _same_file(diagnostic: Any, path: Path) -> bool:
    raw = diagnostic.get("file") if isinstance(diagnostic, dict) else getattr(diagnostic, "file", "")
    try:
        return Path(str(raw)).resolve() == path
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
    *,
    max_turns: int = 8,
    max_calls: int = 20,
    keep_messages: int = 12,
) -> Trajectory:
    """Let the model fix `path` by calling tools, and report what it did.

    Unlike `propose`, this writes. The session that calls it owns the snapshot and
    the revert; everything here can be undone by restoring one file, which is why
    `_Workspace` refuses a write to any other.

    Both bounds are the caller's, not the model's. `max_turns` bounds the
    conversation, `max_calls` bounds the work, and they are separate because a
    single turn can request several tools at once.
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

    space = _Workspace(target, path, original)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"File to fix: {path}\n"
            f"Package root (for run_mypy): {target}\n\n"
            f"mypy --strict reports:\n{_format(diagnostics)}"
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

        if len(space.steps) >= max_calls:
            stopped = "max_calls"
            break

    after = read_file(path)
    final = str(after.data["content"]) if after.ok else original

    return Trajectory(
        path=path, original=original, final=final, changed=final != original,
        steps=tuple(space.steps), turns=turns, stopped=stopped,
    )
