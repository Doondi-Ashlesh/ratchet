"""The agent's tools, as LangChain tools.

Previously these were hand-written JSON schemas plus a hand-written dispatch
table. The schema is now derived from the type hints and the docstring, so a
signature and the description the model reads cannot drift apart - which they did,
silently, the first time a tool was renamed.

The tools are built per session rather than defined at module scope. What makes a
write legal is *which file this particular agent was dispatched to fix*, and that
is not knowable when the module is imported. Closing over the session turns the
policy into something the tool cannot be called around, instead of a check the
caller is trusted to remember.

Nothing here raises. Every outcome, including refusal, is returned as text the
model can act on: an exception gives it a stack trace, a refusal gives it a
decision.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from ratchet.classify import Category, from_result
from ratchet.gate import Measurement, judge
from ratchet.guard import check as guard_check
from ratchet.tools import read_file, replace_once, run_mypy, write_file


class Session:
    """One file's worth of permission, and the record of what was done with it."""

    def __init__(self, target: str, path: str, original: str, before: Measurement) -> None:
        self.target = Path(target).resolve()
        self.path = Path(path).resolve()
        self.original = original
        self.before = before
        self.calls: list[tuple[str, bool, str]] = []
        self.accepted = False

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.calls.append((name, ok, detail))

    @property
    def guard_rejections(self) -> int:
        return sum(1 for n, ok, _ in self.calls if n in ("edit_file", "write_file") and not ok)

    @property
    def wrote(self) -> bool:
        """Whether any edit actually landed. An agent that stops having written
        nothing has not finished the task, whatever it says in its final message."""
        return any(n in ("edit_file", "write_file") and ok for n, ok, _ in self.calls)

    @property
    def self_checked(self) -> bool:
        return any(n == "check_work" for n, _, _ in self.calls)


def _fail(session: Session, name: str, error: str, message: str) -> str:
    session.record(name, False, f"{error}: {message}")
    return json.dumps({"ok": False, "error": error, "message": message})


def _ok(session: Session, name: str, payload: dict[str, Any]) -> str:
    session.record(name, True)
    return json.dumps({"ok": True, **payload}, default=str)


def build_tools(session: Session) -> list[BaseTool]:
    """The three tools this agent gets, bound to one file.

    Deliberately small. There is no shell, no directory listing, no network and no
    way to create a file. Every tool is a capability that has to be justified, and
    the justification for a fourth one has not come up.
    """

    def _apply(name: str, candidate: str) -> str:
        """Guard, then write. The only path to disk, so no tool routes around it."""
        verdict = guard_check(session.original, candidate)
        if not verdict.ok:
            # The write never lands. The model is told why, in the same words the
            # gate would have used, while it can still act on it. A rejection
            # inside the trajectory is feedback; the same rejection afterwards is
            # just a lost session.
            return _fail(session, name, "rejected", verdict.reason)
        written = write_file(str(session.path), candidate)
        if not written.ok:
            return _fail(session, name, written.error_type, written.error_message)
        return _ok(session, name, {"bytes": written.data.get("bytes", 0)})

    @tool
    def read_current_file() -> str:
        """Read the current contents of the file you were asked to fix.

        You are given the file's contents up front, so you only need this after an
        edit, to see the result before anchoring the next one.
        """
        current = read_file(str(session.path))
        if not current.ok:
            return _fail(session, "read_file", current.error_type, current.error_message)
        return _ok(session, "read_file", {"content": current.data["content"]})

    @tool
    def edit_file(old_string: str, new_string: str) -> str:
        """Replace one exact passage of the file you were asked to fix.

        Prefer this over write_file: it is far cheaper and cannot corrupt the parts
        you did not touch. old_string must appear EXACTLY ONCE in the file; if it
        does not, include the surrounding lines until it is unique. Whitespace and
        indentation must match the file exactly.
        """
        current = read_file(str(session.path))
        if not current.ok:
            return _fail(session, "edit_file", current.error_type, current.error_message)

        candidate, result = replace_once(str(current.data["content"]), old_string, new_string)
        if not result.ok:
            return _fail(session, "edit_file", result.error_type, result.error_message)
        return _apply("edit_file", candidate)

    @tool
    def write_whole_file(content: str) -> str:
        """Replace the ENTIRE file with new content. Use edit_file instead unless
        you are restructuring the whole file: this costs far more, and every
        character you retype is a character you can get wrong."""
        return _apply("write_file", content)

    @tool
    def check_work() -> str:
        """Report whether your work WOULD BE ACCEPTED, using the exact rule that
        will judge it.

        The annotation count must fall, and no other error category may rise
        anywhere in the package. Returns the verdict, the errors left in your file,
        and any new errors your edit caused in other files. Call this before you
        finish: if it says REJECTED, you are not done.
        """
        after_diags = from_result(run_mypy(str(session.target)))
        after = Measurement.of(after_diags)
        verdict = judge(session.before, after, Category.ANNOTATION)
        session.accepted = verdict.accepted

        mine = [d for d in after_diags if _same(d.file, session.path)]
        elsewhere = [
            {"file": Path(d.file).name, "line": d.line, "code": d.code, "message": d.message}
            for d in after_diags
            if not _same(d.file, session.path) and d.category is not Category.ANNOTATION
        ]
        return _ok(session, "check_work", {
            "verdict": "ACCEPTED" if verdict.accepted else "REJECTED",
            "reason": verdict.reason,
            "categories_that_rose": [k for k, v in verdict.deltas.items() if v > 0],
            "errors_left_in_your_file": [
                {"line": d.line, "code": d.code, "message": d.message} for d in mine
            ],
            # Capped. This is a package-wide check, and an agent handed eighty of
            # someone else's errors will go and fix them.
            "new_or_other_errors_elsewhere": elsewhere[:15],
        })

    return [read_current_file, edit_file, write_whole_file, check_work]


def _same(raw: str, path: Path) -> bool:
    try:
        return Path(raw).resolve() == path
    except OSError:
        return False
