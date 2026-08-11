"""The coding agent, on LangGraph's prebuilt ReAct loop.

The loop this replaces was hand-written: turn counting, transcript assembly, tool
dispatch, argument parsing, message trimming. It took three rounds of fixes to
reach parity with a single prompt, and every one of those fixes was for a problem
the prebuilt agent had already solved. Writing it once was worth it for
understanding what the framework does; keeping it was not.

What is NOT delegated is the part with no equivalent upstream: the guard on every
write, the gate the agent is judged by, and the fact that a rejected trajectory is
reverted byte for byte. Those live in `agent_tools` and `session`. The agent is a
commodity; the verification around it is the product.

Bounds and context management are middleware rather than hand-written: model call
limits, tool call limits, and clearing stale tool output when the transcript grows.
The hand-rolled versions of all three existed here and were each a bug away from
the framework's.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from ratchet import llm
from ratchet.agent_tools import Session, build_tools
from ratchet.classify import Category, Classified
from ratchet.gate import Measurement
from ratchet.tools import read_file

SYSTEM = """You are fixing missing type annotations in one Python file so that
`mypy --strict` accepts it. You work by calling tools.

PROCEDURE - follow it every time:
1. The file's current contents are given to you below. Do not read it first.
2. Make one targeted change with `edit_file`. Prefer several small edits to one
   large one.
3. Call `check_work`.
4. If it says REJECTED, fix exactly what it names and go back to step 2.
5. Stop only when `check_work` says ACCEPTED, or when it is clear you cannot get
   there.

Stopping before `check_work` accepts means the work is thrown away. A partial fix
scores the same as no fix at all, so there is never a reason to stop early.

HOW YOU ARE JUDGED - the annotation count must FALL, and no other error category
may RISE anywhere in the package. That second half is what usually fails:

- A type you name but never import becomes a NEW error, not a fixed one.
- Annotating a function makes its CALL SITES checkable, so errors can appear in
  OTHER files. `check_work` shows you those. They are your problem.
- If a change you cannot avoid breaks something elsewhere, revert that one change
  and keep the rest. Fixing four of five errors and being accepted beats fixing
  five and being rejected.

RULES - each is enforced by a check, not a preference. A write that breaks one is
refused and never reaches the file:
- Never add `# type: ignore`. Silencing a diagnostic is not fixing it.
- Never use a bare `Any` or `object` as a whole annotation. `dict[str, Any]` is fine.
- Never delete or rewrite code, comments or docstrings.
- Never change what the code does at runtime.
"""


@dataclass(frozen=True)
class Trajectory:
    """What the agent did, and how it ended.

    `stopped` separates an agent that finished from one that was cut off, which
    the file alone cannot tell you: both can leave a correctly edited file, and
    only one of them decided it was done.
    """

    path: str
    original: str
    final: str
    changed: bool
    calls: tuple[tuple[str, bool, str], ...] = ()
    steps: int = 0
    stopped: str = ""
    note: str = ""

    @property
    def guard_rejections(self) -> int:
        return sum(1 for n, ok, _ in self.calls if n in ("edit_file", "write_file") and not ok)

    @property
    def self_checked(self) -> bool:
        return any(n == "check_work" for n, _, _ in self.calls)


def _format(diagnostics: Sequence[Classified]) -> str:
    return "\n".join(
        f"  line {d.line}  {d.code}  {d.message}"
        for d in sorted(diagnostics, key=lambda d: d.line)
    )


def _middleware(max_model_calls: int, max_tool_calls: int) -> list[Any]:
    """The bounds and the context policy, as middleware.

    `exit_behavior="end"` on both limits matters. The alternative raises, and an
    agent stopped mid-repair has still left edits on disk that the gate is about to
    judge; ending the loop lets that judgement happen, while raising would discard
    a trajectory that may well have been acceptable.

    Clearing old tool results rather than dropping whole messages is the part worth
    having taken from upstream: a `check_work` output from six steps ago is the
    largest thing in the transcript and the least useful, while the message that
    requested it still carries the reasoning.
    """
    return [
        # An unattended run makes hundreds of calls, and a rate limit or a slow
        # response is a normal event at that volume. Without this, one of them ends
        # the run and discards every session that already succeeded. Jitter matters:
        # without it, concurrent sessions retry in lockstep and rebuild the spike.
        ModelRetryMiddleware(max_retries=llm.attempts(), backoff_factor=2.0, jitter=True),
        ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior="end"),
        ContextEditingMiddleware(
            edits=[ClearToolUsesEdit(
                trigger=60_000,
                clear_at_least=10_000,
                keep=3,                    # the last few results are the ones being acted on
                clear_tool_inputs=False,   # the arguments say what was attempted; keep them
                exclude_tools=("check_work",),
            )],
        ),
    ]


def work(
    target: str,
    path: str,
    diagnostics: Sequence[Classified],
    before: Measurement,
    *,
    max_model_calls: int = 20,
    max_tool_calls: int = 40,
) -> Trajectory:
    """Let the model fix `path` by calling tools, and report what it did.

    This writes. The caller owns the snapshot and the revert, and every tool is
    bound to this one file, so the whole trajectory can be undone by restoring it.
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

    session = Session(target, path, original, before)
    agent = create_agent(
        llm.chat_model(),
        tools=build_tools(session),
        system_prompt=SYSTEM,
        middleware=_middleware(max_model_calls, max_tool_calls),
    )

    task = HumanMessage(
        f"File to fix: {path}\n\n"
        f"mypy --strict reports these annotation errors in it:\n"
        f"{_format(diagnostics)}\n\n"
        f"--- BEGIN {Path(path).name} ---\n{original}\n--- END {Path(path).name} ---"
    )

    stopped, steps = "model", 0
    try:
        result = agent.invoke(
            {"messages": [task]},
            # A backstop only. The middleware limits above are the real bounds and
            # end the loop cleanly; this catches a cycle they cannot see.
            {"recursion_limit": 2 * (max_model_calls + max_tool_calls)},
        )
        steps = len(result.get("messages", ()))
    except GraphRecursionError:
        # Not an error condition. An agent that runs out of budget mid-repair has
        # produced a partial edit, and the gate judges whatever is on disk. Raising
        # would discard a trajectory that may well have been acceptable.
        stopped = "max_steps"
    except Exception as e:  # noqa: BLE001 - a model failure ends this file, not the run
        stopped = "error"
        session.record("model", False, f"{type(e).__name__}: {e}")

    if session.accepted:
        stopped = "accepted"

    after = read_file(path)
    final = str(after.data["content"]) if after.ok else original

    return Trajectory(
        path=path, original=original, final=final, changed=final != original,
        calls=tuple(session.calls), steps=steps, stopped=stopped,
    )
