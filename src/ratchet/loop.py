"""The multi-session loop, as a LangGraph state machine.

A `while` loop would run these sessions. It would not survive being killed, would
not resume where it stopped, and would have no pause point a human could approve
at. Those three properties are the reason this is a graph: LangGraph supplies
checkpointing, resumption and `interrupt_before` as framework features rather than
as machinery that has to be built and then proven correct.

    preflight ─→ work ⇄ work ─→ finish
                   └──→ escalate

`preflight` measures the target once, builds the queue from files that actually
have annotation work, and drops the ones cross-run history says keep failing.
`work` runs exactly one session and records the verdict. The self-edge is the
cycle. `escalate` is where a run stops for a human.

Every node is traced to LangSmith when LANGSMITH_TRACING is set. That matters more
here than in the single-session case: a run is now dozens of model calls across
many files, and "which file ate the budget" is not answerable from a summary line.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from ratchet import history
from ratchet.classify import actionable, from_result
from ratchet.gate import Measurement
from ratchet.session import run_session
from ratchet.tools import run_mypy

MAX_FAILURES_BEFORE_SKIP = 2


def _keep_last(_old: Any, new: Any) -> Any:
    """Last writer wins. The default for a scalar, stated explicitly so the
    reducer is visible rather than implied."""
    return new


def _extend(old: list[Any], new: list[Any]) -> list[Any]:
    return [*old, *new]


class LoopState(TypedDict, total=False):
    """Graph state. Everything here is JSON-serialisable because the checkpointer
    has to persist it — that constraint is why results are dicts rather than the
    SessionResult dataclass they came from."""

    target: str
    max_attempts: int
    max_files: int
    max_failures: int

    queue: Annotated[list[str], _keep_last]
    results: Annotated[list[dict[str, Any]], _extend]
    skipped: Annotated[list[dict[str, str]], _extend]

    baseline: Annotated[dict[str, int], _keep_last]
    latest: Annotated[dict[str, int], _keep_last]
    escalation: Annotated[str, _keep_last]


def _rel(target: str, path: str) -> str:
    """Relative, and always with forward slashes.

    The history file is committed and shared. Keyed with the OS separator, a
    Windows run writes `gateway\\app.py` and a Linux run looks up `gateway/app.py`,
    so the same file appears twice and every skip rule silently stops working
    across platforms. Observed in the first committed state file.
    """
    try:
        rel = Path(path).resolve().relative_to(Path(target).resolve())
    except ValueError:
        return path.replace("\\", "/")
    return rel.as_posix()


@traceable(name="preflight", run_type="chain")
def preflight(state: LoopState) -> LoopState:
    """Measure once, decide what is worth dispatching, and refuse the rest.

    Two filters, and the second is the point of this whole rung. Files with no
    ANNOTATION work are not the agent's business. Files that history says have
    failed repeatedly are not dispatched again — before this existed, one
    intractable file consumed a dozen model calls across four runs and nothing
    noticed.
    """
    target = state["target"]
    diags = from_result(run_mypy(target))
    baseline = Measurement.of(diags)

    by_file: dict[str, int] = {}
    for c in actionable(diags):
        by_file[c.file] = by_file.get(c.file, 0) + 1

    hist = history.load(target)
    max_failures = state.get("max_failures", MAX_FAILURES_BEFORE_SKIP)

    queue: list[str] = []
    skipped: list[dict[str, str]] = []
    for path, _n in sorted(by_file.items(), key=lambda kv: -kv[1]):
        why = hist.blocked(_rel(target, path), max_failures)
        if why:
            skipped.append({"file": _rel(target, path), "reason": why})
        else:
            queue.append(path)

    limit = state.get("max_files", 0)
    if limit:
        queue = queue[:limit]

    return {
        "queue": queue,
        "skipped": skipped,
        "baseline": dict(baseline.counts),
        "latest": dict(baseline.counts),
    }


@traceable(name="work", run_type="chain")
def work(state: LoopState) -> LoopState:
    """One file, one session. The graph handles the repetition."""
    queue = list(state["queue"])
    path = queue.pop(0)
    target = state["target"]

    result = run_session(target, path, max_attempts=state.get("max_attempts", 3))

    hist = history.load(target)
    rel = _rel(target, path)
    hist.record(rel, kept=result.kept, attempts=result.attempts, reason=result.reason)
    history.save(target, hist)

    return {
        "queue": queue,
        "latest": dict(result.after.counts),
        "results": [
            {
                "file": rel,
                "kept": result.kept,
                "attempts": result.attempts,
                "reason": result.reason,
                "total_after": result.after.total,
            }
        ],
    }


@traceable(name="escalate", run_type="chain")
def escalate(state: LoopState) -> LoopState:
    """Where a run stops and asks for a person.

    Reached when every remaining file has exhausted its attempts. The graph is
    compiled with `interrupt_before=["escalate"]`, so a caller that wants a human
    in the loop gets one here and can resume from the checkpoint afterwards.
    """
    failed = [r for r in state.get("results", []) if not r["kept"]]
    return {
        "escalation": (
            f"{len(failed)} file(s) could not be fixed automatically; "
            f"{len(state.get('skipped', []))} were skipped on prior failures"
        )
    }


def _route(state: LoopState) -> Literal["work", "escalate", "__end__"]:
    """Keep going while there is work, escalate if nothing succeeded, else stop."""
    if state.get("queue"):
        return "work"
    results = state.get("results", [])
    if results and not any(r["kept"] for r in results):
        return "escalate"
    return "__end__"          # the value of langgraph's END, spelled so mypy sees the literal


def build(checkpointer: Any = None, require_approval: bool = False) -> Any:
    """Compile the graph.

    `interrupt_before` is only wired when asked for. A batch run that pauses on
    every escalation is not a batch run, and a pause point nobody resumes is worse
    than no pause point at all.
    """
    g: StateGraph[LoopState] = StateGraph(LoopState)
    g.add_node("preflight", preflight)
    g.add_node("work", work)
    g.add_node("escalate", escalate)

    g.add_edge(START, "preflight")
    g.add_conditional_edges("preflight", _route, {"work": "work", "escalate": "escalate", END: END})
    g.add_conditional_edges("work", _route, {"work": "work", "escalate": "escalate", END: END})
    g.add_edge("escalate", END)

    kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    if require_approval:
        kwargs["interrupt_before"] = ["escalate"]
    return g.compile(**kwargs)


def make_checkpointer(path: str = ":memory:") -> Any:
    """SQLite checkpointer. Pass a file path for a run that survives the process."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver.from_conn_string(path)


def run_loop(
    target: str,
    *,
    max_files: int = 0,
    max_attempts: int = 3,
    max_failures: int = MAX_FAILURES_BEFORE_SKIP,
    thread_id: str = "default",
    checkpointer: Any = None,
) -> LoopState:
    """Drive the graph to completion and return the final state."""
    graph = build(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    initial: LoopState = {
        "target": target,
        "max_files": max_files,
        "max_attempts": max_attempts,
        "max_failures": max_failures,
        "queue": [],
        "results": [],
        "skipped": [],
        "baseline": {},
        "latest": {},
        "escalation": "",
    }
    out: LoopState = graph.invoke(initial, config=config)
    return out


def tracing_enabled() -> bool:
    """Whether LangSmith will actually receive anything, so the CLI can say so
    rather than leaving the user to wonder why the project is empty."""
    flag = os.environ.get("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes")
    return flag and bool(os.environ.get("LANGSMITH_API_KEY"))
