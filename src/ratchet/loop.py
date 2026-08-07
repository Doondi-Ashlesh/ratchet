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
import time
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

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
    order: str
    deadline_s: float
    started: float

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

    # Worst-first moves the count fastest; smallest-first is likelier to succeed.
    # Whole-file rewriting scales with file size, so the biggest file is both the
    # slowest and the most likely to time out or truncate (failure-log 019) — and
    # worst-first dispatches it first. Neither order is right in the abstract; the
    # answer needs accept-rate-by-file-size, which is what `ratchet bench` measures.
    reverse = state.get("order", "worst") == "worst"
    ordered = sorted(by_file.items(), key=lambda kv: kv[1], reverse=reverse)

    queue: list[str] = []
    skipped: list[dict[str, str]] = []
    for path, _n in ordered:
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
        # Stamped here, not at call time, so a resumed run gets the full budget
        # again rather than inheriting an already-expired clock from a checkpoint.
        "started": time.time(),
    }


def _expired(state: LoopState) -> bool:
    budget = float(state.get("deadline_s", 0) or 0)
    started = float(state.get("started", 0) or 0)
    return bool(budget and started and time.time() - started >= budget)


def work(state: LoopState) -> LoopState:
    """One file, one session. The graph handles the repetition.

    The budget is checked between files rather than inside one, because a session
    is the smallest unit that can be kept or reverted. Interrupting mid-session
    would abandon a proposal that the gate had not yet judged.
    """
    queue = list(state["queue"])
    target = state["target"]

    # A session's wall-clock cost is not predictable from the file. The slowest one
    # measured took 530s for a 197-line file: the cost was 10k output tokens of
    # reasoning, not the size of the input, so no file-size heuristic would have
    # caught it. What an unattended run actually needs is a bound on the whole run,
    # after which it stops dispatching and reports what it did — the alternative is
    # what happened before this existed, an outer `timeout` killing the process and
    # discarding every result already earned.
    if _expired(state):
        return {
            "queue": [],
            "skipped": [
                {"file": _rel(target, p), "reason": "run deadline reached"} for p in queue
            ],
        }

    path = queue.pop(0)

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
    # The nodes carry no @traceable decorator on purpose. LangGraph already opens a
    # span per node against whatever tracer is configured, so decorating them too
    # produced two identical spans for every node — visible in LangSmith as a
    # duplicated `preflight` with the same start time and duration. Adding a
    # decorator is the obvious way to "turn on tracing" and the reason it is absent
    # is not visible from the node definitions, so it is recorded here.
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
    order: str = "worst",
    deadline_s: float = 0.0,
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
        "order": order,
        "deadline_s": deadline_s,
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
