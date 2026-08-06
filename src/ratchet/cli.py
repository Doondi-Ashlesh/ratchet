"""Command line entry point.

    ratchet check <path>    measure a codebase and report what work exists, for whom
    ratchet bench <path>    measure the agent's per-file accept rate

Exit codes mirror mypy and ruff so this composes into a pipeline:

    0   nothing to report
    1   errors found
    2   the tool itself could not run

The check report ends with a next step rather than a bare table, because the
ordering is not obvious and is easy to get wrong: config first (failure-log 003),
since an unresolved import both hides errors mypy cannot see and produces
cascading ones that vanish for free once it resolves.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict

from ratchet.bench import DirtyTree, NotAGitRepo, Trial, format_report, run_bench
from ratchet.classify import Category, from_result, summary
from ratchet.tools import run_mypy

EXIT_OK = 0
EXIT_FOUND = 1
EXIT_TOOL_FAILED = 2

_ROUTING = {
    Category.ANNOTATION: "agent may attempt",
    Category.DEFECT: "human decides",
    Category.CONFIG: "fix first, then re-measure",
    Category.CASCADING: "downstream; do not touch",
    Category.UNKNOWN: "escalate, never guess",
}


def _next_step(counts: Mapping[str, int]) -> str:
    """The one instruction that matters, given what was found."""
    if counts[Category.CONFIG.value]:
        return (
            f"next: resolve {counts[Category.CONFIG.value]} config error(s) first — "
            f"they hide errors mypy cannot see, and may clear "
            f"{counts[Category.CASCADING.value]} cascading error(s) for free"
        )
    if counts[Category.ANNOTATION.value]:
        return f"next: {counts[Category.ANNOTATION.value]} annotation error(s) ready for an agent"
    held = counts[Category.DEFECT.value] + counts[Category.UNKNOWN.value]
    if held:
        return f"next: nothing an agent may attempt; {held} need a human"
    return "next: clean"


def _report(path: str, counts: Mapping[str, int], total: int) -> str:
    width = max(len(c.value) for c in Category)
    lines = [f"ratchet check {path}", ""]
    lines += [
        f"  {c.value:<{width}}  {counts[c.value]:>5}   {_ROUTING[c]}" for c in Category
    ]
    lines += ["", f"  {'total':<{width}}  {total:>5}", "", _next_step(counts)]
    return "\n".join(lines)


def _bench(path: str, max_files: int, max_attempts: int, as_json: bool) -> int:
    """Run the benchmark, streaming each trial so a long run is not silent."""

    def announce(t: Trial) -> None:
        if not as_json:
            plural = "" if t.attempts == 1 else "s"
            print(f"  {'keep' if t.kept else 'drop'}  {t.file}  ({t.attempts} attempt{plural})")

    try:
        report = run_bench(path, max_files, max_attempts, on_trial=announce)
    except (NotAGitRepo, DirtyTree) as e:
        print(f"ratchet: {e}", file=sys.stderr)
        return EXIT_TOOL_FAILED

    if as_json:
        print(json.dumps({
            "target": report.target,
            "seconds": report.seconds,
            "accepted": report.accepted,
            "accept_rate": round(report.accept_rate, 4),
            "by_code": {k: {"accepted": a, "total": n} for k, (a, n) in report.by_code().items()},
            "rejections": report.rejections(),
            "trials": [asdict(t) for t in report.trials],
        }, indent=2))
    else:
        print()
        print(format_report(report))

    return EXIT_OK


def _run(args: argparse.Namespace) -> int:
    """Work through a whole package. Imported lazily so `ratchet check` keeps
    working for anyone who installed without the agent extras."""
    from ratchet import loop

    if args.checkpoint:
        with loop.make_checkpointer(args.checkpoint) as saver:
            state = loop.run_loop(
                args.path,
                max_files=args.max_files,
                max_attempts=args.max_attempts,
                max_failures=args.max_failures,
                thread_id=args.thread,
                checkpointer=saver,
            )
    else:
        state = loop.run_loop(
            args.path,
            max_files=args.max_files,
            max_attempts=args.max_attempts,
            max_failures=args.max_failures,
            thread_id=args.thread,
        )

    results = list(state.get("results", []))
    skipped = list(state.get("skipped", []))
    before = sum(state.get("baseline", {}).values())
    after = sum(state.get("latest", {}).values())
    kept = [r for r in results if r["kept"]]

    if args.json:
        print(json.dumps({
            "target": args.path, "before": before, "after": after,
            "sessions": len(results), "kept": len(kept),
            "results": results, "skipped": skipped,
        }, indent=2))
        return EXIT_OK if not results or kept else EXIT_FOUND

    print(f"ratchet run {args.path}\n")
    for r in results:
        mark = "kept   " if r["kept"] else "reverted"
        print(f"  {mark}  {r['file']:<44} {r['attempts']}x  {r['reason'][:60]}")
    for s in skipped:
        print(f"  skipped   {s['file']:<44}     {s['reason'][:60]}")

    print(f"\n  errors  {before} -> {after}")
    print(f"  kept    {len(kept)}/{len(results)} session(s)")
    if state.get("escalation"):
        print(f"\n  {state['escalation']}")
    if not loop.tracing_enabled():
        print("\n  (traces off: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY)")
    return EXIT_OK if kept or not results else EXIT_FOUND


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ratchet",
        description="A harness for long-running agents, built on a metric that only moves one way.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="measure a codebase and report what work exists")
    check.add_argument("path", help="file or directory to type-check")
    check.add_argument("--json", action="store_true", help="machine-readable output")

    bench = sub.add_parser("bench", help="measure the agent's per-file accept rate")
    bench.add_argument("path", help="package to benchmark against")
    bench.add_argument("--max-files", type=int, default=20)
    bench.add_argument("--max-attempts", type=int, default=3)
    bench.add_argument("--json", action="store_true", help="machine-readable output")

    run = sub.add_parser("run", help="work through a codebase, one file per session")
    run.add_argument("path", help="package to work through")
    run.add_argument("--max-files", type=int, default=0, help="0 = every file with work")
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--max-failures", type=int, default=2,
                     help="skip a file after this many failed runs (from committed history)")
    run.add_argument("--thread", default="default", help="checkpoint thread id, for resuming")
    run.add_argument("--checkpoint", default="", help="sqlite path; omit for in-memory")
    run.add_argument("--json", action="store_true", help="machine-readable output")

    args = parser.parse_args(argv)

    if args.command == "bench":
        return _bench(args.path, args.max_files, args.max_attempts, args.json)

    if args.command == "run":
        return _run(args)

    result = run_mypy(args.path)
    if not result.ok:
        # A tool failure is not a finding. Distinguishing them is why run_mypy
        # returns ok=False rather than an empty diagnostics list.
        print(f"ratchet: {result.error_type}: {result.error_message}", file=sys.stderr)
        return EXIT_TOOL_FAILED

    counts = summary(from_result(result))
    total = sum(counts.values())

    if args.json:
        print(json.dumps({"path": args.path, "total": total, "counts": counts}, indent=2))
    else:
        print(_report(args.path, counts, total))

    return EXIT_FOUND if total else EXIT_OK
