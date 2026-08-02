"""`ratchet check` — measure a codebase and report what work exists, for whom.

Exit codes mirror mypy and ruff so this composes into a pipeline:

    0   nothing to report
    1   errors found
    2   the tool itself could not run

The report ends with a next step rather than a bare table, because the ordering
is not obvious and is easy to get wrong: config first (failure-log 003), since an
unresolved import both hides errors mypy cannot see and produces cascading ones
that vanish for free once it resolves.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ratchet",
        description="A harness for long-running agents, built on a metric that only moves one way.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="measure a codebase and report what work exists")
    check.add_argument("path", help="file or directory to type-check")
    check.add_argument("--json", action="store_true", help="machine-readable output")

    args = parser.parse_args(argv)

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
