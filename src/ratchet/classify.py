"""Classify mypy diagnostics by what it would take to fix them, and who may.

Failure-log 002: 17 of SdkAgent's 111 errors were import-not-found — mypy failing
to resolve a package, not a defect in the source. No edit fixes them. Hand all 111
to a model with "make these go away" and it will suppress what it cannot fix. The
count falls. Nothing improves.

Failure-log 004: "fixable" was too coarse. Some editable errors mean the types are
missing (var-annotated, type-arg) — the code is fine, the fix is mechanical. Others
mean mypy believes the CODE IS WRONG (attr-defined, func-returns-value). Both are
editable, but an agent handed the second kind will silence it with a cast, the
count will fall, and a real bug becomes type-checker-approved. So they route
differently: ANNOTATION to the agent, DEFECT to a human.

Two rules hold this together:

  - The default is UNKNOWN, never a workable category. An unrecognized code is
    escalated rather than attempted, because the cost of guessing wrong is an
    agent inventing a suppression for a problem it never understood.
  - The code sets grow only from observed runs. Adding a code you have not seen
    and understood is how a classifier starts lying.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ratchet.tools import Diagnostic, ToolResult


class Category(str, Enum):
    """What kind of work an error represents, and therefore who handles it."""

    ANNOTATION = "annotation"  # types are missing; the code is fine. Agent may attempt.
    DEFECT = "defect"          # mypy believes the code is wrong. Human decides.
    CONFIG = "config"          # environment or mypy config; no source edit helps
    CASCADING = "cascading"    # downstream of a CONFIG error in the same file
    UNKNOWN = "unknown"        # no rule for this code; escalate, never attempt

# These lists grow from observed data, never from guessing. Every code here was
# seen in a real run. Adding a code you have not seen is how a classifier starts
# lying.
_ANNOTATION = frozenset(
    {
        "type-arg",         # dict -> dict[str, Any]
        "no-untyped-def",   # missing signature annotations
        "no-untyped-call",  # calling an unannotated function
        "no-any-return",    # returning Any from a typed function
        "var-annotated",    # bad = []  ->  bad: list[str] = []
    }
)

# mypy is asserting the code is WRONG, not merely undocumented. An agent given
# these will silence them with a cast, the count will fall, and a real bug
# becomes type-checker-approved. Escalate instead.
_DEFECT = frozenset(
    {
        "attr-defined",        # accessing something that may not exist
        "func-returns-value",  # using the result of a function returning None
        "assignment",          # the annotation and the value genuinely disagree
        "arg-type",            # ambiguous: a too-narrow annotation OR a genuinely
                               # wrong call. mypy can't tell you which, so a human does.
    }
)
_CONFIG = frozenset({"import-not-found", "import-untyped"})

# Genuinely ambiguous: `misc` covers "cannot subclass X (has type Any)", which is
# downstream of a failed import — but `misc` also covers unrelated real defects.
# Context decides, which is why the cascade rule exists.
_MAYBE_DOWNSTREAM = frozenset({"misc", "untyped-decorator"})


@dataclass(frozen=True)
class Classified:
    """A diagnostic plus the verdict on who should handle it."""

    file: str
    line: int
    code: str
    message: str
    category: Category
    reason: str


def _categorize(d: Diagnostic, files_with_config: frozenset[str]) -> tuple[Category, str]:
    if d.code in _CONFIG:
        return Category.CONFIG, "mypy cannot resolve the import; install stubs or configure mypy"
    if d.code in _ANNOTATION:
        return Category.ANNOTATION, "types are missing; the code itself is fine"
    if d.code in _DEFECT:
        return Category.DEFECT, "mypy believes this code is wrong; a cast would hide a real bug"
    if d.code in _MAYBE_DOWNSTREAM and d.file in files_with_config:
        return (
            Category.CASCADING,
            "same file has an unresolved import; fix that first, then re-measure",
        )
    return Category.UNKNOWN, f"no rule for code {d.code!r}"


def classify(diagnostics: Iterable[Diagnostic]) -> list[Classified]:
    """Triage diagnostics. Cascade detection is per-file: a possibly-downstream
    error in a file that also has an unresolved import is treated as downstream
    until proven otherwise."""
    diags = list(diagnostics)
    files_with_config = frozenset(d.file for d in diags if d.code in _CONFIG)

    out: list[Classified] = []
    for d in diags:
        category, reason = _categorize(d, files_with_config)
        out.append(
            Classified(
                file=d.file,
                line=d.line,
                code=d.code,
                message=d.message,
                category=category,
                reason=reason,
            )
        )
    return out


def from_result(result: ToolResult) -> list[Classified]:
    """Classify the diagnostics carried by a run_mypy result."""
    raw: list[dict[str, Any]] = result.data.get("diagnostics", [])
    return classify(
        Diagnostic(
            file=str(d.get("file", "")),
            line=int(d.get("line", 0)),
            code=str(d.get("code", "")),
            message=str(d.get("message", "")),
            severity=str(d.get("severity", "error")),
        )
        for d in raw
    )


def summary(classified: Iterable[Classified]) -> dict[str, int]:
    """Counts per category, so a session can see its work split at a glance."""
    out: dict[str, int] = {c.value: 0 for c in Category}
    for item in classified:
        out[item.category.value] += 1
    return out


def actionable(classified: Iterable[Classified]) -> list[Classified]:
    """The only errors an agent is permitted to attempt.

    DEFECT is deliberately excluded despite being editable — silencing a
    suspected bug is exactly the cheating mode the ratchet exists to prevent.
    """
    return [c for c in classified if c.category is Category.ANNOTATION]
