"""Structural checks on a proposal, before the gate ever sees it.

The gate judges one metric. An agent optimising one number is indifferent to
everything that number does not cover, and not maliciously — just indifferent.
Deleting a docstring produces zero mypy errors. So does deleting a function whose
callers happen to be untyped. So does rewriting logic that still type-checks.

Observed live: a proposal for graph.py removed a six-line module docstring while
the prompt forbade exactly that. Had its annotations been correct, the gate would
have accepted it and the documentation would have been silently lost.

So this runs first and works on structure rather than counts. Everything here is
deterministic and comes from the AST, and unlike the gate it can say *what* went
wrong in terms the next attempt can act on: "you removed the docstring on
build_graph" is a correction; "regressed: unknown +2" is a category.

The checks are deliberately about PRESERVATION, not quality. This is not a
reviewer. It asks one question: did the proposal keep everything it was not asked
to change?
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

# Annotations that satisfy the type checker while conveying nothing. `Any` is the
# obvious one and the prompt forbids it; `object` is what the model reached for
# instead, which is the same evasion in different clothing (and worse, since dict
# is invariant, so dict[object, object] rejects the very value it wraps).
_VACUOUS = re.compile(r"\b(?:Any|object)\b")

_TYPE_IGNORE = re.compile(r"#\s*type:\s*ignore")


@dataclass(frozen=True)
class GuardResult:
    """Whether a proposal is structurally safe to measure.

    `violations` are phrased for the model, not for a log line: each one names the
    specific thing that was removed or added, because that is what the next
    attempt has to act on.
    """

    ok: bool
    violations: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        return "; ".join(self.violations) if self.violations else "ok"


def _defs(tree: ast.Module) -> dict[str, ast.AST]:
    """Every top-level and nested function/class, keyed by a qualified name.

    Nested definitions are included because a model that inlines a helper into its
    parent has still deleted the helper, and a name-only comparison at module
    level would not notice.
    """
    found: dict[str, ast.AST] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{prefix}{child.name}"
                found[name] = child
                walk(child, f"{name}.")

    walk(tree, "")
    return found


def _added_lines(original: str, proposed: str) -> list[str]:
    """Lines present in the proposal that were not in the original.

    Set-based rather than a real diff: a `# type: ignore` that already existed is
    not this session's fault, and flagging inherited debt would make the guard
    unusable on any codebase that has some.
    """
    had = set(original.splitlines())
    return [ln for ln in proposed.splitlines() if ln not in had]


def check(original: str, proposed: str) -> GuardResult:
    """Compare a proposal against what it was given. Never raises."""
    violations: list[str] = []

    try:
        before_tree = ast.parse(original)
    except SyntaxError:
        # The original does not parse, so structural comparison is meaningless.
        # Not the proposal's fault, and not something to reject it for.
        return GuardResult(True)

    try:
        after_tree = ast.parse(proposed)
    except SyntaxError as e:
        return GuardResult(False, (f"the file no longer parses: line {e.lineno}: {e.msg}",))

    before_defs = _defs(before_tree)
    after_defs = _defs(after_tree)

    for name in sorted(set(before_defs) - set(after_defs)):
        violations.append(f"you removed `{name}`, which must be preserved")

    if ast.get_docstring(before_tree) and not ast.get_docstring(after_tree):
        violations.append("you removed the module docstring, which must be preserved")

    # get_docstring only accepts these node types, which is exactly what _defs collects.
    documented = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    for name in sorted(set(before_defs) & set(after_defs)):
        before_node, after_node = before_defs[name], after_defs[name]
        if (
            isinstance(before_node, documented)
            and isinstance(after_node, documented)
            and ast.get_docstring(before_node)
            and not ast.get_docstring(after_node)
        ):
            violations.append(f"you removed the docstring on `{name}`, which must be preserved")

    added = _added_lines(original, proposed)

    for line in added:
        if _TYPE_IGNORE.search(line):
            violations.append(f"you added a `# type: ignore`, which is never allowed: {line.strip()}")

    for line in added:
        if _VACUOUS.search(line) and ":" in line:
            violations.append(
                f"you used `Any` or `object` as an annotation, which satisfies the "
                f"checker without saying anything: {line.strip()}"
            )

    return GuardResult(not violations, tuple(violations))
