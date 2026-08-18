"""Structural checks on a proposal, before the gate ever sees it.

The gate judges one metric. An agent optimising one number is indifferent to
everything that number does not cover, and not maliciously - just indifferent.
Deleting a docstring produces zero mypy errors. So does deleting a function whose
callers happen to be untyped. So does rewriting logic that still type-checks.

This used to be a denylist: a rule per incident, added after each new way of
getting through. Removed defs, removed docstrings, `# type: ignore`, bare `Any`,
then bare `object`. A denylist grows forever and is always one incident behind,
which is the exact pattern this project exists to distrust.

It is now four invariants, stated as what an annotation edit is ALLOWED to be:

  1. Only annotations may change. Strip every annotation from both sides; the
     remaining trees must be identical. Imports may be added, because an
     annotation can need one.
  2. No function may become partially annotated.
  3. No annotation may be vacuous.
  4. Nothing may be suppressed.

Rule 1 is the one that carries the weight, and it subsumes every preservation
check that came before it: deleted functions, deleted docstrings, rewritten logic
and changed runtime behaviour are all "something other than an annotation
changed". It cannot be evaded by a form of damage nobody has thought of yet,
because it does not enumerate damage.

Rule 2 exists because a half-annotated function is worse than an unannotated one.
mypy skips the body of an untyped function; give it a return type and the body
becomes checkable, and attributes assigned `None` in a constructor are inferred as
`None`. Measured: one proposal added `-> None` to two functions, fixed 18 errors
and introduced 12, four of them real defects, purely from that inference.

The rules are about PRESERVATION and COMPLETENESS, not quality. This is not a
reviewer. It asks whether the proposal is the kind of change it was asked to make.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

_TYPE_IGNORE = re.compile(r"#\s*type:\s*ignore")
_NOQA = re.compile(r"#\s*noqa")

# Annotations that satisfy the checker while excluding nothing. `Any` is the
# obvious one; `object` is what the model reached for when told not to use `Any`,
# which is the same evasion in different clothing (and worse, since dict is
# invariant, so dict[object, object] rejects the very value it wraps). `None` is
# the third: it is a real type, and as a whole annotation on anything that is not
# a return it means the value can only ever be None.
_VACUOUS_NAMES = frozenset({"Any", "object"})


@dataclass(frozen=True)
class GuardResult:
    """Whether a proposal is structurally safe to measure.

    `violations` are phrased for the model, not for a log line: each one names the
    specific thing that was wrong, because that is what the next attempt has to
    act on.
    """

    ok: bool
    violations: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        return "; ".join(self.violations) if self.violations else "ok"


# ── invariant 1: only annotations may change ─────────────────────────────────


class _StripAnnotations(ast.NodeTransformer):
    """Remove every annotation, leaving the code the annotations describe.

    What survives this is the program: its statements, its names, its docstrings,
    its control flow. Two proposals that differ only in annotations strip to the
    same tree, and anything else that differs is by definition not an annotation
    edit.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.returns = None
        node.type_comment = None
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.returns = None
        node.type_comment = None
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.annotation = None
        node.type_comment = None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST | None:
        # `x: int = 1` becomes `x = 1`. A bare `x: int` declares nothing at
        # runtime, so it strips to nothing at all.
        if node.value is None:
            return None
        return ast.Assign(targets=[node.target], value=node.value, type_comment=None)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node.type_comment = None
        return self.generic_visit(node)


def _is_import(node: ast.stmt) -> bool:
    return isinstance(node, ast.Import | ast.ImportFrom)


def _is_type_checking_block(node: ast.stmt) -> bool:
    """An `if TYPE_CHECKING:` block containing only imports.

    Allowed to appear because it is the standard way to import a name that exists
    only for annotations. Restricted to imports so the allowance cannot be used to
    smuggle in code that runs during type checking.
    """
    if not isinstance(node, ast.If):
        return False
    test = node.test
    name = (
        test.id if isinstance(test, ast.Name)
        else test.attr if isinstance(test, ast.Attribute)
        else ""
    )
    return name == "TYPE_CHECKING" and bool(node.body) and all(_is_import(s) for s in node.body)


def _skeleton(tree: ast.Module) -> str:
    """The program with annotations and imports removed, as a comparable string."""
    stripped = _StripAnnotations().visit(tree)
    stripped.body = [
        s for s in stripped.body
        if not _is_import(s) and not _is_type_checking_block(s)
    ]
    return ast.dump(ast.fix_missing_locations(stripped))


def _imports(tree: ast.Module) -> set[str]:
    """Every name an import binds, not the statements that bind them.

    Comparing statements looked equivalent and was not. Adding `Any` to an
    existing `from typing import Dict, List, Optional` rewrites that line, so a
    statement-level comparison sees the original as removed and rejects a correct
    edit. Measured against a real proposal: the guard refused it for deleting an
    import it had actually extended.

    A rail that cries wolf is a rail people route around, so this compares what
    the imports actually provide: `typing.Any`, `os`, `pd` for a renamed pandas.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            source = "." * node.level + (node.module or "")
            for alias in node.names:
                found.add(f"{source}.{alias.name}" if alias.name != "*" else f"{source}.*")
    return found


# ── invariant 2: no newly partial annotation ─────────────────────────────────

_Func = ast.FunctionDef | ast.AsyncFunctionDef


def _functions(tree: ast.Module) -> dict[str, tuple[_Func, bool]]:
    """Every function, qualified, with whether it is a method.

    Nested definitions are included: a model that inlines a helper into its parent
    has still deleted the helper, and a module-level comparison would not notice.
    """
    found: dict[str, tuple[_Func, bool]] = {}

    def walk(node: ast.AST, prefix: str, in_class: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{prefix}{child.name}"
                found[name] = (child, in_class)
                walk(child, f"{name}.", False)
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.", True)

    walk(tree, "", False)
    return found


def _parameters(func: _Func, is_method: bool) -> list[ast.arg]:
    """Parameters that need annotating. `self` and `cls` never do."""
    args = func.args
    ordered = [*args.posonlyargs, *args.args]
    if is_method and ordered and ordered[0].arg in ("self", "cls"):
        ordered = ordered[1:]
    ordered += args.kwonlyargs
    if args.vararg:
        ordered.append(args.vararg)
    if args.kwarg:
        ordered.append(args.kwarg)
    return ordered


def _annotation_state(func: _Func, is_method: bool) -> str:
    """`full`, `none` or `partial`."""
    params = _parameters(func, is_method)
    annotated = sum(1 for p in params if p.annotation is not None)
    has_return = func.returns is not None

    if has_return and annotated == len(params):
        return "full"
    if not has_return and annotated == 0:
        return "none"
    return "partial"


def _partial_detail(func: _Func, is_method: bool) -> str:
    missing = [p.arg for p in _parameters(func, is_method) if p.annotation is None]
    if missing and func.returns is None:
        return f"parameters {', '.join(missing)} and the return type are unannotated"
    if missing:
        return f"parameters {', '.join(missing)} are unannotated"
    return "the return type is unannotated"


# ── invariant 3: no vacuous annotation ───────────────────────────────────────


def _vacuous(annotation: ast.expr | None, *, is_return: bool) -> str:
    """Name the vacuous annotation, or empty if it says something.

    `dict[str, Any]` is not vacuous. A JSON payload genuinely is a mapping of str
    to anything, and there is no more specific type short of a TypedDict. Flagging
    it was a false positive that fired eleven times on one file, and a rail that
    cries wolf is a rail people route around.

    `-> None` is the exception to the None rule: a function that returns nothing
    is exactly what that means.
    """
    if annotation is None:
        return ""
    if isinstance(annotation, ast.Name) and annotation.id in _VACUOUS_NAMES:
        return annotation.id
    if not is_return:
        if isinstance(annotation, ast.Constant) and annotation.value is None:
            return "None"
        if isinstance(annotation, ast.Name) and annotation.id == "None":
            return "None"
    return ""


def _annotations_of(func: _Func, is_method: bool) -> list[tuple[str, ast.expr | None, bool]]:
    out: list[tuple[str, ast.expr | None, bool]] = [("return", func.returns, True)]
    out += [(p.arg, p.annotation, False) for p in _parameters(func, is_method)]
    return out


def _variable_annotations(tree: ast.Module) -> dict[str, ast.expr]:
    """Annotated assignments, keyed by their target as written."""
    found: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        # Names and attributes cover every target `x: T = v` can legally have, and
        # both unparse without surprises. Anything else is not a variable
        # annotation worth checking.
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name | ast.Attribute):
            found[ast.unparse(node.target)] = node.annotation
    return found


# ── invariant 4: nothing suppressed ──────────────────────────────────────────


def _added_lines(original: str, proposed: str) -> list[str]:
    """Lines in the proposal that were not in the original.

    Set-based rather than a real diff: a `# type: ignore` that already existed is
    not this session's fault, and flagging inherited debt would make the guard
    unusable on any codebase that has some.
    """
    had = set(original.splitlines())
    return [ln for ln in proposed.splitlines() if ln not in had]


def _suppressions(original: str, proposed: str) -> list[str]:
    violations = []
    for line in _added_lines(original, proposed):
        if _TYPE_IGNORE.search(line):
            violations.append(
                f"you added a `# type: ignore`, which is never allowed: {line.strip()}"
            )
        elif _NOQA.search(line):
            violations.append(
                f"you added a `# noqa`, which silences a check rather than fixing it: {line.strip()}"
            )
    return violations


# ── the check ────────────────────────────────────────────────────────────────


def check(original: str, proposed: str) -> GuardResult:
    """Compare a proposal against what it was given. Never raises."""
    try:
        before = ast.parse(original)
    except SyntaxError:
        # The original does not parse, so structural comparison is meaningless.
        # Not the proposal's fault, and not something to reject it for.
        return GuardResult(True)

    try:
        after = ast.parse(proposed)
    except SyntaxError as e:
        return GuardResult(False, (f"the file no longer parses: line {e.lineno}: {e.msg}",))

    violations: list[str] = []
    violations += _only_annotations_changed(before, after)
    violations += _no_new_partial_annotations(before, after)
    violations += _no_vacuous_annotations(before, after)
    violations += _suppressions(original, proposed)

    return GuardResult(not violations, tuple(violations))


def _definitions(tree: ast.Module) -> dict[str, ast.AST]:
    """Every function and class, qualified. Used only to explain a violation."""
    found: dict[str, ast.AST] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{prefix}{child.name}"
                found[name] = child
                walk(child, f"{name}.")

    walk(tree, "")
    return found


def _explain(before: ast.Module, after: ast.Module) -> list[str]:
    """Name what changed, when it is one of the things that changes often.

    The invariant above is what enforces; this only chooses the wording. A model
    told "you removed the docstring on `build_graph`" can act on it, and one told
    "something other than annotations changed" has to go and diff its own work.
    Adding a case here weakens nothing: the catch-all still rejects everything
    these do not describe.
    """
    said: list[str] = []
    before_defs, after_defs = _definitions(before), _definitions(after)

    for name in sorted(set(before_defs) - set(after_defs)):
        said.append(f"you removed `{name}`, which must be preserved")

    if ast.get_docstring(before) and not ast.get_docstring(after):
        said.append("you removed the module docstring, which must be preserved")

    documented = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    for name in sorted(set(before_defs) & set(after_defs)):
        was, now = before_defs[name], after_defs[name]
        if (
            isinstance(was, documented)
            and isinstance(now, documented)
            and ast.get_docstring(was)
            and not ast.get_docstring(now)
        ):
            said.append(f"you removed the docstring on `{name}`, which must be preserved")

    return said


def _only_annotations_changed(before: ast.Module, after: ast.Module) -> list[str]:
    lost = _imports(before) - _imports(after)
    if lost:
        return ["you removed an import; imports may be added but never removed"]

    # Parsed twice because the transformer mutates in place, and the other
    # invariants need the original trees intact.
    if _skeleton(ast.parse(ast.unparse(before))) == _skeleton(ast.parse(ast.unparse(after))):
        return []

    generic = (
        "you changed something other than annotations and imports. Only type "
        "annotations may be added or corrected: no statement, docstring, name or "
        "piece of logic may be added, removed or rewritten"
    )
    return _explain(before, after) or [generic]


def _no_new_partial_annotations(before: ast.Module, after: ast.Module) -> list[str]:
    was = {name: _annotation_state(f, m) for name, (f, m) in _functions(before).items()}
    violations = []

    for name, (func, is_method) in sorted(_functions(after).items()):
        if _annotation_state(func, is_method) != "partial":
            continue
        if was.get(name) == "partial":
            continue  # inherited, not this proposal's doing
        violations.append(
            f"`{name}` is now partly annotated: {_partial_detail(func, is_method)}. "
            f"Annotating only part of a function makes mypy check a body it was "
            f"skipping, which surfaces errors the code did not have. Annotate it "
            f"fully or leave it alone"
        )
    return violations


def _no_vacuous_annotations(before: ast.Module, after: ast.Module) -> list[str]:
    was = {
        f"{name}:{slot}": ast.dump(ann) if ann is not None else ""
        for name, (func, is_method) in _functions(before).items()
        for slot, ann, _ in _annotations_of(func, is_method)
    }
    violations = []

    for name, (func, is_method) in sorted(_functions(after).items()):
        for slot, annotation, is_return in _annotations_of(func, is_method):
            bare = _vacuous(annotation, is_return=is_return)
            if not bare:
                continue
            key = f"{name}:{slot}"
            if annotation is not None and was.get(key) == ast.dump(annotation):
                continue  # already there before this proposal
            violations.append(
                f"`{bare}` on `{name}` ({slot}) says nothing about the value; "
                f"annotate what it actually holds"
            )

    before_vars = _variable_annotations(before)
    for target, annotation in sorted(_variable_annotations(after).items()):
        bare = _vacuous(annotation, is_return=False)
        if not bare:
            continue
        existing = before_vars.get(target)
        if existing is not None and ast.dump(existing) == ast.dump(annotation):
            continue
        violations.append(
            f"`{target}: {bare}` says nothing about the value; annotate what it actually holds"
        )

    return violations
