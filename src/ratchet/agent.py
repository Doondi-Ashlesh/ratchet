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

from collections.abc import Sequence
from dataclasses import dataclass

from ratchet import model
from ratchet.classify import Category, Classified
from ratchet.tools import read_file

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
