#CONTRACT

"""Typed tools — the agent's entire interface to the world.

A tool never raises into the agent. Every outcome, including failure, comes back
as a ToolResult the model can branch on. An exception gives the model a stack
trace; `{"ok": false, "error_type": "file_not_found"}` gives it a decision.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LF = "\n"
CR = "\r"
CRLF = "\r\n"


@dataclass(frozen=True)
class Diagnostic:
    """One mypy finding."""

    file: str
    line: int
    code: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class ToolResult:
    """Every tool returns this. `ok` is the only field the agent must branch on."""

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error_type: str = ""
    error_message: str = ""

#2 The file tools

def read_file(path: str) -> ToolResult:
    p = Path(path)
    if not p.is_file():
        return ToolResult(False, error_type="file_not_found", error_message=str(p))
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            return ToolResult(True, {"path": str(p), "content": f.read()})
    except UnicodeDecodeError as e:
        return ToolResult(False, error_type="not_text", error_message=str(e))


def write_file(path: str, content: str) -> ToolResult:
    p = Path(path)
    if not p.is_file():
        # Deliberately refuses to create files. This agent edits what exists;
        # inventing new modules is a much larger permission than fixing a type.
        return ToolResult(False, error_type="file_not_found", error_message=str(p))
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write(content)
    return ToolResult(True, {"path": str(p), "bytes": len(content)})


def _match_newlines(text: str, like: str) -> str:
    """Rewrite `text` to use the line endings of `like`."""
    body = text.replace(CRLF, LF).replace(CR, LF)
    return body.replace(LF, CRLF) if CRLF in like else body


def replace_once(content: str, old: str, new: str) -> tuple[str, ToolResult]:
    """Anchored replacement. Refuses anything it cannot place unambiguously.

    Uniqueness is the whole safety property. A patch that matches in two places
    is applied to the wrong one roughly half the time, and the result usually
    still parses - which makes it a silent corruption rather than a caught one.
    Refusing is the difference between a failed tool call the agent can retry and
    a bad edit nobody notices.

    Returns the candidate content and a result; on failure the content is the
    unmodified original, so a caller that ignores `ok` cannot write damage.
    """
    if not old:
        return content, ToolResult(
            False, error_type="empty_anchor",
            error_message="old_string must not be empty; use write_file to replace a whole file")
    if old == new:
        return content, ToolResult(
            False, error_type="no_op", error_message="old_string and new_string are identical")

    # The anchor is rewritten to the file's own line endings before matching.
    #
    # Without this the tool is unusable on a CRLF repository, and silently so. A
    # model is shown the file, quotes a passage back with `\n`, and a literal
    # comparison against `\r\n` content never matches - so every multi-line anchor
    # fails with "does not appear in the file", which reads as the model being
    # unable to copy text. Measured: 11 of 16 edits failed this way, every one
    # `not_found`, on targets that were 100% CRLF.
    #
    # The anchor is converted rather than the file. Normalising the content would
    # rewrite line endings in regions nobody edited, turning a two-line change into
    # a whole-file diff on exactly the repositories this is meant to support.
    anchor, replacement = _match_newlines(old, content), _match_newlines(new, content)

    found = content.count(anchor)
    if found == 0 and anchor != old:
        # A file with mixed endings: this region may not use the dominant one.
        anchor, replacement, found = old, new, content.count(old)

    if found == 0:
        return content, ToolResult(
            False, error_type="not_found",
            error_message="old_string does not appear in the file; read it again and quote it exactly")
    if found > 1:
        return content, ToolResult(
            False, error_type="not_unique",
            error_message=f"old_string appears {found} times; include surrounding lines to make it unique")

    return content.replace(anchor, replacement, 1), ToolResult(True, {"replacements": 1})


#3 Oracle :mypy tool

def run_mypy(target: str, strict: bool = True) -> ToolResult:
    """Run mypy and return parsed diagnostics.

    Uses --output=json rather than scraping human output. Regex-parsing a tool that
    can emit structured data is a bug waiting for the first Windows path with a
    colon in it. Prefer the machine format whenever a tool offers one.

    Invoked as `sys.executable -m mypy` rather than bare `mypy`, so it resolves to
    the interpreter's own environment instead of whatever PATH happens to hold.
    """
    # Per-target cache. mypy keys modules by NAME and its default cache lives in the
    # working directory, so two unrelated targets that both contain `a.py` collide
    # and the second run receives the first one's diagnostics — complete with the
    # first one's file paths. A harness whose whole premise is a trustworthy
    # measurement cannot share that cache.
    fingerprint = hashlib.sha256(str(Path(target).resolve()).encode()).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "ratchet-mypy-cache" / fingerprint

    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "--output=json",
        "--no-error-summary",
        f"--cache-dir={cache_dir}",
    ]
    if strict:
        cmd.append("--strict")
    cmd.append(target)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, check=False
        )
    except FileNotFoundError:
        return ToolResult(False, error_type="mypy_not_installed", error_message="pip install mypy")
    except subprocess.TimeoutExpired:
        return ToolResult(False, error_type="timeout", error_message="mypy exceeded 300s")

    diags: list[Diagnostic] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            raw: Any = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line we don't understand is not a crash
        diags.append(
            Diagnostic(
                file=str(raw.get("file", "")),
                line=int(raw.get("line", 0)),
                code=str(raw.get("code") or ""),
                message=str(raw.get("message", "")),
                severity=str(raw.get("severity", "error")),
            )
        )

    errors = [d for d in diags if d.severity == "error"]
    return ToolResult(
        True,
        {
            "error_count": len(errors),
            "diagnostics": [asdict(d) for d in errors],
            "by_file": _tally(d.file for d in errors),
            "by_code": _tally(d.code for d in errors),
            "stderr": proc.stderr[:2000],
        },
    )


def _tally(values: Iterable[str]) -> dict[str, int]:
    """Count occurrences, most frequent first."""
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


#5 The schemas the model sees

# Written by hand rather than generated from the signatures. The description is a
# prompt, not documentation: it is the only place a constraint can be stated before
# the model acts, and it needs wording chosen for a reader who will look for the
# cheapest way to satisfy it. A generator would emit the type and drop exactly that.

SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file and return its exact contents. Read a file "
                "before writing it, and read it again after writing if you need to "
                "know what is actually on disk."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to an existing file."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Overwrite an existing file with complete new contents. Cannot create "
                "files. The write is checked before it lands: if it is rejected the "
                "file on disk is unchanged and the reason is returned to you. A "
                "rejected write is information, not a dead end. Fix the stated problem "
                "and write again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to an existing file."},
                    "content": {
                        "type": "string",
                        "description": "The COMPLETE new file. Not a patch, not an excerpt.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace one exact passage of the file. PREFER THIS over write_file: "
                "it is far cheaper and cannot corrupt the parts you did not touch. "
                "old_string must appear EXACTLY ONCE - include the surrounding lines "
                "if it does not. Whitespace and indentation must match exactly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to an existing file."},
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to replace, quoted verbatim from the file.",
                    },
                    "new_string": {"type": "string", "description": "Text to put in its place."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_work",
            "description": (
                "Type-check and report whether your work WOULD BE ACCEPTED, using the "
                "exact rule that will judge you: the annotation count must fall, and "
                "no other category may rise anywhere in the package. Returns the "
                "verdict, the errors remaining in your file, and any NEW errors your "
                "edit caused in other files. Call this before you finish. If it says "
                "rejected, you are not done."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_NAMES = frozenset(s["function"]["name"] for s in SCHEMAS)


def as_json(result: ToolResult, *, limit: int = 8000) -> str:
    """Render a ToolResult as the string handed back to the model.

    Truncated by length because a tool result goes into the transcript and stays
    there for the rest of the trajectory. One unbounded mypy dump costs its size on
    every subsequent turn, not once. Truncation is announced rather than silent: a
    model that cannot tell a short file from a clipped one will confidently rewrite
    the half it can see and delete the half it cannot.
    """
    payload: dict[str, Any] = {"ok": result.ok}
    if result.ok:
        payload.update(result.data)
    else:
        payload["error"] = result.error_type
        payload["message"] = result.error_message

    body = json.dumps(payload, default=str)
    if len(body) <= limit:
        return body
    return json.dumps({
        "ok": result.ok,
        "truncated": True,
        "note": f"result was {len(body)} chars, showing the first {limit}",
        "partial": body[:limit],
    })
