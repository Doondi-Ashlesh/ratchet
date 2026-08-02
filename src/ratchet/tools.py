#CONTRACT

"""Typed tools — the agent's entire interface to the world.

A tool never raises into the agent. Every outcome, including failure, comes back
as a ToolResult the model can branch on. An exception gives the model a stack
trace; `{"ok": false, "error_type": "file_not_found"}` gives it a decision.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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
        return ToolResult(True, {"path": str(p), "content": p.read_text(encoding="utf-8")})
    except UnicodeDecodeError as e:
        return ToolResult(False, error_type="not_text", error_message=str(e))


def write_file(path: str, content: str) -> ToolResult:
    p = Path(path)
    if not p.is_file():
        # Deliberately refuses to create files. This agent edits what exists;
        # inventing new modules is a much larger permission than fixing a type.
        return ToolResult(False, error_type="file_not_found", error_message=str(p))
    p.write_text(content, encoding="utf-8")
    return ToolResult(True, {"path": str(p), "bytes": len(content)})


#3 Oracle :mypy tool

def run_mypy(target: str, strict: bool = True) -> ToolResult:
    """Run mypy and return parsed diagnostics.

    Uses --output=json rather than scraping human output. Regex-parsing a tool that
    can emit structured data is a bug waiting for the first Windows path with a
    colon in it. Prefer the machine format whenever a tool offers one.
    """
    cmd = ["mypy", "--output=json", "--no-error-summary"]
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
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
