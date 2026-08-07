"""Cross-run memory, committed into the repository being worked on.

Failure-log 017: `graph.py` consumed roughly twelve model calls across four
sessions, failing every time for the same underlying reason, because a session
has no memory and every run rebuilt the same failure from zero.

The store lives at `.ratchet/state.json` inside the target repo rather than in a
database beside the tool. That is deliberate and copied from how Anthropic's
long-running harness persists progress: state that travels with the code survives
a fresh clone, a different machine, and a CI runner with no volume, and it is
reviewable in a diff. A Postgres row is invisible to the person reviewing the pull
request; a committed JSON file is not.

JSON rather than Markdown for the same reason they chose it: a model editing the
repo is measurably less likely to rewrite a file that does not look like prose.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STATE_DIR = ".ratchet"
STATE_FILE = "state.json"


@dataclass
class FileRecord:
    """What has happened to one file across every run so far."""

    attempts: int = 0
    failures: int = 0
    accepted: int = 0
    last_status: str = ""
    last_reason: str = ""


@dataclass
class History:
    """The whole store. `version` exists so a format change is detectable rather
    than silently misread — an unversioned state file is a future outage."""

    version: int = SCHEMA_VERSION
    files: dict[str, FileRecord] = field(default_factory=dict)

    def record(self, rel_path: str, *, kept: bool, attempts: int, reason: str) -> None:
        rec = self.files.setdefault(rel_path, FileRecord())
        rec.attempts += attempts
        rec.last_reason = reason
        if kept:
            rec.accepted += 1
            rec.last_status = "kept"
            rec.failures = 0          # a success clears the streak
        else:
            rec.failures += 1
            rec.last_status = "failed"

    def blocked(self, rel_path: str, max_failures: int) -> str:
        """Why this file should not be dispatched, or '' if it should."""
        rec = self.files.get(rel_path)
        if rec is None or rec.failures < max_failures:
            return ""
        return (
            f"failed {rec.failures}x, last: {rec.last_reason[:90]}"
            if rec.last_reason
            else f"failed {rec.failures}x"
        )


def path_for(target: str) -> Path:
    """State lives at the git root when there is one, else beside the target."""
    p = Path(target).resolve()
    root = p if p.is_dir() else p.parent
    for candidate in [root, *root.parents]:
        if (candidate / ".git").exists():
            return candidate / STATE_DIR / STATE_FILE
    return root / STATE_DIR / STATE_FILE


def load(target: str) -> History:
    """Read the store. A missing or unreadable file is an empty history, never an
    error: a first run on a fresh repo is the normal case, not a failure."""
    f = path_for(target)
    if not f.is_file():
        return History()
    try:
        raw: dict[str, Any] = json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return History()

    if int(raw.get("version", 0)) != SCHEMA_VERSION:
        # A version we do not understand is not something to guess at. Start
        # clean rather than misread fields that may have changed meaning.
        return History()

    files = {
        k: FileRecord(**{f: v[f] for f in FileRecord.__annotations__ if f in v})
        for k, v in raw.get("files", {}).items()
        if isinstance(v, dict)
    }
    return History(version=SCHEMA_VERSION, files=files)


def save(target: str, hist: History) -> Path:
    f = path_for(target)
    f.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SCHEMA_VERSION,
        "files": {k: asdict(v) for k, v in sorted(hist.files.items())},
    }
    with f.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return f
