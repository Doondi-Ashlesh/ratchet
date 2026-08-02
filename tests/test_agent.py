"""Tests for the agent.

Every one of these runs offline. The single non-deterministic call lives behind
`model.set_completer`, so the prompt, the parsing, the routing guard, and every
outcome path can be exercised without a network or an API key.

The prompt test matters more than it looks: if the diagnostics stop reaching the
model, the agent still returns plausible-looking output and nothing else fails.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from ratchet import model
from ratchet.agent import _unfence, propose
from ratchet.classify import Category, Classified

CODE = "def f(x):\n    return x\n"
FIXED = "def f(x: int) -> int:\n    return x\n"


def _write(p: Path, text: str) -> None:
    """Write exactly these bytes. write_text() translates \\n to \\r\\n on Windows,
    so the fixture would not contain what it claims to contain."""
    with p.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


@pytest.fixture(autouse=True)
def _never_call_a_live_model() -> Iterator[None]:
    """A test that forgets to clean up must not leak a fake into the next one —
    or worse, leave a real call enabled."""
    yield
    model.set_completer(None)


def _diag(
    code: str = "no-untyped-def",
    line: int = 1,
    category: Category = Category.ANNOTATION,
) -> Classified:
    return Classified(
        file="a.py",
        line=line,
        code=code,
        message=f"message for {code}",
        category=category,
        reason="r",
    )


# ── fence stripping ──────────────────────────────────────────────────────────

def test_unfence_strips_a_language_tagged_fence() -> None:
    assert _unfence("```python\n" + FIXED + "```") == FIXED.strip()


def test_unfence_strips_a_bare_fence() -> None:
    assert _unfence("```\n" + FIXED + "```") == FIXED.strip()


def test_unfence_leaves_plain_code_alone() -> None:
    assert _unfence(FIXED) == FIXED.strip()


# ── the routing guard ────────────────────────────────────────────────────────

def test_the_agent_refuses_anything_but_annotation_work(tmp_path: Path) -> None:
    """Silently dropping a DEFECT would hide the exact routing mistake the
    classifier exists to prevent, so this is loud."""
    p = tmp_path / "a.py"
    _write(p, CODE)

    with pytest.raises(ValueError, match="ANNOTATION"):
        propose(str(p), [_diag(category=Category.DEFECT)])


# ── outcomes that are not failures ───────────────────────────────────────────

def test_a_missing_file_is_reported_not_raised(tmp_path: Path) -> None:
    out = propose(str(tmp_path / "gone.py"), [_diag()])

    assert out.changed is False
    assert "file_not_found" in out.note


def test_an_empty_response_is_reported(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    _write(p, CODE)
    model.set_completer(lambda _: "   ")

    out = propose(str(p), [_diag()])

    assert out.changed is False
    assert out.note == "model returned nothing"


def test_a_model_that_gave_up_is_reported_as_unchanged(tmp_path: Path) -> None:
    """Returning the file untouched is a real answer: it could not fix these."""
    p = tmp_path / "a.py"
    _write(p, CODE)
    model.set_completer(lambda _: CODE)

    out = propose(str(p), [_diag()])

    assert out.changed is False
    assert "unchanged" in out.note


def test_a_real_edit_is_marked_changed(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    _write(p, CODE)
    model.set_completer(lambda _: "```python\n" + FIXED + "```")

    out = propose(str(p), [_diag()])

    assert out.changed is True
    assert out.proposed == FIXED         # trailing newline preserved, fence removed
    assert out.original == CODE          # kept, so the session can diff or revert
    assert out.note == ""


# ── the prompt ───────────────────────────────────────────────────────────────

def test_the_prompt_carries_the_rules_the_errors_and_the_file(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    _write(p, CODE)
    seen: list[str] = []
    model.set_completer(lambda prompt: seen.append(prompt) or FIXED)

    propose(str(p), [_diag(code="type-arg", line=7)])
    prompt = seen[0]

    assert "type: ignore" in prompt                  # the rule is stated
    assert "line 7  type-arg" in prompt              # the specific error reached it
    assert "message for type-arg" in prompt          # including mypy's own wording
    assert "def f(x):" in prompt                     # and the file content


def test_errors_are_listed_in_line_order(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    _write(p, CODE)
    seen: list[str] = []
    model.set_completer(lambda prompt: seen.append(prompt) or FIXED)

    propose(str(p), [_diag(line=9), _diag(line=2), _diag(line=5)])
    listed = [ln for ln in seen[0].splitlines() if ln.startswith("  line ")]

    assert listed == sorted(listed, key=lambda ln: int(ln.split()[1]))


def test_the_rules_come_before_the_file(tmp_path: Path) -> None:
    """Static prefix first, so a prefix-caching endpoint has something to hit."""
    p = tmp_path / "a.py"
    _write(p, CODE)
    seen: list[str] = []
    model.set_completer(lambda prompt: seen.append(prompt) or FIXED)

    propose(str(p), [_diag()])

    assert seen[0].index("Rules:") < seen[0].index("--- BEGIN")

CRLF_FILE = b"def f(x):\r\n    return x\r\n"


def test_a_proposal_adopts_the_files_line_endings(tmp_path: Path) -> None:
    """The model returns LF; the file is CRLF. Writing that verbatim would turn a
    three-line fix into a whole-file diff."""
    p = tmp_path / "a.py"
    p.write_bytes(CRLF_FILE)
    model.set_completer(lambda _: "def f(x: int) -> int:\n    return x\n")

    out = propose(str(p), [_diag()])

    assert "\r\n" in out.proposed
    assert "\n" not in out.proposed.replace("\r\n", "")   # no bare LF survived
    assert out.proposed.endswith("\r\n")                   # trailing newline kept


def test_identical_content_in_different_endings_is_not_a_change(tmp_path: Path) -> None:
    """Before normalising, a byte-identical answer in LF read as a change and cost
    a full write-measure-reject cycle."""
    p = tmp_path / "a.py"
    p.write_bytes(CRLF_FILE)
    model.set_completer(lambda _: "def f(x):\n    return x\n")

    assert propose(str(p), [_diag()]).changed is False
