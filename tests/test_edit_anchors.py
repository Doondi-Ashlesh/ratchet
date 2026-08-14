"""Tests for anchored replacement, and specifically for line endings.

This file exists because of a bug that made the edit tool unusable on Windows
repositories and looked exactly like a model that could not copy text. The model
is shown a file, quotes a passage back using `\\n`, and a literal comparison
against `\\r\\n` content never matches. Measured live: 11 of 16 edits failed, every
one `not_found`, on targets that were 100% CRLF.

Every test here uses a CRLF fixture, because an LF-only suite is what let it
through the first time.
"""
from __future__ import annotations

from ratchet.tools import replace_once

CRLF = "def f(x):\r\n    return x\r\n"
LF = "def f(x):\n    return x\n"


def test_an_lf_anchor_matches_a_crlf_file() -> None:
    """The bug. A model quotes with `\\n`; the file on disk uses `\\r\\n`."""
    out, result = replace_once(
        CRLF, "def f(x):\n    return x", "def f(x: int) -> int:\n    return x"
    )

    assert result.ok, result.error_message
    assert out == "def f(x: int) -> int:\r\n    return x\r\n"


def test_the_files_line_endings_survive_the_edit() -> None:
    """The anchor is converted, not the file. Normalising the content would
    rewrite endings in regions nobody edited, turning a two-line change into a
    whole-file diff on exactly the repositories this is meant to support."""
    out, result = replace_once(CRLF, "def f(x):\n", "def g(x):\n")

    assert result.ok
    assert "\n" not in out.replace("\r\n", ""), "no bare LF was introduced"


def test_a_crlf_anchor_matches_an_lf_file() -> None:
    """The same mismatch in the other direction, which a Windows-authored anchor
    against a Linux checkout would hit."""
    out, result = replace_once(LF, "def f(x):\r\n", "def g(x):\r\n")

    assert result.ok
    assert out == "def g(x):\n    return x\n"


def test_a_single_line_anchor_still_works() -> None:
    """Single-line anchors never had the bug, so they are the control case."""
    out, result = replace_once(CRLF, "def f(x):", "def f(x: int) -> int:")

    assert result.ok
    assert out.startswith("def f(x: int) -> int:\r\n")


def test_a_genuinely_absent_anchor_is_still_refused() -> None:
    """The fix must not become a fuzzy match. An anchor that is not there has to
    stay an error, or the tool starts guessing where to write."""
    _, result = replace_once(CRLF, "def nope(y):\n    return y", "x")

    assert not result.ok
    assert result.error_type == "not_found"


def test_an_ambiguous_anchor_is_still_refused_on_a_crlf_file() -> None:
    """Uniqueness is the safety property and must survive the newline handling."""
    _, result = replace_once("a = 1\r\na = 1\r\n", "a = 1", "a = 2")

    assert not result.ok
    assert result.error_type == "not_unique"


def test_a_mixed_ending_file_falls_back_to_the_raw_anchor() -> None:
    """A file with both conventions: the dominant one is tried first, then the
    anchor exactly as given, so a region that disagrees with the file's majority
    is still reachable."""
    mixed = "a = 1\r\nb = 2\nc = 3\r\n"
    out, result = replace_once(mixed, "b = 2\nc = 3", "b = 9\nc = 3")

    assert result.ok, result.error_message
    assert "b = 9" in out
