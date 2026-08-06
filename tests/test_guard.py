"""Tests for the structural guard.

The guard exists because the gate judges one number and is blind to everything
that number does not cover. Observed live: a proposal deleted a six-line module
docstring, which produces zero mypy errors, so a correct-looking annotation fix
would have been accepted with the documentation silently gone.

Every check here is about PRESERVATION rather than quality. The guard is not a
reviewer; it asks whether the proposal kept what it was not asked to change.
"""
from __future__ import annotations

from ratchet.guard import check

ORIGINAL = '''"""Module docstring that must survive."""


def keep_me(x: int) -> int:
    """Docstring on a function."""
    return x


class Thing:
    """Docstring on a class."""

    def method(self) -> None:
        """Docstring on a method."""
'''


def test_an_untouched_file_passes() -> None:
    result = check(ORIGINAL, ORIGINAL)

    assert result.ok
    assert result.reason == "ok"


def test_an_annotation_only_change_passes() -> None:
    """The thing the agent is actually supposed to do must not trip the guard."""
    proposed = ORIGINAL.replace("def keep_me(x: int) -> int:", "def keep_me(x: str) -> str:")

    assert check(ORIGINAL, proposed).ok


# ── preservation ─────────────────────────────────────────────────────────────

def test_a_deleted_module_docstring_is_caught() -> None:
    """The live finding. Zero mypy errors, total documentation loss."""
    proposed = ORIGINAL.replace('"""Module docstring that must survive."""\n', "")

    result = check(ORIGINAL, proposed)

    assert not result.ok
    assert "module docstring" in result.reason


def test_a_deleted_function_docstring_names_the_function() -> None:
    proposed = ORIGINAL.replace('    """Docstring on a function."""\n', "")

    result = check(ORIGINAL, proposed)

    assert not result.ok
    assert "`keep_me`" in result.reason


def test_a_deleted_method_docstring_is_caught_by_qualified_name() -> None:
    """Nested definitions are walked, so inlining or gutting a method is visible."""
    proposed = ORIGINAL.replace('        """Docstring on a method."""\n', "        pass\n")

    result = check(ORIGINAL, proposed)

    assert not result.ok
    assert "Thing.method" in result.reason


def test_a_deleted_function_is_caught() -> None:
    proposed = ORIGINAL.replace(
        'def keep_me(x: int) -> int:\n    """Docstring on a function."""\n    return x\n', ""
    )

    result = check(ORIGINAL, proposed)

    assert not result.ok
    assert "removed `keep_me`" in result.reason


# ── suppression ──────────────────────────────────────────────────────────────

def test_an_added_type_ignore_is_caught() -> None:
    proposed = ORIGINAL.replace("    return x", "    return x  # type: ignore")

    result = check(ORIGINAL, proposed)

    assert not result.ok
    assert "type: ignore" in result.reason


def test_a_pre_existing_type_ignore_is_not_this_sessions_fault() -> None:
    """Flagging inherited debt would make the guard unusable on any real codebase."""
    original = ORIGINAL.replace("    return x", "    return x  # type: ignore")

    assert check(original, original).ok


def test_a_vacuous_annotation_is_caught() -> None:
    """The prompt forbade `Any`; the model reached for `object` instead. Same
    evasion, and with dict being invariant, a worse one."""
    proposed = ORIGINAL.replace("def keep_me(x: int) -> int:", "def keep_me(x: object) -> object:")

    result = check(ORIGINAL, proposed)

    assert not result.ok
    assert "says nothing about the value" in result.reason


def test_a_parameterised_generic_containing_Any_is_not_vacuous() -> None:
    """Observed live: this rule fired eleven times on one file, every one of them
    a false positive on `dict[str, Any]`. A JSON payload genuinely is a mapping of
    str to anything, and there is no more specific type short of a TypedDict.

    A rail that cries wolf is a rail people learn to route around, which is the
    exact failure this guard exists to prevent, turned on itself.
    """
    for annotation in (
        "def keep_me(x: int) -> dict[str, Any]:",
        "def keep_me(x: list[Any]) -> int:",
        "def keep_me(x: int) -> Mapping[str, object]:",
    ):
        proposed = ORIGINAL.replace("def keep_me(x: int) -> int:", annotation)

        assert check(ORIGINAL, proposed).ok, f"false positive on: {annotation}"


# ── failure modes of the guard itself ────────────────────────────────────────

def test_a_proposal_that_does_not_parse_is_caught_with_the_line() -> None:
    result = check(ORIGINAL, "def broken(:\n    pass\n")

    assert not result.ok
    assert "no longer parses" in result.reason


def test_an_unparseable_ORIGINAL_is_not_the_proposals_fault() -> None:
    """Structural comparison against a file that does not parse is meaningless,
    and rejecting the proposal for it would blame the wrong party."""
    assert check("def broken(:\n", "anything at all\n").ok


def test_every_violation_is_reported_not_just_the_first() -> None:
    """One round trip should tell the model everything wrong with its attempt."""
    proposed = ORIGINAL.replace('"""Module docstring that must survive."""\n', "").replace(
        "    return x", "    return x  # type: ignore"
    )

    result = check(ORIGINAL, proposed)

    assert "module docstring" in result.reason
    assert "type: ignore" in result.reason
