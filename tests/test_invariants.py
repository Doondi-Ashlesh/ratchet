"""Tests for the guard's four invariants.

The guard used to be a denylist: one rule per incident, always a case behind. It
is now a statement of what an annotation edit is ALLOWED to be. These tests are
written against the invariants rather than against the incidents, which means
several of them describe damage nobody has actually seen yet - that is the point.

The real file this was built from is `views.py`: a proposal that added `-> None`
to two functions, fixed 18 errors and introduced 12, four of them real defects,
entirely because half-annotating a function makes mypy check a body it had been
skipping.
"""
from __future__ import annotations

from ratchet.guard import check

PLAIN = '''"""Module docs."""


def f(x):
    """Does a thing."""
    return x
'''


# ── invariant 1: only annotations may change ─────────────────────────────────


def test_adding_annotations_is_allowed() -> None:
    fixed = '''"""Module docs."""


def f(x: int) -> int:
    """Does a thing."""
    return x
'''
    assert check(PLAIN, fixed).ok


def test_adding_an_import_for_an_annotation_is_allowed() -> None:
    """An annotation frequently needs a name the file does not yet import.
    Forbidding that would make the guard reject correct work."""
    fixed = '''"""Module docs."""
from typing import Any


def f(x: dict[str, Any]) -> int:
    """Does a thing."""
    return x
'''
    assert check(PLAIN, fixed).ok


def test_a_type_checking_block_may_be_added() -> None:
    """The standard way to import a name that exists only for annotations."""
    fixed = '''"""Module docs."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def f(x: "Sequence[int]") -> int:
    """Does a thing."""
    return x
'''
    assert check(PLAIN, fixed).ok


def test_reformatting_without_changing_the_program_is_allowed() -> None:
    """The invariant compares the tree, not the text, so wrapping a long
    signature is not a violation."""
    fixed = '''"""Module docs."""


def f(
    x: int,
) -> int:
    """Does a thing."""
    return x
'''
    assert check(PLAIN, fixed).ok


def test_changing_a_statement_is_refused() -> None:
    """Not a rule about any particular damage. Anything that is not an annotation
    is out of scope by construction."""
    changed = PLAIN.replace("return x", "return x + 1")
    result = check(PLAIN, changed)

    assert not result.ok


def test_adding_a_statement_is_refused() -> None:
    """Damage nobody has seen yet, and the invariant covers it anyway."""
    changed = PLAIN.replace("    return x", "    print(x)\n    return x")

    assert not check(PLAIN, changed).ok


def test_removing_an_import_is_refused() -> None:
    """Imports may be added but never removed: dropping one is how an annotation
    edit silently changes what the module does."""
    before = "import os\n\n\ndef f(x):\n    return os.getcwd()\n"
    after = "def f(x: int) -> str:\n    return os.getcwd()\n"
    result = check(before, after)

    assert not result.ok
    assert "import" in result.reason


def test_a_removed_function_is_named_not_just_refused() -> None:
    """The invariant enforces; the message explains. A model told what it removed
    can fix it; one told 'something changed' has to diff its own work."""
    before = PLAIN + "\n\ndef g(y):\n    return y\n"
    result = check(before, PLAIN)

    assert not result.ok
    assert "`g`" in result.reason


# ── invariant 2: no newly partial annotation ─────────────────────────────────


def test_annotating_only_the_return_type_is_refused() -> None:
    """The views.py failure. Adding `-> None` makes mypy check a body it was
    skipping, and attributes assigned None get inferred as None."""
    half = '''"""Module docs."""


def f(x) -> int:
    """Does a thing."""
    return x
'''
    result = check(PLAIN, half)

    assert not result.ok
    assert "partly annotated" in result.reason
    assert "x" in result.reason


def test_annotating_only_a_parameter_is_refused() -> None:
    half = PLAIN.replace("def f(x):", "def f(x: int):")
    result = check(PLAIN, half)

    assert not result.ok
    assert "partly annotated" in result.reason


def test_self_does_not_need_an_annotation() -> None:
    """A method annotated everywhere except `self` is complete, and demanding
    `self: Foo` would make the guard reject correct work."""
    before = "class A:\n    def m(self, x):\n        return x\n"
    after = "class A:\n    def m(self, x: int) -> int:\n        return x\n"

    assert check(before, after).ok


def test_an_already_partial_function_is_not_this_proposals_fault() -> None:
    """Inherited debt. Flagging it would make the guard unusable on any codebase
    that already has some, which is all of them."""
    before = "def f(x: int):\n    return x\n\n\ndef g(y):\n    return y\n"
    after = "def f(x: int):\n    return x\n\n\ndef g(y: int) -> int:\n    return y\n"

    assert check(before, after).ok


def test_args_and_kwargs_count_as_parameters() -> None:
    before = "def f(*args, **kwargs):\n    return args\n"
    after = "def f(*args: int, **kwargs) -> tuple[int, ...]:\n    return args\n"
    result = check(before, after)

    assert not result.ok
    assert "kwargs" in result.reason


# ── invariant 3: no vacuous annotation ───────────────────────────────────────


def test_a_bare_any_is_refused() -> None:
    result = check(PLAIN, PLAIN.replace("def f(x):", "def f(x: Any) -> Any:"))

    assert not result.ok
    assert "Any" in result.reason


def test_a_bare_object_is_refused() -> None:
    """What the model reached for when told not to use Any."""
    result = check(PLAIN, PLAIN.replace("def f(x):", "def f(x: object) -> object:"))

    assert not result.ok
    assert "object" in result.reason


def test_any_inside_a_generic_is_allowed() -> None:
    """`dict[str, Any]` is not vacuous. Flagging it fired eleven false positives on
    one file, and a rail that cries wolf is a rail people route around."""
    result = check(PLAIN, PLAIN.replace("def f(x):", "def f(x: dict[str, Any]) -> int:"))

    assert result.ok, result.reason


def test_a_bare_none_parameter_is_refused() -> None:
    """A parameter typed `None` can only ever be None. This is the third face of
    the same evasion, and it is what the views.py proposal effectively produced by
    inference."""
    result = check(PLAIN, PLAIN.replace("def f(x):", "def f(x: None) -> int:"))

    assert not result.ok
    assert "None" in result.reason


def test_a_none_return_is_allowed() -> None:
    """A function that returns nothing is exactly what `-> None` means."""
    before = "def f(x):\n    print(x)\n"
    after = "def f(x: int) -> None:\n    print(x)\n"

    assert check(before, after).ok


def test_a_variable_annotated_none_is_refused() -> None:
    before = "def f():\n    x = None\n    return x\n"
    after = "def f() -> None:\n    x: None = None\n    return x\n"
    result = check(before, after)

    assert not result.ok
    assert "None" in result.reason


# ── invariant 4: nothing suppressed ──────────────────────────────────────────


def test_a_type_ignore_is_refused() -> None:
    result = check(PLAIN, PLAIN.replace("    return x", "    return x  # type: ignore"))

    assert not result.ok
    assert "type: ignore" in result.reason


def test_a_noqa_is_refused() -> None:
    """Same evasion, different checker."""
    result = check(PLAIN, PLAIN.replace("    return x", "    return x  # noqa"))

    assert not result.ok
    assert "noqa" in result.reason


def test_an_inherited_suppression_is_not_this_proposals_fault() -> None:
    before = "def f(x):\n    return x  # type: ignore\n"
    after = "def f(x: int) -> int:\n    return x  # type: ignore\n"

    assert check(before, after).ok


# ── the guard must not raise ─────────────────────────────────────────────────


def test_an_unparseable_proposal_is_refused_not_raised() -> None:
    result = check(PLAIN, "def f(x: int -> int:\n")

    assert not result.ok
    assert "parse" in result.reason


def test_an_unparseable_original_is_not_the_proposals_fault() -> None:
    """Structural comparison is meaningless, and this is not something to reject
    the proposal for."""
    assert check("def f(:\n", PLAIN).ok


def test_extending_an_existing_import_line_is_allowed() -> None:
    """A measured false positive. Adding `Any` to an existing
    `from typing import Dict, List` rewrites that line, and comparing whole
    statements read the original as deleted - refusing a correct edit for
    removing an import it had actually extended."""
    before = "from typing import Dict, List\n\n\ndef f(x):\n    return x\n"
    after = (
        "from typing import Any, Dict, List\n\n\n"
        "def f(x: Dict[str, Any]) -> List[int]:\n    return x\n"
    )

    assert check(before, after).ok, check(before, after).reason


def test_dropping_a_name_from_an_import_line_is_still_refused() -> None:
    """The fix must not become 'imports are never checked'."""
    before = "from typing import Dict, List\n\n\ndef f(x):\n    return x\n"
    after = "from typing import Dict\n\n\ndef f(x: Dict[str, int]) -> int:\n    return x\n"

    assert not check(before, after).ok


def test_a_renamed_import_is_tracked_by_the_name_it_binds() -> None:
    before = "import pandas as pd\n\n\ndef f(x):\n    return pd.DataFrame(x)\n"
    after = "def f(x: int) -> int:\n    return pd.DataFrame(x)\n"

    assert not check(before, after).ok
