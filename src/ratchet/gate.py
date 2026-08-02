"""The ratchet: decide whether a session's work may be kept.

A gauge reports the number. A ratchet refuses to let it turn backward. This is
the second half, and it is the piece that makes "the count only goes down" a
property rather than a hope.

The naive rule — accept if the total fell — is wrong, because fixing a config
error can legitimately make the total RISE. When mypy cannot resolve an import it
cannot analyze what depends on it, so it under-reports. Install the stubs and
errors appear that were always there and were merely invisible. That is progress
measured as a regression.

So the verdict depends on what the session was working on:

  CONFIG      config must fall. The total may rise; those are reveals.
  everything  the total must fall AND defect must not rise.
  else

The defect rule is deliberately conservative. A newly revealed defect was not
caused by the agent — but it is still something a human must look at, so the
session is held rather than waved through. Blocking on a problem the agent did
not create is the correct trade when the alternative is blessing it.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ratchet.classify import Category, Classified, summary


@dataclass(frozen=True)
class Measurement:
    """Category counts at a point in time."""

    counts: Mapping[str, int]

    @classmethod
    def of(cls, classified: Iterable[Classified]) -> Measurement:
        return cls(summary(classified))

    def get(self, category: Category) -> int:
        return self.counts.get(category.value, 0)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


@dataclass(frozen=True)
class Verdict:
    """Whether the work may be kept, and why. `reason` is written to be read by a
    human at 2am and by the next session's prompt, so it names the numbers."""

    accepted: bool
    reason: str
    deltas: Mapping[str, int]


def _deltas(before: Measurement, after: Measurement) -> dict[str, int]:
    return {c.value: after.get(c) - before.get(c) for c in Category}


def judge(before: Measurement, after: Measurement, worked_on: Category) -> Verdict:
    """Accept or reject a session's work by comparing measurements around it."""
    deltas = _deltas(before, after)
    moved = deltas[worked_on.value]

    if moved >= 0:
        return Verdict(
            False,
            f"no progress: {worked_on.value} went "
            f"{before.get(worked_on)} -> {after.get(worked_on)}",
            deltas,
        )

    if worked_on is Category.CONFIG:
        # A config fix changes what mypy can SEE. Anything that appeared was
        # always there, so a rising total is not a regression here.
        return Verdict(
            True,
            f"config {before.get(Category.CONFIG)} -> {after.get(Category.CONFIG)}; "
            f"total {before.total} -> {after.total} (reveals allowed)",
            deltas,
        )

    if deltas[Category.DEFECT.value] > 0:
        return Verdict(
            False,
            f"introduced or revealed {deltas[Category.DEFECT.value]} defect(s); "
            "a human decides before this is kept",
            deltas,
        )

    if after.total >= before.total:
        return Verdict(
            False,
            f"{worked_on.value} fell but the total did not: "
            f"{before.total} -> {after.total}",
            deltas,
        )

    return Verdict(
        True,
        f"{worked_on.value} {before.get(worked_on)} -> {after.get(worked_on)}; "
        f"total {before.total} -> {after.total}",
        deltas,
    )
