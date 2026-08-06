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
  everything  the worked category must fall and NO category may rise.
  else

The per-category rule is deliberately conservative, and it replaced a
total-must-fall rule that let a real failure through. Observed live: the model
annotated a function with three type names it never imported, four annotation
errors went away, three unclassifiable ones appeared, and a net of minus one was
enough to keep it. Judging the total lets a session trade errors between
categories; judging each category does not.

It will also reject correct work that surfaces pre-existing problems, because
annotating a function makes its call sites checkable and those may hold real
errors. That is the intended trade: rejection discards and escalates rather than
condemns, and a fix that surfaces three latent bugs should stop the line.
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

    # A source edit cannot reveal anything mypy was previously blind to, the way a
    # config fix can, so no category may grow. This subsumes a total check: if the
    # worked category fell and nothing else rose, the total fell by construction.
    #
    # It does reject correct work that surfaces pre-existing problems — annotating
    # a function makes its call sites checkable, and those may hold real errors.
    # That is the intended trade: rejection discards and escalates, it does not
    # condemn, and a fix that surfaces three latent bugs should stop the line.
    grew = [(c.value, deltas[c.value]) for c in Category if deltas[c.value] > 0]
    if grew:
        detail = ", ".join(f"{name} +{n}" for name, n in grew)
        return Verdict(False, f"regressed: {detail}", deltas)

    return Verdict(
        True,
        f"{worked_on.value} {before.get(worked_on)} -> {after.get(worked_on)}; "
        f"total {before.total} -> {after.total}",
        deltas,
    )
