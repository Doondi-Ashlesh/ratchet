# Failure log

Every failure, its actual root cause, and the fix at the class level rather than
the instance. The third line is the one that matters and the one that's tempting
to skip.

---

## 001 · Estimated 10 --strict errors, actual 111

**observed** — Predicted ~10 errors in SdkAgent (~1.6k LOC, recently written and
reviewed). Actual: 111 across 21 files. Off by 11x.

**root cause** — Estimated from how the code *felt* — clean, reviewed, working —
rather than from rule density. `--strict` adds constraints that correct-looking
code has no reason to satisfy. Feeling clean and being clean are unrelated
properties.

**class fix** — Never estimate static-analysis volume by intuition; run the tool.
For the harness: the session planner reads the real count, never a model's guess
at it.

---

## 002 · 17 of 111 errors are not fixable by editing source

**observed** — `import-not-found` = 17, all third-party imports with no stubs
installed. Also `misc` errors that exist *only* because an import failed
(`Class cannot subclass "BaseModel" (has type "Any")`).

**root cause** — mypy reports environment problems, their downstream effects, and
genuine defects in one undifferentiated stream. Nothing in the record says which
is which.

**class fix** — Triage before dispatch. An agent handed an unfixable error will
make the count fall by suppressing it, which is the cheating mode the ratchet
exists to prevent. Built `classify.py`.

---

## 003 · 22% of errors should never reach the agent

**observed** — 111 errors → 87 annotation, 17 config, 7 cascading. All 7
cascading errors were in files that also had an unresolved import.

**root cause** — Cascading errors are downstream symptoms. Fixing the 17 config
problems removes 24 errors with zero source edits.

**class fix** — **Config before code.** The session planner resolves environment
problems first and re-measures, then dispatches only what remains. Otherwise the
agent spends sessions on errors that would have evaporated.

---

## 004 · Rules derived from one sample did not generalize

**observed** — `unknown: 0` on `src/sdk_agent`, the codebase the rules were built
from. On held-out code (`SdkAgent/tests`): `unknown: 5`.

**root cause** — The rule set was fitted to a sample of one. Zero unknowns on the
source it came from is circular, not validation.

**class fix** — Validate a classifier on data it was not derived from, before
anything depends on it. The fail-closed default meant those 5 were flagged rather
than silently mis-routed — that default is load-bearing, not a nicety.

---

## 005 · "fixable" was too coarse and hid real defects

**observed** — Of the 5 unknown codes, all were editable. But two —
`attr-defined` ("Exception has no attribute errors") and `func-returns-value`
("append does not return a value") — mean mypy believes the code is **wrong**,
not merely unannotated. One of them is a real bug in a test file
(`calls.append(role) or {...}` — works by accident).

**root cause** — Conflated "an edit resolves this" with "an edit is safe here."
An agent given `attr-defined` will silence it with a cast; the count falls and a
real bug becomes type-checker-approved.

**class fix** — Split into ANNOTATION (mechanical, agent may attempt) and DEFECT
(escalate to a human). Route on **what mypy is asserting**, not on whether an
edit is possible.
