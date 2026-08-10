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

---

## 006 · Fixing config changes what is visible, not only what is wrong

**observed** — Installed pydantic stubs against SdkAgent. config 17→13,
cascading 7→3, annotation 87→84, **unknown 0→2**. Total 111→102.

The two new errors were `arg-type` in `contracts.py`. Nothing in the source
changed — only a package was installed. Those errors had always existed and were
invisible while mypy could not resolve pydantic.

**root cause** — mypy under-reports when an import fails; it cannot analyze what
depends on an unresolvable module. So a config fix alters the *denominator*, not
just the numerator.

**class fix** — `judge()` cannot be a subtraction. A CONFIG session is accepted on
config falling, with a rising total allowed, because appearing errors are reveals
rather than regressions. Stated honestly: reveals are proven to occur, their
magnitude is unbounded but unmeasured — here removals won 9 to 2. Bound it by
installing the remaining 13 stubs before the agent depends on this.

---

## 007 · Some codes are irreducibly ambiguous; route them by cost asymmetry

**observed** — Both revealed errors were `arg-type`: pydantic's
`list[ErrorDetails]` passed where `list[dict[str, Any]]` was declared. The code is
correct at runtime — `list` is invariant, so the *annotation* is too narrow. But
`arg-type` is also exactly how a genuinely wrong call reports itself.

**root cause** — mypy cannot distinguish "your annotation is too tight" from "your
call is wrong". Neither can a classifier reading the code alone.

**class fix** — Route by consequence, not by likelihood. A real bug sent to the
agent is silenced with a cast, permanently. An over-narrow annotation sent to a
human costs thirty seconds. So `arg-type` → DEFECT. Volume checked before
committing: 5 defects across 375 errors, 1.3%. Revisit if that grows — the answer
is a function of volume, not principle alone.

---

## 008 · Two of five model ids in SdkAgent do not exist

**observed** — Live call 404'd. `nvidia/nemotron-3-super-120b` is not a real
id; the endpoint serves `nvidia/nemotron-3-super-120b-a12b`. Also
`nvidia/nemotron-3-ultra` → `nvidia/nemotron-3-ultra-550b-a55b`. Only the
nano id was correct.

**root cause** — Ids were written from memory and never exercised. Every
SdkAgent test injects a mock, so the wiring was proven and the world never
was. A live `/solve` would 404 at PLAN.

**class fix** — A hardcoded id will go stale again; correcting it is the
instance fix. The class fix is legibility: the client can already list
`/models`, so a 404 should answer "not found — did you mean X?" rather than
"404 page not found".

---

## 009 · The model routed around the prompt rule instead of breaking it

**observed** — First live proposal changed `-> dict` to
`-> dict[object, object]`. The prompt forbade `Any`; the model used `object`,
which is equally vacuous. Worse, `dict` is invariant, so it introduces a
`return-value` error — net zero errors, and the new one is a defect class.

**root cause** — A prompt is a request. The model complied with the letter of
the rule and discarded its purpose, which no wording prevents.

**class fix** — Enforcement must be a deterministic check. The gate already
rejects this via total-must-decrease, but a guard should name it specifically:
vacuous annotations (`Any`, `object`, `dict[Any, Any]`) are a suppression in
disguise and should be rejected with the reason, so the next attempt gets told
what was wrong.

---

## 010 · read + write silently rewrote every line ending

**observed** — After a rejected session, `git status` showed knowledge.py as
modified while `git diff` was empty and the blob hashes matched. Chasing it
found that `read_file` → `write_file` converts `b"a = 1\nb = 2\n"` into
`b"a = 1\r\nb = 2\r\n"`.

**root cause** — `Path.read_text` / `write_text` do newline translation in text
mode. The revert appeared correct only because SdkAgent's working tree is
already CRLF; on an LF repo the "restore" would have differed on every line.

**class fix** — `newline=""` on both, so the tools move bytes rather than
interpreting them. Test asserts byte-exact round-trip for LF and CRLF inputs.
Fixed in the same pass: the model's proposal is normalised to the file's own
endings and trailing newline before comparison, so `changed` reflects content
rather than formatting. Before that, a byte-identical answer in LF read as a
change and cost a full write-measure-reject cycle.

---

## 011 · mypy's cache leaked results between unrelated targets

**observed** — A session on `<tmp-a>/a.py` received diagnostics whose `file`
field pointed at `<tmp-b>/a.py`. The per-file filter matched nothing, so the
session reported "no annotation work" on a file that plainly had some.

**root cause** — mypy's incremental cache lives in the working directory and keys
modules by *name*. Two unrelated targets each containing a module called `a`
share cache entries, and the second run receives the first one's results —
complete with the first one's paths.

**class fix** — `--cache-dir` per target, keyed by a hash of the resolved path.
Isolation without giving up incrementality. Surfaced by test isolation, but the
real exposure was production: measuring two repos from one working directory
could cross-contaminate, and a ratchet is worth nothing if the measurement is not
trustworthy. Cost is a cold cache per target — suite time went 9s to 22s.

---

## 012 · run_mypy depended on PATH

**observed** — Running the suite through `.venv/Scripts/python.exe` without
activating the venv made every mypy-invoking test fail with
`mypy_not_installed`.

**root cause** — `subprocess.run(["mypy", ...])` resolves by PATH lookup, so the
tool worked or not depending on which shell invoked it. Same class as the empty
`C:\Windows\System32\git` file that shadowed real git for anything using
subprocess.

**class fix** — `[sys.executable, "-m", "mypy", ...]`, which resolves to the
running interpreter's own environment regardless of PATH. Known limitation: this
pins mypy to *Ratchet's* environment, so checking a repo with its own venv and
its own stubs will need that repo's interpreter instead. Already visible:
installing `openai` into Ratchet dropped SdkAgent's measured `config` from 13 to
11 without a line of SdkAgent changing. The measured baseline is not yet a
property of the target alone.

---

## 013 · The gate accepted hallucinated annotations on a one-error margin

**observed** — First accepted live session. The model annotated `build_graph`
with `'BaseCheckpointSaver'` and `'CompiledGraph'`, and `make_checkpointer` with
`'MemorySaver'` — three type names it never imported. Quoted forward references,
so Python never evaluates them and the code still runs; only mypy sees it.

annotation −4, unknown +3, total 100 → 99. **Accepted.**

**root cause** — `judge()` blocked on `defect` rising but not `unknown`, and
otherwise judged the **total**. Judging a net lets a session trade errors between
categories, and a margin of one out of a hundred was enough to keep three
hallucinated annotations.

**class fix** — Per-category, not net: on a non-CONFIG session the worked category
must fall and **no** category may rise. That subsumes the total check — if the
worked category fell and nothing rose, the total fell — so it is one rule where
there were two, and stricter. The reason string now names the offending category
(`regressed: unknown +3`) because the next attempt has to be told what it did.

Accepted cost: this rejects correct work that surfaces pre-existing problems,
since annotating a function makes its call sites checkable. Rejection discards
and escalates rather than condemns, and a fix that surfaces three latent bugs
should stop the line. Rate unknown — two data points, and in this case the three
"reveals" were hallucinations. Log rejections and revisit after ~20 sessions.

**Worth recording:** the harness caught none of this. A human read the deltas and
noticed `unknown +3` next to `total −1`. The next version catches it mechanically,
which is the entire argument for running the thing live before trusting it.

---

## 014 · Three live runs, three failure modes, one root cause

**observed** — Same file, same prompt, temperature 0, three runs:
  1. quoted `'BaseCheckpointSaver'` / `'CompiledGraph'` / `'MemorySaver'` — real
     LangGraph types, never imported. unknown +3.
  2. an unresolvable import. config +1.
  3. `"Optional[Any]"` / `"Any"` with no `typing` import. unknown +6, and it used
     `Any` everywhere despite the prompt forbidding it.

Every run got `annotation: -4` right.

**root cause** — The model writes annotations referencing names not in scope. The
file's imports are in the prompt; it does not reason about them. Temperature 0
narrows the distribution, it does not collapse it — three distinct outputs.

**class fix** — Two, and only the second is enforcement:
  - prompt: state that any type used must already be imported or the import must
    be added. Cheap, and it is still only a request.
  - harness: feed the rejection reason back and retry, bounded. The model is 80%
    correct and misses one consistent thing; discarding that and starting fresh
    throws away the only information that would fix it.

Also: this was only diagnosable because SessionResult now keeps the rejected
proposal. Two runs earlier the evidence was discarded and neither of us could say
what the model had written.


## 015 · The gate is blind to everything the error count cannot see

**observed** — A rejected proposal for `graph.py` had deleted the six-line module
docstring. The prompt forbids it explicitly. Deleting a docstring produces zero
mypy errors, so had the annotations been correct the gate would have accepted it
and the documentation would have been silently lost.

Same run also showed feedback working at the instruction level: the model added
`from typing import Optional, List` and tried to import `CompiledGraph` from
`langgraph.graph`. That name does not exist there. The remaining failure is an
information problem — the model cannot see the library — not an instruction one.

**root cause** — The gate judges one metric. An agent optimising one number is
indifferent to everything the number does not cover: docstrings, comments,
deleted functions, rewritten logic that still type-checks.

**class fix** — A structural guard before the gate, working on the AST rather
than on counts: file parses, top-level definitions preserved, docstrings
preserved, no new `# type: ignore`, no vacuous annotations. It can also give
better feedback than the gate can — "you removed the docstring on build_graph"
is actionable in a way "regressed: unknown +2" is not.


## 016 · Retry with feedback converges — when the task is tractable

**observed** — `output_rails.py`: rejected twice ("no progress", then annotation
84 → 85, actively worse), accepted on attempt 3. The result was correct and
precise: `_PII: dict[str, re.Pattern[str]]`, and an accurate Union for a mixed
return dict. Nothing vacuous, nothing hallucinated, nothing else touched.

**root cause of the earlier failures** — Not the loop. `graph.py` requires
LangGraph's own type names, which the model cannot inspect and was guessing at.
An information gap, not an instruction gap.

**class fix** — None needed; this is the loop working. Recorded because the
retry's cost is only justified if it converges, and now there is evidence it
does: 2 rejections, 1 accept, on a file whose answer was derivable from itself.

---

## 017 · The harness cannot tell a tractable file from an impossible one

**observed** — `graph.py` consumed ~12 API calls and ~20 mypy runs across four
sessions, failing every time for the same underlying reason.
`output_rails.py` succeeded in 3 attempts. Nothing in the harness distinguished
them, and nothing stopped it retrying the first indefinitely.

**root cause** — A session has no memory. Every run starts from zero and rebuilds
the same failure.

**class fix** — The multi-session loop needs a durable record per file: attempts,
verdicts, reasons. A file that has failed N times for the same reason gets
escalated rather than re-dispatched. This is the structured handoff artifact from
Anthropic's harness work, arriving here as a requirement derived from measurement
rather than from reading about it.

---

## 018 · Every model call failed with a bare "Connection error"

**observed** — Live runs that had worked for days started failing with
`openai.APIConnectionError: Connection error.` and nothing else. Isolating it
outside the harness showed a plain client failing and a client verifying against
the OS certificate store succeeding immediately, listing 102 models.

**root cause** — The network began inspecting TLS. The proxy's root certificate
is installed in the Windows certificate store and absent from certifi's bundle,
which is what the openai client uses by default. Every call therefore failed
certificate verification, and the error names none of that.

**class fix** — Verify against the OS store via `truststore`. SdkAgent already had
this fix and Ratchet never got it, which is the real lesson: a fix applied to one
project is not a fix, it is a note. The error message remains the weak point —
"Connection error" for a certificate failure is the same class of unhelpful as
"404 page not found" for a wrong model id.

---

## 019 · One timeout ended a run and discarded the sessions that had succeeded

**observed** — With TLS fixed, the loop reached `gateway/app.py` — the largest
file, dispatched first because the queue is ordered worst-first — and the request
timed out at 120s. `APITimeoutError` propagated out of the session, out of the
graph node, and killed the entire run.

**root cause** — Two independent gaps. The model plane had no retry, so a normal
event at this call volume was fatal. And whole-file rewriting scales with file
size: the model must return every line it did not change, so the largest files are
both the slowest and the most likely to truncate.

**class fix** — Retry transient failures (429, 5xx, timeouts, connection errors)
with jittered backoff; fail fast on anything the caller caused. Raise the default
timeout to 300s. Truncation is explicitly *not* retried — a cut-off response is a
budget problem and the retry reproduces the cut.

Unfixed and now measurable: worst-first ordering sends the hardest file first,
which maximises count reduction and minimises the chance the first session
succeeds. Whether that is the right trade needs the accept-rate-by-file-size
number, not an opinion.

---

## 020 · An accepted change quietly deleted a blank line

**observed** — First successful multi-file run: 4 files, 4 accepted, errors
96 → 92. The diff on `nodes/plan.py` shows the annotation added *and* a blank
line removed, leaving one blank line before a top-level `def` where PEP 8 wants
two. mypy does not care, so the gate accepted it.

**root cause** — Same shape as failure-log 015, which produced the guard: the
gate measures one metric and is blind to everything the metric does not cover.
The guard was then taught about docstrings, `type: ignore`, and vacuous
annotations — a list of the damage seen so far, which does not generalise.
Formatting was simply the next thing nobody had thought of.

**class fix** — Stop enumerating style rules in the guard. The target repo
already declares its own: a `ruff`/`black`/`flake8` config is a machine-readable
statement of what counts as damage there. Run the repo's own linter as a second
oracle and reject a proposal that makes it worse, exactly as the gate does with
mypy. The guard keeps only the checks no linter can express — deleted docstrings,
removed definitions, suppressions.

Also observed: with the vacuous rule narrowed to bare `Any`/`object`, the model
moved to `dict[str, object]`. That is a legitimate annotation for a genuinely
heterogeneous mapping, so allowing it is correct — but the drift is worth
watching. Each time a rule tightens, the next proposal lands just inside it.

---

## 021 · The TLS fix was written per-client, so it only fixed one client

**observed** — With `LANGSMITH_TRACING=true` and a key set, every trace upload
failed: `CERTIFICATE_VERIFY_FAILED ... unable to get local issuer certificate`.
The run itself worked. The model calls went through and the traces did not.

**root cause** — Failure-log 018 diagnosed this correctly (a network that
inspects TLS, root in the OS store, absent from certifi) and then fixed it in the
wrong place: an `httpx` context handed to the OpenAI client. LangSmith's uploader
is a different client — `requests`/`urllib3`, its own certifi bundle — so it was
never covered. The fix was scoped to the symptom that was visible at the time.

**class fix** — `truststore.inject_into_ssl()` once at the CLI entry point,
which patches the `ssl` module process-wide and therefore covers every client,
including ones not imported yet. Not done on module import: a library has no
business monkey-patching global `ssl`, an application does.

The general lesson is about the shape of the first fix, not about TLS. "Configure
this client correctly" scales with the number of clients. "Configure the layer
they all sit on" does not.

---

## 022 · Tracing was wired, enabled, and reported zero tokens

**observed** — First traces to arrive in LangSmith showed every model span as
`tokens in/out = 0/0`. The spans existed, were named correctly, and carried no
usage data at all — so the one question worth asking a trace ("what did this
run cost, and where") was unanswerable.

**root cause** — `@traceable(run_type="llm")` on a function returning `str`.
The decorator records what the function returns; the function had already thrown
the response object away and kept `choices[0].message.content`. Declaring
`run_type="llm"` makes LangSmith *render* a span as a model call. It does not
make usage data appear.

A second defect in the same area: every graph node also carried `@traceable`,
while LangGraph opens a span per node on its own. Every node was therefore
recorded twice, with identical start times and durations.

**class fix** — Stop hand-instrumenting what the library instruments. The client
is wrapped with `langsmith.wrappers.wrap_openai`, so usage is read by the code
that owns the response shape, and the node decorators are gone. Result on the
next run: `in/out = 3540/6146`, `ls_model_name`, `ls_temperature`, and the
`langgraph_node` that spent it, all without Ratchet parsing anything.

Wrapping is guarded by a deliberately blind `except`. Instrumentation must never
be able to break the call it instruments: traces are diagnostic, the completion
is the product.

---

## 023 · A run that cannot finish is indistinguishable from a run that hung

**observed** — Two consecutive unattended runs were killed by an outer `timeout`
after nine minutes with no output at all. Both had done real work — the state
file showed accepted sessions — but nothing was reported, because the summary
only prints at the end.

**root cause** — The traces answered it in one line: a single completion took
**530 seconds** and produced **10,048 output tokens** — for a 197-line file. The
cost was reasoning tokens, not input size. This matters because the obvious fix,
a file-size cap, would not have caught it: that file was the second smallest in
the package. Per-call wall-clock is not predictable from anything visible before
the call.

**class fix** — Bound the run, not the file. `--deadline` stops dispatching new
files once the budget is spent and reports what was already earned, with the
remainder recorded as `run deadline reached`. Checked between sessions, never
inside one, because a session is the smallest unit the gate can keep or revert;
interrupting mid-session abandons a proposal nobody has judged.

This is the first fix in this log that was found by reading a trace rather than
by reading a diff. Three runs were spent guessing at model timeouts before
tracing worked; the answer took one query afterwards.

---

## 024 · Giving the model tools changed the failure mode, not just the interface

**observed** — First live trajectory of the tool-calling agent, on an 85-line
file with 5 annotation errors. Eight turns, `37,602` input and `17,390` output
tokens, and **zero progress**: `errors 5 -> 5`, file byte-identical at the end.
Two `write_file` calls were rejected by the guard with the same message,
`the file no longer parses: line 26: unterminated string literal`.

The single-shot agent had been fixing files of this size in one or two calls.

**first wrong diagnosis** — This is the signature of a truncated response
(log 019), so the assumption was that the model was being cut off mid-write.
It was not. `finish_reason` was `tool_calls`, not `length`, and the file
continued correctly for 59 lines *after* the corruption. Nothing was cut off.

**root cause** — The channel, not the model. Comparing the same line across a
successful write and a failed one:

```
ok    ...pip\s+install\s+(.+)\")\n_VERSION = re.compile(...
bad   ...pip\s+install\s+(.+)_VERSION = re.compile(...
```

A `\")` and a newline are missing. Returning a whole source file as a JSON tool
argument requires the model to escape every quote and backslash correctly across
thousands of tokens, and this file is unusually dense with regex literals
(`r"(?:^|[!\s])pip\s+install\s+(.+)"`). It dropped an escape sequence. The same
file returned as plain text by `propose()` needs no escaping and has never failed
this way, so the defect belongs to how the content was carried, not to the model's
understanding of the task.

**why the retry did not help** — It emitted the identical corruption twice, at
the same line. At temperature 0 the same context regenerates the same tokens, so
feeding back "line 26 is unterminated" cannot fix it: this is not a reasoning
error the model can act on. A bounded retry loop assumes attempts are independent
draws. Where the failure is deterministic, retrying is just paying twice.

Note the tension with log 013, which recorded that temperature 0 is *not*
determinism. Both hold. A temp-0 call is not guaranteed to repeat, and it repeats
often enough that a retry strategy cannot rely on getting a different answer.

**class fix** — Not yet applied; recorded before choosing, because the obvious fix
is the wrong one. Making `write_file` accept a patch instead of a whole file
shrinks the escaping surface, but does not remove it, and it trades one corruption
class for a harder one: a mis-anchored patch that still parses.

What the evidence actually supports is narrower. First, the guard has to run on
every write and not only at the end, which it already does — that is the only
reason this trajectory cost tokens instead of leaving a broken file on disk.
Second, a retry after a deterministic failure must change something about the
request, or it is not a retry.

**what held** — Both bad writes were refused, the file on disk was never touched,
and the trajectory ended byte-identical to its start. The agent gained an action
space and the ability to corrupt a file, and the checks written for the
single-shot agent caught it on the first live run without modification.
