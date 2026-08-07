# ratchet

[![CI](https://github.com/Doondi-Ashlesh/ratchet/actions/workflows/ci.yml/badge.svg)](https://github.com/Doondi-Ashlesh/ratchet/actions/workflows/ci.yml)

A harness for long-running LLM agents, built around a progress metric that can only move one way.

The first thing it drives is `mypy --strict` on a Python codebase, one file per session, because that gives an objective oracle and a number that is hard to fake.

---

## The problem

An agent asked to "fix the type errors" will fix some of them and quietly suppress the rest. The error count goes down. Nothing improved.

You cannot prompt your way out of this. A model told not to write `# type: ignore` mostly complies, and "mostly" is not a property you can build on. Enforcement has to be a deterministic check that the model does not control.

Ratchet is that check, plus the machinery around it.

---

## How it works

```
run_mypy   ->  measure        what is wrong
classify   ->  triage         who may touch it
agent      ->  propose        the model's only job
session    ->  apply          write it to disk
run_mypy   ->  re-measure     what happened
gate       ->  judge          keep it, or revert byte for byte
```

Every step except `propose` is deterministic. The model proposes; the harness disposes.

That is one session, one file. A codebase needs many, and the loop over them is a LangGraph state machine rather than a `while`:

```
        ┌───────────────┐
START ─→│   preflight   │  measure once, build the queue,
        └───────┬───────┘  drop what history says keeps failing
                │
                ▼
        ┌───────────────┐
        │     work      │◀─┐  one file, one session
        └───────┬───────┘  │
                │──────────┘  queue not empty
                │
       ┌────────┴────────┐
       ▼                 ▼
  ┌──────────┐         END
  │ escalate │  nothing succeeded; stop and ask a person
  └──────────┘
```

A `while` loop would run these sessions. It would not survive being killed, would not resume where it stopped, and would have no pause point a human could approve at. Those three are framework features here — checkpointing, resumption, and `interrupt_before` — rather than machinery to build and then prove correct.

### Triage

Not every error is work an agent should attempt. Errors are routed by what it would actually take to fix them:

| Category | Meaning | Goes to |
|---|---|---|
| `annotation` | types are missing, the code is fine | the agent |
| `defect` | mypy believes the code is **wrong** | a human |
| `config` | unresolved import; no source edit helps | fix first, re-measure |
| `cascading` | downstream of a config error | nobody, fix the root |
| `unknown` | no rule for this code | a human, never guessed at |

The default is `unknown`, never a workable category. An agent handed an error it cannot fix will silence it instead, which is exactly the failure this exists to prevent.

### The gate

On a source-editing session, the worked category must fall and **no** category may rise. Judging the net total lets a session trade errors between categories, which is not a hypothetical: an early live run fixed four annotation errors, introduced three unclassifiable ones, and was accepted on a margin of one.

Config sessions are the exception. Fixing an unresolved import changes what mypy can *see*, so errors that appear were always there. A rising total is allowed there and only there.

### Retry

A rejected attempt is retried with the specific diagnostics it introduced, bounded, always starting again from the original file rather than patching its own broken output. Exhausting the attempts is an escalation with a record of what was tried, not a silent give-up.

---

## Install

```bash
git clone https://github.com/Doondi-Ashlesh/ratchet.git
cd ratchet
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e ".[dev]"
```

`ratchet check` has **no third-party runtime dependencies**. The agent needs one:

```bash
pip install -e ".[agent]"
```

---

## Usage

### Measure a codebase

```bash
ratchet check path/to/package
```

```
ratchet check src/sdk_agent

  annotation     84   agent may attempt
  defect          2   human decides
  config         13   fix first, then re-measure
  cascading       3   downstream; do not touch
  unknown         0   escalate, never guess

  total         102

next: resolve 13 config error(s) first - they hide errors mypy cannot see,
      and may clear 3 cascading error(s) for free
```

Exit codes follow mypy and ruff, so it composes into a pipeline:

| Code | Meaning |
|---|---|
| `0` | nothing to report |
| `1` | errors found |
| `2` | the tool itself could not run |

That third one matters. A pipeline has to tell "mypy is not installed" apart from "your code is clean", and both are silence otherwise.

`--json` gives the same result machine-readable.

### Work through a codebase

```bash
ratchet run path/to/package --max-files 3 --deadline 240
```

```
ratchet run src/sdk_agent

  kept      nodes/verify.py       1x  annotation 78 -> 76; total 91 -> 89
  reverted  catalog/catalog.py    1x  exhausted 1 attempts; escalating. Last: guard: you removed t
  skipped   graph.py                  run deadline reached

  errors  91 -> 89
  kept    1/2 session(s)
```

| Flag | Why it exists |
|---|---|
| `--max-files` | bound the blast radius of an unattended run |
| `--max-attempts` | per-file retry budget |
| `--max-failures` | stop dispatching a file that has failed this often *across runs* |
| `--order` | `worst` moves the count fastest, `smallest` is likelier to succeed |
| `--deadline` | seconds; stop dispatching past this and report what was earned |
| `--checkpoint` | SQLite path, so a killed run resumes instead of restarting |
| `--thread` | which run to resume |

`--deadline` is the one that is not obvious. Per-call wall-clock is not predictable from anything you can see beforehand: the slowest session measured took 530 seconds on the *second smallest* file in the package, because the cost was 10k output tokens of reasoning rather than input size. A file-size cap would not have caught it. Bounding the run does.

### Memory across runs

Every verdict is written to `.ratchet/state.json` **in the target repository**, and it is meant to be committed. A file that has failed twice is not dispatched a third time, and the reason it is being skipped travels with the code, visible in a diff, rather than living in a cache on whoever ran it last.

### Observability

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=ratchet
```

Traces go to LangSmith. Nodes are not decorated — LangGraph opens a span per node on its own, and decorating them too recorded everything twice. The OpenAI client is wrapped with `langsmith.wrappers.wrap_openai`, so token counts, model id and temperature are read by the code that owns the response shape:

```
chain  work         success  160.7s
chain  complete     success  156.3s
llm    ChatOpenAI   success  154.1s   in/out=1265/10667
```

Each model span carries the `langgraph_node` that spent it, which is what makes "which file ate the budget" a query rather than a guess. Ratchet parses none of this itself. Instrumentation is wrapped in a deliberately blind `except`: traces are diagnostic, the completion is the product, and observability must never be able to break the thing it observes.

### Run one session



```python
from ratchet.session import run_session

result = run_session(target="src/mypackage", path="src/mypackage/graph.py")

print(result.kept)      # was the work kept
print(result.reason)    # regressed: unknown +3
print(result.attempts)  # how many tries
print(result.proposed)  # what the model actually wrote, kept even on rejection
```

Configure the model:

```bash
NVIDIA_API_KEY=...                                   # required for live calls
NIM_BASE_URL=https://integrate.api.nvidia.com/v1     # default
RATCHET_MODEL=nvidia/nemotron-3-super-120b-a12b      # default
```

Any OpenAI-compatible endpoint works. The provider is a base-URL change.

---

## Status

Early, but it runs unattended against a real model and a real codebase now.

**Working:** measurement, triage, the gate, bounded retry, the multi-file loop, cross-run memory, checkpoint/resume, a wall-clock budget, LangSmith tracing, the CLI.

Measured on a 2,300-line target: `errors 96 → 92`, four of four sessions kept, then `91 → 89` on a later bounded run. Every accepted change is a real annotation; every rejected one was reverted byte for byte.

**Not built yet:** a separate evaluator model, a clean-tree precondition on `run`, and the second oracle described in [failure log 020](docs/failure-log.md) — running the target's own linter so the guard stops enumerating style rules it learned one at a time.

The measured baseline is not yet a property of the target alone, because mypy runs in Ratchet's environment. Installing a package into Ratchet changes what it reports about your repo. See [failure log 012](docs/failure-log.md).

---

## What running it live taught us

The [failure log](docs/failure-log.md) is the most useful document in this repository. Every design decision above came out of a measurement that contradicted an expectation. A few:

**Guessing does not work.** Predicted ~10 `--strict` errors in a 1,600-line codebase that had just been reviewed. Actual: 111. Feeling clean and being clean are unrelated properties.

**A fifth of the errors were not the agent's to fix.** 17 of 111 were unresolved imports, and 7 more existed only because of those. Fixing the imports removed 24 errors with no source edit at all.

**Rules fitted to one codebase do not generalise.** Zero unknowns on the source they were derived from; five on held-out code. The fail-closed default is what made that visible instead of silently mis-routed.

**A prompt is a request.** Told not to use `Any`, the model returned `dict[object, object]` instead. Equally vacuous, and because `dict` is invariant it swapped one error for another.

**Temperature 0 is not determinism.** Same file, same prompt, three runs, three distinct failure modes. All three shared one root cause: annotations referencing names never imported.

**A fix scoped to the symptom does not hold.** TLS verification was fixed by handing one client a custom context. It worked, and then trace uploads failed identically, because that was a different client. "Configure this client" scales with the number of clients; patching the layer underneath them does not.

**Declaring a span an LLM call does not make it one.** `@traceable(run_type="llm")` on a function returning `str` rendered correctly in the UI and reported `tokens 0/0` — the response object had already been discarded. Hand-instrumenting what the library instruments cost the exact number the tracing was added to get.

**The harness caught what a human read past.** Reading a diff, `-> dict[object, object]` and three quoted type names looked entirely reasonable. The checker rejected both in about a second.

---

## Development

```bash
ruff check .
mypy
pytest -q
```

All three run in CI on every pull request, and `main` is protected.

`mypy --strict` is enforced on Ratchet itself. A tool that drives other codebases to strict-clean while exempting its own would not deserve to be believed.

---

## Design notes

- **Tools never raise into the agent.** Every outcome, including failure, comes back as a typed result with a stable `error_type`. An exception gives a model a stack trace; typed failure gives it a decision.
- **mypy runs as a subprocess**, not an imported API, which buys crash isolation, an enforceable timeout, and version independence from the target.
- **The whole target is re-measured**, not just the edited file. A change in one file can create errors in another, and a session that checked only its own would export the mess to a neighbour and report success.
- **Rejected work is reverted byte for byte.** Reads and writes disable newline translation; without that, a "revert" on an LF repo differs on every line.
- **Rejected proposals are kept.** A refused attempt is evidence the next one needs, not garbage.
