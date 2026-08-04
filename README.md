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

Early. The single-file loop works end to end against a real model and a real codebase.

**Working:** measurement, triage, the gate, one session with bounded retry, the CLI.

**Not built yet:** the multi-session loop, resumable state across runs, a separate evaluator, a clean-tree precondition.

The measured baseline is not yet a property of the target alone, because mypy runs in Ratchet's environment. Installing a package into Ratchet changes what it reports about your repo. See [failure log 012](docs/failure-log.md).

---

## What running it live taught us

The [failure log](docs/failure-log.md) is the most useful document in this repository. Every design decision above came out of a measurement that contradicted an expectation. A few:

**Guessing does not work.** Predicted ~10 `--strict` errors in a 1,600-line codebase that had just been reviewed. Actual: 111. Feeling clean and being clean are unrelated properties.

**A fifth of the errors were not the agent's to fix.** 17 of 111 were unresolved imports, and 7 more existed only because of those. Fixing the imports removed 24 errors with no source edit at all.

**Rules fitted to one codebase do not generalise.** Zero unknowns on the source they were derived from; five on held-out code. The fail-closed default is what made that visible instead of silently mis-routed.

**A prompt is a request.** Told not to use `Any`, the model returned `dict[object, object]` instead. Equally vacuous, and because `dict` is invariant it swapped one error for another.

**Temperature 0 is not determinism.** Same file, same prompt, three runs, three distinct failure modes. All three shared one root cause: annotations referencing names never imported.

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
