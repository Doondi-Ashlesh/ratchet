"""Model plane — the single place an LLM is called from.

Confined to one module on purpose. Every other part of Ratchet is deterministic,
and keeping the one non-deterministic component behind a single function means
the entire test suite runs offline: `set_completer()` swaps in a fake and nothing
else in the codebase knows the difference.

Served through NIM's OpenAI-compatible endpoint, so the provider is a base-URL
change rather than a rewrite. Temperature 0 to narrow the distribution — not for
reproducibility, which a model cannot give you (batching and floating-point
nondeterminism mean a temp-0 call is not guaranteed to repeat). Ratchet's
reproducibility comes from mypy and the gate, both of which are programs.

Retries and backoff went in after a run failed, not before: a rate limit or a slow
response is a normal event across dozens of calls, and one of them ending an
unattended run discards every session that already succeeded.

Configure:
    NVIDIA_API_KEY          required for live calls
    NIM_BASE_URL            default https://integrate.api.nvidia.com/v1
    RATCHET_MODEL           default nvidia/nemotron-3-super-120b-a12b
    RATCHET_TIMEOUT_S       default 300
    RATCHET_MODEL_ATTEMPTS  default 3
    RATCHET_BACKOFF_S       default 1.0
"""
from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from langsmith import traceable

Completer = Callable[[str], str]
Conversant = Callable[[list[dict[str, Any]]], "Reply"]

_injected: Completer | None = None
_injected_chat: Conversant | None = None
_usage = {"input": 0, "output": 0, "calls": 0}


def usage() -> dict[str, int]:
    """Cumulative token counts since the last reset. Read by the benchmark."""
    return dict(_usage)


def reset_usage() -> None:
    _usage.update(input=0, output=0, calls=0)


class Truncated(RuntimeError):
    """The response hit the output limit and was cut off mid-generation.

    Distinct from a bad answer, and worth its own type. A half-file is not a
    proposal. Observed live: a truncated response arrived as "unterminated string
    literal at line 51", which reads as the model writing broken Python when it
    had actually been cut off. Silently returning the fragment turns a budget
    constraint into an apparent competence problem.
    """


def set_completer(fn: Completer | None) -> None:
    """Inject a fake completer, or pass None to restore live calls."""
    global _injected
    _injected = fn


def model_id() -> str:
    return os.environ.get("RATCHET_MODEL", "nvidia/nemotron-3-super-120b-a12b")


def _base_url() -> str:
    return os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")


_cached: tuple[tuple[str, str, float], Any] | None = None


def reset_client() -> None:
    """Drop the cached client. Needed after changing base URL, key or timeout."""
    global _cached
    _cached = None


def _client() -> Any:
    """Built once and reused. A fresh client per call opens a new connection pool
    every time, which on a loop making dozens of calls is both slower and a new
    chance for the TLS handshake to fail."""
    global _cached
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError(
            "No NVIDIA_API_KEY. Set it for live calls, or inject a fake with "
            "model.set_completer(fn) for offline use."
        )
    signature = (_base_url(), key, _timeout())
    if _cached is not None and _cached[0] == signature:
        return _cached[1]
    client = _wrap(_build(key))
    _cached = (signature, client)
    return client


def _wrap(client: Any) -> Any:
    """Let LangSmith instrument the client rather than tracing by hand.

    The hand-rolled span reported `tokens 0/0` in the UI, because a `@traceable`
    function that returns a plain string gives LangSmith no response object to read
    usage off. `wrap_openai` reads the actual response, so model id, token counts
    and cost are attributed by the same code that knows the response shape — and
    keeps working when that shape changes.

    Instrumentation must never be able to break the call it instruments, so any
    failure here degrades to an untraced client rather than propagating. Traces are
    diagnostic; the completion is the product.
    """
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except Exception:  # noqa: BLE001 - deliberately blind; see docstring
        return client


def _build(key: str) -> Any:
    from openai import OpenAI  # lazy: only needed for live calls

    # Verify against the OS certificate store rather than certifi's bundle.
    # On a network that inspects TLS, the proxy's root is installed in the OS
    # store and absent from certifi, so every call fails with a bare
    # "Connection error" that reads as the endpoint being down. Observed live:
    # runs that had worked for days started failing when the network changed,
    # and nothing in the error named the cause.
    try:
        import ssl

        import httpx
        import truststore

        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return OpenAI(
            base_url=_base_url(),
            api_key=key,
            http_client=httpx.Client(verify=ctx, timeout=_timeout()),
        )
    except ImportError:
        return OpenAI(base_url=_base_url(), api_key=key, timeout=_timeout())


def _timeout() -> float:
    return float(os.environ.get("RATCHET_TIMEOUT_S", "300"))


# ── transient-failure handling ────────────────────────────────────────────────
# An unattended loop makes dozens of calls. A rate limit or a slow response is a
# normal event at that volume, and without this one of them ends the whole run
# and discards every session that already succeeded.

_TRANSIENT = {
    "APITimeoutError", "APIConnectionError", "RateLimitError",
    "InternalServerError", "ConnectError", "ReadTimeout", "TimeoutException",
}
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

_sleep: Callable[[float], None] = time.sleep


def set_sleep(fn: Callable[[float], None] | None) -> None:
    """Swap the backoff sleep so retry tests cost no wall-clock."""
    global _sleep
    _sleep = fn or time.sleep


class ModelUnavailable(RuntimeError):
    """Every attempt failed transiently. Distinct from a bad answer: nothing is
    wrong with the request, so the caller should back off rather than rephrase."""


def _is_transient(e: BaseException) -> bool:
    """Retry only what a retry can fix. A 400 will fail identically forever, and
    retrying it spends the budget to receive the same error."""
    status = getattr(e, "status_code", None) or getattr(
        getattr(e, "response", None), "status_code", None
    )
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS
    return type(e).__name__ in _TRANSIENT


def _max_attempts() -> int:
    return max(1, int(os.environ.get("RATCHET_MODEL_ATTEMPTS", "3")))


def _backoff(attempt: int) -> float:
    """Exponential with jitter. Without the jitter, concurrent sessions retry in
    lockstep and rebuild the spike that caused the rate limit."""
    base = float(os.environ.get("RATCHET_BACKOFF_S", "1.0"))
    return float(base * (2**attempt) * (0.5 + random.random() / 2))


_T = TypeVar("_T")


def _with_retries(call: Callable[[], _T]) -> _T:
    """Run `call`, retrying only what a retry can fix.

    Shared by both entry points. Duplicating this per call shape is how one of them
    ends up with a retry policy the other quietly lacks.
    """
    last: BaseException | None = None
    for attempt in range(_max_attempts()):
        try:
            return call()
        except Truncated:
            raise                                   # a budget problem; retrying repeats it
        except Exception as e:
            if not _is_transient(e):
                raise
            last = e
            if attempt < _max_attempts() - 1:
                _sleep(_backoff(attempt))
    raise ModelUnavailable(
        f"{model_id()} failed {_max_attempts()}x "
        f"(last: {type(last).__name__ if last else 'unknown'})"
    ) from last


def complete(prompt: str) -> str:
    """One completion. Returns the raw text, unparsed.

    Parsing belongs to the caller: the agent knows it asked for a Python file,
    so it knows how to read the answer. This layer only moves strings.

    Raises Truncated if the response was cut off. That is the one piece of the
    response this layer does interpret, because the caller cannot tell a complete
    answer from half of one by looking at the text.
    """
    if _injected is not None:
        _usage["calls"] += 1
        return _injected(prompt)
    # A nested def rather than a lambda: `@traceable` types the call as a protocol,
    # and mypy resolves its return through a normal call but not through a lambda,
    # which silently widens the retry helper's type variable to Any.
    def attempt() -> str:
        return _live_call(prompt)

    return _with_retries(attempt)


@traceable(name="complete", run_type="chain")
def _live_call(prompt: str) -> str:
    """The traced boundary. Split out from `complete` so the injected-fake path
    never opens a span — a run whose traces are half real calls and half test
    doubles is worse than no traces.

    A chain span, not an llm one: the wrapped client emits the llm span underneath
    with real token attribution. This span exists for what the wrapper cannot see —
    that a retry happened, and that these attempts belong to one logical completion.
    """
    resp = _client().chat.completions.create(
        model=model_id(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    # Counted before the truncation check: a cut-off response was still paid for,
    # and a cost measurement that ignores failed attempts flatters the numbers.
    used = getattr(resp, "usage", None)
    _usage["calls"] += 1
    if used is not None:
        _usage["input"] += int(getattr(used, "prompt_tokens", 0) or 0)
        _usage["output"] += int(getattr(used, "completion_tokens", 0) or 0)

    choice = resp.choices[0]

    if choice.finish_reason == "length":
        raise Truncated(
            f"{model_id()} hit its output limit; the file is likely too large to "
            f"return whole"
        )

    return str(choice.message.content or "")


# ── tool calling ──────────────────────────────────────────────────────────────
# The agent loop needs the model to choose actions rather than return prose. The
# provider's objects are converted here and nowhere else: the SDK's message shape
# is a dependency this module already carries, and letting it leak into the agent
# would mean a provider change edits the agent too.


@dataclass(frozen=True)
class ToolCall:
    """One action the model asked for.

    `arguments` is already parsed. `malformed` is set when it could not be, which
    is an outcome rather than an error: the model emitted something, the loop must
    tell it what was wrong, and raising here would end a trajectory over a stray
    brace.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    malformed: str = ""


@dataclass(frozen=True)
class Reply:
    """One turn from the model. No tool calls means it considers itself finished."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def done(self) -> bool:
        return not self.tool_calls


def set_conversant(fn: Conversant | None) -> None:
    """Inject a fake tool-calling model, or None to restore live calls."""
    global _injected_chat
    _injected_chat = fn


def converse(messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> Reply:
    """One turn of a tool-calling conversation.

    Stateless on purpose. The transcript belongs to the caller, because the caller
    is the one that has to decide what to drop when it grows too long, and that
    decision needs to be visible where the loop is rather than hidden in here.
    """
    if _injected_chat is not None:
        _usage["calls"] += 1
        return _injected_chat(list(messages))
    def attempt() -> Reply:
        return _live_chat(list(messages), list(tools))

    return _with_retries(attempt)


def _parse_arguments(raw: str) -> tuple[dict[str, Any], str]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        return {}, f"arguments were not valid JSON: {e}"
    if not isinstance(parsed, dict):
        return {}, f"arguments must be an object, got {type(parsed).__name__}"
    return parsed, ""


@traceable(name="turn", run_type="chain")
def _live_chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
    resp = _client().chat.completions.create(
        model=model_id(),
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )

    used = getattr(resp, "usage", None)
    _usage["calls"] += 1
    if used is not None:
        _usage["input"] += int(getattr(used, "prompt_tokens", 0) or 0)
        _usage["output"] += int(getattr(used, "completion_tokens", 0) or 0)

    choice = resp.choices[0]

    # A truncated turn is worse here than in `complete`. There, half a file arrives
    # and is obviously unusable. Here the model may have been cut off mid-tool-call,
    # so the arguments parse cleanly and describe an action it never finished
    # choosing. Treating it as a normal turn would execute that.
    if choice.finish_reason == "length":
        raise Truncated(f"{model_id()} hit its output limit mid-turn")

    calls = []
    for call in choice.message.tool_calls or ():
        arguments, malformed = _parse_arguments(call.function.arguments)
        calls.append(
            ToolCall(id=call.id, name=call.function.name, arguments=arguments, malformed=malformed)
        )

    return Reply(content=choice.message.content or "", tool_calls=tuple(calls))
