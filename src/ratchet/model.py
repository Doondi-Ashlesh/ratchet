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

import os
import random
import time
from collections.abc import Callable
from typing import Any

from langsmith import traceable

Completer = Callable[[str], str]

_injected: Completer | None = None
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

    last: BaseException | None = None
    for attempt in range(_max_attempts()):
        try:
            return _live_call(prompt)
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
