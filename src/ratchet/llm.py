"""Model plane, built on NVIDIA's own LangChain integration.

This replaces a hand-written OpenAI client, a hand-written retry loop, a
hand-written tool-call parser and hand-written JSON tool schemas. All four
existed because the agent loop needed them before there was a reason to reach for
a framework; none of them was ever the point of this project, and each one was a
place for a bug that someone else had already fixed.

`ChatNVIDIA` is the maintained integration for NIM endpoints, so model listing,
tool binding, streaming, token accounting and the OpenAI-compatible plumbing come
from the people who ship the endpoint. Retry policy comes from LangChain's
`with_retry`. What remains here is configuration and a test seam.

Configure:
    NVIDIA_API_KEY          required for live calls
    NIM_BASE_URL            default https://integrate.api.nvidia.com/v1
    RATCHET_MODEL           default nvidia/nemotron-3-super-120b-a12b
    RATCHET_MODEL_ATTEMPTS  default 3
    RATCHET_THINKING        default off
    RATCHET_TIMEOUT_S       default 300
    RATCHET_MAX_TOKENS      default 8192
"""
from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


class Truncated(RuntimeError):
    """The response hit the output limit and was cut off mid-generation.

    Worth its own type. A half-file is not a proposal, and returning the fragment
    turns a budget constraint into an apparent competence problem: observed live,
    a truncated response arrived as "unterminated string literal at line 51", which
    reads as the model being unable to write Python.
    """


def finish_reason(message: BaseMessage) -> str:
    """Why the model stopped, across providers that spell it differently."""
    meta = getattr(message, "response_metadata", {}) or {}
    return str(meta.get("finish_reason") or meta.get("stop_reason") or "")


_injected: BaseChatModel | None = None


def set_model(model: BaseChatModel | None) -> None:
    """Inject a chat model, or None to restore live calls.

    The single seam that keeps the test suite offline. It takes a `BaseChatModel`
    rather than a bespoke callable, so the fakes shipped with LangChain work here
    without a custom double having to imitate a provider's response shape - which
    is exactly where a hand-rolled test double drifts from the real thing.
    """
    global _injected
    _injected = model


def model_id() -> str:
    return os.environ.get("RATCHET_MODEL", "nvidia/nemotron-3-super-120b-a12b")


def base_url() -> str:
    return os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")


def thinking_enabled() -> bool:
    """Whether the model deliberates before answering.

    Off by default, and configurable rather than hard-coded because whether
    reasoning helps on this task is a measurable question, not an opinion. Set
    RATCHET_THINKING=on to benchmark the other side of it.
    """
    return os.environ.get("RATCHET_THINKING", "off").strip().lower() in {"on", "1", "true"}


def timeout_s() -> float:
    """Seconds to wait for one model call.

    The replacement library defaults to 60, which is not enough: a single agent
    turn on a real file routinely runs longer, and the failure looks like the
    endpoint being unreachable rather than a deadline being too short.
    """
    return float(os.environ.get("RATCHET_TIMEOUT_S", "300"))


def max_tokens() -> int:
    """Output budget for one call.

    The replacement library defaults to 1024. A whole-file write does not fit in
    that, and a response cut off mid-file is indistinguishable from a model that
    cannot write Python (log 019).
    """
    return int(os.environ.get("RATCHET_MAX_TOKENS", "8192"))


def attempts() -> int:
    """How many times a failed model call is retried. Read by the agent, which owns
    the retry policy."""
    return max(1, int(os.environ.get("RATCHET_MODEL_ATTEMPTS", "3")))


def chat_model(**overrides: Any) -> BaseChatModel:
    """The configured model.

    Temperature 0 to narrow the distribution, not for reproducibility - a model
    cannot give you that, and Ratchet's reproducibility comes from mypy and the
    gate, both of which are programs.

    Deliberately unwrapped. Retry policy is `ModelRetryMiddleware` on the agent,
    not a decorator here: `create_agent` accepts a chat model rather than an
    arbitrary runnable, and wrapping it would put the retry somewhere the agent
    cannot see it. Keeping this a plain model also means the test seam can inject
    one without reproducing a wrapper's behaviour.
    """
    if _injected is not None:
        return _injected

    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError(
            "No NVIDIA_API_KEY. Set it for live calls, or inject a model with "
            "llm.set_model(...) for offline use."
        )

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    settings: dict[str, Any] = {
        "model": model_id(),
        "api_key": key,
        "base_url": base_url(),
        "temperature": 0,
    }

    # Reasoning off by default, and this is not only a cost decision.
    #
    # Nemotron is a reasoning model. Left on, it returns its deliberation as the
    # assistant's CONTENT with no tool call attached, and an agent loop reads a
    # message with no tool call as "finished". Measured live: three files in a row
    # reported "no change written" after two steps, because the model had narrated
    # what it intended to do instead of doing it.
    #
    # The cost difference is the same size as the correctness one. On an identical
    # prompt: 423 output tokens with reasoning on, 56 with it off.
    #
    # Passed as a plain keyword rather than `extra_body`, which this integration
    # rejects. It is forwarded to the endpoint as a model kwarg, which is where NIM
    # expects it.
    settings["chat_template_kwargs"] = {"enable_thinking": thinking_enabled()}

    # Both of these were configured on the old hand-written client and were lost in
    # the swap, because the replacement has its own defaults and neither is a
    # constructor argument. Measured cost of the loss: 30 of 80 model calls in one
    # benchmark failed with `Read timed out. (read timeout=60)`, which surfaced as
    # sessions that burned ten minutes, reported zero tokens, and wrote nothing.
    #
    # The lesson is not "restore the settings". Adopting a library means adopting
    # its defaults, and the defaults it ships are the ones its usual callers want,
    # not necessarily the ones the previous code had.
    settings["max_tokens"] = max_tokens()
    settings.update(overrides)

    model = ChatNVIDIA(**settings)

    # `timeout` is not a field on the chat model; it lives on the client underneath,
    # where 60s is the default. Set after construction because there is no
    # constructor path to it. Private attribute, so it is asserted in the tests: if
    # a future version moves it, the suite should say so rather than the next
    # benchmark quietly losing a third of its calls again.
    model._client.timeout = timeout_s()

    checked: BaseChatModel = model
    return checked
