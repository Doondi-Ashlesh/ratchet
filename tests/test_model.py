"""Tests for the model plane's failure handling.

An unattended loop makes dozens of calls, so a rate limit or a slow response is a
normal event rather than an exception. Before this existed, one timeout on the
first file ended a run and discarded every session that had already succeeded.

No network and no wall-clock: the client is injected and the backoff sleep is
swapped out.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from ratchet import model


class _Transient(Exception):
    """Stands in for a 429 or a 503, classified by status_code as the real ones are."""

    def __init__(self, status: int = 503) -> None:
        self.status_code = status
        super().__init__(f"transient {status}")


class _Permanent(Exception):
    def __init__(self) -> None:
        self.status_code = 400
        super().__init__("bad request")


class _Resp:
    def __init__(self, text: str, finish: str = "stop") -> None:
        msg = type("M", (), {"content": text})()
        self.choices = [type("C", (), {"message": msg, "finish_reason": finish})()]
        self.usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()


class _FakeClient:
    """Replays a scripted sequence of outcomes."""

    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.calls = 0
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs: Any) -> _Resp:
        outcome = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if isinstance(outcome, _Resp) else _Resp(outcome)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    model.set_completer(None)
    model.set_sleep(lambda _s: None)
    yield
    model.set_sleep(None)
    model.set_completer(None)


def _script(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> _FakeClient:
    client = _FakeClient(outcomes)
    monkeypatch.setattr(model, "_client", lambda: client)
    return client


def test_a_transient_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _script(monkeypatch, [_Transient(429), _Transient(503), "ok"])

    assert model.complete("p") == "ok"
    assert client.calls == 3


def test_a_permanent_failure_fails_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrying a 400 spends the budget to receive the same error."""
    client = _script(monkeypatch, [_Permanent()])

    with pytest.raises(Exception) as exc:
        model.complete("p")

    assert getattr(exc.value, "status_code", None) == 400
    assert client.calls == 1


def test_exhausting_every_attempt_raises_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct from a bad answer: the caller should back off, not rephrase."""
    _script(monkeypatch, [_Transient()])

    with pytest.raises(model.ModelUnavailable, match="failed"):
        model.complete("p")


def test_truncation_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cut-off response is a budget problem. Retrying it produces the same cut."""
    client = _script(monkeypatch, [_Resp("half a file", finish="length")])

    with pytest.raises(model.Truncated):
        model.complete("p")

    assert client.calls == 1


def test_attempt_count_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RATCHET_MODEL_ATTEMPTS", "1")
    client = _script(monkeypatch, [_Transient()])

    with pytest.raises(model.ModelUnavailable):
        model.complete("p")

    assert client.calls == 1


def test_backoff_grows_and_is_jittered() -> None:
    """Lockstep retries across concurrent sessions rebuild the spike that caused
    the rate limit in the first place."""
    early = [model._backoff(0) for _ in range(20)]
    later = [model._backoff(2) for _ in range(20)]

    assert min(later) > max(early)
    assert len(set(early)) > 1


def test_transient_classification_covers_status_codes_and_type_names() -> None:
    assert model._is_transient(_Transient(429))
    assert model._is_transient(_Transient(503))
    assert not model._is_transient(_Permanent())
    assert model._is_transient(type("APITimeoutError", (Exception,), {})())
    assert not model._is_transient(ValueError("nope"))


def test_usage_counts_what_was_billed_not_what_was_attempted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport failure returns no response, so nothing was generated and
    nothing was billed — counting it would inflate the cost figure. A truncated
    response was generated and paid for, so it counts even though it is unusable.
    """
    model.reset_usage()
    _script(monkeypatch, [_Transient(), "ok"])
    model.complete("p")

    assert model.usage()["calls"] == 1          # the failed attempt cost nothing

    model.reset_usage()
    _script(monkeypatch, [_Resp("half a file", finish="length")])
    with pytest.raises(model.Truncated):
        model.complete("p")

    assert model.usage()["calls"] == 1          # unusable, but billed
    assert model.usage()["output"] == 5
