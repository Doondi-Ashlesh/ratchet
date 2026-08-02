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

Deliberately absent: retries, backoff, fallback tiers, token accounting. Those
are real and will be needed, but adding them before a single live call has been
made would be designing for imagined failures. They go in when a run fails.

Configure:
    NVIDIA_API_KEY   required for live calls
    NIM_BASE_URL     default https://integrate.api.nvidia.com/v1
    RATCHET_MODEL    default nvidia/nemotron-3-super-120b
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

Completer = Callable[[str], str]

_injected: Completer | None = None


def set_completer(fn: Completer | None) -> None:
    """Inject a fake completer, or pass None to restore live calls."""
    global _injected
    _injected = fn


def model_id() -> str:
    return os.environ.get("RATCHET_MODEL", "nvidia/nemotron-3-super-120b-a12b")


def _base_url() -> str:
    return os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")


def _client() -> Any:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError(
            "No NVIDIA_API_KEY. Set it for live calls, or inject a fake with "
            "model.set_completer(fn) for offline use."
        )
    from openai import OpenAI  # lazy: only needed for live calls

    return OpenAI(base_url=_base_url(), api_key=key, timeout=120)


def complete(prompt: str) -> str:
    """One completion. Returns the raw text, unparsed.

    Parsing belongs to the caller: the agent knows it asked for a Python file,
    so it knows how to read the answer. This layer only moves strings.
    """
    if _injected is not None:
        return _injected(prompt)

    resp = _client().chat.completions.create(
        model=model_id(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return str(resp.choices[0].message.content or "")
