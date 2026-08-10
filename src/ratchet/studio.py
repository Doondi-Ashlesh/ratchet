"""Entry point for LangGraph Studio and the LangGraph server.

`ratchet run` is the unattended interface: point it at a package and read the
report afterwards. This is the attended one. It exposes the same graph over
LangGraph's server so a run can be started by hand, watched node by node, paused
at the escalation, inspected, and resumed from a checkpoint.

Nothing here is a second implementation. It is the same `build()` the CLI drives,
so anything seen in Studio is the behaviour that runs in production rather than a
demonstration of it.

    langgraph dev

Then set `target` to the package you want worked on. Everything else has a
default; the flags on `ratchet run` map to the same names.
"""
from __future__ import annotations

from typing import Any

from ratchet.certs import use_os_certificates
from ratchet.loop import build

# The server is a second application entry point, so it needs the same TLS setup
# `main()` does. Without it the graph still runs and every trace upload fails, which
# is the quietest possible failure: the work looks fine and the record of it is
# silently lost. Fourth client to need this (failure-log 021, 023).
use_os_certificates()

# Checkpointing is deliberately not configured here. The server supplies its own
# persistence, and passing one as well would give a run two competing stores and a
# thread id that means something different to each.
orchestrator: Any = build(require_approval=True)
"""The multi-file loop, compiled to pause before it escalates.

`require_approval=True` where the CLI defaults to False, because the two
interfaces exist for opposite reasons. A batch run that stops for a human is not a
batch run. An interactive session is where a human is already present, and the
escalation is the one point where the harness has decided it cannot proceed alone.
"""
