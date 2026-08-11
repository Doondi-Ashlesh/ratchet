"""TLS trust, configured once per process.

Its own module because it is needed by every entry point and belongs to none of
them. Four separate clients have now failed on this: the model SDK, the LangSmith
uploader, git, and the LangGraph server. Each failure looked like a different
problem - "connection error", "please confirm your internet connection", a push
rejected - and each was the same cause.

On a network that inspects TLS, the proxy's root certificate is installed in the
OS store and absent from the bundle Python libraries ship with. Patching the ssl
module once covers every client in the process, including ones not yet imported,
which is the difference between fixing the class and fixing it per client.
"""
from __future__ import annotations


def use_os_certificates() -> None:
    """Verify TLS against the OS certificate store, process-wide.

    Call from application entry points only. A library has no business patching
    global `ssl` on import; an application may, and must do it before the first
    client is constructed.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
