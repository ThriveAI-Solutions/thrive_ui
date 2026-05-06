"""Local-only observability for the Pydantic AI agent.

Per spec §8.7: Logfire instrumentation is OTel-based. We use it for
per-tool spans but configure it to NEVER transmit to logfire.pydantic.dev.
Span data is HIPAA-sensitive (may contain PHI in tool arguments) and must
stay on the locked-down server.

If a downstream OTel collector (Jaeger, OpenObserve) is configured via
OTEL_EXPORTER_OTLP_ENDPOINT, spans flow there. Otherwise spans are
in-process only.
"""

from __future__ import annotations
import os
import logfire


def _send_to_logfire_setting() -> bool:
    """Hardcoded False. DO NOT change without re-reading spec §8.7."""
    return False


_configured = False


def configure_observability() -> None:
    global _configured
    if _configured:
        return

    logfire.configure(
        send_to_logfire=_send_to_logfire_setting(),
        service_name="thrive-agent",
        environment=os.getenv("THRIVE_ENV", "local"),
    )
    logfire.instrument_pydantic_ai()
    _configured = True
