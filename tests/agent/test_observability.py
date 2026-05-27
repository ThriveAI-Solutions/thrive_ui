"""Per spec §8.7 — Logfire MUST NOT transmit to logfire.pydantic.dev.

This test guards against a future change that accidentally re-enables
cloud telemetry.
"""

import pytest
from agent.observability import configure_observability, _send_to_logfire_setting


def test_send_to_logfire_is_false_by_default():
    assert _send_to_logfire_setting() is False


def test_configure_observability_does_not_raise():
    configure_observability()
    configure_observability()  # idempotent
