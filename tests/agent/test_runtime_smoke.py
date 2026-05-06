"""Smoke test the runtime entry. Real end-to-end Streamlit testing
lives in manual smoke runs; this test asserts the imports compose."""

from unittest.mock import patch, MagicMock


def test_runtime_imports_compose():
    from agent.runtime import run_agentic_message_flow

    assert callable(run_agentic_message_flow)
