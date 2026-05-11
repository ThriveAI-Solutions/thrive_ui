"""After Phase 3 PR 1 the role gate on executed-SQL / tool-data display is removed.

These tests assert that the tool-call renderer surfaces SQL and result rows
for any role, including PATIENT — the durable directive is "give everybody
the highest permissions" (see memory/feedback_default_permissive_roles.md).
"""

from __future__ import annotations
import json
from unittest.mock import MagicMock, patch

import pytest


def _payload_with_sql_and_rows():
    return {
        "tool_name": "get_patient_clinical_data",
        "arguments": {"query": {"domain": "labs"}},
        "result_summary": "row_count=3",
        "success": True,
        "elapsed_ms": 42,
        "error": None,
        "sql_executed": [{"sql": "SELECT 1", "params": {}}],
        "result_payload": {"items": [{"a": 1}, {"a": 2}, {"a": 3}]},
    }


@pytest.mark.parametrize("role_value", [0, 1, 2, 3, None])
def test_executed_sql_renders_regardless_of_role(role_value):
    """Patient (3) used to be blocked; after PR 1, every role sees SQL."""
    from utils.renderers.tool_call import render_tool_call_message

    message = MagicMock(content=json.dumps(_payload_with_sql_and_rows()))

    with patch("utils.renderers.tool_call.st") as mock_st:
        mock_st.session_state = {"user_role": role_value}
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=None)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        render_tool_call_message(message, index=0)

    # The renderer should have called st.expander at least twice with labels
    # that include "View SQL" and "View rows" — proving the gated branch fired.
    labels = [call.args[0] for call in mock_st.expander.call_args_list]
    assert any("View SQL" in label for label in labels), labels
    assert any("View rows" in label for label in labels), labels


def _msg_with(payload: dict):
    m = MagicMock()
    m.content = json.dumps(payload)
    return m


_PAYLOAD_LEGACY = {
    "tool_name": "find_patient",
    "arguments": {"last_name": "Smith"},
    "result_summary": "match_count=2",
    "success": True,
    "elapsed_ms": 12,
    "error": None,
    "reliability_note": None,
}


def test_renderer_handles_missing_observability_fields():
    """Old persisted messages (pre-this-feature) won't have sql_executed
    or result_payload. Renderer must not crash."""
    from utils.renderers.tool_call import render_tool_call_message

    with patch("utils.renderers.tool_call.st") as st_mod:
        st_mod.session_state = {"user_role": 0}
        st_mod.expander.return_value.__enter__ = MagicMock(return_value=None)
        st_mod.expander.return_value.__exit__ = MagicMock(return_value=False)
        render_tool_call_message(_msg_with(_PAYLOAD_LEGACY), index=0)
        # Reaches the end without raising — that's the assertion.
