"""Tool-call card surfaces SQL + raw rows behind role-gated expanders."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from utils.renderers.tool_call import render_tool_call_message


def _msg_with(payload: dict):
    m = MagicMock()
    m.content = json.dumps(payload)
    return m


_PAYLOAD = {
    "tool_name": "find_patient",
    "arguments": {"last_name": "Smith"},
    "result_summary": "match_count=2",
    "success": True,
    "elapsed_ms": 12,
    "error": None,
    "reliability_note": None,
    "sql_executed": [
        {
            "sql": "SELECT ... FROM dw.internal_patient_profile_v WHERE last_name = :ln",
            "params": {"ln": "Smith"},
        }
    ],
    "result_payload": {
        "matches": [
            {"source_id": "src-1", "display_name": "John Smith", "dob": "1962-05-01"},
            {"source_id": "src-2", "display_name": "Jane Smith", "dob": "1985-03-12"},
        ],
        "total_unique": 2,
        "truncated": False,
    },
}


def test_renderer_calls_gate_with_session_role():
    """The renderer must consult the role gate using st.session_state.user_role."""
    with (
        patch("utils.renderers.tool_call.st") as st_mod,
        patch("utils.renderers.tool_call.role_can_see_query_details") as gate,
    ):
        gate.return_value = False
        st_mod.session_state = {"user_role": 1}
        render_tool_call_message(_msg_with(_PAYLOAD), index=0)
        gate.assert_called_once_with(1)


def test_admin_sees_sql_expander():
    """When gate allows, st.expander must be called for 'View SQL'."""
    with (
        patch("utils.renderers.tool_call.st") as st_mod,
        patch("utils.renderers.tool_call.role_can_see_query_details", return_value=True),
    ):
        st_mod.session_state = {"user_role": 0}
        render_tool_call_message(_msg_with(_PAYLOAD), index=0)
        # Collect all labels passed to st.expander.
        labels = [c.args[0] for c in st_mod.expander.call_args_list]
        assert any("SQL" in label for label in labels), labels


def test_admin_sees_rows_expander():
    with (
        patch("utils.renderers.tool_call.st") as st_mod,
        patch("utils.renderers.tool_call.role_can_see_query_details", return_value=True),
    ):
        st_mod.session_state = {"user_role": 0}
        render_tool_call_message(_msg_with(_PAYLOAD), index=0)
        labels = [c.args[0] for c in st_mod.expander.call_args_list]
        assert any("row" in label.lower() or "data" in label.lower() for label in labels), labels


def test_non_admin_does_not_see_sql_or_rows_expanders():
    with (
        patch("utils.renderers.tool_call.st") as st_mod,
        patch("utils.renderers.tool_call.role_can_see_query_details", return_value=False),
    ):
        st_mod.session_state = {"user_role": 1}
        render_tool_call_message(_msg_with(_PAYLOAD), index=0)
        labels = [c.args[0] for c in st_mod.expander.call_args_list]
        # The outer card expander still fires (with the tool name); the
        # inner SQL/Rows ones must NOT.
        sql_labels = [l for l in labels if "SQL" in l]
        row_labels = [l for l in labels if "row" in l.lower() or "data" in l.lower()]
        assert sql_labels == []
        assert row_labels == []


def test_renderer_handles_missing_observability_fields():
    """Old persisted messages (pre-this-feature) won't have sql_executed
    or result_payload. Renderer must not crash."""
    legacy = dict(_PAYLOAD)
    legacy.pop("sql_executed")
    legacy.pop("result_payload")
    with (
        patch("utils.renderers.tool_call.st") as st_mod,
        patch("utils.renderers.tool_call.role_can_see_query_details", return_value=True),
    ):
        st_mod.session_state = {"user_role": 0}
        render_tool_call_message(_msg_with(legacy), index=0)
        # Reaches the end without raising — that's the assertion.
