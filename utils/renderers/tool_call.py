"""Render TOOL_CALL messages as collapsible Streamlit cards.

Stored payload (Message.content) is JSON:
    {
        "tool_name": "find_patient",
        "arguments": {...},
        "result_summary": "match_count=3" | null,
        "success": true|false|null,
        "elapsed_ms": 120 | null,
        "error": null | "...",
    }

`null` for result_summary/success/elapsed_ms means the call is in-flight;
the rendered card shows a spinner.
"""

from __future__ import annotations
import json
import streamlit as st

from agent.observability_gate import role_can_see_query_details


_TOOL_EMOJI = {
    "find_patient": "🔍",
    "get_patient_clinical_data": "🩺",
    "list_patient_documents": "📄",
    "search_codes": "🔢",
    "search_patients_by_criteria": "👥",
    "search_knowledge_base": "📚",
    "run_sql": "🗄️",
    "make_chart": "📊",
    "summarize_results": "📝",
}


def _status_icon(payload: dict) -> str:
    if payload.get("error"):
        return "❌"
    if payload.get("success") is True:
        return "✓"
    if payload.get("success") is False:
        return "⚠"
    return "⏳"


def render_tool_call_message(message, index: int) -> None:
    payload = json.loads(message.content)
    tool_name = payload.get("tool_name", "<unknown>")
    emoji = _TOOL_EMOJI.get(tool_name, "🔧")
    status = _status_icon(payload)
    summary = payload.get("result_summary") or "running…"
    label = f"{emoji} {tool_name} — {status} {summary}"

    with st.expander(label, expanded=False):
        st.markdown("**Arguments**")
        st.json(payload.get("arguments", {}))
        if payload.get("result_summary"):
            st.markdown("**Result summary**")
            st.code(payload["result_summary"])
        if payload.get("reliability_note"):
            st.warning(f"⚠ {payload['reliability_note']}", icon="⚠️")
        if payload.get("error"):
            st.error(payload["error"])
        if payload.get("elapsed_ms"):
            st.caption(f"{payload['elapsed_ms']} ms")

        # Role-gated observability: executed SQL + raw row data.
        # Admins see by default; doctors/others only if [agent].
        # expose_query_details_to includes their role.
        user_role = st.session_state.get("user_role")
        if role_can_see_query_details(user_role):
            sql_log = payload.get("sql_executed") or []
            if sql_log:
                with st.expander(f"View SQL ({len(sql_log)} statement(s))", expanded=False):
                    for i, entry in enumerate(sql_log, 1):
                        st.code(entry.get("sql", ""), language="sql")
                        if entry.get("params"):
                            st.caption(f"params: {entry['params']}")

            result_payload = payload.get("result_payload")
            if isinstance(result_payload, dict):
                rows = _rows_from_payload(result_payload)
                if rows:
                    with st.expander(f"View rows ({len(rows)})", expanded=False):
                        try:
                            import pandas as pd

                            st.dataframe(pd.DataFrame(rows), use_container_width=True)
                        except Exception:
                            st.json(rows)
                else:
                    with st.expander("View raw data", expanded=False):
                        st.json(result_payload)


def _rows_from_payload(payload: dict) -> list:
    """Return the row-like list inside a tool's result payload, if any.

    Tools wrap their rows under domain-specific keys: matches (find_patient),
    items (ClinicalResult), documents (DocumentIndexResult). Returns the
    first list-of-dicts value found; empty list otherwise.
    """
    for key in ("items", "matches", "documents"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            return value
    return []
