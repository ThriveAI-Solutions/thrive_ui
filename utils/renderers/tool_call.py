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
