"""Render PATIENT_CHOOSER messages as clickable button list.

Stored payload (Message.content) is JSON of PatientSearchResults.
Click sets st.session_state.selected_patient_source_id and triggers rerun.
"""

from __future__ import annotations

import json
from datetime import datetime

import streamlit as st


def _format_match(m: dict) -> str:
    parts = [m.get("display_name", "<unknown>")]
    if m.get("dob"):
        parts.append(f"b. {m['dob']}")
    if m.get("facilities_seen"):
        parts.append(f"@ {', '.join(m['facilities_seen'])}")
    if m.get("most_recent_activity"):
        parts.append(f"last seen {m['most_recent_activity']}")
    return " — ".join(parts)


def render_patient_chooser(message, index: int) -> None:
    payload = json.loads(message.content)
    matches = payload.get("matches", [])
    total = payload.get("total_unique", len(matches))
    truncated = payload.get("truncated", False)

    st.markdown(f"**Which patient?** ({total} unique{' — more available' if truncated else ''})")
    # Use message.id (persisted PK) so keys stay stable as session_state
    # message history is trimmed and `index` shifts.
    msg_id = getattr(message, "id", None) or index
    for i, m in enumerate(matches):
        if st.button(
            _format_match(m),
            key=f"chooser-msg{msg_id}-{i}-{m.get('source_id')}",
        ):
            st.session_state["selected_patient_source_id"] = m["source_id"]
            st.session_state["selected_patient_display_name"] = m.get("display_name", "")
            st.session_state["selected_patient_dob"] = m.get("dob")
            st.session_state["selection_origin"] = "user_click"
            st.session_state["selected_at"] = datetime.now().isoformat()
            st.rerun()
