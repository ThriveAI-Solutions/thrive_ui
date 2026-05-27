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
            previous_source_id = st.session_state.get("selected_patient_source_id")
            parent_run_id = st.session_state.get("agent_current_run_id")
            st.session_state["selected_patient_source_id"] = m["source_id"]
            st.session_state["selected_patient_display_name"] = m.get("display_name", "")
            st.session_state["selected_patient_dob"] = m.get("dob")
            st.session_state["selection_origin"] = "user_click"
            st.session_state["selected_at"] = datetime.now().isoformat()
            try:
                from orm.agent_logging_functions import log_patient_selection

                log_patient_selection(
                    session_id=st.session_state.get("agent_session_id", ""),
                    user_id=int(st.session_state.get("user_id") or 0),
                    source_id=m["source_id"],
                    display_name=m.get("display_name", ""),
                    selection_origin="user_click",
                    action="selected",
                    previous_source_id=previous_source_id,
                    run_id=parent_run_id,
                )
            except Exception:
                pass
            st.session_state["agent_parent_run_id"] = parent_run_id
            st.session_state["agent_resume_reason"] = "patient_selection_resume"
            # Re-fire the original question so the agent continues with
            # the slot now filled. agent.runtime stashes this before each
            # run; if missing (e.g. stale chooser from a prior session),
            # the user can simply re-ask.
            pending = st.session_state.get("pending_user_question")
            if pending:
                st.session_state["my_question"] = pending
            st.rerun()
