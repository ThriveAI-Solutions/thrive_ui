"""Build AgentDeps from Streamlit session state.

Called at the top of every agent run. Reads:
- st.session_state.user (id, role)
- st.session_state.selected_patient_*
- st.session_state.last_dataframe / last_sql
- @st.cache_resource singletons for analytics_db, rag, sqlite_session

Returns a fresh AgentDeps. Per spec §8.1: AgentDeps is per-run, not
persisted; persistence lives in session_state.
"""

from __future__ import annotations
from datetime import date, datetime
from typing import Optional
import uuid
import streamlit as st

from agent.deps import AgentDeps, SelectedPatient
from agent.audit import AuditLogger
from agent.db.analytics_adapter import AnalyticsDbAdapter


@st.cache_resource
def _analytics_db() -> AnalyticsDbAdapter:
    return AnalyticsDbAdapter.from_streamlit_secrets()


@st.cache_resource
def _rag():
    """Lazy import to avoid Streamlit picking up Chroma at module load."""
    from agent.rag.chroma_adapter import ChromaRagAdapter
    import chromadb

    chroma_path = st.secrets.get("rag_model", {}).get("chroma_path", "./chromadb")
    return ChromaRagAdapter(chromadb.PersistentClient(path=chroma_path))


def _selected_patient_from_session() -> Optional[SelectedPatient]:
    src = st.session_state.get("selected_patient_source_id")
    if not src:
        return None
    raw_dob = st.session_state.get("selected_patient_dob")
    parsed_dob: Optional[date] = None
    if raw_dob:
        try:
            parsed_dob = date.fromisoformat(raw_dob) if isinstance(raw_dob, str) else raw_dob
        except ValueError:
            parsed_dob = None
    selected_at_raw = st.session_state.get("selected_at")
    selected_at = (
        datetime.fromisoformat(selected_at_raw)
        if isinstance(selected_at_raw, str)
        else (selected_at_raw or datetime.now())
    )
    return SelectedPatient(
        source_id=src,
        display_name=st.session_state.get("selected_patient_display_name", ""),
        dob=parsed_dob,
        selected_at=selected_at,
        selection_origin=st.session_state.get("selection_origin", "user_click"),
    )


def build_agent_deps(sqlite_session) -> AgentDeps:
    user = st.session_state["user"]
    if "agent_session_id" not in st.session_state:
        st.session_state["agent_session_id"] = str(uuid.uuid4())
    session_id = st.session_state["agent_session_id"]

    return AgentDeps(
        user_id=user.id,
        user_role=user.role.role,
        session_id=session_id,
        selected_patient=_selected_patient_from_session(),
        last_dataframe=st.session_state.get("last_dataframe"),
        last_sql=st.session_state.get("last_sql"),
        last_query_meta=None,
        analytics_db=_analytics_db(),
        rag=_rag(),
        sqlite_session=sqlite_session,
        audit_logger=AuditLogger(
            session=sqlite_session,
            session_id=session_id,
            user_id=user.id,
            user_role=int(user.role.role.value),
        ),
    )
