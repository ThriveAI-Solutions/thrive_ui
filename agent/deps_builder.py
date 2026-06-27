"""Build AgentDeps from Streamlit session state.

Called at the top of every agent run. Reads:
- st.session_state.cookies["user_id"] (JSON-encoded int) — see orm/functions.py
- st.session_state.user_role (int value of RoleTypeEnum)
- st.session_state.selected_patient_*
- st.session_state["df"] (Vanna and the agent share this key; written by
  the runner's _sync_last_dataframe_to_session_state at turn boundaries
  and consumed by slash-command magic functions too)
- st.session_state.last_sql
- @st.cache_resource singletons for analytics_db, rag, sqlite_session

Returns a fresh AgentDeps. Per spec §8.1: AgentDeps is per-run, not
persisted; persistence lives in session_state.

Note: this codebase shreds user attributes into individual session_state
keys rather than storing a single User ORM object. We read those keys
here rather than assuming a `user` object exists.
"""

from __future__ import annotations
import json
from datetime import date, datetime
from typing import Optional
import uuid
import streamlit as st

from agent.deps import AgentDeps, SelectedPatient
from agent.run_logger import AgentRunLogger
from agent.logging_config import AgentLoggingConfig
from agent.db.analytics_adapter import AnalyticsDbAdapter
from orm.models import RoleTypeEnum
from utils.enums import RoleType


@st.cache_resource
def _analytics_db() -> AnalyticsDbAdapter:
    return AnalyticsDbAdapter.from_streamlit_secrets()


@st.cache_resource
def _rag():
    """Lazy import to avoid Streamlit picking up Chroma at module load."""
    from agent.rag.chroma_adapter import ChromaRagAdapter
    import chromadb
    from chromadb.config import Settings

    chroma_path = st.secrets.get("rag_model", {}).get("chroma_path", "./chromadb")
    return ChromaRagAdapter(chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False)))


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


def _user_id_from_session() -> int:
    """Read user_id from cookies (where login flow stores it).

    Falls back to st.session_state.user_id for tests/dev that prefer the
    direct key.
    """
    direct = st.session_state.get("user_id")
    if direct is not None:
        return int(direct)
    cookies = st.session_state.get("cookies")
    if cookies is not None:
        raw = cookies.get("user_id")
        if raw is not None:
            try:
                return int(json.loads(raw))
            except (TypeError, ValueError, json.JSONDecodeError):
                return int(raw)
    raise RuntimeError(
        "Cannot determine user_id — neither st.session_state.user_id nor "
        "st.session_state.cookies['user_id'] is set. Is the user logged in?"
    )


def _latest_user_message_id() -> Optional[int]:
    """Return the id of the most recent USER-role Message in session state.

    Links the AgentRun to the user question that triggered it so the per-query
    audit view's join (``AgentRun.user_message_id == Message.id``) populates
    the agentic half of its UNION ALL instead of falling through to legacy.
    """
    messages = st.session_state.get("messages") or []
    for msg in reversed(messages):
        if getattr(msg, "role", None) == RoleType.USER.value:
            return getattr(msg, "id", None)
    return None


def _user_role_from_session() -> RoleTypeEnum:
    """Read user_role (int) from session and convert to enum.

    orm/functions.py stores the enum's int value; we round-trip back to
    the enum for type-safe consumption inside tools.
    """
    raw = st.session_state.get("user_role")
    if raw is None:
        raise RuntimeError("Cannot determine user_role — st.session_state.user_role is not set.")
    return RoleTypeEnum(int(raw))


def build_agent_deps(sqlite_session) -> AgentDeps:
    user_id = _user_id_from_session()
    user_role = _user_role_from_session()
    if "agent_session_id" not in st.session_state:
        st.session_state["agent_session_id"] = str(uuid.uuid4())
    session_id = st.session_state["agent_session_id"]

    config = AgentLoggingConfig.from_streamlit()
    run_id = str(uuid.uuid4())
    st.session_state["agent_current_run_id"] = run_id
    group_id = st.session_state.get("current_group_id")
    run_logger = None
    if config.enabled:
        run_logger = AgentRunLogger(
            session=sqlite_session,
            config=config,
            run_id=run_id,
            session_id=session_id,
            user_id=user_id,
            user_role=int(user_role.value),
            group_id=group_id,
        )

    return AgentDeps(
        user_id=user_id,
        user_role=user_role,
        session_id=session_id,
        selected_patient=_selected_patient_from_session(),
        last_dataframe=st.session_state.get("df"),
        last_sql=st.session_state.get("last_sql"),
        last_query_meta=None,
        analytics_db=_analytics_db(),
        rag=_rag(),
        sqlite_session=sqlite_session,
        run_logger=run_logger,
        group_id=group_id,
        user_message_id=_latest_user_message_id(),
        parent_run_id=st.session_state.get("agent_parent_run_id"),
        resume_reason=st.session_state.get("agent_resume_reason"),
    )
