import pytest
from unittest.mock import MagicMock, patch
from agent.deps import AgentDeps
from orm.models import RoleTypeEnum


@patch("agent.deps_builder._analytics_db")
@patch("agent.deps_builder._rag")
def test_build_agent_deps_with_no_selection(rag_mock, db_mock):
    """user_id comes from direct session key, user_role from int->enum."""
    db_mock.return_value = MagicMock()
    rag_mock.return_value = MagicMock()
    fake_session_state = {
        "user_id": 1,
        "user_role": RoleTypeEnum.DOCTOR.value,
    }
    sqlite = MagicMock()

    with patch("agent.deps_builder.st") as st:
        st.session_state = fake_session_state
        from agent.deps_builder import build_agent_deps

        deps = build_agent_deps(sqlite)

    assert isinstance(deps, AgentDeps)
    assert deps.user_id == 1
    assert deps.user_role == RoleTypeEnum.DOCTOR
    assert deps.selected_patient is None


@patch("agent.deps_builder._analytics_db")
@patch("agent.deps_builder._rag")
def test_build_agent_deps_reads_cookie_user_id(rag_mock, db_mock):
    """Real login flow stores user_id in cookies as a JSON-encoded int."""
    db_mock.return_value = MagicMock()
    rag_mock.return_value = MagicMock()
    cookies = MagicMock()
    cookies.get.return_value = "42"  # json.loads("42") == 42
    fake_session_state = {
        "cookies": cookies,
        "user_role": RoleTypeEnum.NURSE.value,
    }
    sqlite = MagicMock()

    with patch("agent.deps_builder.st") as st:
        st.session_state = fake_session_state
        from agent.deps_builder import build_agent_deps

        deps = build_agent_deps(sqlite)

    assert deps.user_id == 42
    assert deps.user_role == RoleTypeEnum.NURSE


def test_build_deps_attaches_run_logger(monkeypatch):
    import streamlit as st
    from agent.deps_builder import build_agent_deps
    from agent.run_logger import AgentRunLogger
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from orm.models import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    st.session_state.clear()
    st.session_state["user_id"] = 1
    st.session_state["user_role"] = 1
    st.session_state["current_group_id"] = "g-xyz"

    monkeypatch.setattr("agent.deps_builder._analytics_db", lambda: object())
    monkeypatch.setattr("agent.deps_builder._rag", lambda: object())

    deps = build_agent_deps(session)
    assert isinstance(deps.run_logger, AgentRunLogger)
    assert deps.run_logger.run_id
    assert deps.group_id == "g-xyz"


def test_build_deps_run_logger_none_when_disabled(monkeypatch):
    import streamlit as st
    from agent.deps_builder import build_agent_deps
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from orm.models import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    st.session_state.clear()
    st.session_state["user_id"] = 1
    st.session_state["user_role"] = 1
    monkeypatch.setattr("agent.deps_builder._analytics_db", lambda: object())
    monkeypatch.setattr("agent.deps_builder._rag", lambda: object())
    monkeypatch.setattr(
        "agent.logging_config.AgentLoggingConfig.from_streamlit",
        classmethod(lambda cls: cls(mode="disabled")),
    )

    deps = build_agent_deps(session)
    assert deps.run_logger is None
