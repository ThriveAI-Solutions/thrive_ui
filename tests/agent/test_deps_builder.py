import pytest
from unittest.mock import MagicMock, patch
from agent.deps import AgentDeps


@patch("agent.deps_builder._analytics_db")
@patch("agent.deps_builder._rag")
def test_build_agent_deps_with_no_selection(rag_mock, db_mock):
    db_mock.return_value = MagicMock()
    rag_mock.return_value = MagicMock()
    user = MagicMock(id=1)
    user.role.role = MagicMock(value=1)
    fake_session_state = {"user": user}
    sqlite = MagicMock()

    with patch("agent.deps_builder.st") as st:
        st.session_state = fake_session_state
        from agent.deps_builder import build_agent_deps

        deps = build_agent_deps(sqlite)

    assert isinstance(deps, AgentDeps)
    assert deps.user_id == 1
    assert deps.selected_patient is None
