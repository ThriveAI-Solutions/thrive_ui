"""Per-user Vanna-fallback opt-in (Feature #232 of Epic #228).

Covers:
  - ORM column exists with default False.
  - Round-trip persistence.
  - set_user_preferences_in_session_state mirrors the column to
    st.session_state.vanna_fallback_enabled on login.
  - save_user_settings writes the session-state value back to the DB.
"""

from __future__ import annotations

import types
from contextlib import nullcontext

from orm.functions import (
    create_user,
    save_user_settings,
    set_user_preferences_in_session_state,
)
from orm.models import User, UserRole


def _seed_user(session_factory, *, username: str = "fallback_test_user") -> int:
    with session_factory() as session:
        role_id = session.query(UserRole).filter(UserRole.role_name == "Doctor").one().id
    assert (
        create_user(
            username,
            "pw",
            "Fall",
            "Back",
            role_id,
            email=f"{username}@example.com",
            organization="Acme",
        )
        is True
    )
    with session_factory() as session:
        return session.query(User).filter(User.username == username).one().id


def _fake_st():
    st = types.SimpleNamespace()

    class _State(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

        def __setattr__(self, k, v):
            self[k] = v

    st.session_state = _State()
    st.session_state["cookies"] = {}
    st.chat_message = lambda *_a, **_k: nullcontext()
    st.empty = lambda: None
    st.rerun = lambda: None
    st.toast = lambda *_a, **_k: None
    st.markdown = lambda *_a, **_k: None
    st.error = lambda *_a, **_k: None
    return st


def test_vanna_fallback_enabled_column_exists_and_defaults_false(in_memory_orm_session):
    user_id = _seed_user(in_memory_orm_session)
    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.id == user_id).one()
        # Column exists and default kicks in (False) for a freshly-created row.
        assert hasattr(user, "vanna_fallback_enabled")
        assert user.vanna_fallback_enabled is False


def test_vanna_fallback_enabled_round_trip(in_memory_orm_session):
    user_id = _seed_user(in_memory_orm_session)

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.id == user_id).one()
        user.vanna_fallback_enabled = True
        session.commit()

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.id == user_id).one()
        assert user.vanna_fallback_enabled is True


def test_set_user_preferences_loads_vanna_fallback_into_session_state(monkeypatch, in_memory_orm_session):
    import json as _json

    user_id = _seed_user(in_memory_orm_session)

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.id == user_id).one()
        user.vanna_fallback_enabled = True
        session.commit()

    fake_st = _fake_st()
    fake_st.session_state["cookies"] = {"user_id": _json.dumps(user_id)}
    import orm.functions as fns

    monkeypatch.setattr(fns, "st", fake_st)

    set_user_preferences_in_session_state()

    assert fake_st.session_state.vanna_fallback_enabled is True


def test_save_user_settings_writes_vanna_fallback_back_to_db(monkeypatch, in_memory_orm_session):
    import json as _json

    user_id = _seed_user(in_memory_orm_session)

    fake_st = _fake_st()
    fake_st.session_state["cookies"] = {"user_id": _json.dumps(user_id)}
    # Populate every key save_user_settings reads so it doesn't AttributeError.
    fake_st.session_state.update(
        {
            "show_sql": True,
            "show_table": True,
            "show_plotly_code": True,
            "show_chart": True,
            "show_question_history": True,
            "show_summary": True,
            "voice_input": False,
            "speak_summary": False,
            "show_suggested": False,
            "show_followup": False,
            "show_elapsed_time": True,
            "show_thinking_process": False,
            "llm_fallback": False,
            "vanna_fallback_enabled": True,
            "confirm_magic_commands": True,
            "show_community_engagement": False,
            "min_message_id": 0,
            "selected_llm_provider": None,
            "selected_llm_model": None,
        }
    )

    import orm.functions as fns

    monkeypatch.setattr(fns, "st", fake_st)

    save_user_settings()

    with in_memory_orm_session() as session:
        user = session.query(User).filter(User.id == user_id).one()
        assert user.vanna_fallback_enabled is True
