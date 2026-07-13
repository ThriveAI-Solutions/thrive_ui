from types import SimpleNamespace
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm.evaluation_models import AgentRunFeedback  # noqa: F401
from orm.models import AgentRun, Base
from views import agent_feedback


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeSt:
    def __init__(self):
        self.session_state = {"user_id": 1}
        self.buttons = []

    def columns(self, spec):
        return [_Ctx() for _ in spec]

    def button(self, label, **kwargs):
        self.buttons.append((label, kwargs))
        return False

    def popover(self, *args, **kwargs):
        return _Ctx()

    def markdown(self, *args, **kwargs):
        return None

    def radio(self, *args, **kwargs):
        return agent_feedback._CATEGORY_PLACEHOLDER

    def text_area(self, *args, **kwargs):
        return ""

    def info(self, *args, **kwargs):
        return None



def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _run(session, *, group_id="group-1", user_id=1, status="success", final_message_id=7):
    row = AgentRun(
        run_id="run-1",
        session_id="session-1",
        group_id=group_id,
        user_id=user_id,
        user_role=1,
        status=status,
        success=status == "success",
        final_message_id=final_message_id,
        logging_mode="full",
    )
    session.add(row)
    session.commit()
    return row


def test_render_controls_once_for_owned_completed_final_message(monkeypatch):
    session = _session()
    _run(session)
    fake_st = _FakeSt()
    monkeypatch.setattr(agent_feedback, "st", fake_st)
    monkeypatch.setattr(agent_feedback, "SessionLocal", lambda: session)

    message = SimpleNamespace(id=7, group_id="group-1")
    agent_feedback.render_agent_feedback_for_group([message])

    labels = [label for label, _ in fake_st.buttons]
    assert labels.count("👍") == 1
    assert labels.count("Submit") == 1


def test_render_skips_intermediate_error_only_or_non_owner_groups(monkeypatch):
    session = _session()
    _run(session, final_message_id=99)
    fake_st = _FakeSt()
    monkeypatch.setattr(agent_feedback, "st", fake_st)
    monkeypatch.setattr(agent_feedback, "SessionLocal", lambda: session)

    agent_feedback.render_agent_feedback_for_group([SimpleNamespace(id=7, group_id="group-1")])

    assert fake_st.buttons == []


def test_up_button_invokes_agent_feedback_not_vanna_training(monkeypatch):
    session = _session()
    _run(session)
    fake_st = _FakeSt()
    fake_st.button = MagicMock(side_effect=lambda label, **kwargs: label == "👍")
    fake_st.toast = MagicMock()
    fake_st.rerun = MagicMock()
    monkeypatch.setattr(agent_feedback, "st", fake_st)
    monkeypatch.setattr(agent_feedback, "SessionLocal", lambda: session)

    agent_feedback.render_agent_feedback_for_group([SimpleNamespace(id=7, group_id="group-1")])

    row = session.query(AgentRunFeedback).one()
    assert row.rating == "up"
