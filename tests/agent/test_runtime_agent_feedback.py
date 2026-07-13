from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agent import runtime
from agent.state import AgentResponse, FinalResponseEvent
from orm.models import AgentRun, Base, Message
from utils.enums import MessageType, RoleType


class _FakeSt:
    def __init__(self):
        self.session_state = {}


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_final_response_render_backfills_agent_run_final_message_id(monkeypatch):
    session = _session()
    session.add(
        AgentRun(
            run_id="run-1",
            session_id="session-1",
            group_id="group-1",
            user_id=1,
            user_role=1,
            status="success",
            success=True,
            logging_mode="full",
        )
    )
    session.commit()

    next_id = {"value": 40}

    def fake_add_message(message: Message):
        next_id["value"] += 1
        message.id = next_id["value"]
        return message

    monkeypatch.setattr(runtime, "st", _FakeSt())
    monkeypatch.setattr(runtime, "SessionLocal", lambda: session)
    monkeypatch.setattr("utils.chat_bot_helper.add_message", fake_add_message)

    event = FinalResponseEvent(
        response=AgentResponse(text="final answer"),
        all_messages=[],
        usage={},
        run_id="run-1",
    )
    runtime._render_event(event, {"thinking": {}, "text": {}})

    run = session.query(AgentRun).filter_by(run_id="run-1").one()
    assert run.final_message_id == 41


def test_final_response_uses_streamed_text_message_when_deduped(monkeypatch):
    session = _session()
    session.add(
        AgentRun(
            run_id="run-1",
            session_id="session-1",
            group_id="group-1",
            user_id=1,
            user_role=1,
            status="success",
            success=True,
            logging_mode="full",
        )
    )
    session.commit()
    calls = []

    def fake_add_message(message: Message):
        calls.append(message)
        message.id = 99
        return message

    monkeypatch.setattr(runtime, "st", _FakeSt())
    monkeypatch.setattr(runtime, "SessionLocal", lambda: session)
    monkeypatch.setattr("utils.chat_bot_helper.add_message", fake_add_message)

    state = {"thinking": {}, "text": {}, "last_persisted_text": "final answer", "last_persisted_text_message_id": 55}
    event = FinalResponseEvent(
        response=AgentResponse(text="final answer"),
        all_messages=[],
        usage={},
        run_id="run-1",
    )
    runtime._render_event(event, state)

    assert calls == []
    assert session.query(AgentRun).filter_by(run_id="run-1").one().final_message_id == 55


def test_agent_final_messages_are_plain_text_type():
    message = Message(RoleType.ASSISTANT, "answer", MessageType.TEXT, user_id=1, group_id="group-1")
    assert message.type == MessageType.TEXT.value
