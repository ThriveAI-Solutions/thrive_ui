from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm.agent_feedback import (
    AGENT_FEEDBACK_CATEGORIES,
    AgentFeedbackError,
    clear_agent_run_feedback,
    get_agent_run_feedback,
    set_agent_run_feedback,
)
from orm.evaluation_models import AgentRunFeedback, AgentRunFeedbackEvent  # noqa: F401
from orm.models import AgentRun, Base


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _run(session, *, user_id=1, status="success", final_message_id=10, run_id="run-1"):
    row = AgentRun(
        run_id=run_id,
        session_id="session-1",
        group_id="group-1",
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


def test_owner_can_create_update_and_clear_feedback_with_audit_events():
    session = _session()
    _run(session)

    up = set_agent_run_feedback(session, run_id="run-1", user_id=1, rating="up")
    assert up.rating == "up"
    assert up.category is None

    down = set_agent_run_feedback(
        session,
        run_id="run-1",
        user_id=1,
        rating="down",
        category=AGENT_FEEDBACK_CATEGORIES[0],
        comment="missed the latest lab",
    )
    assert down.rating == "down"
    assert down.category == AGENT_FEEDBACK_CATEGORIES[0]
    assert down.comment == "missed the latest lab"
    assert session.query(AgentRunFeedback).count() == 1
    assert session.query(AgentRunFeedbackEvent).count() == 2

    clear_agent_run_feedback(session, run_id="run-1", user_id=1)
    assert get_agent_run_feedback(session, run_id="run-1", user_id=1) is None
    assert session.query(AgentRunFeedbackEvent).count() == 3
    assert session.query(AgentRunFeedbackEvent).order_by(AgentRunFeedbackEvent.id.desc()).first().new_rating is None


def test_non_owner_cannot_mutate_or_read_feedback():
    session = _session()
    _run(session, user_id=1)

    set_agent_run_feedback(session, run_id="run-1", user_id=1, rating="up")

    assert get_agent_run_feedback(session, run_id="run-1", user_id=2) is None
    try:
        set_agent_run_feedback(session, run_id="run-1", user_id=2, rating="down", category=AGENT_FEEDBACK_CATEGORIES[0])
    except AgentFeedbackError as exc:
        assert "owner" in str(exc)
    else:
        raise AssertionError("non-owner feedback mutation should fail")


def test_down_requires_approved_category_and_comment_max_500():
    session = _session()
    _run(session)

    for kwargs in (
        {"category": None, "comment": None},
        {"category": "not approved", "comment": None},
        {"category": AGENT_FEEDBACK_CATEGORIES[0], "comment": "x" * 501},
    ):
        try:
            set_agent_run_feedback(session, run_id="run-1", user_id=1, rating="down", **kwargs)
        except AgentFeedbackError:
            pass
        else:
            raise AssertionError(f"invalid feedback accepted: {kwargs}")


def test_feedback_requires_completed_final_message():
    session = _session()
    _run(session, status="open", final_message_id=None)

    try:
        set_agent_run_feedback(session, run_id="run-1", user_id=1, rating="up")
    except AgentFeedbackError as exc:
        assert "completed" in str(exc)
    else:
        raise AssertionError("incomplete run feedback should fail")
