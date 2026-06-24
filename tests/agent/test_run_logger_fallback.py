"""Tests for the post-finalize ``mark_run_fallback_invoked`` helper
(Feature #233 of Epic #228).

The helper is module-level (not a method on ``AgentRunLogger``) because
the fallback fires AFTER the original logger's session is closed by the
agent runtime — we need a fresh session opened on the Streamlit script
thread.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agent.run_logger import mark_run_fallback_invoked
from orm.models import AgentRun, AgentRunEvent, Base


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _seed_run(session, *, run_id: str = "r1", status: str = "success") -> AgentRun:
    run = AgentRun(
        run_id=run_id,
        session_id="s1",
        user_id=1,
        user_role=1,
        question="q",
        status=status,
        success=(status == "success"),
        logging_mode="full",
    )
    session.add(run)
    # Seed a couple of pre-existing events so the helper's seq calculation
    # has something to work against.
    for seq in (1, 2, 3):
        session.add(
            AgentRunEvent(
                run_id=run_id,
                seq=seq,
                event_type=f"placeholder_{seq}",
            )
        )
    session.commit()
    return run


def test_mark_run_fallback_invoked_updates_status_and_sql():
    s = _session()
    _seed_run(s)

    mark_run_fallback_invoked(s, run_id="r1", fallback_sql="SELECT 1")

    run = s.query(AgentRun).filter_by(run_id="r1").one()
    assert run.status == "fallback_invoked"
    assert run.fallback_sql == "SELECT 1"
    assert run.success is True  # original success flag preserved


def test_mark_run_fallback_invoked_handles_missing_sql():
    s = _session()
    _seed_run(s)

    mark_run_fallback_invoked(s, run_id="r1", fallback_sql=None)

    run = s.query(AgentRun).filter_by(run_id="r1").one()
    assert run.status == "fallback_invoked"
    assert run.fallback_sql is None


def test_mark_run_fallback_invoked_silently_handles_missing_run():
    s = _session()
    # No row seeded — the helper must NOT raise.
    mark_run_fallback_invoked(s, run_id="does-not-exist", fallback_sql="SELECT 1")
    # And no row appeared.
    assert s.query(AgentRun).count() == 0


def test_mark_run_fallback_invoked_emits_event_row():
    s = _session()
    _seed_run(s)  # writes 3 placeholder events at seq 1,2,3.

    mark_run_fallback_invoked(s, run_id="r1", fallback_sql="SELECT 42")

    ev = s.query(AgentRunEvent).filter_by(event_type="fallback_invoked").one()
    assert ev.seq == 4  # one past the highest pre-existing seq
    assert ev.payload_summary == "fallback_sql_chars=9"


def test_mark_run_fallback_invoked_event_seq_for_run_without_prior_events():
    s = _session()
    # Seed an AgentRun without seeding any events.
    run = AgentRun(
        run_id="r-lonely",
        session_id="s",
        user_id=1,
        user_role=1,
        question="q",
        status="success",
        success=True,
        logging_mode="full",
    )
    s.add(run)
    s.commit()

    mark_run_fallback_invoked(s, run_id="r-lonely", fallback_sql="X")

    ev = s.query(AgentRunEvent).filter_by(run_id="r-lonely").one()
    assert ev.seq == 1
    assert ev.event_type == "fallback_invoked"
