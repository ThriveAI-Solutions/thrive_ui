from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm.models import (
    AgentRun,
    AgentRunEvent,
    AgentPatientAccess,
    PatientSelectionEvent,
    ToolCall,
    Base,
)


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_agent_run_roundtrip():
    s = _session()
    run = AgentRun(
        run_id="r1",
        session_id="s1",
        user_id=1,
        user_role=1,
        question="hi",
        llm_provider="anthropic",
        llm_model="claude",
        tool_call_count=0,
        event_count=1,
        status="open",
        success=False,
        logging_mode="full",
        schema_version=1,
    )
    s.add(run)
    s.commit()
    assert s.query(AgentRun).one().run_id == "r1"


def test_agent_run_event_roundtrip():
    s = _session()
    ev = AgentRunEvent(run_id="r1", seq=1, event_type="run_started", payload_truncated=False)
    s.add(ev)
    s.commit()
    assert s.query(AgentRunEvent).one().event_type == "run_started"


def test_tool_call_has_new_columns():
    s = _session()
    tc = ToolCall(
        session_id="s1",
        user_id=1,
        user_role=1,
        tool_name="run_sql",
        arguments_json="{}",
        result_summary="row_count=1",
        elapsed_ms=5,
        success=True,
        run_id="r1",
        call_index=1,
        turn_index=1,
        attempt_index=1,
        result_json='{"row_count":1}',
        result_truncated=False,
        sql_executed_json="[]",
        sql_executed_truncated=False,
    )
    s.add(tc)
    s.commit()
    row = s.query(ToolCall).one()
    assert row.run_id == "r1"
    assert row.result_json == '{"row_count":1}'
    assert row.call_index == 1


def test_patient_access_and_selection_roundtrip():
    s = _session()
    s.add(
        AgentPatientAccess(
            run_id="r1",
            session_id="s1",
            user_id=1,
            source_id="src-1",
            access_type="pinned_at_run_start",
            access_origin="run_context",
        )
    )
    s.add(
        PatientSelectionEvent(
            session_id="s1",
            user_id=1,
            source_id="src-1",
            selection_origin="user_click",
            action="selected",
        )
    )
    s.commit()
    assert s.query(AgentPatientAccess).one().source_id == "src-1"
    assert s.query(PatientSelectionEvent).one().action == "selected"
