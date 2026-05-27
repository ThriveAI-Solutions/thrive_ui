import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agent.logging_config import AgentLoggingConfig
from agent.run_logger import AgentRunLogger
from orm.models import AgentPatientAccess, AgentRun, AgentRunEvent, Base, ToolCall


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _logger(session, mode="full"):
    return AgentRunLogger(
        session=session,
        config=AgentLoggingConfig(mode=mode),
        run_id="r1",
        session_id="s1",
        user_id=1,
        user_role=1,
    )


def test_start_run_creates_open_run_and_first_event():
    s = _session()
    lg = _logger(s)
    lg.start_run(
        question="who is Ann?", llm_provider="anthropic", llm_model="claude", selected_patient=None, group_id="g1"
    )
    run = s.query(AgentRun).one()
    assert run.status == "open"
    assert run.question == "who is Ann?"
    assert run.group_id == "g1"
    ev = s.query(AgentRunEvent).filter_by(event_type="run_started").one()
    assert ev.seq == 1


def test_pinned_patient_at_start_writes_access_row():
    s = _session()
    lg = _logger(s)
    lg.start_run(
        question="q",
        llm_provider="x",
        llm_model="y",
        selected_patient={"source_id": "src-1", "display_name": "Ann", "selection_origin": "user_click"},
        group_id=None,
    )
    acc = s.query(AgentPatientAccess).one()
    assert acc.source_id == "src-1"
    assert acc.access_type == "pinned_at_run_start"


def test_log_event_increments_seq():
    s = _session()
    lg = _logger(s)
    lg.start_run(question="q", llm_provider="x", llm_model="y", selected_patient=None, group_id=None)
    lg.log_event("thinking_completed", payload={"text": "hmm"}, turn_index=1, elapsed_ms=10)
    seqs = [e.seq for e in s.query(AgentRunEvent).order_by(AgentRunEvent.seq).all()]
    assert seqs == [1, 2]


def test_log_tool_completed_writes_toolcall_event_and_access():
    s = _session()
    lg = _logger(s)
    lg.start_run(question="q", llm_provider="x", llm_model="y", selected_patient=None, group_id=None)
    lg.log_tool_completed(
        tool_name="find_patient",
        tool_call_id="tc-1",
        turn_index=1,
        arguments={"first_name": "Ann"},
        result_obj={"matches": [{"source_id": "src-1", "display_name": "Ann B"}], "total_unique": 1},
        sql_executed=[{"sql": "SELECT 1", "params": {}}],
        elapsed_ms=12,
        success=True,
        error=None,
        selected_patient_source_id=None,
    )
    tc = s.query(ToolCall).one()
    assert tc.run_id == "r1"
    assert tc.call_index == 1
    assert tc.tool_call_id == "tc-1"
    assert json.loads(tc.result_json)["total_unique"] == 1  # full result stored
    assert json.loads(tc.arguments_json)["first_name"] == "Ann"  # verbatim
    assert s.query(AgentRunEvent).filter_by(event_type="tool_call_completed").count() == 1
    assert s.query(AgentPatientAccess).filter_by(access_type="tool_result").one().source_id == "src-1"


def test_scrubbed_mode_hashes_sql_literals_and_drops_full_result():
    s = _session()
    lg = _logger(s, mode="scrubbed")
    lg.start_run(question="q", llm_provider="x", llm_model="y", selected_patient=None, group_id=None)
    lg.log_tool_completed(
        tool_name="run_sql",
        tool_call_id="tc-1",
        turn_index=1,
        arguments={"sql": "SELECT * FROM p WHERE name = 'John Smith'"},
        result_obj={"row_count": 1, "data_availability": "data_present", "dataframe": [{"name": "John Smith"}]},
        sql_executed=[],
        elapsed_ms=1,
        success=True,
        error=None,
        selected_patient_source_id=None,
    )
    tc = s.query(ToolCall).one()
    assert "John Smith" not in tc.arguments_json
    assert tc.result_json is None  # scrubbed: no full payload
    assert "row_count=1" in tc.result_summary


def test_finalize_run_sets_terminal_status_and_totals():
    s = _session()
    lg = _logger(s)
    lg.start_run(question="q", llm_provider="x", llm_model="y", selected_patient=None, group_id=None)
    lg.log_tool_completed(
        tool_name="t",
        tool_call_id="c",
        turn_index=1,
        arguments={},
        result_obj={"row_count": 0},
        sql_executed=[],
        elapsed_ms=1,
        success=True,
        error=None,
        selected_patient_source_id=None,
    )
    lg.finalize_run(
        status="success",
        final_answer_text="done",
        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        total_elapsed_ms=99,
        cap_reached=None,
    )
    run = s.query(AgentRun).one()
    assert run.status == "success"
    assert run.success is True
    assert run.final_answer_text == "done"
    assert run.input_tokens == 10 and run.total_tokens == 15
    assert run.tool_call_count == 1
    assert run.total_elapsed_ms == 99
    assert run.completed_at is not None
    assert s.query(AgentRunEvent).filter_by(event_type="run_completed").count() == 1


def test_finalize_failure_records_error_and_terminal_event():
    s = _session()
    lg = _logger(s)
    lg.start_run(question="q", llm_provider="x", llm_model="y", selected_patient=None, group_id=None)
    lg.finalize_run(
        status="failed",
        final_answer_text=None,
        usage=None,
        total_elapsed_ms=5,
        cap_reached=None,
        error_type="ValueError",
        error="boom",
        stack_trace="trace",
    )
    run = s.query(AgentRun).one()
    assert run.status == "failed"
    assert run.success is False
    assert run.error_type == "ValueError"
    assert s.query(AgentRunEvent).filter_by(event_type="run_failed").count() == 1
    assert s.query(AgentRunEvent).filter_by(event_type="run_completed").count() == 1


def test_logging_failure_never_raises(monkeypatch):
    s = _session()
    lg = _logger(s)
    lg.start_run(question="q", llm_provider="x", llm_model="y", selected_patient=None, group_id=None)
    # Force a write error; the call must swallow it.
    monkeypatch.setattr(lg, "session", None)
    lg.log_event("thinking_completed", payload={"t": "x"}, turn_index=1, elapsed_ms=1)  # no raise
