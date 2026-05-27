from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from orm.models import Base, AgentRun, AgentRunEvent, ToolCall, AgentPatientAccess, AdminAction


def _seed(monkeypatch):
    from orm import agent_logging_functions as alf

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(alf, "SessionLocal", Session)
    s = Session()
    s.add(
        AgentRun(
            run_id="r1",
            session_id="s1",
            user_id=1,
            user_role=1,
            question="q1",
            llm_provider="anthropic",
            llm_model="claude",
            tool_call_count=2,
            event_count=5,
            total_elapsed_ms=1200,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            status="success",
            success=True,
            logging_mode="full",
            schema_version=1,
            review_status="unreviewed",
            selected_patient_source_id="src-1",
        )
    )
    s.add(
        AgentRun(
            run_id="r2",
            session_id="s1",
            user_id=1,
            user_role=1,
            question="q2",
            llm_model="claude",
            status="failed",
            success=False,
            tool_call_count=0,
            event_count=2,
            logging_mode="full",
            schema_version=1,
            review_status="unreviewed",
        )
    )
    s.add(AgentRunEvent(run_id="r1", seq=1, event_type="run_started", payload_truncated=False))
    s.add(AgentRunEvent(run_id="r1", seq=2, event_type="run_completed", payload_truncated=False))
    s.add(
        ToolCall(
            session_id="s1",
            user_id=1,
            user_role=1,
            tool_name="run_sql",
            arguments_json="{}",
            result_summary="row_count=3",
            elapsed_ms=50,
            success=True,
            run_id="r1",
            call_index=1,
            attempt_index=1,
        )
    )
    s.add(
        AgentPatientAccess(
            run_id="r1",
            session_id="s1",
            user_id=1,
            source_id="src-1",
            display_name="Ann",
            access_type="tool_result",
            access_origin="tool",
        )
    )
    s.commit()
    return alf


def test_get_agent_run_stats(monkeypatch):
    alf = _seed(monkeypatch)
    stats = alf.get_agent_run_stats(days=30)
    assert stats["total_runs"] == 2
    assert stats["success_rate"] == 50.0
    assert stats["avg_tool_calls"] == 1.0  # (2 + 0) / 2


def test_get_recent_runs(monkeypatch):
    alf = _seed(monkeypatch)
    rows = alf.get_recent_agent_runs(days=30, limit=10)
    assert len(rows) == 2
    assert {r["run_id"] for r in rows} == {"r1", "r2"}


def test_get_run_detail_bundles_events_and_tools(monkeypatch):
    alf = _seed(monkeypatch)
    detail = alf.get_agent_run_detail("r1")
    assert detail["run"]["question"] == "q1"
    assert [e["event_type"] for e in detail["events"]] == ["run_started", "run_completed"]
    assert detail["tool_calls"][0]["tool_name"] == "run_sql"


def test_get_tool_breakdown(monkeypatch):
    alf = _seed(monkeypatch)
    rows = alf.get_tool_breakdown(days=30)
    assert any(r["tool_name"] == "run_sql" and r["count"] == 1 for r in rows)


def test_get_patient_access_for_source(monkeypatch):
    alf = _seed(monkeypatch)
    rows = alf.get_patient_access(source_id="src-1", days=30)
    assert rows[0]["source_id"] == "src-1"


def test_set_run_review_status(monkeypatch):
    alf = _seed(monkeypatch)
    alf.set_run_review_status("r1", review_status="verified", reviewed_by=9, notes="looks right")
    detail = alf.get_agent_run_detail("r1")
    assert detail["run"]["review_status"] == "verified"
    assert detail["run"]["review_notes"] == "looks right"


def test_log_agent_run_viewed_writes_admin_action(monkeypatch):
    alf = _seed(monkeypatch)
    assert alf.log_agent_run_viewed(admin_id=9, run_id="r1") is True
    from orm import agent_logging_functions as alf_module

    s = alf_module.SessionLocal()
    row = s.query(AdminAction).one()
    assert row.action_type == "agent_run_view"
    assert row.target_entity_type == "agent_run"
    assert row.target_entity_id == "r1"
