"""Tests for the per-query audit data-access layer (Epic #190, Phase 1).

The per-query view unfolds each user question into one row per *query
unit*:

  * Legacy question (no AgentRun) → 1 row, sql from the assistant SQL Message
  * Agentic question → 1 row per ToolCall (0–N SQL statements per row,
    decoded from ToolCall.sql_executed_json)

Shape and filter semantics covered here are the contract that Phases 2–4
of Epic #190 build on.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm.models import (
    AgentPatientAccess,
    AgentRun,
    Base,
    Message,
    RoleTypeEnum,
    ToolCall,
    User,
    UserRole,
)
from utils.enums import MessageType, RoleType


# ---------------------------------------------------------------------------
# Fixtures & seed helpers (mirrors tests/orm/test_question_audit_scope.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def session_factory(monkeypatch):
    """In-memory SQLite with SessionLocal patched on orm.logging_functions."""
    from orm import logging_functions as lf

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(lf, "SessionLocal", Session)
    return Session


def _seed_roles(session):
    session.add_all(
        [
            UserRole(id=1, role_name="Admin", description="Admin", role=RoleTypeEnum.ADMIN),
            UserRole(id=4, role_name="Patient", description="Patient", role=RoleTypeEnum.PATIENT),
        ]
    )
    session.commit()


def _seed_user(session, *, id, username, org="TestOrg", role_id=4):
    session.add(
        User(
            id=id,
            user_role_id=role_id,
            username=username,
            first_name=username,
            last_name="X",
            password="hash:abc",
            email=f"{username}@x.io",
            organization=org,
        )
    )
    session.commit()


def _add_question(
    session,
    *,
    user_id,
    content,
    when,
    assistant_sql=None,
    assistant_sql_elapsed=None,
    assistant_error=None,
):
    """Seed a user-role Message with a clamped created_at and optional legacy
    assistant SQL/error rows. Returns the user Message.id.
    """
    session.add(
        Message(
            role=RoleType.USER,
            content=content,
            type=MessageType.TEXT,
            user_id=user_id,
        )
    )
    session.commit()
    last = session.query(Message).order_by(Message.id.desc()).first()
    last.created_at = when
    session.commit()
    user_message_id = last.id

    if assistant_sql is not None:
        m = Message(
            role=RoleType.ASSISTANT,
            content=assistant_sql,
            type=MessageType.SQL,
            user_id=user_id,
            question=content,
            elapsed_time=assistant_sql_elapsed,
        )
        session.add(m)
        session.commit()
        # Bump created_at one second after the user message so ordering is
        # deterministic in fixture results.
        am = session.query(Message).order_by(Message.id.desc()).first()
        am.created_at = when + timedelta(seconds=1)
        session.commit()
    if assistant_error is not None:
        m = Message(
            role=RoleType.ASSISTANT,
            content=assistant_error,
            type=MessageType.ERROR,
            user_id=user_id,
            question=content,
        )
        session.add(m)
        session.commit()

    return user_message_id


def _add_agent_run(
    session,
    *,
    run_id,
    user_id,
    user_message_id,
    selected_patient_source_id=None,
    logging_mode="full",
    created_at=None,
):
    ar = AgentRun(
        run_id=run_id,
        session_id=f"sess-{run_id}",
        user_id=user_id,
        user_role=RoleTypeEnum.PATIENT.value,
        user_message_id=user_message_id,
        selected_patient_source_id=selected_patient_source_id,
        tool_call_count=0,
        event_count=0,
        status="completed",
        success=True,
        review_status="unreviewed",
        logging_mode=logging_mode,
        schema_version=1,
    )
    session.add(ar)
    session.commit()
    if created_at is not None:
        row = session.query(AgentRun).filter_by(run_id=run_id).first()
        row.created_at = created_at
        session.commit()


def _add_tool_call(
    session,
    *,
    run_id,
    user_id,
    tool_name,
    call_index=0,
    tool_call_uid=None,
    sql_executed=None,
    result_summary="{}",
    elapsed_ms=42,
    success=True,
    error=None,
):
    """Seed a ToolCall. ``sql_executed`` may be a list[str] (encoded to JSON),
    a raw JSON string, or None.
    """
    if isinstance(sql_executed, list):
        sql_executed_json = json.dumps(sql_executed)
    else:
        sql_executed_json = sql_executed
    tc = ToolCall(
        session_id=f"sess-{run_id}",
        user_id=user_id,
        user_role=RoleTypeEnum.PATIENT.value,
        tool_name=tool_name,
        arguments_json="{}",
        result_summary=result_summary,
        elapsed_ms=elapsed_ms,
        success=success,
        error=error,
        run_id=run_id,
        tool_call_id=tool_call_uid,
        call_index=call_index,
        sql_executed_json=sql_executed_json,
    )
    session.add(tc)
    session.commit()
    return tc.id


def _add_patient_access(
    session,
    *,
    run_id,
    user_id,
    source_id,
    display_name=None,
    tool_call_uid=None,
    access_type="read",
    access_origin="tool",
):
    session.add(
        AgentPatientAccess(
            run_id=run_id,
            tool_call_id=tool_call_uid,
            session_id=f"sess-{run_id}",
            user_id=user_id,
            source_id=source_id,
            display_name=display_name,
            access_type=access_type,
            access_origin=access_origin,
        )
    )
    session.commit()


def _filters(**overrides):
    base = {"usernames": [], "orgs": [], "days": 30, "search": None}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# (1) Row shape & per-pipeline emission
# ---------------------------------------------------------------------------


def test_legacy_question_yields_one_row_with_assistant_sql(session_factory):
    from orm.logging_functions import PER_QUERY_ROW_KEYS, get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    _add_question(
        s,
        user_id=10,
        content="legacy q",
        when=datetime(2026, 6, 1, 12, 0, 0),
        assistant_sql="SELECT 1",
        assistant_sql_elapsed=Decimal("1.250000"),
    )

    page = get_per_query_audit_page(_filters(), page=1, page_size=50)
    assert page["total"] == 1
    assert len(page["items"]) == 1
    item = page["items"][0]
    assert set(item.keys()) == set(PER_QUERY_ROW_KEYS)
    assert item["pipeline"] == "legacy"
    assert item["sql_statements"] == ["SELECT 1"]
    assert item["tool_call_id"] is None
    assert item["tool_name"] is None
    assert item["run_id"] is None
    assert item["scope"] == "Legacy/Unknown"
    assert item["elapsed_ms"] == 1250
    assert item["patients_touched"] == []
    assert item["success"] is True


def test_legacy_question_with_no_assistant_sql_still_emits_row(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    _add_question(s, user_id=10, content="bare q", when=datetime(2026, 6, 1, 12, 0, 0))

    page = get_per_query_audit_page(_filters(), page=1, page_size=50)
    assert page["total"] == 1
    item = page["items"][0]
    assert item["pipeline"] == "legacy"
    assert item["sql_statements"] == []
    assert item["non_sql_summary"] is None
    assert item["elapsed_ms"] is None
    assert item["success"] is None  # neither sql nor error → no status info


def test_agentic_question_with_three_tool_calls_emits_three_rows(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    mid = _add_question(s, user_id=10, content="agentic q", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(s, run_id="run-1", user_id=10, user_message_id=mid)
    _add_tool_call(
        s,
        run_id="run-1",
        user_id=10,
        tool_name="run_sql",
        call_index=0,
        tool_call_uid="tc-0",
        sql_executed=["SELECT 1"],
    )
    _add_tool_call(
        s,
        run_id="run-1",
        user_id=10,
        tool_name="get_patient_clinical_data",
        call_index=1,
        tool_call_uid="tc-1",
        sql_executed=["SELECT * FROM enc", "SELECT * FROM lab"],
    )
    _add_tool_call(
        s,
        run_id="run-1",
        user_id=10,
        tool_name="summarize_results",
        call_index=2,
        tool_call_uid="tc-2",
        sql_executed=[],
        result_summary="3 rows summarised",
    )

    page = get_per_query_audit_page(_filters(), page=1, page_size=50)
    assert page["total"] == 3
    rows = page["items"]
    # All three share the question + run + message id.
    assert {r["user_message_id"] for r in rows} == {mid}
    assert {r["run_id"] for r in rows} == {"run-1"}
    assert {r["pipeline"] for r in rows} == {"agentic"}
    # Ordering: call_index ascending within the run.
    assert [r["call_index"] for r in rows] == [0, 1, 2]
    # SQL counts per row.
    assert [len(r["sql_statements"]) for r in rows] == [1, 2, 0]
    # The non-SQL row carries its result summary; SQL rows do not.
    assert rows[0]["non_sql_summary"] is None
    assert rows[1]["non_sql_summary"] is None
    assert rows[2]["non_sql_summary"] == "3 rows summarised"


def test_agentic_sql_executed_json_preserves_order(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    mid = _add_question(s, user_id=10, content="multi-sql", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(s, run_id="run-2", user_id=10, user_message_id=mid)
    _add_tool_call(
        s,
        run_id="run-2",
        user_id=10,
        tool_name="run_sql",
        call_index=0,
        tool_call_uid="tc-0",
        sql_executed=["SELECT A", "SELECT B", "SELECT C", "SELECT D"],
    )

    page = get_per_query_audit_page(_filters(), page=1, page_size=50)
    assert page["items"][0]["sql_statements"] == ["SELECT A", "SELECT B", "SELECT C", "SELECT D"]


# ---------------------------------------------------------------------------
# (2) Patient-touch decoration
# ---------------------------------------------------------------------------


def test_patients_touched_attached_per_tool_call(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    mid = _add_question(s, user_id=10, content="touch q", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(s, run_id="run-3", user_id=10, user_message_id=mid)
    _add_tool_call(
        s,
        run_id="run-3",
        user_id=10,
        tool_name="get_patient_clinical_data",
        call_index=0,
        tool_call_uid="tc-pa",
        sql_executed=["SELECT 1"],
    )
    _add_patient_access(
        s, run_id="run-3", user_id=10, source_id="pat-001", display_name="Alice A", tool_call_uid="tc-pa"
    )
    _add_patient_access(s, run_id="run-3", user_id=10, source_id="pat-002", display_name="Bob B", tool_call_uid="tc-pa")

    page = get_per_query_audit_page(_filters(), page=1, page_size=50)
    pa = page["items"][0]["patients_touched"]
    assert {p["source_id"] for p in pa} == {"pat-001", "pat-002"}
    assert {p["display_name"] for p in pa} == {"Alice A", "Bob B"}


def test_patients_touched_empty_for_legacy_rows(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    _add_question(
        s,
        user_id=10,
        content="legacy",
        when=datetime(2026, 6, 1, 12, 0, 0),
        assistant_sql="SELECT 1",
    )

    page = get_per_query_audit_page(_filters(), page=1, page_size=50)
    assert page["items"][0]["patients_touched"] == []


# ---------------------------------------------------------------------------
# (3) Filters
# ---------------------------------------------------------------------------


def _seed_mixed_pipelines(s):
    """Seeds one legacy + one agentic question for the same user."""
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    _add_question(
        s,
        user_id=10,
        content="legacy q",
        when=datetime(2026, 6, 1, 12, 0, 0),
        assistant_sql="SELECT legacy",
    )
    agentic_mid = _add_question(s, user_id=10, content="agentic q", when=datetime(2026, 6, 1, 12, 5, 0))
    _add_agent_run(s, run_id="run-pip", user_id=10, user_message_id=agentic_mid)
    _add_tool_call(
        s,
        run_id="run-pip",
        user_id=10,
        tool_name="run_sql",
        call_index=0,
        tool_call_uid="tc-p0",
        sql_executed=["SELECT agentic"],
    )


def test_pipelines_filter_agentic_only(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_mixed_pipelines(s)

    page = get_per_query_audit_page(_filters(pipelines=["agentic"]), page=1, page_size=50)
    assert all(r["pipeline"] == "agentic" for r in page["items"])
    assert page["total"] == 1


def test_pipelines_filter_legacy_only(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_mixed_pipelines(s)

    page = get_per_query_audit_page(_filters(pipelines=["legacy"]), page=1, page_size=50)
    assert all(r["pipeline"] == "legacy" for r in page["items"])
    assert page["total"] == 1


def test_source_ids_filter_excludes_legacy_and_other_runs(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    # Two agentic questions; only run-A touched pat-001.
    mid_a = _add_question(s, user_id=10, content="qA", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(s, run_id="run-A", user_id=10, user_message_id=mid_a)
    _add_tool_call(
        s, run_id="run-A", user_id=10, tool_name="run_sql", call_index=0, tool_call_uid="tcA", sql_executed=["SELECT 1"]
    )
    _add_patient_access(s, run_id="run-A", user_id=10, source_id="pat-001")

    mid_b = _add_question(s, user_id=10, content="qB", when=datetime(2026, 6, 1, 12, 5, 0))
    _add_agent_run(s, run_id="run-B", user_id=10, user_message_id=mid_b)
    _add_tool_call(
        s, run_id="run-B", user_id=10, tool_name="run_sql", call_index=0, tool_call_uid="tcB", sql_executed=["SELECT 2"]
    )
    _add_patient_access(s, run_id="run-B", user_id=10, source_id="pat-999")

    # Plus one legacy question that should never qualify under source_ids.
    _add_question(
        s,
        user_id=10,
        content="legacy q",
        when=datetime(2026, 6, 1, 12, 10, 0),
        assistant_sql="SELECT legacy",
    )

    page = get_per_query_audit_page(_filters(source_ids=["pat-001"]), page=1, page_size=50)
    assert page["total"] == 1
    assert page["items"][0]["run_id"] == "run-A"


def test_tool_names_filter_restricts_to_named_tools(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    mid = _add_question(s, user_id=10, content="mixed tools", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(s, run_id="run-t", user_id=10, user_message_id=mid)
    _add_tool_call(
        s, run_id="run-t", user_id=10, tool_name="run_sql", call_index=0, tool_call_uid="tc0", sql_executed=["SELECT 1"]
    )
    _add_tool_call(
        s,
        run_id="run-t",
        user_id=10,
        tool_name="search_knowledge_base",
        call_index=1,
        tool_call_uid="tc1",
        sql_executed=[],
        result_summary="kb hit",
    )

    page = get_per_query_audit_page(_filters(tool_names=["run_sql"]), page=1, page_size=50)
    assert all(r["tool_name"] == "run_sql" for r in page["items"])
    assert page["total"] == 1


def test_scope_filter_patient_carries_over_to_per_row_view(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    # Patient-scope run (selected_patient_source_id set) with 2 tool calls.
    mid_p = _add_question(s, user_id=10, content="patient q", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(s, run_id="run-pat", user_id=10, user_message_id=mid_p, selected_patient_source_id="pat-001")
    _add_tool_call(
        s,
        run_id="run-pat",
        user_id=10,
        tool_name="run_sql",
        call_index=0,
        tool_call_uid="tcp0",
        sql_executed=["SELECT 1"],
    )
    _add_tool_call(
        s,
        run_id="run-pat",
        user_id=10,
        tool_name="run_sql",
        call_index=1,
        tool_call_uid="tcp1",
        sql_executed=["SELECT 2"],
    )
    # Pop-Health run.
    mid_o = _add_question(s, user_id=10, content="cohort q", when=datetime(2026, 6, 1, 12, 5, 0))
    _add_agent_run(s, run_id="run-pop", user_id=10, user_message_id=mid_o)
    _add_tool_call(
        s,
        run_id="run-pop",
        user_id=10,
        tool_name="search_patients_by_criteria",
        call_index=0,
        tool_call_uid="tco0",
        sql_executed=["SELECT cohort"],
    )

    page = get_per_query_audit_page(_filters(scopes=["Patient"]), page=1, page_size=50)
    assert page["total"] == 2  # 2 tool calls in the Patient run
    assert all(r["scope"] == "Patient" for r in page["items"])
    assert {r["run_id"] for r in page["items"]} == {"run-pat"}


# ---------------------------------------------------------------------------
# (4) Pagination
# ---------------------------------------------------------------------------


def test_pagination_offset_limit_and_total(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    # 11 user questions; mix of legacy + agentic
    base = datetime(2026, 6, 1, 12, 0, 0)
    n = 0
    for i in range(7):
        _add_question(
            s,
            user_id=10,
            content=f"legacy-{i}",
            when=base - timedelta(minutes=n),
            assistant_sql=f"SELECT {i}",
        )
        n += 1
    for i in range(4):
        mid = _add_question(s, user_id=10, content=f"agentic-{i}", when=base - timedelta(minutes=n))
        _add_agent_run(s, run_id=f"run-{i}", user_id=10, user_message_id=mid)
        _add_tool_call(
            s,
            run_id=f"run-{i}",
            user_id=10,
            tool_name="run_sql",
            call_index=0,
            tool_call_uid=f"tc-{i}",
            sql_executed=[f"SELECT a{i}"],
        )
        n += 1

    page2 = get_per_query_audit_page(_filters(), page=2, page_size=5)
    assert page2["total"] == 11
    assert len(page2["items"]) == 5


# ---------------------------------------------------------------------------
# (5) Logging-mode handling (full / scrubbed / disabled)
# ---------------------------------------------------------------------------


def test_disabled_logging_mode_emits_sentinel(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    mid = _add_question(s, user_id=10, content="disabled q", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(s, run_id="run-d", user_id=10, user_message_id=mid, logging_mode="disabled")
    # Even in disabled mode there can be carrier ToolCalls (the run_logger
    # writes the row before short-circuiting payload columns). Seed one with
    # NULL sql_executed_json to mimic that.
    _add_tool_call(
        s,
        run_id="run-d",
        user_id=10,
        tool_name="run_sql",
        call_index=0,
        tool_call_uid="tcd",
        sql_executed=None,
    )

    page = get_per_query_audit_page(_filters(), page=1, page_size=50)
    item = page["items"][0]
    assert item["pipeline"] == "agentic"
    assert item["logging_mode"] == "disabled"
    assert item["sql_statements"] == []
    assert item["non_sql_summary"] == "(logging disabled)"


def test_scrubbed_mode_passes_logging_mode_through(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    mid = _add_question(s, user_id=10, content="scrubbed q", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(s, run_id="run-sc", user_id=10, user_message_id=mid, logging_mode="scrubbed")
    # In scrubbed mode the SQL literals are hashed by the writer; we pass
    # through whatever the writer stored.
    _add_tool_call(
        s,
        run_id="run-sc",
        user_id=10,
        tool_name="run_sql",
        call_index=0,
        tool_call_uid="tcs",
        sql_executed=["SELECT * FROM t WHERE id = '<hash:abc>'"],
    )

    page = get_per_query_audit_page(_filters(), page=1, page_size=50)
    item = page["items"][0]
    assert item["logging_mode"] == "scrubbed"
    assert item["sql_statements"] == ["SELECT * FROM t WHERE id = '<hash:abc>'"]
    assert item["non_sql_summary"] is None


# ---------------------------------------------------------------------------
# (6) Defensive decode + multi-run-per-message edge cases
# ---------------------------------------------------------------------------


def test_malformed_sql_executed_json_returns_empty_list(session_factory, caplog):
    import logging

    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    mid = _add_question(s, user_id=10, content="bad json", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(s, run_id="run-bad", user_id=10, user_message_id=mid)
    _add_tool_call(
        s,
        run_id="run-bad",
        user_id=10,
        tool_name="run_sql",
        call_index=0,
        tool_call_uid="tcbad",
        sql_executed="not-json{[",
    )

    with caplog.at_level(logging.WARNING):
        page = get_per_query_audit_page(_filters(), page=1, page_size=50)
    assert page["items"][0]["sql_statements"] == []
    assert any("malformed JSON" in rec.message for rec in caplog.records)


def test_two_runs_same_user_message_id_both_appear_with_deterministic_order(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    mid = _add_question(s, user_id=10, content="retried q", when=datetime(2026, 6, 1, 12, 0, 0))
    _add_agent_run(
        s,
        run_id="run-first",
        user_id=10,
        user_message_id=mid,
        created_at=datetime(2026, 6, 1, 12, 0, 5),
    )
    _add_tool_call(
        s,
        run_id="run-first",
        user_id=10,
        tool_name="run_sql",
        call_index=0,
        tool_call_uid="tcA",
        sql_executed=["SELECT first"],
    )
    _add_agent_run(
        s,
        run_id="run-second",
        user_id=10,
        user_message_id=mid,
        created_at=datetime(2026, 6, 1, 12, 0, 10),
    )
    _add_tool_call(
        s,
        run_id="run-second",
        user_id=10,
        tool_name="run_sql",
        call_index=0,
        tool_call_uid="tcB",
        sql_executed=["SELECT second"],
    )

    page = get_per_query_audit_page(_filters(), page=1, page_size=50)
    assert page["total"] == 2
    # Ordering: same asked_at, same user_message_id, so secondary sort is by
    # run_id ascending — "run-first" then "run-second".
    assert [r["run_id"] for r in page["items"]] == ["run-first", "run-second"]


# ---------------------------------------------------------------------------
# (7) Export cap
# ---------------------------------------------------------------------------


def test_export_caps_at_max_audit_export_rows(session_factory, monkeypatch):
    from orm import logging_functions as lf
    from orm.logging_functions import get_per_query_audit_export

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")

    # Seed 60 legacy questions.
    base = datetime(2026, 6, 1, 12, 0, 0)
    for i in range(60):
        _add_question(
            s,
            user_id=10,
            content=f"q-{i}",
            when=base - timedelta(minutes=i),
            assistant_sql=f"SELECT {i}",
        )

    monkeypatch.setattr(lf, "MAX_AUDIT_EXPORT_ROWS", 50)
    rows = get_per_query_audit_export(_filters())
    assert len(rows) == 50


# ---------------------------------------------------------------------------
# (8) Ordering: question time desc, then call_index within an agentic question
# ---------------------------------------------------------------------------


def test_ordering_reverse_chronological_by_question_then_call_index(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    older = datetime(2026, 6, 1, 11, 0, 0)
    newer = datetime(2026, 6, 1, 13, 0, 0)
    _add_question(s, user_id=10, content="older legacy", when=older, assistant_sql="SELECT old")
    mid = _add_question(s, user_id=10, content="newer agentic", when=newer)
    _add_agent_run(s, run_id="run-n", user_id=10, user_message_id=mid)
    _add_tool_call(
        s, run_id="run-n", user_id=10, tool_name="run_sql", call_index=2, tool_call_uid="tc2", sql_executed=["B"]
    )
    _add_tool_call(
        s, run_id="run-n", user_id=10, tool_name="run_sql", call_index=0, tool_call_uid="tc0", sql_executed=["A"]
    )
    _add_tool_call(
        s, run_id="run-n", user_id=10, tool_name="run_sql", call_index=1, tool_call_uid="tc1", sql_executed=["AB"]
    )

    page = get_per_query_audit_page(_filters(), page=1, page_size=50)
    # Newer question first; within it, call_index ascending; then legacy.
    pipelines = [(r["pipeline"], r["call_index"]) for r in page["items"]]
    assert pipelines == [
        ("agentic", 0),
        ("agentic", 1),
        ("agentic", 2),
        ("legacy", None),
    ]
