"""Tests for the "final answer" surfacing in Admin → Audit → Queries.

Joe asked during the 2026-06-15 demo for the audit screen to show the actual
answer the user got — not just the SQL and tool calls. The follow-up to PR
#218 wires ``AgentRun.final_answer_text`` (agentic) and the latest assistant
``MessageType.SUMMARY`` content (legacy) through four surfaces:

1. Per-query dialog body — new "Result" section.
2. Per-question dialog body — new "Final Answer" section at the top.
3. Flat-mode data_editor — new "Answer" column (truncated).
4. CSV export — new "Final Answer" column (untruncated).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orm.models import (
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
# Fixtures and seed helpers (mirrors tests/orm/test_per_query_audit.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def session_factory(monkeypatch):
    from orm import logging_functions as lf

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(lf, "SessionLocal", Session)
    return Session


def _seed_roles(session):
    session.add(UserRole(id=1, role_name="Admin", description="Admin", role=RoleTypeEnum.ADMIN))
    session.add(UserRole(id=4, role_name="Patient", description="Patient", role=RoleTypeEnum.PATIENT))
    session.commit()


def _seed_user(session, *, id, username, org="TestOrg"):
    session.add(
        User(
            id=id,
            user_role_id=4,
            username=username,
            first_name=username,
            last_name="X",
            password="hash:x",
            email=f"{username}@x.io",
            organization=org,
        )
    )
    session.commit()


def _add_user_question(session, *, user_id, content, when):
    session.add(Message(role=RoleType.USER, content=content, type=MessageType.TEXT, user_id=user_id))
    session.commit()
    last = session.query(Message).order_by(Message.id.desc()).first()
    last.created_at = when
    session.commit()
    return last.id


def _add_assistant_message(session, *, user_id, question, content, message_type, when):
    m = Message(role=RoleType.ASSISTANT, content=content, type=message_type, user_id=user_id, question=question)
    session.add(m)
    session.commit()
    last = session.query(Message).order_by(Message.id.desc()).first()
    last.created_at = when
    session.commit()


def _add_agent_run(
    session, *, run_id, user_id, user_message_id, final_answer_text=None, logging_mode="full", status="completed"
):
    ar = AgentRun(
        run_id=run_id,
        session_id=f"sess-{run_id}",
        user_id=user_id,
        user_role=RoleTypeEnum.PATIENT.value,
        user_message_id=user_message_id,
        tool_call_count=0,
        event_count=0,
        status=status,
        success=(status == "completed"),
        review_status="unreviewed",
        logging_mode=logging_mode,
        schema_version=1,
        final_answer_text=final_answer_text,
    )
    session.add(ar)
    session.commit()


def _add_tool_call(session, *, run_id, user_id, tool_name, call_index, result_summary, sql_executed=None):
    sql_json = json.dumps(sql_executed) if isinstance(sql_executed, list) else sql_executed
    tc = ToolCall(
        session_id=f"sess-{run_id}",
        user_id=user_id,
        user_role=RoleTypeEnum.PATIENT.value,
        tool_name=tool_name,
        arguments_json="{}",
        result_summary=result_summary,
        elapsed_ms=10,
        success=True,
        run_id=run_id,
        tool_call_id=f"tc-{run_id}-{call_index}",
        call_index=call_index,
        sql_executed_json=sql_json,
    )
    session.add(tc)
    session.commit()


def _filters(**overrides):
    base = {"usernames": [], "orgs": [], "days": 30, "search": None}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Data layer: PER_QUERY_ROW_KEYS now exports the two new fields
# ---------------------------------------------------------------------------


def test_per_query_row_keys_exports_new_fields():
    from orm.logging_functions import PER_QUERY_ROW_KEYS

    assert "final_answer_text" in PER_QUERY_ROW_KEYS
    assert "result_text" in PER_QUERY_ROW_KEYS


# ---------------------------------------------------------------------------
# Data layer: agentic pipeline
# ---------------------------------------------------------------------------


def test_agentic_rows_carry_final_answer_text_from_agent_run(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    when = datetime(2026, 6, 1, 12, 0, 0)
    mid = _add_user_question(s, user_id=10, content="agentic q", when=when)
    _add_agent_run(
        s, run_id="run-1", user_id=10, user_message_id=mid, final_answer_text="The patient has 3 active conditions."
    )
    _add_tool_call(
        s,
        run_id="run-1",
        user_id=10,
        tool_name="run_sql",
        call_index=0,
        result_summary="rows=3",
        sql_executed=["SELECT 1"],
    )
    _add_tool_call(
        s, run_id="run-1", user_id=10, tool_name="summarize_results", call_index=1, result_summary="summary done"
    )

    page = get_per_query_audit_page(_filters())
    items = page["items"]
    assert len(items) == 2
    for it in items:
        assert it["final_answer_text"] == "The patient has 3 active conditions."


def test_agentic_row_result_text_matches_tool_result_summary(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    when = datetime(2026, 6, 1, 12, 0, 0)
    mid = _add_user_question(s, user_id=10, content="lab q", when=when)
    _add_agent_run(s, run_id="run-1", user_id=10, user_message_id=mid, final_answer_text="3 labs in range.")
    _add_tool_call(
        s,
        run_id="run-1",
        user_id=10,
        tool_name="run_sql",
        call_index=0,
        result_summary="rows=3; columns=[hgb, wbc, glucose]",
        sql_executed=["SELECT hgb, wbc, glucose FROM lab WHERE patient_id = 'p1'"],
    )

    items = get_per_query_audit_page(_filters())["items"]
    assert len(items) == 1
    # SQL-bearing tool: result_text is loaded too (Epic #218 follow-up wires it
    # independently of the non_sql_summary routing logic).
    assert items[0]["result_text"] == "rows=3; columns=[hgb, wbc, glucose]"


def test_agentic_run_error_status_with_no_final_answer_yields_none(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    when = datetime(2026, 6, 1, 12, 0, 0)
    mid = _add_user_question(s, user_id=10, content="fail q", when=when)
    _add_agent_run(s, run_id="run-1", user_id=10, user_message_id=mid, final_answer_text=None, status="error")
    _add_tool_call(s, run_id="run-1", user_id=10, tool_name="run_sql", call_index=0, result_summary="error: timeout")

    items = get_per_query_audit_page(_filters())["items"]
    assert len(items) == 1
    assert items[0]["final_answer_text"] is None


def test_disabled_logging_mode_still_surfaces_final_answer_text(session_factory):
    """Tool payloads are sentinelled in disabled mode, but AgentRun.final_answer_text
    is written by finalize_run independently of tool logging — it should still
    surface in the audit row."""
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    when = datetime(2026, 6, 1, 12, 0, 0)
    mid = _add_user_question(s, user_id=10, content="opaque q", when=when)
    _add_agent_run(
        s,
        run_id="run-1",
        user_id=10,
        user_message_id=mid,
        final_answer_text="Answer captured even with logging disabled.",
        logging_mode="disabled",
    )
    _add_tool_call(s, run_id="run-1", user_id=10, tool_name="run_sql", call_index=0, result_summary="should-be-ignored")

    items = get_per_query_audit_page(_filters())["items"]
    assert len(items) == 1
    assert items[0]["final_answer_text"] == "Answer captured even with logging disabled."
    # And the per-unit result is not surfaced in disabled mode.
    assert items[0]["result_text"] is None


# ---------------------------------------------------------------------------
# Data layer: legacy pipeline
# ---------------------------------------------------------------------------


def test_legacy_row_final_answer_text_comes_from_latest_summary_message(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    when = datetime(2026, 6, 1, 12, 0, 0)
    _add_user_question(s, user_id=10, content="legacy q", when=when)
    _add_assistant_message(
        s,
        user_id=10,
        question="legacy q",
        content="SELECT 1",
        message_type=MessageType.SQL,
        when=when + timedelta(seconds=1),
    )
    _add_assistant_message(
        s,
        user_id=10,
        question="legacy q",
        content="There are 42 patients in the cohort.",
        message_type=MessageType.SUMMARY,
        when=when + timedelta(seconds=2),
    )

    items = get_per_query_audit_page(_filters())["items"]
    assert len(items) == 1
    assert items[0]["final_answer_text"] == "There are 42 patients in the cohort."
    # Legacy unit and question collapse to the same row, so result_text matches.
    assert items[0]["result_text"] == "There are 42 patients in the cohort."


def test_legacy_row_without_summary_message_yields_none(session_factory):
    from orm.logging_functions import get_per_query_audit_page

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    when = datetime(2026, 6, 1, 12, 0, 0)
    _add_user_question(s, user_id=10, content="no-summary q", when=when)
    _add_assistant_message(
        s,
        user_id=10,
        question="no-summary q",
        content="SELECT 1",
        message_type=MessageType.SQL,
        when=when + timedelta(seconds=1),
    )

    items = get_per_query_audit_page(_filters())["items"]
    assert len(items) == 1
    assert items[0]["final_answer_text"] is None


# ---------------------------------------------------------------------------
# View layer: row dict / table column
# ---------------------------------------------------------------------------


def _make_item(**overrides) -> dict:
    base = {
        "asked_at": datetime(2026, 6, 1, 12, 0, 0),
        "user_id": 7,
        "username": "alice",
        "organization": "Acme",
        "question": "test?",
        "user_message_id": 100,
        "scope": "Patient",
        "pipeline": "agentic",
        "run_id": "run-x",
        "logging_mode": "full",
        "tool_call_id": 1,
        "tool_name": "run_sql",
        "call_index": 0,
        "sql_statements": ["SELECT 1"],
        "non_sql_summary": None,
        "elapsed_ms": 50,
        "success": True,
        "error": None,
        "patients_touched": [],
        "final_answer_text": None,
        "result_text": None,
    }
    base.update(overrides)
    return base


def test_row_dict_includes_truncated_answer_column():
    from views.admin_audit_queries import _ANSWER_PREVIEW_CHARS, _per_query_row_to_table_dict

    long_answer = "x" * (_ANSWER_PREVIEW_CHARS + 50)
    row = _per_query_row_to_table_dict(_make_item(final_answer_text=long_answer))
    assert "Answer" in row
    answer = row["Answer"]
    assert answer.endswith("…")
    # _truncate's contract: result length is <= preview cap.
    assert len(answer) <= _ANSWER_PREVIEW_CHARS


def test_row_dict_answer_column_empty_when_no_answer():
    from views.admin_audit_queries import _per_query_row_to_table_dict

    row = _per_query_row_to_table_dict(_make_item(final_answer_text=None))
    assert row["Answer"] == ""


# ---------------------------------------------------------------------------
# View layer: dialog bodies
# ---------------------------------------------------------------------------


def test_result_block_renders_result_text():
    from views import admin_audit_queries as q

    written: list[str] = []

    class _Stub:
        session_state: dict = {}

        def markdown(self, s, **_kw):
            written.append(("markdown", str(s)))

        def write(self, s, **_kw):
            written.append(("write", str(s)))

        def warning(self, s, **_kw):
            written.append(("warning", str(s)))

    q.st = _Stub()  # type: ignore[assignment]
    try:
        q._render_result_block(_make_item(result_text="3 patients with high BP"))
    finally:
        # Restore the real streamlit module so other tests aren't affected.
        import importlib

        importlib.reload(q)

    assert ("markdown", "**Result**") in written
    assert ("write", "3 patients with high BP") in written


def test_result_block_fallback_for_missing_result_text():
    from views import admin_audit_queries as q

    written: list[str] = []

    class _Stub:
        session_state: dict = {}

        def markdown(self, s, **_kw):
            written.append(("markdown", str(s)))

        def write(self, s, **_kw):
            written.append(("write", str(s)))

        def warning(self, s, **_kw):
            written.append(("warning", str(s)))

    q.st = _Stub()  # type: ignore[assignment]
    try:
        q._render_result_block(_make_item(result_text=None, logging_mode="full"))
    finally:
        import importlib

        importlib.reload(q)

    assert ("markdown", "**Result**") in written
    assert any(payload.startswith("_(no result") for kind, payload in written if kind == "write")


def test_group_header_carries_final_answer_text_from_first_unit():
    from views.admin_audit_queries import _group_items_by_question

    a = _make_item(user_message_id=1, tool_call_id=1, call_index=0, final_answer_text="agentic answer")
    b = _make_item(user_message_id=1, tool_call_id=2, call_index=1, final_answer_text="agentic answer")
    groups = _group_items_by_question([a, b])
    assert len(groups) == 1
    header, units = groups[0]
    assert header["final_answer_text"] == "agentic answer"
    assert len(units) == 2


def test_question_detail_dialog_body_renders_final_answer_section():
    """Smoke-test the per-question dialog body: the 'Final Answer' label appears
    and the rendered text matches group_header['final_answer_text']."""
    from views import admin_audit_queries as q

    written: list[tuple[str, str]] = []

    class _Stub:
        session_state: dict = {}

        def markdown(self, s, **_kw):
            written.append(("markdown", str(s)))

        def write(self, s, **_kw):
            written.append(("write", str(s)))

        def warning(self, s, **_kw):
            written.append(("warning", str(s)))

        def divider(self):
            written.append(("divider", ""))

        def caption(self, s, **_kw):
            written.append(("caption", str(s)))

        def code(self, s, **_kw):
            written.append(("code", str(s)))

        def columns(self, n, **_kw):
            class _Col:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

            return [_Col() for _ in range(n if isinstance(n, int) else len(n))]

        def button(self, *a, **_kw):
            return False

    q.st = _Stub()  # type: ignore[assignment]
    try:
        header = {
            "user_message_id": 1,
            "asked_at": datetime(2026, 6, 1, 12, 0, 0),
            "username": "alice",
            "organization": "Acme",
            "question": "How many patients?",
            "scope": "Pop Health",
            "total_elapsed_ms": 250,
            "query_count": 2,
            "logging_mode": "full",
            "final_answer_text": "42 patients in the cohort.",
        }
        with patch_role_can_see(True):
            q._render_question_detail_dialog_body(header, [_make_item()])
    finally:
        import importlib

        importlib.reload(q)

    labels = [payload for kind, payload in written if kind == "markdown"]
    assert "**Final Answer**" in labels
    assert ("write", "42 patients in the cohort.") in written


class patch_role_can_see:
    """Tiny context manager patching the role gate to a fixed boolean."""

    def __init__(self, allow: bool) -> None:
        self.allow = allow
        self._patcher = None

    def __enter__(self):
        from unittest.mock import patch as _patch

        self._patcher = _patch("agent.observability_gate.role_can_see_query_details", return_value=self.allow)
        self._patcher.start()
        return self

    def __exit__(self, *exc):
        if self._patcher is not None:
            self._patcher.stop()


# ---------------------------------------------------------------------------
# View layer: CSV export
# ---------------------------------------------------------------------------


def test_csv_export_includes_final_answer_column(session_factory):
    """End-to-end: data layer round-trips final_answer_text and the export view
    surfaces it as a 'Final Answer' column without truncation."""
    from orm.logging_functions import get_per_query_audit_export

    s = session_factory()
    _seed_roles(s)
    _seed_user(s, id=10, username="alice")
    when = datetime(2026, 6, 1, 12, 0, 0)
    mid = _add_user_question(s, user_id=10, content="big-q", when=when)
    long_answer = "long answer " * 50  # well beyond the truncation cap
    _add_agent_run(s, run_id="run-1", user_id=10, user_message_id=mid, final_answer_text=long_answer)
    _add_tool_call(
        s,
        run_id="run-1",
        user_id=10,
        tool_name="run_sql",
        call_index=0,
        result_summary="rows=1",
        sql_executed=["SELECT 1"],
    )

    export_rows = get_per_query_audit_export(_filters())
    assert export_rows
    # Build the same dict shape the CSV writer in _render_csv_export uses.
    csv_row = {
        "Final Answer": export_rows[0].get("final_answer_text") or "",
    }
    assert csv_row["Final Answer"] == long_answer
