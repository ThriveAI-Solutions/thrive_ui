"""ToolCallCompleted events must carry the SQL that was executed and a
serialized snapshot of the result payload, so the renderer can surface
both behind a role-gated expander.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.deps import AgentDeps
from agent.runner import AgenticRunner
from agent.state import ToolCallCompleted


_SQL_FILE = Path(__file__).parents[2] / "agent" / "redshift_synthetic.sql"


def _make_threadsafe_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    sql = _SQL_FILE.read_text()
    with engine.begin() as conn:
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    return engine


def _scripted_find_then_final():
    turn = {"n": 0}

    async def behavior(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        turn["n"] += 1
        if turn["n"] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="find_patient",
                        args={"last_name": "Smith"},
                        tool_call_id="call-1",
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "text": "ok",
                        "followups": [],
                        "artifacts": [],
                        "clear_selection": False,
                        "cap_reached": False,
                    },
                    tool_call_id="call-2",
                )
            ]
        )

    return FunctionModel(behavior, model_name="scripted")


def _deps(engine):
    return AgentDeps(
        user_id=1,
        user_role=MagicMock(value=1),
        session_id="s",
        selected_patient=None,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=AnalyticsDbAdapter(engine=engine, dialect="sqlite"),
        rag=MagicMock(),
        sqlite_session=None,
        audit_logger=MagicMock(),
    )


@pytest.mark.asyncio
async def test_completed_event_carries_executed_sql():
    """find_patient runs two SELECTs (one for matches, one for related
    source_ids per match). Both should land on the event."""
    runner = AgenticRunner(model=_scripted_find_then_final())
    completed: list[ToolCallCompleted] = []
    async for ev in runner.stream("smith", deps=_deps(_make_threadsafe_db())):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)
    assert completed, "no ToolCallCompleted yielded"
    assert hasattr(completed[0], "sql_executed")
    sql_list = completed[0].sql_executed
    assert isinstance(sql_list, list)
    assert sql_list, "sql_executed is empty even though find_patient ran SQL"
    # The SQL should contain the patient view at minimum.
    combined = "\n".join(s["sql"] for s in sql_list)
    assert "internal_patient_profile_v" in combined


@pytest.mark.asyncio
async def test_completed_event_carries_result_payload():
    runner = AgenticRunner(model=_scripted_find_then_final())
    completed: list[ToolCallCompleted] = []
    async for ev in runner.stream("smith", deps=_deps(_make_threadsafe_db())):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)
    assert completed
    assert hasattr(completed[0], "result_payload")
    payload = completed[0].result_payload
    # find_patient returns a PatientSearchResults — must serialize to dict
    # with the matches list visible.
    assert isinstance(payload, dict)
    assert "matches" in payload
    assert isinstance(payload["matches"], list)
    assert payload["matches"], "no matches in serialized payload"


@pytest.mark.asyncio
async def test_sql_log_resets_between_tools():
    """Running two tools in one agent run must not leak SQL between them."""
    # Use TestModel pattern — script find_patient twice via a custom behavior.
    turn = {"n": 0}

    async def behavior(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        turn["n"] += 1
        if turn["n"] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="find_patient",
                        args={"last_name": "Smith"},
                        tool_call_id="c1",
                    )
                ]
            )
        if turn["n"] == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="find_patient",
                        args={"last_name": "Doe"},
                        tool_call_id="c2",
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "text": "done",
                        "followups": [],
                        "artifacts": [],
                        "clear_selection": False,
                        "cap_reached": False,
                    },
                    tool_call_id="c3",
                )
            ]
        )

    runner = AgenticRunner(model=FunctionModel(behavior, model_name="x"))
    completed: list[ToolCallCompleted] = []
    async for ev in runner.stream("x", deps=_deps(_make_threadsafe_db())):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)
    assert len(completed) == 2
    # Each tool's SQL log should reflect only its own params.
    first_combined = "\n".join(f"{s['sql']} {s['params']}" for s in completed[0].sql_executed)
    second_combined = "\n".join(f"{s['sql']} {s['params']}" for s in completed[1].sql_executed)
    assert "smith" in first_combined.lower()
    assert "doe" in second_combined.lower()
    assert "doe" not in first_combined.lower()
    assert "smith" not in second_combined.lower()
