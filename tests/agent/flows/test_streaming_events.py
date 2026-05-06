"""Streaming events from AgenticRunner.stream() must fire in order:
ToolCallStarted → ToolCallCompleted → FinalResponseEvent.

Uses FunctionModel like test_disambiguation_flow.
"""

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
from agent.state import (
    AgentResponse,
    CapReachedEvent,
    FinalResponseEvent,
    ToolCallCompleted,
    ToolCallStarted,
)


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


def _scripted_llm():
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
                    ),
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "text": "Found 3 patients.",
                        "followups": [],
                        "artifacts": [],
                        "clear_selection": False,
                        "cap_reached": False,
                    },
                    tool_call_id="call-2",
                ),
            ]
        )

    return FunctionModel(behavior, model_name="scripted")


def _deps(engine, audit_logger=None) -> AgentDeps:
    return AgentDeps(
        user_id=1,
        user_role=MagicMock(value=1),
        session_id="s1",
        selected_patient=None,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=AnalyticsDbAdapter(engine=engine, dialect="sqlite"),
        rag=MagicMock(),
        sqlite_session=None,
        audit_logger=audit_logger or MagicMock(),
    )


@pytest.mark.asyncio
async def test_stream_yields_started_completed_then_final_in_order():
    engine = _make_threadsafe_db()
    runner = AgenticRunner(model=_scripted_llm())
    events: list = []
    async for ev in runner.stream("Smith", deps=_deps(engine)):
        events.append(ev)

    started_idxs = [i for i, e in enumerate(events) if isinstance(e, ToolCallStarted)]
    completed_idxs = [i for i, e in enumerate(events) if isinstance(e, ToolCallCompleted)]
    final_idxs = [i for i, e in enumerate(events) if isinstance(e, FinalResponseEvent)]

    assert started_idxs, f"no ToolCallStarted in {events}"
    assert completed_idxs, f"no ToolCallCompleted in {events}"
    assert final_idxs, f"no FinalResponseEvent in {events}"
    assert started_idxs[0] < completed_idxs[0] < final_idxs[0]


@pytest.mark.asyncio
async def test_stream_started_event_carries_tool_name_and_args():
    engine = _make_threadsafe_db()
    runner = AgenticRunner(model=_scripted_llm())
    started: list[ToolCallStarted] = []
    async for ev in runner.stream("Smith", deps=_deps(engine)):
        if isinstance(ev, ToolCallStarted):
            started.append(ev)

    assert any(e.tool_name == "find_patient" for e in started)
    fp = next(e for e in started if e.tool_name == "find_patient")
    assert fp.arguments.get("last_name") == "Smith"


@pytest.mark.asyncio
async def test_stream_completed_event_has_positive_elapsed_ms():
    engine = _make_threadsafe_db()
    runner = AgenticRunner(model=_scripted_llm())
    completed: list[ToolCallCompleted] = []
    async for ev in runner.stream("Smith", deps=_deps(engine)):
        if isinstance(ev, ToolCallCompleted):
            completed.append(ev)

    assert completed
    assert all(c.elapsed_ms >= 0 for c in completed)
    assert all(c.success is True for c in completed)


@pytest.mark.asyncio
async def test_stream_final_event_carries_agent_response():
    engine = _make_threadsafe_db()
    runner = AgenticRunner(model=_scripted_llm())
    finals: list[FinalResponseEvent] = []
    async for ev in runner.stream("Smith", deps=_deps(engine)):
        if isinstance(ev, FinalResponseEvent):
            finals.append(ev)

    assert len(finals) == 1
    assert isinstance(finals[0].response, AgentResponse)
    assert "patient" in finals[0].response.text.lower()


@pytest.mark.asyncio
async def test_stream_calls_audit_logger_once_per_tool_call():
    engine = _make_threadsafe_db()
    audit_logger = MagicMock()
    runner = AgenticRunner(model=_scripted_llm())
    deps = _deps(engine, audit_logger=audit_logger)
    async for _ in runner.stream("Smith", deps=deps):
        pass

    # Exactly one audit call for find_patient. final_result is the
    # output-bearing tool and should not be audited.
    audited_tools = [c.kwargs.get("tool_name") for c in audit_logger.log.call_args_list]
    assert "find_patient" in audited_tools
    assert "final_result" not in audited_tools
