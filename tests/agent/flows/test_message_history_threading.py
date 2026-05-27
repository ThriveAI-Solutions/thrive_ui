"""runner.stream() and runner.run() must thread message_history through
to pydantic-ai so multi-turn conversations work.

Two behaviors covered:
1. Caller-supplied message_history is passed to the underlying Agent.
2. After a run, the runner exposes the resulting transcript so the
   runtime can persist it back to st.session_state.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.deps import AgentDeps
from agent.runner import AgenticRunner
from agent.state import FinalResponseEvent


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


def _final_only_model(captured_messages: list):
    """Replies with final_result immediately. Records the messages it
    received so the test can assert prior turns were threaded in."""

    async def behavior(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured_messages.append(list(messages))
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
                    tool_call_id="c1",
                ),
            ]
        )

    return FunctionModel(behavior, model_name="final-only")


def _deps(engine) -> AgentDeps:
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
        run_logger=MagicMock(),
    )


@pytest.mark.asyncio
async def test_run_passes_message_history_to_agent():
    engine = _make_threadsafe_db()
    captured: list[list[ModelMessage]] = []
    runner = AgenticRunner(model=_final_only_model(captured))
    prior = [
        ModelRequest(parts=[UserPromptPart(content="earlier question")]),
    ]
    await runner.run("follow-up", deps=_deps(engine), message_history=prior)
    assert captured, "model never invoked"
    flat_text = ""
    for m in captured[0]:
        for part in getattr(m, "parts", []):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                flat_text += content
    assert "earlier question" in flat_text


@pytest.mark.asyncio
async def test_stream_passes_message_history_to_agent():
    engine = _make_threadsafe_db()
    captured: list[list[ModelMessage]] = []
    runner = AgenticRunner(model=_final_only_model(captured))
    prior = [
        ModelRequest(parts=[UserPromptPart(content="earlier streamed question")]),
    ]
    async for _ in runner.stream("follow-up", deps=_deps(engine), message_history=prior):
        pass
    assert captured, "model never invoked"
    flat_text = ""
    for m in captured[0]:
        for part in getattr(m, "parts", []):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                flat_text += content
    assert "earlier streamed question" in flat_text


@pytest.mark.asyncio
async def test_stream_final_event_carries_all_messages():
    """The runtime needs the post-run transcript to persist into
    session_state. The FinalResponseEvent must surface it."""
    engine = _make_threadsafe_db()
    captured: list[list[ModelMessage]] = []
    runner = AgenticRunner(model=_final_only_model(captured))
    finals: list[FinalResponseEvent] = []
    async for ev in runner.stream("hi", deps=_deps(engine)):
        if isinstance(ev, FinalResponseEvent):
            finals.append(ev)
    assert len(finals) == 1
    assert hasattr(finals[0], "all_messages"), "FinalResponseEvent missing all_messages"
    msgs = finals[0].all_messages
    assert isinstance(msgs, list)
    # Should at minimum contain the user prompt + the model's response.
    assert any(isinstance(m, ModelRequest) for m in msgs)
    assert any(isinstance(m, ModelResponse) for m in msgs)
