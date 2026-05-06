"""Per spec §11.2 — selection flow end-to-end with FunctionModel."""

import json
import pytest
from unittest.mock import MagicMock
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
)

from agent.runner import AgenticRunner
from agent.deps import AgentDeps
from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.state import AgentResponse


_SQL_FILE = Path(__file__).parents[2] / "agent" / "redshift_synthetic.sql"


def _make_threadsafe_db():
    """In-memory SQLite with StaticPool so cross-thread connections share data.

    Pydantic AI runs sync tools in anyio's thread-pool executor; without
    StaticPool each thread opens a new :memory: database and finds no tables.
    """
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


def _scripted_llm(call_log: list[dict]):
    """Returns a FunctionModel that:
    - Turn 1: calls find_patient(last_name='Smith')
    - Turn 2 (after find_patient result): calls the 'final_result' output tool
      with an AgentResponse asking the user to select.

    The agent is configured with output_type=AgentResponse so Pydantic AI
    exposes a 'final_result' output tool; text responses are rejected.
    """
    turn = {"n": 0}

    async def behavior(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        turn["n"] += 1
        call_log.append({"turn": turn["n"], "messages": len(messages)})
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
        # Turn 2: return a structured AgentResponse via the output tool.
        response_payload = {
            "text": "Found 3 patients named Smith. Please select.",
            "followups": [],
            "artifacts": [],
            "clear_selection": False,
            "cap_reached": False,
        }
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args=response_payload,
                    tool_call_id="call-2",
                ),
            ]
        )

    return FunctionModel(behavior, model_name="scripted")


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
        audit_logger=MagicMock(),
    )


@pytest.mark.asyncio
async def test_disambiguation_flow():
    engine = _make_threadsafe_db()
    log: list[dict] = []
    runner = AgenticRunner(model=_scripted_llm(log))
    response = await runner.run("show me Smith's record", deps=_deps(engine))
    assert isinstance(response, AgentResponse)
    assert "Smith" in response.text or "patient" in response.text.lower()
    assert len(log) == 2
