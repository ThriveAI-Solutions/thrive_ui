from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from agent.db.analytics_adapter import AnalyticsDbAdapter
from agent.deps import AgentDeps
from agent.logging_config import AgentLoggingConfig
from agent.run_logger import AgentRunLogger
from agent.runner import AgenticRunner
from orm.models import AgentRun, AgentRunEvent, Base, ToolCall


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
                parts=[ToolCallPart(tool_name="find_patient", args={"last_name": "Smith"}, tool_call_id="call-1")]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={"text": "Found 3 patients.", "followups": [], "clear_selection": False, "cap_reached": False},
                    tool_call_id="call-2",
                )
            ]
        )

    return FunctionModel(behavior, model_name="scripted")


@pytest.fixture
def logged_deps():
    app_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(app_engine)
    session = sessionmaker(bind=app_engine)()
    run_logger = AgentRunLogger(
        session=session, config=AgentLoggingConfig(), run_id="run-e2e", session_id="s1", user_id=1, user_role=1
    )
    deps = AgentDeps(
        user_id=1,
        user_role=MagicMock(value=1),
        session_id="s1",
        selected_patient=None,
        last_dataframe=None,
        last_sql=None,
        last_query_meta=None,
        analytics_db=AnalyticsDbAdapter(engine=_make_threadsafe_db(), dialect="sqlite"),
        rag=MagicMock(),
        sqlite_session=session,
        run_logger=run_logger,
        group_id="g1",
        user_message_id=None,
        parent_run_id=None,
        resume_reason=None,
    )
    runner = AgenticRunner(model=_scripted_llm())
    return session, deps, runner


@pytest.mark.asyncio
async def test_full_run_persists_run_events_and_toolcalls(logged_deps):
    session, deps, runner = logged_deps
    async for _ in runner.stream("find patient Ann", deps=deps):
        pass
    run = session.query(AgentRun).filter_by(run_id=deps.run_logger.run_id).one()
    assert run.status in ("success", "cap_reached")
    events = session.query(AgentRunEvent).order_by(AgentRunEvent.seq).all()
    assert events[0].event_type == "run_started"
    assert events[-1].event_type == "run_completed"
    assert session.query(ToolCall).filter_by(run_id=run.run_id).count() >= 1
