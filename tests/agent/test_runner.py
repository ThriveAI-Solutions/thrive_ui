"""Phase 1 runner tests using Pydantic AI's TestModel for deterministic
LLM behavior. Per spec §11.2."""

import pytest
from pydantic_ai.models.test import TestModel
from agent.runner import AgenticRunner
from agent.deps import AgentDeps
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_runner_constructs_with_test_model():
    runner = AgenticRunner(model=TestModel())
    assert runner._agent is not None


@pytest.mark.asyncio
async def test_runner_run_returns_agent_response():
    runner = AgenticRunner(model=TestModel(call_tools=[]))
    deps = MagicMock(spec=AgentDeps)
    response = await runner.run("hello", deps=deps)
    assert hasattr(response, "text")
