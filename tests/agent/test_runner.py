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


def test_runner_registers_phase2_tools():
    """list_patient_documents and search_codes must be registered alongside Phase 1 tools."""
    from agent.runner import AgenticRunner
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel
    from agent.deps import AgentDeps
    from agent.state import AgentResponse
    from agent.system_prompt import SYSTEM_PROMPT

    runner = AgenticRunner.__new__(AgenticRunner)  # bypass __init__ to avoid model wiring
    runner._agent = Agent(
        model=TestModel(),
        deps_type=AgentDeps,
        output_type=AgentResponse,
        system_prompt=SYSTEM_PROMPT,
    )
    runner._register_tools()
    tool_names = set(runner._agent._function_toolset.tools.keys())
    assert "find_patient" in tool_names
    assert "get_patient_clinical_data" in tool_names
    assert "search_knowledge_base" in tool_names
    assert "list_patient_documents" in tool_names
    assert "search_codes" in tool_names
