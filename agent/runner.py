"""AgenticRunner — owns the Pydantic AI Agent and tool registrations.

Per spec §6: agent.iter() exposes node-level events for streaming.
Per §12.4: hard cap of 7 tool calls and 30s wall-clock per run.
"""

from __future__ import annotations
import asyncio
from typing import Any, AsyncIterator, Optional

from pydantic_ai import Agent
from pydantic_ai.models import Model

from agent.deps import AgentDeps
from agent.models import build_model
from agent.state import (
    AgentResponse,
    StreamEvent,
    ToolCallStarted,
    ToolCallCompleted,
    FinalResponseEvent,
    CapReachedEvent,
)
from agent.system_prompt import SYSTEM_PROMPT
from agent.tools.find_patient import find_patient
from agent.tools.get_patient_clinical_data import get_patient_clinical_data
from agent.tools.search_knowledge_base import search_knowledge_base


_MAX_TOOL_CALLS = 7
_MAX_WALL_CLOCK_S = 30.0


class AgenticRunner:
    """Owns the Pydantic AI Agent and its tool registrations.

    A single runner is constructed at process start and reused across
    requests via Streamlit's @st.cache_resource. Per-request state lives
    in AgentDeps, not on the runner.
    """

    def __init__(self, model: Optional[Model] = None) -> None:
        self._agent: Agent[AgentDeps, AgentResponse] = Agent(
            model=model or build_model(),
            deps_type=AgentDeps,
            output_type=AgentResponse,
            system_prompt=SYSTEM_PROMPT,
            retries=2,
        )
        self._register_tools()

    def _register_tools(self) -> None:
        self._agent.tool(find_patient)
        self._agent.tool(get_patient_clinical_data)
        self._agent.tool(search_knowledge_base)
        # Phase 2+ tools register here as they ship.

    async def run(self, question: str, deps: AgentDeps) -> AgentResponse:
        """Single-shot run; collects the final AgentResponse without streaming.

        Used by tests and any caller that doesn't need step-by-step streaming.
        """
        result = await self._agent.run(question, deps=deps)
        return result.output

    async def stream(
        self,
        question: str,
        deps: AgentDeps,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming variant. Yields ToolCallStarted / ToolCallCompleted /
        FinalResponseEvent / CapReachedEvent in order.

        Used by chat_bot_helper to render tool-call cards as they happen.
        """
        start = asyncio.get_event_loop().time()
        tool_count = 0

        async with self._agent.iter(question, deps=deps) as run:
            async for node in run:
                if asyncio.get_event_loop().time() - start > _MAX_WALL_CLOCK_S:
                    yield CapReachedEvent(reason="wall_clock")
                    break

                # Pydantic AI exposes specific node types; we use the
                # has-tool-calls / has-output convention rather than
                # importing private node classes.
                tool_calls = getattr(node, "tool_calls", None) or []
                for tc in tool_calls:
                    tool_count += 1
                    if tool_count > _MAX_TOOL_CALLS:
                        yield CapReachedEvent(reason="tool_count")
                        return
                    yield ToolCallStarted(
                        tool_name=getattr(tc, "tool_name", "<unknown>"),
                        arguments=dict(getattr(tc, "args", {}) or {}),
                    )

                output = getattr(node, "output", None)
                if isinstance(output, AgentResponse):
                    yield FinalResponseEvent(response=output)

        # Fallback: if we exited the loop without yielding a final response,
        # surface whatever the run produced.
        if hasattr(run, "result") and run.result is not None:
            output = run.result.output
            if isinstance(output, AgentResponse):
                yield FinalResponseEvent(response=output)
