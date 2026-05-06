"""AgenticRunner — owns the Pydantic AI Agent and tool registrations.

Per spec §6: agent.iter() exposes node-level events for streaming.
Per §12.4: hard caps on tool count and wall-clock per run.

Caps are tunable via secrets.toml:
    [agent]
    max_tool_calls = 7         # default
    max_wall_clock_s = 120.0   # default; raise for slower local models
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


_DEFAULT_MAX_TOOL_CALLS = 7
_DEFAULT_MAX_WALL_CLOCK_S = 120.0


def _max_tool_calls() -> int:
    try:
        import streamlit as st

        return int(st.secrets.get("agent", {}).get("max_tool_calls", _DEFAULT_MAX_TOOL_CALLS))
    except Exception:
        return _DEFAULT_MAX_TOOL_CALLS


def _max_wall_clock_s() -> float:
    try:
        import streamlit as st

        return float(st.secrets.get("agent", {}).get("max_wall_clock_s", _DEFAULT_MAX_WALL_CLOCK_S))
    except Exception:
        return _DEFAULT_MAX_WALL_CLOCK_S


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
        wall_clock_cap = _max_wall_clock_s()
        tool_call_cap = _max_tool_calls()

        async with self._agent.iter(question, deps=deps) as run:
            async for node in run:
                if asyncio.get_event_loop().time() - start > wall_clock_cap:
                    yield CapReachedEvent(reason="wall_clock")
                    break

                # Pydantic AI exposes specific node types; we use the
                # has-tool-calls / has-output convention rather than
                # importing private node classes.
                tool_calls = getattr(node, "tool_calls", None) or []
                for tc in tool_calls:
                    tool_count += 1
                    if tool_count > tool_call_cap:
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
