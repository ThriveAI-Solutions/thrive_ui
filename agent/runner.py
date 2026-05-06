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
import json
from typing import Any, AsyncIterator, Optional

from pydantic_ai import (
    Agent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
)
from pydantic_ai.models import Model

from agent.audit import summarize_result
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
# pydantic-ai exposes a synthetic 'final_result' tool for output-typed agents;
# it is not a user-facing tool and should not appear in the audit log or UI.
_OUTPUT_TOOL_NAMES = {"final_result"}


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


def _normalize_args(raw: Any) -> dict:
    """Pydantic AI gives args as either a dict or a JSON string depending on
    the model. Normalize to dict for downstream consumers (audit, UI)."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_raw": raw}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {}


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
        loop = asyncio.get_event_loop()
        start = loop.time()
        tool_count = 0
        wall_clock_cap = _max_wall_clock_s()
        tool_call_cap = _max_tool_calls()
        # tool_call_id -> {tool_name, args, started_at, started_perf}
        pending: dict[str, dict[str, Any]] = {}

        async with self._agent.iter(question, deps=deps) as run:
            async for node in run:
                if loop.time() - start > wall_clock_cap:
                    yield CapReachedEvent(reason="wall_clock")
                    return

                if Agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as handle_stream:
                        async for event in handle_stream:
                            if isinstance(event, FunctionToolCallEvent):
                                tool_name = event.part.tool_name
                                if tool_name in _OUTPUT_TOOL_NAMES:
                                    continue
                                tool_count += 1
                                if tool_count > tool_call_cap:
                                    yield CapReachedEvent(reason="tool_count")
                                    return
                                args = _normalize_args(event.part.args)
                                pending[event.part.tool_call_id] = {
                                    "tool_name": tool_name,
                                    "args": args,
                                    "started_perf": loop.time(),
                                }
                                yield ToolCallStarted(
                                    tool_name=tool_name,
                                    arguments=args,
                                )
                            elif isinstance(event, FunctionToolResultEvent):
                                info = pending.pop(event.tool_call_id, None)
                                if info is None:
                                    continue
                                elapsed_ms = max(0, int((loop.time() - info["started_perf"]) * 1000))
                                result_content = getattr(event.result, "content", None)
                                # Best-effort: scrub via audit summarizer.
                                # If summarize_result rejects (e.g., raw DataFrame),
                                # fall back to a minimal type tag.
                                try:
                                    summary = summarize_result(info["tool_name"], result_content)
                                except Exception:
                                    summary = f"result_type={type(result_content).__name__}"

                                # Persist to audit (per spec §8.6).
                                audit = getattr(deps, "audit_logger", None)
                                if audit is not None:
                                    selected = getattr(deps, "selected_patient", None)
                                    sel_src = getattr(selected, "source_id", None) if selected is not None else None
                                    try:
                                        audit.log(
                                            tool_name=info["tool_name"],
                                            selected_patient_source_id=sel_src,
                                            arguments=info["args"],
                                            result_obj=result_content if isinstance(result_content, dict) else {},
                                            elapsed_ms=elapsed_ms,
                                            success=True,
                                            error=None,
                                        )
                                    except Exception:
                                        # Audit failure must never break the run.
                                        pass

                                yield ToolCallCompleted(
                                    tool_name=info["tool_name"],
                                    result_summary=summary,
                                    success=True,
                                    elapsed_ms=elapsed_ms,
                                    error=None,
                                )

                elif Agent.is_end_node(node):
                    output = getattr(node.data, "output", None)
                    if isinstance(output, AgentResponse):
                        yield FinalResponseEvent(response=output)
                        return

        # Fallback: if iteration exited without an End node (shouldn't happen
        # in practice), surface whatever the run produced.
        if hasattr(run, "result") and run.result is not None:
            output = run.result.output
            if isinstance(output, AgentResponse):
                yield FinalResponseEvent(response=output)
