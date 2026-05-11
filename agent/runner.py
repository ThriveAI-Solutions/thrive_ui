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
from agent.instructions import selection_instructions
from agent.models import build_model
from agent.state import (
    AgentResponse,
    CapReachedEvent,
    FinalResponseEvent,
    PatientChooserEvent,
    StreamEvent,
    ToolCallCompleted,
    ToolCallStarted,
)
from agent.system_prompt import SYSTEM_PROMPT
from agent.tools.find_patient import find_patient
from agent.tools.get_patient_clinical_data import get_patient_clinical_data
from agent.tools.list_patient_documents import list_patient_documents
from agent.tools.make_chart import make_chart
from agent.tools.run_sql import run_sql
from agent.tools.search_codes import search_codes
from agent.tools.search_knowledge_base import search_knowledge_base
from agent.tools.summarize_results import summarize_results


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


def _to_jsonable(obj: Any) -> Any:
    """Coerce a Pydantic model to a plain dict so the UI / audit can read it."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


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


def _sync_last_dataframe_to_session_state(
    last_dataframe: Optional[Any],
    session_state: dict,
) -> None:
    """Mirror deps.last_dataframe into st.session_state['df'] for
    slash-command compatibility (Vanna and the agent share the same key).

    None means 'this turn produced no dataframe' — leave any prior
    session_state['df'] in place. An empty DataFrame IS a valid result
    (e.g., 'no records found') and overwrites the prior value.
    """
    if last_dataframe is None:
        return
    session_state["df"] = last_dataframe


class AgenticRunner:
    """Owns the Pydantic AI Agent and its tool registrations.

    A single runner is constructed at process start and reused across
    requests via Streamlit's @st.cache_resource. Per-request state lives
    in AgentDeps, not on the runner.
    """

    def __init__(self, model: Optional[Model] = None) -> None:
        # retries=5: small local models (qwen3.6:27b, gemma) often need
        # several attempts to produce a valid date_range / array shape
        # for the discriminated-union clinical query. Two retries was
        # too tight — well-routed runs were aborting on a 3rd malformed
        # call. See the regression run notes in the Phase 2 plan.
        self._agent: Agent[AgentDeps, AgentResponse] = Agent(
            model=model or build_model(),
            deps_type=AgentDeps,
            output_type=AgentResponse,
            system_prompt=SYSTEM_PROMPT,
            retries=5,
        )
        self._agent.instructions(selection_instructions)
        self._register_tools()

    def _register_tools(self) -> None:
        self._agent.tool(find_patient)
        self._agent.tool(get_patient_clinical_data)
        self._agent.tool(search_knowledge_base)
        self._agent.tool(list_patient_documents)
        self._agent.tool(search_codes)
        self._agent.tool(run_sql)
        self._agent.tool(make_chart)
        self._agent.tool(summarize_results)

    async def run(
        self,
        question: str,
        deps: AgentDeps,
        message_history: Optional[list[Any]] = None,
    ) -> AgentResponse:
        """Single-shot run; collects the final AgentResponse without streaming.

        Used by tests and any caller that doesn't need step-by-step streaming.
        Pass `message_history` from a prior run's `FinalResponseEvent.all_messages`
        to continue a multi-turn conversation.
        """
        result = await self._agent.run(
            question,
            deps=deps,
            message_history=message_history or None,
        )
        return result.output

    async def stream(
        self,
        question: str,
        deps: AgentDeps,
        message_history: Optional[list[Any]] = None,
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

        async with self._agent.iter(
            question,
            deps=deps,
            message_history=message_history or None,
        ) as run:
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
                                # summarize_result wants a dict. Pydantic results
                                # (ClinicalResult, DocumentIndexResult, …) need to
                                # be dumped first, otherwise the summarizer falls
                                # through to the type-tag branch and the audit /
                                # streamed event lose data_availability and the
                                # row-count signals downstream graders rely on.
                                try:
                                    summary_input = (
                                        result_content
                                        if isinstance(result_content, dict)
                                        else (
                                            result_content.model_dump(mode="json")
                                            if hasattr(result_content, "model_dump")
                                            else result_content
                                        )
                                    )
                                    summary = summarize_result(info["tool_name"], summary_input)
                                except Exception:
                                    summary = f"result_type={type(result_content).__name__}"

                                reliability_note = None
                                rn_attr = getattr(result_content, "reliability_note", None)
                                if isinstance(rn_attr, str) and rn_attr.strip():
                                    reliability_note = rn_attr

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
                                            result_obj=result_content
                                            if isinstance(result_content, dict)
                                            else (
                                                result_content.model_dump(mode="json")
                                                if hasattr(result_content, "model_dump")
                                                else {}
                                            ),
                                            elapsed_ms=elapsed_ms,
                                            success=True,
                                            error=None,
                                        )
                                    except Exception:
                                        # Audit failure must never break the run.
                                        pass

                                # Pop the analytics adapter's SQL log so
                                # the next tool call starts clean. Tools
                                # run sequentially; whatever ran since the
                                # last pop is attributable to this tool.
                                sql_executed: list[dict] = []
                                adapter = getattr(deps, "analytics_db", None)
                                if adapter is not None and hasattr(adapter, "pop_sql_log"):
                                    try:
                                        sql_executed = adapter.pop_sql_log()
                                    except Exception:
                                        sql_executed = []

                                # Reuse the same dict we built for summarize_result
                                # so we don't dump the model twice.
                                result_payload: dict | None = None
                                try:
                                    if isinstance(summary_input, dict):
                                        result_payload = summary_input
                                except Exception:
                                    result_payload = None

                                yield ToolCallCompleted(
                                    tool_name=info["tool_name"],
                                    result_summary=summary,
                                    success=True,
                                    elapsed_ms=elapsed_ms,
                                    error=None,
                                    reliability_note=reliability_note,
                                    sql_executed=sql_executed,
                                    result_payload=result_payload,
                                )

                                # Auto-surface the disambiguation chooser
                                # right after find_patient succeeds, so the
                                # UI does not depend on the model attaching
                                # an artifact to final_result (smaller models
                                # often skip that step).
                                if info["tool_name"] == "find_patient":
                                    payload = _to_jsonable(result_content)
                                    if isinstance(payload, dict) and payload.get("total_unique", 0) > 0:
                                        yield PatientChooserEvent(payload=payload)

                elif Agent.is_end_node(node):
                    output = getattr(node.data, "output", None)
                    if isinstance(output, AgentResponse):
                        all_msgs = list(run.result.all_messages()) if getattr(run, "result", None) is not None else []
                        try:
                            import streamlit as st

                            _sync_last_dataframe_to_session_state(deps.last_dataframe, st.session_state)
                        except Exception:
                            pass
                        yield FinalResponseEvent(response=output, all_messages=all_msgs)
                        return

        # Fallback: if iteration exited without an End node (shouldn't happen
        # in practice), surface whatever the run produced.
        if hasattr(run, "result") and run.result is not None:
            output = run.result.output
            if isinstance(output, AgentResponse):
                try:
                    import streamlit as st

                    _sync_last_dataframe_to_session_state(deps.last_dataframe, st.session_state)
                except Exception:
                    pass
                yield FinalResponseEvent(
                    response=output,
                    all_messages=list(run.result.all_messages()),
                )
