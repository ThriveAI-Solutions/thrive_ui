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
import hashlib
import json
from typing import Any, AsyncIterator, Optional

from pydantic_ai import (
    Agent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
)
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.models import Model

from agent.audit import summarize_result
from agent.deps import AgentDeps
from agent.instructions import selection_instructions
from agent.models import build_model
from agent.state import (
    AgentResponse,
    AssistantTextCompletedEvent,
    AssistantTextDeltaEvent,
    CapReachedEvent,
    CohortSampleEvent,
    FinalResponseEvent,
    PatientChooserEvent,
    StreamEvent,
    ThinkingCompletedEvent,
    ThinkingDeltaEvent,
    ToolCallCompleted,
    ToolCallStarted,
)
from agent.system_prompt import SYSTEM_PROMPT
from agent.tools.find_patient import find_patient
from agent.tools.get_patient_clinical_data import get_patient_clinical_data
from agent.tools.list_patient_documents import list_patient_documents
from agent.tools.make_chart import make_chart
from agent.tools.run_sql import run_sql, _augment_run_sql_description
from agent.tools.search_codes import search_codes
from agent.tools.search_knowledge_base import search_knowledge_base
from agent.tools.summarize_results import summarize_results
from agent.tools.search_patients_by_criteria import search_patients_by_criteria


_DEFAULT_MAX_TOOL_CALLS = 7
_DEFAULT_MAX_WALL_CLOCK_S = 120.0


def _logger_run_id(logger: Any) -> Optional[str]:
    """Extract ``logger.run_id`` only if it's a real string. Production
    AgentRunLogger always provides a string; defensive isinstance guard
    keeps existing tests with MagicMock loggers working (the mocked
    ``logger.run_id`` is a MagicMock, which pydantic rejects)."""
    if logger is None:
        return None
    rid = getattr(logger, "run_id", None)
    return rid if isinstance(rid, str) else None


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


def _usage_dict(run: Any) -> Optional[dict]:
    """Best-effort token usage from a pydantic-ai run. Returns None if absent."""
    try:
        usage = run.usage()
    except Exception:
        return None
    if usage is None:
        return None
    inp = getattr(usage, "input_tokens", None)
    if inp is None:
        inp = getattr(usage, "request_tokens", None)
    out = getattr(usage, "output_tokens", None)
    if out is None:
        out = getattr(usage, "response_tokens", None)
    tot = getattr(usage, "total_tokens", None)
    if tot is None and (inp is not None or out is not None):
        tot = (inp or 0) + (out or 0)
    if inp is None and out is None and tot is None:
        return None
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": tot}


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
        model_obj = model or build_model()
        self._agent: Agent[AgentDeps, AgentResponse] = Agent(
            model=model_obj,
            deps_type=AgentDeps,
            output_type=AgentResponse,
            system_prompt=SYSTEM_PROMPT,
            retries=5,
        )
        self._agent.instructions(selection_instructions)
        self._register_tools()
        self._system_prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        tool_names = ",".join(
            sorted(
                (
                    "find_patient",
                    "get_patient_clinical_data",
                    "search_knowledge_base",
                    "list_patient_documents",
                    "search_codes",
                    "run_sql",
                    "make_chart",
                    "summarize_results",
                    "search_patients_by_criteria",
                )
            )
        )
        self._tool_schema_hash = hashlib.sha256(tool_names.encode("utf-8")).hexdigest()
        self._provider_name = getattr(model_obj, "system", None) or getattr(model_obj, "_system", None)
        self._model_name = getattr(model_obj, "model_name", None) or str(type(model_obj).__name__)

    def _register_tools(self) -> None:
        self._agent.tool(find_patient)
        self._agent.tool(get_patient_clinical_data)
        self._agent.tool(search_knowledge_base)
        self._agent.tool(list_patient_documents)
        self._agent.tool(search_codes)
        self._agent.tool(run_sql, prepare=_augment_run_sql_description)
        self._agent.tool(make_chart)
        self._agent.tool(summarize_results)
        self._agent.tool(search_patients_by_criteria)

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
        # Parity with stream(): mirror deps.last_dataframe into session_state
        # so a subsequent turn (or a magic-function slash command) can pick it
        # up. Wrapped in try/except for non-Streamlit callers (scripts/tests).
        try:
            import streamlit as st

            _sync_last_dataframe_to_session_state(deps.last_dataframe, st.session_state)
        except Exception:
            pass
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
        turn_index = 0  # increments per model_request_node so the renderer
        # can scope a transient placeholder to one turn's reasoning
        wall_clock_cap = _max_wall_clock_s()
        tool_call_cap = _max_tool_calls()
        # tool_call_id -> {tool_name, args, started_at, started_perf}
        pending: dict[str, dict[str, Any]] = {}
        logger = getattr(deps, "run_logger", None)

        if logger is not None:
            selected_patient = None
            try:
                sel = getattr(deps, "selected_patient", None)
                if sel is not None:
                    selected_patient = {
                        "source_id": getattr(sel, "source_id", None),
                        "display_name": getattr(sel, "display_name", None),
                        "dob": sel.dob.isoformat() if getattr(sel, "dob", None) else None,
                        "selection_origin": getattr(sel, "selection_origin", None),
                    }
            except Exception:
                selected_patient = None
            logger.start_run(
                question=question,
                llm_provider=self._provider_name,
                llm_model=self._model_name,
                selected_patient=selected_patient,
                group_id=getattr(deps, "group_id", None),
                parent_run_id=getattr(deps, "parent_run_id", None),
                resume_reason=getattr(deps, "resume_reason", None),
                user_message_id=getattr(deps, "user_message_id", None),
                system_prompt_hash=self._system_prompt_hash,
                tool_schema_hash=self._tool_schema_hash,
                message_history=None if not message_history else f"{len(message_history)} prior messages",
            )

        try:
            async with self._agent.iter(
                question,
                deps=deps,
                message_history=message_history or None,
            ) as run:
                async for node in run:
                    if loop.time() - start > wall_clock_cap:
                        if logger is not None:
                            logger.finalize_run(
                                status="cap_reached",
                                final_answer_text=None,
                                usage=None,
                                total_elapsed_ms=int((loop.time() - start) * 1000),
                                cap_reached="wall_clock",
                            )
                        yield CapReachedEvent(reason="wall_clock")
                        return

                    if Agent.is_model_request_node(node):
                        # Stream reasoning + plain text deltas from the model so
                        # the UI shows signs of life instead of dead air. Tool
                        # call parts are NOT consumed here — they're handled by
                        # the call_tools_node branch below via the normal
                        # FunctionToolCallEvent flow.
                        turn_index += 1
                        thinking_buf = ""
                        thinking_started_perf: float | None = None
                        text_buf = ""
                        try:
                            async with node.stream(run.ctx) as req_stream:
                                async for ev in req_stream:
                                    if isinstance(ev, PartStartEvent):
                                        if isinstance(ev.part, ThinkingPart):
                                            if thinking_started_perf is None:
                                                thinking_started_perf = loop.time()
                                            initial = ev.part.content or ""
                                            if initial:
                                                thinking_buf += initial
                                                yield ThinkingDeltaEvent(
                                                    delta=initial,
                                                    turn_index=turn_index,
                                                )
                                        elif isinstance(ev.part, TextPart):
                                            initial = ev.part.content or ""
                                            if initial:
                                                text_buf += initial
                                                yield AssistantTextDeltaEvent(
                                                    delta=initial,
                                                    turn_index=turn_index,
                                                )
                                    elif isinstance(ev, PartDeltaEvent):
                                        if isinstance(ev.delta, ThinkingPartDelta):
                                            chunk = ev.delta.content_delta or ""
                                            if chunk:
                                                if thinking_started_perf is None:
                                                    thinking_started_perf = loop.time()
                                                thinking_buf += chunk
                                                yield ThinkingDeltaEvent(
                                                    delta=chunk,
                                                    turn_index=turn_index,
                                                )
                                        elif isinstance(ev.delta, TextPartDelta):
                                            chunk = ev.delta.content_delta or ""
                                            if chunk:
                                                text_buf += chunk
                                                yield AssistantTextDeltaEvent(
                                                    delta=chunk,
                                                    turn_index=turn_index,
                                                )
                                    # PartEndEvent and other events are ignored:
                                    # we close the buffers after the stream exits.
                        except AssertionError as e:
                            # FunctionModel test stubs without stream_function raise
                            # AssertionError mentioning `stream_function` on entry.
                            # Catch only that exact case so real invariant failures
                            # in pydantic-ai (malformed deltas, broken adapters)
                            # still propagate instead of being silently swallowed.
                            # Iteration advances internally even when we skip the
                            # streaming surface, so the run completes normally
                            # without any thinking events for this turn.
                            if "stream_function" not in str(e):
                                raise
                        if thinking_buf:
                            elapsed_ms = max(
                                0,
                                int((loop.time() - (thinking_started_perf or loop.time())) * 1000),
                            )
                            yield ThinkingCompletedEvent(
                                text=thinking_buf,
                                elapsed_ms=elapsed_ms,
                                turn_index=turn_index,
                            )
                            if logger is not None:
                                logger.log_event(
                                    "thinking_completed",
                                    payload={"text": thinking_buf},
                                    turn_index=turn_index,
                                    elapsed_ms=elapsed_ms,
                                )
                        if text_buf:
                            # End-of-turn flush so the streamed narration survives
                            # rerun as a persisted MessageType.TEXT row. Renderer
                            # dedupes against FinalResponseEvent.response.text to
                            # avoid double-rendering when the model echoes the
                            # final answer here as well as via final_result.
                            yield AssistantTextCompletedEvent(
                                text=text_buf,
                                turn_index=turn_index,
                            )
                            if logger is not None:
                                logger.log_event(
                                    "assistant_text_completed",
                                    payload={"text": text_buf},
                                    turn_index=turn_index,
                                )
                        continue

                    if Agent.is_call_tools_node(node):
                        async with node.stream(run.ctx) as handle_stream:
                            async for event in handle_stream:
                                if isinstance(event, FunctionToolCallEvent):
                                    tool_name = event.part.tool_name
                                    if tool_name in _OUTPUT_TOOL_NAMES:
                                        continue
                                    tool_count += 1
                                    if tool_count > tool_call_cap:
                                        if logger is not None:
                                            logger.finalize_run(
                                                status="cap_reached",
                                                final_answer_text=None,
                                                usage=None,
                                                total_elapsed_ms=int((loop.time() - start) * 1000),
                                                cap_reached="tool_count",
                                            )
                                        yield CapReachedEvent(reason="tool_count")
                                        return
                                    args = _normalize_args(event.part.args)
                                    started_event_seq = None
                                    if logger is not None:
                                        started_event_seq = logger.log_tool_started(
                                            tool_name=tool_name,
                                            tool_call_id=event.part.tool_call_id,
                                            turn_index=turn_index,
                                            arguments=args,
                                        )
                                    pending[event.part.tool_call_id] = {
                                        "tool_name": tool_name,
                                        "args": args,
                                        "started_perf": loop.time(),
                                        "turn_index": turn_index,
                                        "started_event_seq": started_event_seq,
                                    }
                                    yield ToolCallStarted(
                                        tool_name=tool_name,
                                        arguments=args,
                                    )
                                elif isinstance(event, FunctionToolResultEvent):
                                    info = pending.pop(event.part.tool_call_id, None)
                                    if info is None:
                                        continue
                                    elapsed_ms = max(0, int((loop.time() - info["started_perf"]) * 1000))
                                    # pydantic-ai 2.0: FunctionToolResultEvent.result was renamed to
                                    # .part (the ToolReturnPart|RetryPromptPart). .part.content holds
                                    # the tool's return object — keep using the part (not the event's
                                    # normalized .content) so model_dump()/reliability_note still work.
                                    result_content = getattr(event.part, "content", None)
                                    # summarize_result wants a dict. Pydantic results
                                    # (ClinicalResult, DocumentIndexResult, …) need to
                                    # be dumped first, otherwise the summarizer falls
                                    # through to the type-tag branch and the streamed
                                    # event lose data_availability and the
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

                                    # Pop SQL once, before logging, so it rides along
                                    # on the tool row + the card. Tools run
                                    # sequentially; whatever ran since the last pop is
                                    # attributable to this tool.
                                    sql_executed: list[dict] = []
                                    adapter = getattr(deps, "analytics_db", None)
                                    if adapter is not None and hasattr(adapter, "pop_sql_log"):
                                        try:
                                            sql_executed = adapter.pop_sql_log()
                                        except Exception:
                                            sql_executed = []

                                    # Persist the enriched tool-call row (per spec §8.6).
                                    if logger is not None:
                                        selected = getattr(deps, "selected_patient", None)
                                        sel_src = getattr(selected, "source_id", None) if selected is not None else None
                                        result_for_log = (
                                            result_content
                                            if isinstance(result_content, dict)
                                            else (
                                                result_content.model_dump(mode="json")
                                                if hasattr(result_content, "model_dump")
                                                else {}
                                            )
                                        )
                                        logger.log_tool_completed(
                                            tool_name=info["tool_name"],
                                            tool_call_id=event.part.tool_call_id,
                                            turn_index=info.get("turn_index"),
                                            arguments=info["args"],
                                            result_obj=result_for_log,
                                            sql_executed=sql_executed,
                                            elapsed_ms=elapsed_ms,
                                            success=True,
                                            error=None,
                                            selected_patient_source_id=sel_src,
                                            started_event_seq=info.get("started_event_seq"),
                                        )

                                    # Reuse the same dict we built for summarize_result
                                    # so we don't dump the model twice.
                                    result_payload = summary_input if isinstance(summary_input, dict) else None

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
                                            if logger is not None:
                                                logger.log_chooser_candidates(payload)

                                    # Auto-surface the cohort sample as a DataFrame
                                    # after search_patients_by_criteria succeeds, for
                                    # the same reason as the find_patient chooser:
                                    # don't depend on the LLM attaching an artifact.
                                    if info["tool_name"] == "search_patients_by_criteria":
                                        payload = _to_jsonable(result_content)
                                        if isinstance(payload, dict) and (
                                            (isinstance(payload.get("sample"), list) and payload["sample"])
                                            or (isinstance(payload.get("buckets"), list) and payload["buckets"])
                                        ):
                                            yield CohortSampleEvent(payload=payload)

                    elif Agent.is_end_node(node):
                        output = getattr(node.data, "output", None)
                        if isinstance(output, AgentResponse):
                            all_msgs = (
                                list(run.result.all_messages()) if getattr(run, "result", None) is not None else []
                            )
                            try:
                                import streamlit as st

                                _sync_last_dataframe_to_session_state(deps.last_dataframe, st.session_state)
                            except Exception:
                                pass
                            usage = _usage_dict(run)
                            if logger is not None:
                                logger.finalize_run(
                                    status="success",
                                    final_answer_text=output.text,
                                    usage=usage,
                                    total_elapsed_ms=int((loop.time() - start) * 1000),
                                    cap_reached=None,
                                )
                            yield FinalResponseEvent(
                                response=output,
                                all_messages=all_msgs,
                                usage=usage,
                                run_id=_logger_run_id(logger),
                            )
                            return
        except Exception as exc:
            if logger is not None:
                import traceback

                logger.finalize_run(
                    status="failed",
                    final_answer_text=None,
                    usage=None,
                    total_elapsed_ms=int((loop.time() - start) * 1000),
                    cap_reached=None,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    stack_trace=traceback.format_exc(),
                )
            raise

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
                usage = _usage_dict(run)
                if logger is not None:
                    logger.finalize_run(
                        status="success",
                        final_answer_text=output.text,
                        usage=usage,
                        total_elapsed_ms=int((loop.time() - start) * 1000),
                        cap_reached=None,
                    )
                yield FinalResponseEvent(
                    response=output,
                    all_messages=list(run.result.all_messages()),
                    usage=usage,
                    run_id=logger.run_id if logger is not None else None,
                )
