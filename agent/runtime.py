"""Streamlit-side runtime for the agentic message flow.

Replaces the Vanna call chain inside chat_bot_helper.normal_message_flow
when the user has agentic_mode=True.
"""

from __future__ import annotations
import asyncio
import json
import queue
import threading
from typing import Any, Coroutine

import streamlit as st

from agent.deps_builder import build_agent_deps
from agent.fallback import should_fallback
from agent.observability import configure_observability
from agent.runner import AgenticRunner
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
from utils.enums import MessageType, RoleType
from utils.quick_logger import get_logger
from orm.models import Message, SessionLocal


logger = get_logger(__name__)


_FALLBACK_BANNER_TEXT = "_💡 The agent didn't query data — falling back to direct SQL._"


def _is_fallback_feature_enabled() -> bool:
    """True only when the per-deploy secrets flag AND the per-user opt-in
    are both on. Either flag missing or off → False. Until Feature #232
    lands the `vanna_fallback_enabled` session-state key stays unset, so
    this returns False by default and the fallback path is dormant.
    """
    try:
        secrets_enabled = bool(st.secrets.get("agent", {}).get("fallback", {}).get("enabled", False))
    except Exception:
        secrets_enabled = False
    user_opt_in = bool(st.session_state.get("vanna_fallback_enabled", False))
    return secrets_enabled and user_opt_in


def _is_scrubbed_logging() -> bool:
    """Read `[agent_logging].mode` from secrets. Returns True only when
    explicitly set to "scrubbed"; any other value (including missing
    section) returns False.
    """
    try:
        return st.secrets.get("agent_logging", {}).get("mode") == "scrubbed"
    except Exception:
        return False


def _maybe_invoke_vanna_fallback(
    question: str,
    final_text: str,
    tool_events: list[ToolCallStarted | ToolCallCompleted],
) -> None:
    """Post-stream fallback hook. Decides whether the agent's turn should
    be retried via the legacy Vanna pipeline, and if so renders an
    append-style banner and invokes _run_vanna_flow.

    Runs on the Streamlit script thread. The classifier LLM call is
    bridged to the persistent asyncio loop via _run_async; the banner
    persistence and the Vanna invocation both need ScriptRunContext so
    they stay on this thread.

    Failure-tolerant by design: any exception during the decision or
    the Vanna invocation is logged and swallowed. The original agent
    reply remains visible — the Epic's Acceptance Criteria require
    "classifier failure does not block the user."
    """
    if not _is_fallback_feature_enabled():
        return

    try:
        with st.spinner("Checking whether to retry with direct SQL..."):
            decision = _run_async(
                should_fallback(
                    question=question,
                    final_text=final_text,
                    tool_events=tool_events,
                    feature_enabled=True,
                    scrubbed=_is_scrubbed_logging(),
                )
            )
    except Exception:
        logger.exception("Fallback decision failed; skipping fallback")
        return

    if not decision.invoke:
        return

    # Lazy import to avoid a circular dependency: utils.chat_bot_helper
    # imports agent.runtime via the dispatcher at module top, so a
    # top-level `from utils.chat_bot_helper import ...` here would
    # close the cycle.
    from utils.chat_bot_helper import _run_vanna_flow, add_message

    try:
        add_message(
            Message(
                RoleType.ASSISTANT,
                _FALLBACK_BANNER_TEXT,
                MessageType.TEXT,
            )
        )
        _run_vanna_flow(question)
    except BaseException as exc:
        # Streamlit's RerunException / StopException are control-flow
        # signals raised by _run_vanna_flow's st.rerun() — let them
        # propagate so the script runner can act on them. Mirrors the
        # safety net in chat_bot_helper.normal_message_flow:1342.
        if type(exc).__name__ in ("RerunException", "StopException"):
            raise
        if not isinstance(exc, Exception):
            raise
        logger.exception("Vanna fallback invocation failed")


@st.cache_resource
def _runner() -> AgenticRunner:
    configure_observability()
    return AgenticRunner()


_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_THREAD: threading.Thread | None = None
_LOOP_LOCK = threading.Lock()


def _persistent_loop() -> asyncio.AbstractEventLoop:
    """Process-wide event loop, running forever on its own daemon thread.

    Pydantic AI's OpenAI / Anthropic providers hold long-lived httpx
    connection pools whose TCP transports are bound to the loop on
    which they were first awaited. asyncio.run() creates and closes a
    fresh loop per call, leaving the cached client with dead transports
    on the second request — visible as
    "RuntimeError: unable to perform operation on <TCPTransport closed>".
    Pinning a single loop keeps those transports alive.

    The loop runs via run_forever() on a dedicated thread rather than
    being driven with run_until_complete from each caller. Callers
    submit coroutines with run_coroutine_threadsafe (see _run_async).
    This is what makes concurrent Streamlit script threads safe: two
    sessions (or a rerun that fires while a prior run is still
    unwinding) would otherwise both call run_until_complete on this one
    loop, and the second raises "this event loop is already running".
    A single run_forever loop multiplexes any number of submissions.
    """
    global _LOOP, _LOOP_THREAD
    with _LOOP_LOCK:
        if _LOOP is None or _LOOP.is_closed():
            _LOOP = asyncio.new_event_loop()
            _LOOP_THREAD = threading.Thread(
                target=_LOOP.run_forever,
                name="agent-asyncio-loop",
                daemon=True,
            )
            _LOOP_THREAD.start()
        return _LOOP


def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine to completion on the persistent loop, from any thread.

    Submits to the dedicated loop thread and blocks the caller on the
    result. Works whether or not the calling thread has its own running
    loop (e.g. pytest-asyncio), and is safe under concurrency: the loop
    multiplexes overlapping submissions instead of colliding the way
    run_until_complete would.

    Note the coroutine BODY executes on the loop thread, which has no
    Streamlit ScriptRunContext. Code that must touch st.session_state or
    render widgets has to run on the calling script thread instead — see
    run_agentic_message_flow, which keeps streaming on the loop thread
    but renders on the caller.
    """
    return asyncio.run_coroutine_threadsafe(coro, _persistent_loop()).result()


def run_agentic_message_flow(my_question: str) -> None:
    """Drop-in replacement for the Vanna branch in normal_message_flow.

    Runs the agent with streaming and persists each tool call and the
    final response as orm.Message rows. The USER row for `my_question`
    is already in session_state.messages — set_question() at
    chat_bot_helper.py:535 wrote it before the rerun landed us here.

    Multi-turn continuity: prior `agent_message_history` (list of
    pydantic-ai ModelMessage) is threaded into the run; the post-run
    transcript is written back. The original question is also stashed
    as `pending_user_question` so the patient-chooser click handler
    can re-trigger this flow with the same prompt.
    """
    # Stash the question for chooser-click resume before any errors can
    # short-circuit us out.
    st.session_state["pending_user_question"] = my_question

    # Outer try ensures my_question is always cleared even if the agent
    # call raises — otherwise the chat_bot.py loop would re-fire the
    # same question on the next rerun and the user would see no response.
    try:
        # The session is created here but used ONLY on the loop thread
        # (audit writes, commit, close all live in produce()). SQLite's
        # default check_same_thread forbids using one connection from two
        # threads, and SQLAlchemy sessions aren't thread-safe — so we must
        # not touch it from this (calling) thread once produce() owns it.
        sqlite_session = SessionLocal()
        deps = build_agent_deps(sqlite_session)
        runner = _runner()
        prior_history = st.session_state.get("agent_message_history") or None

        # Hand events from the loop thread back to this script thread. We
        # render here, not inside the coroutine: _render_event calls
        # st.* / add_message, which need this thread's ScriptRunContext.
        # The loop thread has none.
        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        async def produce() -> None:
            # Runs on the dedicated loop thread. Owns the session lifecycle
            # so all of its DB I/O stays on one thread (see note above).
            try:
                async for event in runner.stream(my_question, deps=deps, message_history=prior_history):
                    events.put(("event", event))
                sqlite_session.commit()
            finally:
                sqlite_session.close()
                events.put(("done", None))

        future = asyncio.run_coroutine_threadsafe(produce(), _persistent_loop())

        # Per-flow state for live placeholders. Two buckets keyed by
        # turn_index so thinking and plain assistant text get distinct
        # chat_message containers (different headers, different lifecycles).
        # The finally block drains anything still open if the stream dies
        # mid-flight — otherwise a partial "🤔 Thinking..." panel would
        # stay frozen on screen.
        renderer_state: dict[str, Any] = {"thinking": {}, "text": {}}
        # Collected for the post-stream Vanna-fallback decision (Epic #228 / #231).
        # We accumulate tool-call events and the final assistant text on the
        # script thread as they're rendered, then hand them to the decision
        # engine after future.result() so we don't double-render anything.
        tool_events: list[ToolCallStarted | ToolCallCompleted] = []
        final_response_text: str = ""
        try:
            while True:
                try:
                    kind, event = events.get(timeout=1.0)
                except queue.Empty:
                    # No event this second. produce() always emits its "done"
                    # sentinel from a finally, so the only way the queue goes
                    # quiet for good is a dead loop thread (interpreter
                    # shutdown, or the loop closed mid-run). Bail in that case
                    # so the script thread can't wedge forever. A slow-but-
                    # alive producer keeps the future pending, so this never
                    # fires during a long LLM turn between events.
                    if future.done():
                        break
                    continue
                if kind == "done":
                    break
                _render_event(event, renderer_state)
                if isinstance(event, (ToolCallStarted, ToolCallCompleted)):
                    tool_events.append(event)
                elif isinstance(event, FinalResponseEvent):
                    final_response_text = event.response.text
            # Surface any exception raised inside produce() (and thus the
            # agent run) to the caller, matching the old _run_async behavior.
            future.result()
            # runner.stream's own df-sync ran on the loop thread with no
            # ScriptRunContext, so it was a no-op there (it's wrapped in a
            # bare except). Redo it here on the script thread so magic
            # functions and the Vanna flow see the final DataFrame.
            _sync_last_dataframe_to_session_state(deps)
            # Post-stream Vanna fallback hook (Epic #228 / #231). No-op
            # unless the per-deploy secrets flag AND the per-user opt-in
            # are both on. Failures are logged and swallowed so the
            # agent's reply still stands.
            _maybe_invoke_vanna_fallback(my_question, final_response_text, tool_events)
        finally:
            _drain_placeholders(renderer_state)
    finally:
        # Parity with the Vanna flow at chat_bot_helper.py:1417.
        st.session_state["my_question"] = None


def _sync_last_dataframe_to_session_state(deps: Any) -> None:
    """Mirror deps.last_dataframe into st.session_state['df'] on the script
    thread. runner.stream attempts this itself, but under the loop-thread
    design that attempt has no ScriptRunContext and silently no-ops, so we
    repeat it here where session_state is real. Tolerant of deps shapes
    without last_dataframe (e.g. the exception-path unit test)."""
    df = getattr(deps, "last_dataframe", None)
    if df is not None:
        st.session_state["df"] = df


def _drain_placeholders(state: dict[str, Any]) -> None:
    """Clear any open transient placeholders. Called from the finally block of
    the consumer loop in run_agentic_message_flow (and from FinalResponseEvent /
    CapReachedEvent handling) so a mid-stream death (exception, network drop,
    cap reached) doesn't leave a partial "🤔 Thinking..." panel frozen on
    screen. Idempotent — safe to call multiple times."""
    for bucket in ("thinking", "text"):
        slots = state.get(bucket) or {}
        for slot in slots.values():
            try:
                slot["outer"].empty()
            except Exception:
                pass
        slots.clear()


def _open_slot(state: dict[str, Any], bucket: str, turn_index: int) -> dict[str, Any]:
    """Return the existing live-update slot for (bucket, turn_index), or
    create one anchored to a fresh outer st.empty().

    The chat_message bubble is rendered INSIDE the outer placeholder on
    each delta via `outer.container()`, so calling `outer.empty()` later
    removes both the bubble's avatar and its content. The prior pattern
    of placing st.empty() inside a chat_message left orphaned empty
    avatar bubbles after the placeholder was cleared.
    """
    slot = state[bucket].get(turn_index)
    if slot is None:
        outer = st.empty()
        slot = {"outer": outer, "buf": ""}
        state[bucket][turn_index] = slot
    return slot


def _write_slot(slot: dict[str, Any], header: str | None = None) -> None:
    """Re-render the slot's buffer into the outer placeholder as a fresh
    chat_message + markdown. Called on every delta. `header` (e.g.
    "🤔 **Thinking...**") is prepended when present."""
    body = (f"{header}\n\n{slot['buf']}") if header else slot["buf"]
    with slot["outer"].container():
        with st.chat_message(RoleType.ASSISTANT.value):
            st.markdown(body)


def _render_event(event: StreamEvent, state: dict[str, Any] | None = None) -> None:
    from utils.chat_bot_helper import add_message

    state = state if state is not None else {"thinking": {}, "text": {}}
    state.setdefault("thinking", {})
    state.setdefault("text", {})

    if isinstance(event, ThinkingDeltaEvent):
        slot = _open_slot(state, "thinking", event.turn_index)
        slot["buf"] += event.delta
        # Epic #222 / Feature #226: when the user has the toggle off, render
        # a single static placeholder for the turn instead of the live
        # chunk-by-chunk stream. The slot is still opened so the matching
        # ThinkingCompletedEvent can outer.empty() the bubble, and the
        # buffer still accumulates so any downstream consumer sees parity
        # with the on-path. ``static_rendered`` makes the off-path
        # idempotent across repeated deltas in the same turn.
        if not st.session_state.get("show_thinking_process", False):
            if not slot.get("static_rendered"):
                with slot["outer"].container():
                    with st.chat_message(RoleType.ASSISTANT.value):
                        st.caption("🤔 Thinking…")
                slot["static_rendered"] = True
            return
        _write_slot(slot, header="🤔 **Thinking...**")
        return

    if isinstance(event, AssistantTextDeltaEvent):
        # Plain assistant narration between tool calls. Lives in its own
        # bucket with no "Thinking..." header so the user can tell the two
        # apart at a glance. Persisted on the matching AssistantTextCompletedEvent.
        slot = _open_slot(state, "text", event.turn_index)
        slot["buf"] += event.delta
        _write_slot(slot)
        return

    if isinstance(event, AssistantTextCompletedEvent):
        slot = state["text"].pop(event.turn_index, None)
        if slot is not None:
            slot["outer"].empty()
        if event.text.strip():
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    event.text,
                    MessageType.TEXT,
                )
            )
            # Remember the last persisted narration so FinalResponseEvent
            # can skip its own response.text row if the model echoed the
            # same content via TextPart on the final turn.
            state["last_persisted_text"] = event.text
        return

    if isinstance(event, ThinkingCompletedEvent):
        slot = state["thinking"].pop(event.turn_index, None)
        if slot is not None:
            # Drop the transient bubble; the persisted MessageType.THINKING
            # row that follows replaces it visually.
            slot["outer"].empty()
        if event.text.strip():
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    event.text,
                    MessageType.THINKING,
                    elapsed_time=event.elapsed_ms / 1000.0,
                )
            )
        return

    if isinstance(event, ToolCallStarted):
        # In-flight tool call card.
        payload = {
            "tool_name": event.tool_name,
            "arguments": event.arguments,
            "result_summary": None,
            "success": None,
            "elapsed_ms": None,
            "error": None,
        }
        add_message(
            Message(
                RoleType.ASSISTANT,
                json.dumps(payload),
                MessageType.TOOL_CALL,
            )
        )
        return

    if isinstance(event, ToolCallCompleted):
        # Completed tool call card.
        payload = {
            "tool_name": event.tool_name,
            "arguments": {},
            "result_summary": event.result_summary,
            "success": event.success,
            "elapsed_ms": event.elapsed_ms,
            "error": event.error,
            "reliability_note": event.reliability_note,
            "sql_executed": event.sql_executed,
            "result_payload": event.result_payload,
        }
        add_message(
            Message(
                RoleType.ASSISTANT,
                json.dumps(payload),
                MessageType.TOOL_CALL,
            )
        )
        return

    if isinstance(event, PatientChooserEvent):
        # Auto-surfaced by runner.stream() right after find_patient succeeds,
        # independent of any artifact the model may attach to final_result.
        add_message(
            Message(
                RoleType.ASSISTANT,
                json.dumps(event.payload),
                MessageType.PATIENT_CHOOSER,
            )
        )
        return

    if isinstance(event, CohortSampleEvent):
        # Auto-surfaced by runner.stream() right after
        # search_patients_by_criteria succeeds with a non-empty sample OR
        # breakdown buckets. Renders as a DataFrame message regardless of
        # whether the LLM attaches a DataFrameArtifact to the final response.
        import pandas as pd

        buckets = event.payload.get("buckets", [])
        sample = event.payload.get("sample", [])
        rows = buckets or sample
        if rows:
            df = pd.DataFrame(rows)
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    df.to_json(orient="records", date_format="iso"),
                    MessageType.DATAFRAME,
                )
            )
        return

    if isinstance(event, FinalResponseEvent):
        # Drain any open transient placeholders before the final TEXT row
        # lands, otherwise a plain-text live preview would briefly double up
        # with the persisted answer.
        _drain_placeholders(state)
        response = event.response
        # Dedupe: if the model already streamed the same final text as a
        # TextPart on the last turn, AssistantTextCompletedEvent already
        # persisted it. Skip the redundant row so the user doesn't see the
        # answer twice. Strip-equality is enough — both come from the same
        # model output and won't drift in whitespace meaningfully.
        last_streamed = state.get("last_persisted_text", "")
        if response.text.strip() and response.text.strip() != last_streamed.strip():
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    response.text,
                    MessageType.TEXT,
                )
            )
        # Honor clear_selection: drop both the slot AND the conversation
        # history, since "start fresh" means no carry-over context.
        if response.clear_selection:
            for k in (
                "selected_patient_source_id",
                "selected_patient_display_name",
                "selected_patient_dob",
                "selection_origin",
                "selected_at",
                "agent_message_history",
                "pending_user_question",
            ):
                st.session_state.pop(k, None)
        else:
            # Persist the post-run transcript so the next turn (whether
            # a follow-up or a chooser-click resume) continues the same
            # conversation rather than starting over.
            if event.all_messages:
                st.session_state["agent_message_history"] = list(event.all_messages)
        return

    if isinstance(event, CapReachedEvent):
        # Cap fires mid-stream; clear any frozen "Thinking..." panels first.
        _drain_placeholders(state)
        add_message(
            Message(
                RoleType.ASSISTANT,
                f"Stopped: cap reached ({event.reason}).",
                MessageType.ERROR,
            )
        )
