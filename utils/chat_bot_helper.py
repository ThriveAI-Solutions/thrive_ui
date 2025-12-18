import ast
import hashlib
import json
import logging
import random
import time
import uuid
from io import StringIO

import pandas as pd
import streamlit as st
from ethical_guardrails_lib import get_ethical_guideline

from orm.functions import save_user_settings, set_user_preferences_in_session_state
from orm.models import Message
from utils.communicate import speak
from utils.enums import MessageType, RoleType
from utils.vanna_calls import VannaService, remove_from_file_training, write_to_file_and_training
import re

logger = logging.getLogger(__name__)

# Patterns for non-recoverable SQL errors that won't benefit from retries
NON_RECOVERABLE_ERROR_PATTERNS = [
    r"relation.*does not exist",
    r"table.*does not exist",
    r"column.*does not exist",
    r"permission denied",
    r"access denied",
    r"authentication failed",
    r"database.*does not exist",
    r"schema.*does not exist",
]


def is_non_recoverable_error(error_message: str | None) -> bool:
    """Check if an error is non-recoverable and shouldn't be retried."""
    if not error_message:
        return False
    error_lower = error_message.lower()
    for pattern in NON_RECOVERABLE_ERROR_PATTERNS:
        if re.search(pattern, error_lower):
            return True
    return False

# Expose a module-level symbol `vn` for tests that patch utils.chat_bot_helper.vn
# It lazily resolves the VannaService instance on first use via get_vn()
vn = None


def get_vanna_service():
    """Get VannaService instance, ensuring user preferences are loaded first."""
    # Ensure user preferences are loaded into session state
    set_user_preferences_in_session_state()
    return VannaService.from_streamlit_session()


# Get the VannaService instance when needed, not at module import time
def get_vn():
    """Get the VannaService instance, ensuring it's created with correct user context."""
    # Always load user preferences first
    set_user_preferences_in_session_state()

    global vn
    # If we have cookies (real app/session), prefer a per-session instance and ignore module cache
    has_cookies = hasattr(st.session_state, "cookies") and st.session_state.cookies is not None
    if has_cookies:
        if not hasattr(st.session_state, "_vn_instance") or st.session_state._vn_instance is None:
            st.session_state._vn_instance = get_vanna_service()
        vn = st.session_state._vn_instance
        return vn

    # Test-only path: honor a patched module-level vn when no cookies available
    if vn is not None:
        return vn
    if not hasattr(st.session_state, "_vn_instance") or st.session_state._vn_instance is None:
        st.session_state._vn_instance = get_vanna_service()
        vn = st.session_state._vn_instance
    else:
        # Keep the module-level vn in sync so tests can patch it
        vn = st.session_state._vn_instance
    return st.session_state._vn_instance


def call_llm(my_question: str):
    vn_instance = get_vn()
    response = vn_instance.submit_prompt(
        "You are a helpful AI assistant trained to provide detailed and accurate responses. Be concise yet informative, and maintain a friendly and professional tone. If asked about controversial topics, provide balanced and well-researched information without expressing personal opinions.",
        my_question,
    )
    add_message(
        Message(role=RoleType.ASSISTANT, content=response, type=MessageType.TEXT, group_id=get_current_group_id())
    )


def get_llm_stream_generator(my_question: str):
    """Return a generator that yields LLM response chunks for ephemeral streaming.

    Falls back to a single-shot response if streaming is unsupported.
    """
    vn_instance = get_vn()
    system_msg = (
        "You are a helpful AI assistant trained to provide detailed and accurate responses. "
        "Be concise yet informative, and maintain a friendly and professional tone. "
        "If asked about controversial topics, provide balanced and well-researched information without expressing personal opinions."
    )

    # Prefer underlying VN streaming if available (Ollama path)
    try:
        underlying = getattr(vn_instance, "vn", None)
        if underlying is not None and hasattr(underlying, "stream_submit_prompt"):
            prompt = [underlying.system_message(system_msg), underlying.user_message(my_question)]
            return underlying.stream_submit_prompt(prompt)
    except Exception:
        pass

    # Fallback to a simple one-shot generator
    def _fallback_gen():
        content = vn_instance.submit_prompt(system_msg, my_question)
        yield str(content)

    return _fallback_gen()


def get_llm_sql_thought_stream(my_question: str):
    """Return a generator that streams an ephemeral narration while SQL is being prepared.

    Uses underlying VN streaming when available; otherwise falls back to a short, single message.
    """
    vn_instance = get_vn()

    try:
        # Prefer streaming actual SQL derivation using a single call
        if hasattr(vn_instance, "stream_generate_sql"):
            return vn_instance.stream_generate_sql(my_question)
    except Exception:
        pass

    def _fallback_gen():
        yield "Analyzing your question and preparing a SQL query..."

    return _fallback_gen()


def get_summary_stream_generator(question: str, df: pd.DataFrame):
    """Return a generator that streams the summary as it is produced by the backend.

    Prefer a single-call streaming path; fallback to non-streaming summary if unavailable.
    This yields only content tokens; use get_summary_event_stream for CoT + content.
    """
    vn_instance = get_vn()
    # Check manual cache first to avoid regenerating summaries
    try:
        cache_key = create_summary_cache_key(question, df)
        manual_cache = getattr(st.session_state, "manual_summary_cache", None) or {}
        cached = manual_cache.get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 2 and isinstance(cached[0], str):
            cached_summary, cached_elapsed = cached

            def _cached_gen():
                try:
                    try:
                        st.session_state["streamed_summary"] = cached_summary
                        st.session_state["streamed_summary_for_question"] = question
                        st.session_state["streamed_summary_elapsed_time"] = cached_elapsed
                    except Exception:
                        setattr(st.session_state, "streamed_summary", cached_summary)
                        setattr(st.session_state, "streamed_summary_for_question", question)
                        setattr(st.session_state, "streamed_summary_elapsed_time", cached_elapsed)
                except Exception:
                    pass
                yield cached_summary

            return _cached_gen()
    except Exception:
        pass
    try:
        if hasattr(vn_instance, "stream_generate_summary"):
            # Wrap the streaming generator to persist manual cache after completion
            def _wrapped():
                try:
                    for chunk in vn_instance.stream_generate_summary(question, df):
                        yield chunk
                finally:
                    try:
                        try:
                            summary_text = st.session_state.get("streamed_summary", "")
                        except Exception:
                            summary_text = getattr(st.session_state, "streamed_summary", "")
                        try:
                            elapsed = st.session_state.get("streamed_summary_elapsed_time", 0)
                        except Exception:
                            elapsed = getattr(st.session_state, "streamed_summary_elapsed_time", 0)
                        key = create_summary_cache_key(question, df)
                        manual_cache = getattr(st.session_state, "manual_summary_cache", None) or {}
                        # Only cache non-empty summaries; avoid poisoning cache with blanks
                        if isinstance(summary_text, str) and summary_text.strip() != "":
                            manual_cache[key] = (summary_text, elapsed or 0)
                        else:
                            # Ensure we do not persist an empty/failed entry
                            if key in manual_cache:
                                del manual_cache[key]
                        try:
                            st.session_state["manual_summary_cache"] = manual_cache
                        except Exception:
                            setattr(st.session_state, "manual_summary_cache", manual_cache)
                    except Exception:
                        pass

            return _wrapped()
    except Exception:
        pass

    def _fallback_gen():
        summary, _elapsed = vn_instance.generate_summary(question=question, df=df)
        try:
            try:
                st.session_state["streamed_summary"] = summary or ""
                st.session_state["streamed_summary_for_question"] = question
                st.session_state["streamed_summary_elapsed_time"] = _elapsed or 0
            except Exception:
                setattr(st.session_state, "streamed_summary", summary or "")
                setattr(st.session_state, "streamed_summary_for_question", question)
                setattr(st.session_state, "streamed_summary_elapsed_time", _elapsed or 0)
            # Store in manual cache for reuse
            cache_key_inner = create_summary_cache_key(question, df)
            manual_cache_inner = getattr(st.session_state, "manual_summary_cache", None) or {}
            if isinstance(summary, str) and summary.strip() != "":
                manual_cache_inner[cache_key_inner] = (summary, _elapsed or 0)
            else:
                if cache_key_inner in manual_cache_inner:
                    del manual_cache_inner[cache_key_inner]
            try:
                st.session_state["manual_summary_cache"] = manual_cache_inner
            except Exception:
                setattr(st.session_state, "manual_summary_cache", manual_cache_inner)
        except Exception:
            pass
        yield summary or ""

    return _fallback_gen()


def get_summary_event_stream(question: str, df: pd.DataFrame, think: bool = False):
    """Yield (kind, text) where kind in {"thinking","content"} while streaming.

    When think=True, upstream may emit thinking tokens; otherwise only content.
    """
    vn_instance = get_vn()
    # Cache short-circuit: if we already have a summary for this question+df, avoid streaming
    try:
        cache_key = create_summary_cache_key(question, df)
        manual_cache = getattr(st.session_state, "manual_summary_cache", None) or {}
        cached = manual_cache.get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 2 and isinstance(cached[0], str):
            cached_summary, cached_elapsed = cached

            def _noop_cached():
                try:
                    try:
                        st.session_state["streamed_summary"] = cached_summary
                        st.session_state["streamed_summary_for_question"] = question
                        st.session_state["streamed_summary_elapsed_time"] = cached_elapsed
                    except Exception:
                        setattr(st.session_state, "streamed_summary", cached_summary)
                        setattr(st.session_state, "streamed_summary_for_question", question)
                        setattr(st.session_state, "streamed_summary_elapsed_time", cached_elapsed)
                except Exception:
                    pass
                if False:
                    yield ("content", "")

            return _noop_cached()
    except Exception:
        pass
    if hasattr(vn_instance, "summary_event_stream"):
        # Wrap to store final content into manual cache once streaming finishes
        def _wrapped_events():
            try:
                for kind_text in vn_instance.summary_event_stream(question, df, think=think):
                    yield kind_text
            finally:
                try:
                    try:
                        summary_text = st.session_state.get("streamed_summary", "")
                    except Exception:
                        summary_text = getattr(st.session_state, "streamed_summary", "")
                    try:
                        elapsed = st.session_state.get("streamed_summary_elapsed_time", 0)
                    except Exception:
                        elapsed = getattr(st.session_state, "streamed_summary_elapsed_time", 0)
                    key = create_summary_cache_key(question, df)
                    manual_cache = getattr(st.session_state, "manual_summary_cache", None) or {}
                    if isinstance(summary_text, str) and summary_text.strip() != "":
                        manual_cache[key] = (summary_text, elapsed or 0)
                    else:
                        if key in manual_cache:
                            del manual_cache[key]
                    try:
                        st.session_state["manual_summary_cache"] = manual_cache
                    except Exception:
                        setattr(st.session_state, "manual_summary_cache", manual_cache)
                except Exception:
                    pass

        return _wrapped_events()

    # Fallback: proxy to content-only stream
    def _proxy():
        for chunk in get_summary_stream_generator(question, df):
            yield ("content", chunk)

    return _proxy()


def create_summary_cache_key(question: str, df: pd.DataFrame) -> str:
    """Create a stable cache key for summary caching based on question and DataFrame signature.

    Uses a SHA-1 hash of columns, length, and a preview of up to 20 rows to avoid huge keys.
    """
    try:
        # Prefer SQL signature if available in session (most stable across identical queries)
        try:
            sql_sig = st.session_state.get("current_sql_for_summary")
        except Exception:
            sql_sig = getattr(st.session_state, "current_sql_for_summary", None)
        if isinstance(sql_sig, str) and len(sql_sig.strip()) > 0:
            payload = json.dumps({"q": question, "sql": sql_sig.strip()}, ensure_ascii=False)
            return hashlib.sha1(payload.encode("utf-8")).hexdigest()
        cols = list(df.columns) if df is not None else []
        nrows = int(df.shape[0]) if df is not None else 0
        preview = df.head(20).to_json(orient="records", date_format="iso") if df is not None else ""
        payload = json.dumps({"q": question, "cols": cols, "n": nrows, "p": preview}, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()
    except Exception:
        # Fallback to question-only key if df processing fails
        return hashlib.sha1((question or "").encode("utf-8")).hexdigest()


def get_chart(my_question, sql, df):
    vn_instance = vn if vn is not None else get_vn()
    elapsed_sum = 0
    code = None
    if vn_instance.should_generate_chart(question=my_question, sql=sql, df=df):
        result = vn_instance.generate_plotly_code(question=my_question, sql=sql, df=df)
        if not isinstance(result, tuple) or len(result) != 2:
            # Defensive: some tests may patch to return a single value
            code, elapsed_time = result, 0
        else:
            code, elapsed_time = result
        elapsed_sum += elapsed_time if elapsed_time is not None else 0

        if st.session_state.get("show_plotly_code", False):
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    code,
                    MessageType.PYTHON,
                    sql,
                    my_question,
                    df,
                    elapsed_time,
                    group_id=get_current_group_id(),
                )
            )

        if code is not None and code != "":
            plot_result = vn_instance.generate_plot(code=code, df=df)
            if not isinstance(plot_result, tuple) or len(plot_result) != 2:
                fig, elapsed_time = plot_result, 0
            else:
                fig, elapsed_time = plot_result
            elapsed_sum += elapsed_time if elapsed_time is not None else 0
            if fig is not None:
                add_message(
                    Message(
                        RoleType.ASSISTANT,
                        fig,
                        MessageType.PLOTLY_CHART,
                        sql,
                        my_question,
                        None,
                        elapsed_sum,
                        group_id=get_current_group_id(),
                    )
                )
            else:
                add_message(
                    Message(
                        RoleType.ASSISTANT,
                        "I couldn't generate a chart",
                        MessageType.ERROR,
                        sql,
                        my_question,
                        None,
                        elapsed_sum,
                        group_id=get_current_group_id(),
                    )
                )
        else:
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    "I couldn't generate a chart",
                    MessageType.ERROR,
                    sql,
                    my_question,
                    None,
                    elapsed_sum,
                    group_id=get_current_group_id(),
                )
            )
    else:
        add_message(
            Message(
                RoleType.ASSISTANT,
                "I was unable to generate a chart for this question.",
                MessageType.ERROR,
                sql,
                my_question,
                None,
                0,
                group_id=get_current_group_id(),
            )
        )


def set_question(question: str, render=True):
    if question is None:
        if len(st.session_state.messages) > 0:
            st.session_state.min_message_id = st.session_state.messages[-1].id
            save_user_settings()
            # Clear questions history when resetting
            st.session_state.my_question = None
            st.session_state.messages = None

    else:
        # Start a new group for this question/answer flow
        group_id = start_new_group()

        # Set question
        st.session_state.my_question = question
        # Clear any lingering error state from prior flows so it doesn't leak
        try:
            st.session_state["pending_sql_error"] = False
            st.session_state["last_run_sql_error"] = None
            st.session_state["last_failed_sql"] = None
            st.session_state["show_failed_sql_open"] = False
        except Exception:
            pass
        add_message(Message(RoleType.USER, question, MessageType.TEXT, group_id=group_id), render)


def get_unique_messages():
    # Assuming st.session_state.messages is a list of dictionaries
    messages = st.session_state.messages

    # Filter messages to ensure uniqueness based on the "content" field
    seen_content = set()
    unique_messages = []
    for message in messages:
        if message.role == RoleType.USER.value:
            content = message.content
            if content not in seen_content:
                seen_content.add(content)
                unique_messages.append(message)

    return unique_messages


def set_feedback(index: int, value: str):
    message = st.session_state.messages[index]
    message.feedback = value
    message.save()
    new_entry = {
        "question": st.session_state.messages[index].question,
        "query": st.session_state.messages[index].query,
    }
    if st.session_state.cookies.get("role_name") == "Admin":
        if value == "up":
            write_to_file_and_training(new_entry)
        else:
            remove_from_file_training(new_entry)


def generate_guid():
    return str(uuid.uuid4())


def generate_group_id():
    """Generate a new group ID for message flows."""
    return str(uuid.uuid4())


def get_current_group_id():
    """Get the current group ID from session state, or create a new one if none exists."""
    if not hasattr(st.session_state, "current_group_id") or st.session_state.current_group_id is None:
        st.session_state.current_group_id = generate_group_id()
    return st.session_state.current_group_id


def start_new_group():
    """Start a new message group by generating a new group ID."""
    st.session_state.current_group_id = generate_group_id()
    return st.session_state.current_group_id


def group_messages_by_id(messages: list) -> list:
    """Group consecutive messages by their group_id.

    Returns a list of tuples: (group_id, [messages_in_group])
    Messages without a group_id are treated as individual groups.
    """
    if not messages:
        return []

    groups = []
    current_group_id = None
    current_group_messages = []

    for message in messages:
        msg_group_id = getattr(message, "group_id", None)

        if msg_group_id is None:
            # Message has no group_id - treat as individual group
            if current_group_messages:
                groups.append((current_group_id, current_group_messages))
            groups.append((None, [message]))
            current_group_id = None
            current_group_messages = []
        elif msg_group_id == current_group_id:
            # Same group - add to current group
            current_group_messages.append(message)
        else:
            # New group - save current group and start new one
            if current_group_messages:
                groups.append((current_group_id, current_group_messages))
            current_group_id = msg_group_id
            current_group_messages = [message]

    # Don't forget the last group
    if current_group_messages:
        groups.append((current_group_id, current_group_messages))

    return groups


def get_message_group_css() -> str:
    """Generate global CSS styling for message group containers.

    Uses custom styling to override Streamlit's default container border
    with a left border accent for visual distinction between groups.
    Only targets innermost containers (those without nested bordered containers).
    """
    border_color = "#0b5258"  # Primary theme color

    return """
        <style>
            /* Only style innermost bordered containers - exclude parents that have nested bordered containers */
            div[data-testid="stVerticalBlock"]:has([data-testid="stChatMessage"]):not(:has(div[data-testid="stVerticalBlock"]:has([data-testid="stChatMessage"]))) {
                border: none !important;
                border-left: 4px solid """ + border_color + """ !important;
                border-radius: 0 8px 8px 0 !important;
                background-color: rgba(11, 82, 88, 0.03) !important;
                padding-left: 1rem !important;
            }

            /* Alternating background for visual distinction between groups */
            div[data-testid="stVerticalBlock"]:has([data-testid="stChatMessage"]):not(:has(div[data-testid="stVerticalBlock"]:has([data-testid="stChatMessage"]))):nth-of-type(even) {
                background-color: rgba(11, 82, 88, 0.06) !important;
            }
        </style>
    """


def render_message_group(messages: list, group_index: int, start_index: int):
    """Render a group of messages within a styled container.

    Args:
        messages: List of messages in this group
        group_index: Index of the group for CSS styling
        start_index: Starting index for individual message rendering
    """
    # Only apply grouping if there's a valid group_id (indicating it's part of a Q&A flow)
    has_group_id = messages and getattr(messages[0], "group_id", None) is not None

    if has_group_id and len(messages) > 0:
        # Use Streamlit's native container with border=True
        # This creates a proper wrapper that contains all child elements
        with st.container(border=True):
            # Render each message in the group
            for i, message in enumerate(messages):
                render_message(message, start_index + i)
    else:
        # No grouping - render messages individually
        for i, message in enumerate(messages):
            render_message(message, start_index + i)


def get_followup_questions(my_question, sql, df):
    vn_instance = get_vn()
    followup_questions = vn_instance.generate_followup_questions(question=my_question, sql=sql, df=df)

    add_message(Message(RoleType.ASSISTANT, followup_questions, MessageType.FOLLOWUP, sql, my_question))


# --- Private helper functions for rendering specific message types ---


def _render_sql(message: Message, index: int):
    if st.session_state.get("show_elapsed_time", True) and message.elapsed_time is not None:
        st.write(f"Elapsed Time: {message.elapsed_time}")
    st.code(message.content, language="sql", line_numbers=True)


def _render_python(message: Message, index: int):
    st.code(message.content, language="python", line_numbers=True)


def _render_line_chart(message: Message, index: int):
    if st.session_state.get("show_elapsed_time", True) and message.elapsed_time is not None:
        st.write(f"Elapsed Time: {message.elapsed_time}")
    df = pd.read_json(StringIO(message.dataframe))
    columns = df.columns.tolist()
    st.line_chart(
        df, x=columns[0], y=columns[1], color="#0b5258"
    )  # Assuming first column is x-axis and rest are y-axis


def _render_bar_chart(message: Message, index: int):
    if st.session_state.get("show_elapsed_time", True) and message.elapsed_time is not None:
        st.write(f"Elapsed Time: {message.elapsed_time}")
    df = pd.read_json(StringIO(message.dataframe))
    columns = df.columns.tolist()
    st.bar_chart(df, x=columns[0], y=columns[1], color="#0b5258")  # Assuming first column is x-axis and rest are y-axis


def _render_area_chart(message: Message, index: int):
    if st.session_state.get("show_elapsed_time", True) and message.elapsed_time is not None:
        st.write(f"Elapsed Time: {message.elapsed_time}")
    df = pd.read_json(StringIO(message.dataframe))
    columns = df.columns.tolist()
    st.area_chart(
        df, x=columns[0], y=columns[1], color="#0b5258"
    )  # Assuming first column is x-axis and rest are y-axis


def _render_scatter_chart(message: Message, index: int):
    if st.session_state.get("show_elapsed_time", True) and message.elapsed_time is not None:
        st.write(f"Elapsed Time: {message.elapsed_time}")
    df = pd.read_json(StringIO(message.dataframe))
    columns = df.columns.tolist()
    st.scatter_chart(
        df, x=columns[0], y=columns[1], color="#0b5258"
    )  # Assuming first column is x-axis and rest are y-axis


def _render_plotly_chart(message: Message, index: int):
    if st.session_state.get("show_elapsed_time", True) and message.elapsed_time is not None:
        st.write(f"Elapsed Time: {message.elapsed_time}")
    message.content = message.content.replace("#000001", "#0b5258")  # Replace color code for consistency
    chart = json.loads(message.content)
    st.plotly_chart(chart, key=f"message_{message.id}")


def _render_error(message: Message, index: int):
    if st.session_state.get("show_elapsed_time", True) and message.elapsed_time is not None:
        st.write(f"Elapsed Time: {message.elapsed_time}")
    # Short error messages are displayed directly; long ones (stack traces) use collapsible
    error_length_threshold = 300
    if len(message.content) <= error_length_threshold:
        # Short, user-friendly error - display directly in warning
        st.warning(message.content)
    else:
        # Long error (likely stack trace) - use collapsible to reduce visual clutter
        st.warning("An error occurred while processing your request.")
        with st.expander("View error details", expanded=False):
            st.code(message.content, language="text")


def _render_dataframe(message: Message, index: int):
    if st.session_state.get("show_elapsed_time", True) and message.elapsed_time is not None:
        st.write(f"Query Execution Time: {message.elapsed_time:.3f}s")
    df = pd.read_json(StringIO(message.content), convert_dates=True)
    st.dataframe(df, key=f"message_{message.id}")


def _render_summary_actions_popover(message: Message, index: int, my_df: pd.DataFrame):
    # Helper for the popover content within summary messages
    with st.popover("Actions", use_container_width=True):
        st.button("Speak Summary", key=f"speak_summary_{message.id}", on_click=lambda: speak(message.content))
        st.button(
            "Follow-up Questions",
            key=f"follow_up_questions_{message.id}",
            on_click=lambda: get_followup_questions(message.question, message.query, my_df),
        )
        if st.button("Generate Table", key=f"table_{message.id}"):
            # Ensure DataFrame is converted to JSON string for the Message constructor if it expects that
            df_json_content = my_df.to_json(date_format="iso")
            add_message(
                Message(RoleType.ASSISTANT, df_json_content, MessageType.DATAFRAME, message.query, message.question),
                False,
            )

        # Use the already parsed DataFrame instead of parsing again
        if len(my_df.columns) >= 2:
            cols = st.columns((1, 1, 1, 1, 1))
            with cols[0]:
                st.button(
                    "AI Generate Plotly",
                    key=f"graph_{message.id}",
                    on_click=lambda: get_chart(message.question, message.query, my_df),
                )
            with cols[1]:
                if st.button("Line Chart", key=f"line_chart_{message.id}"):
                    add_message(
                        Message(
                            RoleType.ASSISTANT,
                            message.dataframe,
                            MessageType.ST_LINE_CHART,
                            message.query,
                            message.question,
                            message.dataframe,
                            0,
                        ),
                        False,
                    )
            with cols[2]:
                if st.button("Bar Chart", key=f"bar_chart_{message.id}"):
                    add_message(
                        Message(
                            RoleType.ASSISTANT,
                            message.dataframe,
                            MessageType.ST_BAR_CHART,
                            message.query,
                            message.question,
                            message.dataframe,
                            0,
                        ),
                        False,
                    )
            with cols[3]:
                if st.button("Area Chart", key=f"area_chart_{message.id}"):
                    add_message(
                        Message(
                            RoleType.ASSISTANT,
                            message.dataframe,
                            MessageType.ST_AREA_CHART,
                            message.query,
                            message.question,
                            message.dataframe,
                            0,
                        ),
                        False,
                    )
            with cols[4]:
                if st.button("Scatter Chart", key=f"scatter_chart_{message.id}"):
                    add_message(
                        Message(
                            RoleType.ASSISTANT,
                            message.dataframe,
                            MessageType.ST_SCATTER_CHART,
                            message.query,
                            message.question,
                            message.dataframe,
                            0,
                        ),
                        False,
                    )

        with st.expander("Show SQL"):
            st.code(message.query, language="sql", line_numbers=True)


def _render_summary(message: Message, index: int):
    if st.session_state.get("show_elapsed_time", True) and message.elapsed_time is not None:
        st.write(f"Elapsed Time: {message.elapsed_time}")
    st.markdown(message.content)

    cols = st.columns(
        [0.1, 0.1, 0.6]
    )  # Note: Original was [0.1, 0.1, 0.6]. If popover is wider, might need adjustment.
    with cols[0]:
        st.button(
            "👍",
            key=f"thumbs_up_{message.id}",
            type="primary" if message.feedback == "up" else "secondary",
            on_click=set_feedback,
            args=(index, "up"),
        )
    with cols[1]:
        st.button(
            "👎",
            key=f"thumbs_down_{message.id}",
            type="primary" if message.feedback == "down" else "secondary",
            on_click=set_feedback,
            args=(index, "down"),
        )
    with cols[2]:
        # message.dataframe is expected to be a JSON string from the Message object.
        my_df = pd.read_json(StringIO(message.dataframe), orient="records") if message.dataframe else pd.DataFrame()
        _render_summary_actions_popover(message, index, my_df)


def _render_followup(message: Message, index: int):
    if len(message.content) > 0:
        st.text("Here are some possible follow-up questions")
        try:
            # Attempt to evaluate the content string as a Python literal (e.g., a list)
            content_object = ast.literal_eval(message.content)
            # Ensure content_array is a list, even if content_object is a single string or other non-list iterable
            content_array = list(content_object) if isinstance(content_object, (list, tuple, set)) else []
        except (ValueError, SyntaxError):
            # If content is not a valid literal (e.g. plain string not representing a list, or malformed)
            # Handle as an empty list or log a warning. For now, treat as empty.
            content_array = []
            # Optionally, log this: logger.warning(f"Could not parse follow-up questions from content: {message.content}")

        for question_text in content_array[:5]:  # Max 5 follow-up questions
            if isinstance(question_text, str) and len(question_text) > 0:
                question_value = question_text.strip()
                if "/" in question_text:
                    question_value = "/" + question_text.partition("/")[2]
                st.button(
                    question_text,
                    on_click=set_question,
                    args=(question_value,),
                    key=generate_guid(),
                    use_container_width=True,
                )
            # Optionally, add an else to log/warn about non-string items if expected.


def _render_thinking(message: Message, index: int):
    """Render thinking messages in an expandable container."""
    with st.expander("🤔 AI Thinking Process", expanded=False):
        st.markdown(message.content)
        if st.session_state.get("show_elapsed_time", True) and message.elapsed_time is not None:
            st.caption(f"Thinking time: {message.elapsed_time:.3f}s")


def _render_default(message: Message, index: int):
    st.markdown(message.content)


# --- Registry of rendering functions ---
MESSAGE_RENDERERS = {
    MessageType.SQL.value: _render_sql,
    MessageType.PYTHON.value: _render_python,
    MessageType.PLOTLY_CHART.value: _render_plotly_chart,
    MessageType.ST_LINE_CHART.value: _render_line_chart,
    MessageType.ST_BAR_CHART.value: _render_bar_chart,
    MessageType.ST_AREA_CHART.value: _render_area_chart,
    MessageType.ST_SCATTER_CHART.value: _render_scatter_chart,
    MessageType.ERROR.value: _render_error,
    MessageType.DATAFRAME.value: _render_dataframe,
    MessageType.SUMMARY.value: _render_summary,
    MessageType.FOLLOWUP.value: _render_followup,
    MessageType.TEXT.value: _render_default,  # Explicitly map TEXT to default
    MessageType.THINKING.value: _render_thinking,
}


def render_message(message: Message, index: int):
    with st.chat_message(message.role):
        renderer = MESSAGE_RENDERERS.get(message.type, _render_default)
        renderer(message, index)


def add_message(message: Message, render=True):
    message = message.save()
    st.session_state.messages.append(message)

    # Manage session state memory by keeping only the most recent messages
    from utils.config_helper import get_max_session_messages

    max_messages = get_max_session_messages()

    if len(st.session_state.messages) > max_messages:
        # Remove oldest messages to stay within limit
        messages_to_remove = len(st.session_state.messages) - max_messages
        st.session_state.messages = st.session_state.messages[messages_to_remove:]
        logger.info(
            f"Trimmed {messages_to_remove} messages from session state. Kept most recent {max_messages} messages."
        )

    if len(st.session_state.messages) > 0 and render:
        render_message(st.session_state.messages[-1], len(st.session_state.messages) - 1)


def add_acknowledgement():
    acknowledgements = [
        "That's an excellent question. Let me think about that for a moment.",
        "Interesting point! Let me analyze this for you.",
        "Great question! Let me dive into that.",
        "I see where you're coming from. Let me process this.",
        "That's a thoughtful question. Let me work through it.",
        "Good question! Let me gather the relevant information.",
        "I appreciate the depth of your question. Let me consider it carefully.",
        "That's a valid and insightful question. Let me provide a detailed response.",
        "You've raised an important point. Let me think this through.",
        "I like the way you're thinking. Let me explore this further for you.",
    ]

    # write an acknowledgment message to
    random_acknowledgment = random.choice(acknowledgements)
    with st.chat_message(RoleType.ASSISTANT.value):
        st.write(random_acknowledgment)


def normal_message_flow(my_question: str):
    # check guardrails here
    guardrail_sentence, guardrail_score = get_ethical_guideline(my_question)
    logger.debug(
        "Ethical Guardrails triggered: Question=%s Score=%s Response=%s",
        my_question,
        guardrail_score,
        guardrail_sentence,
    )
    if guardrail_score == 2:
        logger.info(
            "Ethical Guardrails triggered: Question=%s Score=%s Response=%s",
            my_question,
            guardrail_score,
            guardrail_sentence,
        )
        add_message(
            Message(
                RoleType.ASSISTANT,
                guardrail_sentence,
                MessageType.ERROR,
                "",
                my_question,
                group_id=get_current_group_id(),
            )
        )
        st.stop()
    if guardrail_score >= 3:
        logger.warning(
            "Ethical Guardrails triggered: Question=%s Score=%s Response=%s",
            my_question,
            guardrail_score,
            guardrail_sentence,
        )
        add_message(
            Message(
                RoleType.ASSISTANT,
                guardrail_sentence,
                MessageType.ERROR,
                "",
                my_question,
                group_id=get_current_group_id(),
            )
        )
        st.stop()

    add_acknowledgement()

    # Initialize variables
    sql = None
    elapsed_time = 0

    # Check if we have a thinking model (Ollama with thinking support)
    vn_instance = get_vn()
    has_thinking_model = False
    try:
        # Check if this is an Ollama-based model that might support thinking
        underlying = getattr(vn_instance, "vn", None)
        if underlying and hasattr(underlying, "ollama_client") and underlying.ollama_client is not None:
            has_thinking_model = True
    except Exception:
        pass

    # If we have a thinking model, display the thinking stream
    thinking_text = ""  # Initialize outside the try block for use later
    if has_thinking_model and hasattr(vn_instance, "stream_generate_sql"):
        try:
            thinking_chunks = []

            # Create a placeholder for real-time thinking display
            with st.chat_message(RoleType.ASSISTANT.value):
                thinking_placeholder = st.empty()

                # Stream the SQL generation and show thinking in real-time
                stream_gen = vn_instance.stream_generate_sql(my_question)
                for chunk in stream_gen:
                    thinking_chunks.append(chunk)
                    # Update the thinking display in real-time
                    thinking_placeholder.markdown("🤔 **Thinking...**\n\n" + "".join(thinking_chunks))

                # After streaming completes, get the cached result
                if hasattr(st.session_state, "streamed_sql"):
                    sql = st.session_state.streamed_sql
                    elapsed_time = st.session_state.get("streamed_sql_elapsed_time", 0)
                    thinking_text = st.session_state.get("streamed_thinking", "")

                    # Phase 1: Graceful transition - show completion indicator
                    if thinking_text and thinking_text.strip():
                        # Show "Done thinking" state for a brief moment
                        thinking_placeholder.markdown(
                            "✅ **Done thinking**\n\n" + "".join(thinking_chunks)
                        )
                        # Brief delay for visual continuity (1.5 seconds)
                        time.sleep(1.5)

                    # Clear the placeholder - we'll add the thinking as a proper message
                    thinking_placeholder.empty()

            # Phase 2: Persistent display - add thinking to chat history as collapsible expander
            # If we collected thinking text, add it as a proper message
            if thinking_text and thinking_text.strip():
                add_message(
                    Message(
                        RoleType.ASSISTANT,
                        thinking_text,
                        MessageType.THINKING,
                        "",
                        my_question,
                        None,
                        elapsed_time,
                        group_id=get_current_group_id(),
                    ),
                    render=True,  # Render so it appears in chat history as collapsible expander
                )
        except Exception as e:
            logger.warning(f"Failed to stream SQL generation: {e}")

    # If no thinking model or streaming failed, use regular generation
    # Import retry config
    from utils.config_helper import get_max_sql_retries

    max_retries = get_max_sql_retries()
    attempt = 1
    df = None
    sql_elapsed_time = 0
    last_error_msg = None
    last_failed_sql = None

    # Check if this is a manual retry (user clicked retry button after auto-retries exhausted)
    if st.session_state.get("use_retry_context"):
        # Manual retry starts fresh from attempt 1
        st.session_state["use_retry_context"] = False
        last_failed_sql = st.session_state.get("retry_failed_sql")
        last_error_msg = st.session_state.get("retry_error_msg")
        st.session_state["retry_failed_sql"] = None
        st.session_state["retry_error_msg"] = None
        # For manual retry, generate with context from attempt 2
        if sql is None:
            sql, elapsed_time = get_vn().generate_sql_retry(
                question=my_question,
                failed_sql=last_failed_sql,
                error_message=last_error_msg,
                attempt_number=2,
            )

    # Auto-retry loop for SQL generation and execution
    while attempt <= max_retries + 1:  # +1 because first attempt is not a "retry"
        # Generate SQL on first attempt if not already generated (e.g., from thinking model)
        if sql is None:
            if attempt == 1:
                sql, elapsed_time = get_vn().generate_sql(question=my_question)
            else:
                # Show retry status to user
                with st.chat_message(RoleType.ASSISTANT.value):
                    st.info(f"Attempt {attempt}/{max_retries + 1}: Trying a different approach...")
                logger.info(
                    f"SQL retry attempt {attempt}/{max_retries + 1} for question: {my_question}. "
                    f"Previous error: {last_error_msg}"
                )
                sql, elapsed_time = get_vn().generate_sql_retry(
                    question=my_question,
                    failed_sql=last_failed_sql,
                    error_message=last_error_msg,
                    attempt_number=attempt,
                )

        if not sql:
            # SQL generation failed completely, break out of retry loop
            break

        # Validate SQL
        if not get_vn().is_sql_valid(sql=sql):
            logger.debug(f"sql is not valid on attempt {attempt}")
            last_failed_sql = sql
            last_error_msg = "SQL syntax validation failed"
            sql = None
            attempt += 1
            continue

        # Show SQL if enabled (only on first attempt to avoid clutter)
        if attempt == 1 and st.session_state.get("show_sql", True):
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    sql,
                    MessageType.SQL,
                    sql,
                    my_question,
                    None,
                    elapsed_time,
                    group_id=get_current_group_id(),
                )
            )

        # Execute SQL
        rs = get_vn().run_sql(sql=sql)
        if isinstance(rs, tuple) and len(rs) == 2:
            df, sql_elapsed_time = rs
        else:
            df = rs
            try:
                sql_elapsed_time = st.session_state.get("last_sql_elapsed_time", 0)
            except Exception:
                sql_elapsed_time = 0

        # Check if execution succeeded
        if isinstance(df, pd.DataFrame):
            # Success! Break out of retry loop
            logger.info(f"SQL execution succeeded on attempt {attempt}")
            break

        # Execution failed - prepare for retry
        last_error_msg = st.session_state.get("last_run_sql_error")
        last_failed_sql = st.session_state.get("last_failed_sql") or sql

        # Check if error is non-recoverable
        if is_non_recoverable_error(last_error_msg):
            logger.info(f"Non-recoverable error detected, skipping auto-retry: {last_error_msg}")
            break

        # Log the retry attempt
        logger.warning(
            f"SQL execution failed on attempt {attempt}/{max_retries + 1}. "
            f"Error: {last_error_msg}. SQL: {last_failed_sql[:200]}..."
        )

        # Preserve the last SQL that was tried before resetting for regeneration
        final_sql = sql
        sql = None  # Reset SQL for next iteration to force regeneration
        attempt += 1

    st.session_state.my_question = None

    # Determine the final SQL: either the successful one or the last failed one
    final_sql = sql if sql else (last_failed_sql if last_failed_sql else None)

    if final_sql and isinstance(df, pd.DataFrame):
        # SQL executed successfully
        st.session_state["df"] = df

        # Show SQL if enabled and this was a successful retry (SQL wasn't shown during first attempt)
        if attempt > 1 and st.session_state.get("show_sql", True):
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    final_sql,
                    MessageType.SQL,
                    final_sql,
                    my_question,
                    None,
                    elapsed_time,
                    group_id=get_current_group_id(),
                )
            )

        if st.session_state.get("show_table", True):
            df = st.session_state.get("df")
            add_message(
                Message(
                    RoleType.ASSISTANT,
                    df,
                    MessageType.DATAFRAME,
                    final_sql,
                    my_question,
                    None,
                    sql_elapsed_time,
                    group_id=get_current_group_id(),
                )
            )

        if st.session_state.get("show_chart", True):
            get_chart(my_question, final_sql, df)

        # Successful data path: clear error flags ONLY if we have a real DataFrame result
        try:
            if isinstance(st.session_state.get("df"), pd.DataFrame):
                st.session_state["pending_sql_error"] = False
                st.session_state["last_run_sql_error"] = None
                st.session_state["last_failed_sql"] = None
        except Exception:
            pass

        # Only generate summary if SQL ran and produced a DataFrame
        _df_for_summary = st.session_state.get("df")
        if isinstance(_df_for_summary, pd.DataFrame) and (
            st.session_state.get("show_summary", True) or st.session_state.get("speak_summary", True)
        ):
            # Provide SQL signature for stable summary cache keys and reset streamed summary state
            try:
                st.session_state["current_sql_for_summary"] = final_sql
                st.session_state["streamed_summary"] = None
                st.session_state["streamed_summary_for_question"] = None
                st.session_state["streamed_summary_elapsed_time"] = 0
            except Exception:
                pass
            # Try streaming summary first
            summary = None
            elapsed_time = 0

            try:
                # Stream summary content only (no thinking)
                with st.chat_message(RoleType.ASSISTANT.value):
                    # Use a visible status/spinner while generating the summary
                    try:
                        status_cm = st.status("Generating summary...", expanded=False)
                    except Exception:
                        status_cm = None

                    if status_cm is not None:
                        with status_cm:
                            event_stream = get_summary_event_stream(my_question, _df_for_summary, think=False)
                            # Prefer Streamlit write_stream when available for typewriter effect
                            if hasattr(st, "write_stream"):
                                def _content_only():
                                    for kind, text in event_stream:
                                        if kind == "content":
                                            yield text
                                st.write_stream(_content_only())
                            else:
                                # Fallback to manual placeholder loop when write_stream is unavailable (tests)
                                summary_placeholder = st.empty()
                                content_chunks = []
                                for kind, text in event_stream:
                                    if kind == "content":
                                        content_chunks.append(text)
                                        summary_placeholder.markdown("".join(content_chunks))
                                summary_placeholder.empty()
                    else:
                        # No status context available; fallback to manual rendering
                        event_stream = get_summary_event_stream(my_question, _df_for_summary, think=False)
                        if hasattr(st, "write_stream"):
                            def _content_only2():
                                for kind, text in event_stream:
                                    if kind == "content":
                                        yield text
                            st.write_stream(_content_only2())
                        else:
                            summary_placeholder = st.empty()
                            content_chunks = []
                            for kind, text in event_stream:
                                if kind == "content":
                                    content_chunks.append(text)
                                    summary_placeholder.markdown("".join(content_chunks))
                            summary_placeholder.empty()

                    # Get the final summary from session state after streaming completes
                    if hasattr(st.session_state, "streamed_summary"):
                        summary = st.session_state.streamed_summary
                        elapsed_time = st.session_state.get("streamed_summary_elapsed_time", 0)

            except Exception as e:
                logger.warning(f"Failed to stream summary: {e}")
                # Fall back to non-streaming
                summary, elapsed_time = get_vn().generate_summary(question=my_question, df=df)

            # Add the summary message only if we have non-empty content
            if summary is not None and str(summary).strip() != "":
                if st.session_state.get("show_summary", True):
                    add_message(
                        Message(
                            RoleType.ASSISTANT,
                            summary,
                            MessageType.SUMMARY,
                            final_sql,
                            my_question,
                            df,
                            elapsed_time,
                            group_id=get_current_group_id(),
                        )
                    )

                if st.session_state.get("speak_summary", True):
                    speak(summary)
            else:
                # Do not add a blank/failed summary message per guidance; optionally speak a brief notice
                if st.session_state.get("speak_summary", True):
                    speak("Summary is unavailable for this result")

        if st.session_state.get("show_followup", True):
            get_followup_questions(my_question, final_sql, df)

        # Clear the question from session state after successful processing
        st.session_state.my_question = None

        # Trigger a rerun to properly refresh the UI
        st.rerun()
    elif final_sql:
        # SQL was generated but execution/validation failed after all retries
        error_msg = last_error_msg or st.session_state.get("last_run_sql_error")
        failed_sql = last_failed_sql or st.session_state.get("last_failed_sql")

        # Add error message to messages for test verification and history
        add_message(
            Message(
                RoleType.ASSISTANT,
                f"SQL failed: {error_msg}" if error_msg else "SQL failed after retries",
                MessageType.ERROR,
                failed_sql,
                my_question,
                None,
                elapsed_time,
                group_id=get_current_group_id(),
            )
        )

        # Show SQL that was tried (if not already shown)
        if attempt > 1 or not st.session_state.get("show_sql", True):
            if st.session_state.get("show_sql", True):
                add_message(
                    Message(
                        RoleType.ASSISTANT,
                        failed_sql,
                        MessageType.SQL,
                        failed_sql,
                        my_question,
                        None,
                        elapsed_time,
                        group_id=get_current_group_id(),
                    )
                )

        with st.chat_message(RoleType.ASSISTANT.value):
            # Use warning with collapsible details for less intrusive error display
            if attempt > 1:
                st.warning(f"I couldn't execute the SQL after {attempt - 1} automatic retries.")
            else:
                st.warning("I couldn't execute the generated SQL.")
            # Collapsible error details section
            with st.expander("View error details", expanded=False):
                if error_msg:
                    st.markdown(f"**Database error:** {error_msg}")
                if failed_sql:
                    st.markdown("**Failed SQL:**")
                    st.code(failed_sql, language="sql", line_numbers=True)
            # Action buttons remain outside expander for easy access
            cols = st.columns([0.2, 0.8])
            with cols[0]:
                retry_clicked = st.button("Retry", type="primary", key="retry_inline")

        if retry_clicked:
            # Persist retry intent and context, then rerun so it flows through normal pipeline
            st.session_state["use_retry_context"] = True
            st.session_state["retry_failed_sql"] = failed_sql
            st.session_state["retry_error_msg"] = error_msg
            # Set a persistent flag so the panel can be re-rendered if needed
            st.session_state["pending_sql_error"] = False
            st.session_state["pending_question"] = my_question
            st.session_state["my_question"] = my_question
            st.rerun()
        else:
            # Mark pending error so a persistent panel can be shown if page reruns without action
            st.session_state["pending_sql_error"] = True
            st.session_state["pending_question"] = my_question
            st.stop()
    else:
        add_message(
            Message(
                RoleType.ASSISTANT,
                "I wasn't able to generate SQL for that question",
                MessageType.ERROR,
                final_sql,
                my_question,
                group_id=get_current_group_id(),
            )
        )
        if st.session_state.get("llm_fallback", True):
            call_llm(my_question)

        # Clear the question from session state after error processing
        st.session_state.my_question = None

        # Trigger a rerun to properly refresh the UI
        st.rerun()
