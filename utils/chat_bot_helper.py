import ast
import json
import logging
import random
import uuid
from io import StringIO
from ethical_guardrails_lib import get_ethical_guideline
import pandas as pd
import streamlit as st

from orm.functions import save_user_settings, set_user_preferences_in_session_state
from orm.models import Message
from utils.communicate import speak

logger = logging.getLogger(__name__)
from utils.enums import MessageType, RoleType
from utils.vanna_calls import VannaService, remove_from_file_training, write_to_file_and_training


def get_vanna_service():
    """Get VannaService instance, ensuring user preferences are loaded first."""
    # Ensure user preferences are loaded into session state
    set_user_preferences_in_session_state()
    return VannaService.from_streamlit_session()


# Get the VannaService instance when needed, not at module import time
def get_vn():
    """Get the VannaService instance, ensuring it's created with correct user context."""
    if not hasattr(st.session_state, "_vn_instance") or st.session_state._vn_instance is None:
        st.session_state._vn_instance = get_vanna_service()
    return st.session_state._vn_instance


def call_llm(my_question: str):
    vn_instance = get_vn()
    response = vn_instance.submit_prompt(
        "You are a helpful AI assistant trained to provide detailed and accurate responses. Be concise yet informative, and maintain a friendly and professional tone. If asked about controversial topics, provide balanced and well-researched information without expressing personal opinions.",
        my_question,
    )
    add_message(Message(role=RoleType.ASSISTANT, content=response, type=MessageType.ERROR, group_id=get_current_group_id()))


def get_chart(my_question, sql, df):
    vn_instance = get_vn()
    elapsed_sum = 0
    code = None
    if vn_instance.should_generate_chart(question=my_question, sql=sql, df=df):
        code, elapsed_time = vn_instance.generate_plotly_code(question=my_question, sql=sql, df=df)
        elapsed_sum += elapsed_time if elapsed_time is not None else 0

        if st.session_state.get("show_plotly_code", False):
            add_message(Message(RoleType.ASSISTANT, code, MessageType.PYTHON, sql, my_question, df, elapsed_time, group_id=get_current_group_id()))

        if code is not None and code != "":
            fig, elapsed_time = vn_instance.generate_plot(code=code, df=df)
            elapsed_sum += elapsed_time if elapsed_time is not None else 0
            if fig is not None:
                add_message(
                    Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, my_question, None, elapsed_sum, group_id=get_current_group_id())
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
                        group_id=get_current_group_id()
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
                    group_id=get_current_group_id()
                )
            )
    else:
        add_message(
            Message(
                RoleType.ASSISTANT,
                "I was unable to generate a chart for this question. Please ask another question.",
                MessageType.ERROR,
                sql,
                my_question,
                None,
                0,
                group_id=get_current_group_id()
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
    if not hasattr(st.session_state, 'current_group_id') or st.session_state.current_group_id is None:
        st.session_state.current_group_id = generate_group_id()
    return st.session_state.current_group_id


def start_new_group():
    """Start a new message group by generating a new group ID."""
    st.session_state.current_group_id = generate_group_id()
    return st.session_state.current_group_id


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
    st.error(message.content)


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
            df_json_content = my_df.to_json(date_format='iso')
            add_message(
                Message(RoleType.ASSISTANT, df_json_content, MessageType.DATAFRAME, message.query, message.question),
                False,
            )

        # Use the already parsed DataFrame instead of parsing again
        if len(my_df.columns) >= 2:
            cols = st.columns((1, 1, 1, 1, 1))
            with cols[0]:
                st.button(
                    "Generate Plotly",
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
    st.code(message.content, language=None, wrap_lines=True)  # wrap_lines is not a valid arg for st.code

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
        logger.info(f"Trimmed {messages_to_remove} messages from session state. Kept most recent {max_messages} messages.")
    
    if len(st.session_state.messages) > 0 and render:
        render_message(st.session_state.messages[-1], len(st.session_state.messages) - 1)

def normal_message_flow(my_question:str):
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
        add_message(Message(RoleType.ASSISTANT, guardrail_sentence, MessageType.ERROR, "", my_question, group_id=get_current_group_id()))
        call_llm(my_question)
        st.stop()
    if guardrail_score >= 3:
        logger.warning(
            "Ethical Guardrails triggered: Question=%s Score=%s Response=%s",
            my_question,
            guardrail_score,
            guardrail_sentence,
        )
        add_message(Message(RoleType.ASSISTANT, guardrail_sentence, MessageType.ERROR, "", my_question, group_id=get_current_group_id()))
        st.stop()

    # write an acknowledgment message to
    random_acknowledgment = random.choice(acknowledgements)
    with st.chat_message(RoleType.ASSISTANT.value):
        st.write(random_acknowledgment)

    if st.session_state.get("use_retry_context"):
        sql, elapsed_time = get_vn().generate_sql_retry(
            question=my_question,
            failed_sql=st.session_state.get("retry_failed_sql"),
            error_message=st.session_state.get("retry_error_msg"),
        )
        # Clear retry context after use
        st.session_state["use_retry_context"] = False
        st.session_state["retry_failed_sql"] = None
        st.session_state["retry_error_msg"] = None
    else:
        sql, elapsed_time = get_vn().generate_sql(question=my_question)
    st.session_state.my_question = None

    if sql:
        if get_vn().is_sql_valid(sql=sql):
            if st.session_state.get("show_sql", True):
                add_message(Message(RoleType.ASSISTANT, sql, MessageType.SQL, sql, my_question, None, elapsed_time, group_id=get_current_group_id()))
        else:
            logger.debug("sql is not valid")
            add_message(Message(RoleType.ASSISTANT, sql, MessageType.ERROR, sql, my_question, None, elapsed_time, group_id=get_current_group_id()))
            # TODO: not sure if calling the LLM here is the correct spot or not, it seems to be necessary
            if st.session_state.get("llm_fallback", True):
                logger.debug("fallback to LLM")
                call_llm(my_question)
            st.stop()

        # Query limiting is now handled inside the run_sql method via LIMIT clause
        df, sql_elapsed_time = get_vn().run_sql(sql=sql)

        # if sql doesn't return a dataframe, offer retry with LLM guidance
        if not isinstance(df, pd.DataFrame):
            error_msg = st.session_state.get("last_run_sql_error")
            failed_sql = st.session_state.get("last_failed_sql")
            with st.chat_message(RoleType.ASSISTANT.value):
                st.error("I couldn't execute the generated SQL.")
                if error_msg:
                    st.caption(f"Database error: {error_msg}")
                cols = st.columns([0.2, 0.8])
                with cols[0]:
                    retry_clicked = st.button("Retry", type="primary", key="retry_inline")
                with cols[1]:
                    show_sql_clicked = st.button("Show Failed SQL", key="show_failed_sql_inline")
                if show_sql_clicked and failed_sql:
                    st.code(failed_sql, language="sql", line_numbers=True)

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
            st.session_state["df"] = df

        if st.session_state.get("show_table", True):
            df = st.session_state.get("df")
            add_message(Message(RoleType.ASSISTANT, df, MessageType.DATAFRAME, sql, my_question, None, sql_elapsed_time, group_id=get_current_group_id()))

        if st.session_state.get("show_chart", True):
            get_chart(my_question, sql, df)

        if st.session_state.get("show_summary", True) or st.session_state.get("speak_summary", True):
            summary, elapsed_time = get_vn().generate_summary(question=my_question, df=df)
            if summary is not None:
                if st.session_state.get("show_summary", True):
                    add_message(
                        Message(RoleType.ASSISTANT, summary, MessageType.SUMMARY, sql, my_question, df, elapsed_time, group_id=get_current_group_id())
                    )

                if st.session_state.get("speak_summary", True):
                    speak(summary)
            else:
                add_message(
                    Message(
                        RoleType.ASSISTANT,
                        "Could not generate a summary",
                        MessageType.SUMMARY,
                        sql,
                        my_question,
                        df,
                        elapsed_time,
                        group_id=get_current_group_id(),
                    )
                )
                if st.session_state.get("speak_summary", True):
                    speak("Could not generate a summary")

        if st.session_state.get("show_followup", True):
            get_followup_questions(my_question, sql, df)
    else:
        add_message(
            Message(
                RoleType.ASSISTANT,
                "I wasn't able to generate SQL for that question",
                MessageType.ERROR,
                sql,
                my_question,
                group_id=get_current_group_id(),
            )
        )
        if st.session_state.get("llm_fallback", True):
            call_llm(my_question)
