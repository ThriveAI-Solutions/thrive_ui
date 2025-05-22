import ast
import json
import logging
import random
import time
import uuid
from io import StringIO
import pandas as pd
import streamlit as st
from ethical_guardrails_lib import get_ethical_guideline
from orm.functions import get_recent_messages, save_user_settings, set_user_preferences_in_session_state
from orm.models import Message
from utils.communicate import listen, speak
from utils.enums import MessageType, RoleType
from utils.llm_calls import ask_message
from utils.vanna_calls import VannaService, remove_from_file_training, write_to_file_and_training

logger = logging.getLogger(__name__)

# Initialize VannaService singleton
vn = VannaService.get_instance()

set_user_preferences_in_session_state()

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

# Initialize session state variables
if "messages" not in st.session_state or st.session_state.messages == []:
    st.session_state.messages = get_recent_messages()
if st.session_state.messages is None:
    st.session_state.messages = []


def generate_guid():
    return str(uuid.uuid4())


def get_followup_questions(my_question, sql, df):
    followup_questions = vn.generate_followup_questions(question=my_question, sql=sql, df=df)

    add_message(Message(RoleType.ASSISTANT, followup_questions, MessageType.FOLLOWUP, sql, my_question))


def get_chart(my_question, sql, df):
    elapsed_sum = 0
    code = None
    if vn.should_generate_chart(question=my_question, sql=sql, df=df):
        code, elapsed_time = vn.generate_plotly_code(question=my_question, sql=sql, df=df)
        elapsed_sum += elapsed_time if elapsed_time is not None else 0

        if st.session_state.get("show_plotly_code", False):
            add_message(Message(RoleType.ASSISTANT, code, MessageType.PYTHON, sql, my_question, df, elapsed_time))

        if code is not None and code != "":
            fig, elapsed_time = vn.generate_plot(code=code, df=df)
            elapsed_sum += elapsed_time if elapsed_time is not None else 0
            if fig is not None:
                add_message(
                    Message(RoleType.ASSISTANT, fig, MessageType.PLOTLY_CHART, sql, my_question, None, elapsed_sum)
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
                    )
                )
    else:
        # If a chart should not be generated, inform the user.
        add_message(
            Message(
                RoleType.ASSISTANT,
                "I am unable to generate a chart for this data. The data might be unsuitable for visualization (e.g., empty or non-numeric).",
                MessageType.ERROR,
                sql,
                my_question,
                None,
                elapsed_sum,  # elapsed_sum would be 0 here
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
        # Set question
        st.session_state.my_question = question
        add_message(Message(RoleType.USER, question, MessageType.TEXT), render)


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
    if value == "up":
        write_to_file_and_training(new_entry)
    else:
        remove_from_file_training(new_entry)


# --- Private helper functions for rendering specific message types ---


def _render_sql(message: Message, index: int):
    if st.session_state.get("show_elapsed_time", True) and message.elapsed_time is not None:
        st.write(f"Elapsed Time: {message.elapsed_time}")
    st.code(message.content, language="sql", line_numbers=True)


def _render_python(message: Message, index: int):
    st.code(message.content, language="python", line_numbers=True)


def _render_plotly_chart(message: Message, index: int):
    if st.session_state.get("show_elapsed_time", True) and message.elapsed_time is not None:
        st.write(f"Elapsed Time: {message.elapsed_time}")
    chart = json.loads(message.content)
    st.plotly_chart(chart, key=f"message_{index}")


def _render_error(message: Message, index: int):
    st.error(message.content)


def _render_dataframe(message: Message, index: int):
    df = pd.read_json(StringIO(message.content))
    st.dataframe(df, key=f"message_{index}")


def _render_summary_actions_popover(message: Message, index: int, my_df: pd.DataFrame):
    # Helper for the popover content within summary messages
    with st.popover("Actions", use_container_width=True):
        st.button("Speak Summary", key=f"speak_summary_{index}", on_click=lambda: speak(message.content))
        st.button(
            "Follow-up Questions",
            key=f"follow_up_questions_{index}",
            on_click=lambda: get_followup_questions(message.question, message.query, my_df),
        )
        if st.button("Generate Table", key=f"table_{index}"):
            # Ensure DataFrame is converted to JSON string for the Message constructor if it expects that
            df_json_content = my_df.to_json(orient="records")
            add_message(
                Message(RoleType.ASSISTANT, df_json_content, MessageType.DATAFRAME, message.query, message.question)
            )
        st.button(
            "Generate Graph",
            key=f"graph_{index}",
            on_click=lambda: get_chart(message.question, message.query, my_df),
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
            key=f"thumbs_up_{index}",
            type="primary" if message.feedback == "up" else "secondary",
            on_click=set_feedback,
            args=(index, "up"),
        )
    with cols[1]:
        st.button(
            "👎",
            key=f"thumbs_down_{index}",
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
                st.button(
                    question_text,
                    on_click=set_question,
                    args=(question_text,),
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
    if len(st.session_state.messages) > 0 and render:
        render_message(st.session_state.messages[-1], len(st.session_state.messages) - 1)

def call_llm(my_question:str):
    response = ask_message(Message(RoleType.ASSISTANT, my_question, MessageType.SQL))
    add_message(Message(role=RoleType.ASSISTANT, content=response, type=MessageType.ERROR))

######### Sidebar settings #########
with st.sidebar.expander("Settings"):
    st.checkbox("Show SQL", key="show_sql")
    st.checkbox("Show Table", key="show_table")
    # st.checkbox("Show Plotly Code", value=False, key="show_plotly_code")
    st.checkbox("Show Chart", key="show_chart")
    st.checkbox("Show Elapsed Time", key="show_elapsed_time")
    st.checkbox("Show Question History", key="show_question_history")
    st.checkbox("Voice Input", key="voice_input")
    st.checkbox("Speak Summary", key="speak_summary")
    st.checkbox("Show Suggested Questions", key="show_suggested")
    st.checkbox("Show Follow-up Questions", key="show_followup")
    if "openai_api" in st.secrets.ai_keys and "openai_model" in st.secrets.ai_keys:
        st.checkbox("LLM Fallback on Error", key="llm_fallback")
    st.button("Save", on_click=save_user_settings, use_container_width=True)

st.sidebar.button("Reset", on_click=lambda: set_question(None), use_container_width=True, type="primary")

if st.session_state.get("voice_input", True):
    with st.sidebar.popover("🎤 Speak Your Question", use_container_width=True):
        if st.button("Listen", use_container_width=True):
            text = listen()
            if text:
                st.toast(f"Recognized text: {text}")
            else:
                st.error("No input detected.")
            if text:
                set_question(text, False)

# Show suggested questions
if st.session_state.get("show_suggested", True):
    with st.sidebar.popover("Click to show suggested questions", use_container_width=True):
        questions = vn.generate_questions()
        for i, question in enumerate(questions):
            time.sleep(0.05)
            button = st.button(
                question,
                on_click=set_question,
                args=(question, False),
                key=f"suggested_question_{i}",
                use_container_width=True,
            )

# Display questions history in sidebar
if st.session_state.get("show_question_history", True):
    with st.sidebar:
        st.title("Question History")
    filtered_messages = get_unique_messages()
    if len(filtered_messages) > 0:
        for past_question in filtered_messages:
            st.sidebar.button(
                past_question.content, on_click=set_question, args=(past_question.content,), use_container_width=True
            )
    else:
        st.sidebar.text("No questions asked yet")

# for debugging
# st.sidebar.write(st.session_state)
######### Sidebar settings #########

st.title("Thrive AI")

if st.session_state.messages == []:
    with st.chat_message(RoleType.ASSISTANT.value):
        st.markdown("Ask me a question about your data")

# Populate messages in the chat message component everytime the streamlit is run
index = 0
for message in st.session_state.messages:
    render_message(message, index)
    index = index + 1

# Always show chat input
chat_input = st.chat_input("Ask me a question about your data")

######### Handle new chat input #########
if chat_input:
    set_question(chat_input)

# Get question from session state
my_question = st.session_state.get("my_question", None)

if my_question:
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
        add_message(Message(RoleType.ASSISTANT, guardrail_sentence, MessageType.ERROR, "", my_question))
        call_llm(my_question)
        st.stop()
    if guardrail_score >= 3:
        logger.warning(
            "Ethical Guardrails triggered: Question=%s Score=%s Response=%s",
            my_question,
            guardrail_score,
            guardrail_sentence,
        )
        add_message(Message(RoleType.ASSISTANT, guardrail_sentence, MessageType.ERROR, "", my_question))
        st.stop()

    # write an acknowledgment message to
    random_acknowledgment = random.choice(acknowledgements)
    with st.chat_message(RoleType.ASSISTANT.value):
        st.write(random_acknowledgment)

    sql, elapsed_time = vn.generate_sql(question=my_question)
    st.session_state.my_question = None

    if sql:
        if vn.is_sql_valid(sql=sql):
            if st.session_state.get("show_sql", True):
                add_message(Message(RoleType.ASSISTANT, sql, MessageType.SQL, sql, my_question, None, elapsed_time))
        else:
            add_message(Message(RoleType.ASSISTANT, sql, MessageType.ERROR, sql, my_question, None, elapsed_time))
            # TODO: not sure if calling the LLM here is the correct spot or not, it seems to be necessary
            if st.session_state.get("llm_fallback", True):
                call_llm(my_question)
            st.stop()

        df = vn.run_sql(sql=sql)

        # if sql doesn't return a dataframe, stop
        if not isinstance(df, pd.DataFrame):
            st.stop()
        else:
            st.session_state["df"] = df

        if st.session_state.get("show_table", True):
            df = st.session_state.get("df")
            add_message(Message(RoleType.ASSISTANT, df, MessageType.DATAFRAME, sql, my_question))

        if st.session_state.get("show_chart", True):
            get_chart(my_question, sql, df)

        if st.session_state.get("show_summary", True) or st.session_state.get("speak_summary", True):
            summary, elapsed_time = vn.generate_summary(question=my_question, df=df)
            if summary is not None:
                if st.session_state.get("show_summary", True):
                    add_message(
                        Message(RoleType.ASSISTANT, summary, MessageType.SUMMARY, sql, my_question, df, elapsed_time)
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
            )
        )
        if st.session_state.get("llm_fallback", True):
            call_llm(my_question)
######### Handle new chat input #########
