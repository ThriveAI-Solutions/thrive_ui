import logging
import random
import time
import pandas as pd
import streamlit as st
from ethical_guardrails_lib import get_ethical_guideline
from utils.chat_bot_helper import (
    set_question,
    vn,
    get_unique_messages,
    render_message,
    add_message,
    call_llm,
    get_chart,
    get_followup_questions,
)
from orm.functions import get_recent_messages, save_user_settings, set_user_preferences_in_session_state
from orm.models import Message
from utils.communicate import listen, speak
from utils.enums import MessageType, RoleType
from utils.magic_functions import is_magic_do_magic

logger = logging.getLogger(__name__)

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


######### Sidebar settings #########


def save_settings_on_click():
    """Update session state with temporary settings values and save to database"""
    # Update session state with temporary values
    st.session_state.show_sql = st.session_state.get("temp_show_sql", st.session_state.show_sql)
    st.session_state.show_table = st.session_state.get("temp_show_table", st.session_state.show_table)
    st.session_state.show_chart = st.session_state.get("temp_show_chart", st.session_state.show_chart)
    st.session_state.show_elapsed_time = st.session_state.get(
        "temp_show_elapsed_time", st.session_state.show_elapsed_time
    )
    st.session_state.show_question_history = st.session_state.get(
        "temp_show_question_history", st.session_state.show_question_history
    )
    st.session_state.voice_input = st.session_state.get("temp_voice_input", st.session_state.voice_input)
    st.session_state.speak_summary = st.session_state.get("temp_speak_summary", st.session_state.speak_summary)
    st.session_state.show_suggested = st.session_state.get("temp_show_suggested", st.session_state.show_suggested)
    st.session_state.show_followup = st.session_state.get("temp_show_followup", st.session_state.show_followup)
    st.session_state.llm_fallback = st.session_state.get("temp_llm_fallback", st.session_state.llm_fallback)
    # Handle show_plotly_code even though it's not currently in the UI
    st.session_state.show_plotly_code = st.session_state.get(
        "temp_show_plotly_code", st.session_state.get("show_plotly_code", False)
    )

    # Save to database
    save_user_settings()


st.logo(image="assets/logo.png", size="medium", icon_image="assets/icon.jpg")
with st.sidebar.expander("Settings"):
    st.checkbox("Show SQL", value=st.session_state.get("show_sql", True), key="temp_show_sql")
    st.checkbox("Show Table", value=st.session_state.get("show_table", True), key="temp_show_table")
    # st.checkbox("Show Plotly Code", value=False, key="show_plotly_code")
    st.checkbox("Show Chart", value=st.session_state.get("show_chart", False), key="temp_show_chart")
    st.checkbox(
        "Show Elapsed Time", value=st.session_state.get("show_elapsed_time", True), key="temp_show_elapsed_time"
    )
    st.checkbox(
        "Show Question History",
        value=st.session_state.get("show_question_history", True),
        key="temp_show_question_history",
    )
    st.checkbox("Voice Input", value=st.session_state.get("voice_input", False), key="temp_voice_input")
    st.checkbox("Speak Summary", value=st.session_state.get("speak_summary", False), key="temp_speak_summary")
    st.checkbox(
        "Show Suggested Questions", value=st.session_state.get("show_suggested", False), key="temp_show_suggested"
    )
    st.checkbox(
        "Show Follow-up Questions", value=st.session_state.get("show_followup", False), key="temp_show_followup"
    )
    st.checkbox("LLM Fallback on Error", value=st.session_state.get("llm_fallback", False), key="temp_llm_fallback")
    st.button("Save", on_click=save_settings_on_click, use_container_width=True)

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
                past_question.content,
                on_click=set_question,
                args=(past_question.content,),
                use_container_width=True,
            )
    else:
        st.sidebar.text("No questions asked yet")

# for debugging
# st.sidebar.write(st.session_state)
######### Sidebar settings #########
# st.title("Thrive AI")

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
    magic_response = is_magic_do_magic(my_question)
    if magic_response == True:
        st.stop()

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
            logger.debug("sql is not valid")
            add_message(Message(RoleType.ASSISTANT, sql, MessageType.ERROR, sql, my_question, None, elapsed_time))
            # TODO: not sure if calling the LLM here is the correct spot or not, it seems to be necessary
            if st.session_state.get("llm_fallback", True):
                logger.debug("fallback to LLM")
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
