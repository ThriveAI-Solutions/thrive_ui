import io
import logging
import time

import pandas as pd
import streamlit as st

from orm.functions import get_recent_messages, save_user_settings, set_user_preferences_in_session_state
from utils.chat_bot_helper import get_unique_messages, get_vn, normal_message_flow, render_message, set_question
from utils.communicate import listen, speak
from utils.enums import MessageType, RoleType, ThemeType
from utils.magic_functions import is_magic_do_magic


def get_themed_asset_path(asset_name):
    theme = st.session_state.get("user_theme", ThemeType.HEALTHELINK.value).lower()
    return f"assets/themes/{theme}/{asset_name}"


def load_community_questions(csv_file):
    """Load questions from a CSV file and store them in session state."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Validate the CSV has a 'question' column
        if 'question' not in df.columns:
            st.error("CSV must have a 'question' column")
            return

        # Get the list of questions
        questions = df['question'].dropna().tolist()

        if not questions:
            st.error("No questions found in CSV")
            return

        # Store questions in session state
        st.session_state.community_questions = questions
        st.success(f"✅ Loaded {len(questions)} community questions")

    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")


logger = logging.getLogger(__name__)

set_user_preferences_in_session_state()

# Initialize session state variables
if "messages" not in st.session_state or st.session_state.messages == []:
    st.session_state.messages = get_recent_messages()
if st.session_state.messages is None:
    st.session_state.messages = []

# Manage session state memory by limiting messages for performance
from utils.config_helper import get_max_session_messages

max_messages = get_max_session_messages()
if len(st.session_state.messages) > max_messages:
    messages_to_remove = len(st.session_state.messages) - max_messages
    st.session_state.messages = st.session_state.messages[messages_to_remove:]
    logger.info(
        f"Session startup: Trimmed {messages_to_remove} messages from session state. Kept most recent {max_messages} messages."
    )


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


st.logo(image=get_themed_asset_path("logo.png"), size="large", icon_image="assets/icon.jpg")

# Display current LLM
try:
    vn_instance = get_vn()
    if vn_instance:
        llm_name = vn_instance.get_llm_name()
        st.sidebar.info(f"🤖 **Current LLM:** {llm_name}")
except Exception:
    pass

with st.sidebar.expander("Settings"):
    st.checkbox("Show SQL", value=st.session_state.get("show_sql", True), key="temp_show_sql")
    st.checkbox("Show Table", value=st.session_state.get("show_table", True), key="temp_show_table")
    # st.checkbox("Show Plotly Code", value=False, key="show_plotly_code")
    st.checkbox("Show AI Chart", value=st.session_state.get("show_chart", False), key="temp_show_chart")
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

st.sidebar.button("Clear", on_click=lambda: set_question(None), use_container_width=True, type="primary")

# Community Engagement CSV Upload
with st.sidebar.popover("📊 Community Engagement", use_container_width=True):
    st.markdown("**Upload a CSV file with questions**")
    st.caption("CSV should have a 'question' column")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        key="community_csv_uploader",
        help="Upload a CSV file with a 'question' column"
    )

    if uploaded_file is not None:
        if st.button("Load Questions", use_container_width=True, type="primary"):
            load_community_questions(uploaded_file)
            st.rerun()

    # Show sample format
    with st.expander("View Sample Format"):
        st.code("""question
What is the average patient age?
How many patients visited last month?
What are the top diagnoses?""", language="csv")

# Display Community Questions in sidebar
if st.session_state.get("community_questions"):
    with st.sidebar:
        st.title("Community Questions")
        community_questions = st.session_state.get("community_questions", [])

        # Add "Ask All Questions" button at the top
        if st.button("🚀 Ask All Questions", use_container_width=True, type="primary", key="ask_all_community_questions"):
            st.session_state.processing_community_questions = True
            st.session_state.community_question_index = 0
            set_question(community_questions[0], False)
            st.rerun()

        # Display individual question buttons
        for idx, question in enumerate(community_questions):
            st.sidebar.button(
                question,
                on_click=set_question,
                args=(question, False),
                key=f"community_question_{idx}",
                use_container_width=True,
            )

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
        questions = get_vn().generate_questions()
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

# Populate messages in a dedicated container so we can keep a footer below
messages_container = st.container()
with messages_container:
    index = 0 
    for message in st.session_state.messages:
        render_message(message, index)
        index = index + 1

# Footer placeholder that always stays at the end
tail_placeholder = st.empty()

# Always show chat input
chat_input = st.chat_input("Ask me a question about your data")

######### Handle new chat input #########
if chat_input:
    set_question(chat_input)

# Get question from session state
my_question = st.session_state.get("my_question", None)

# If we have a pending SQL error from a prior run, render a persistent retry panel
# Gate on an actual stored error message to avoid showing stale panels
if (
    not my_question
    and st.session_state.get("pending_sql_error", False)
    and st.session_state.get("last_run_sql_error")
):
    pending_question = st.session_state.get("pending_question")
    error_msg = st.session_state.get("last_run_sql_error")
    failed_sql = st.session_state.get("last_failed_sql")
    with st.chat_message(RoleType.ASSISTANT.value):
        # Use warning with collapsible details for less intrusive error display
        st.warning("I couldn't execute the generated SQL.")
        # Collapsible error details section
        with st.expander("View error details", expanded=False):
            if error_msg:
                st.markdown(f"**Database error:** {error_msg}")
            if failed_sql:
                st.markdown("**Failed SQL:**")
                st.code(failed_sql, language="sql", line_numbers=True)
        # Action button remains outside expander for easy access
        retry_clicked = st.button("Retry", type="primary", key="retry_persist")

    if retry_clicked:
        st.session_state["use_retry_context"] = True
        st.session_state["retry_failed_sql"] = failed_sql
        st.session_state["retry_error_msg"] = error_msg
        st.session_state["my_question"] = pending_question
        # Clear the pending panel and open state
        st.session_state["pending_sql_error"] = False
        st.session_state["show_failed_sql_open"] = False
        st.rerun()
    else:
        st.stop()

if my_question:
    magic_response = is_magic_do_magic(my_question)
    if magic_response == True:
        st.stop()

    normal_message_flow(my_question)
    # normal_message_flow calls st.rerun() at the end, so code after this won't execute

# Check if we need to continue processing community questions after a rerun
# This runs when my_question is None (cleared after processing)
if not my_question and st.session_state.get("processing_community_questions", False):
    community_questions = st.session_state.get("community_questions", [])
    current_index = st.session_state.get("community_question_index", 0)

    # Move to next question
    next_index = current_index + 1

    if next_index < len(community_questions):
        # Update index and process next question
        st.session_state.community_question_index = next_index
        with st.chat_message(RoleType.ASSISTANT.value):
            st.info(f"Processing question {next_index + 1} of {len(community_questions)}...")
        time.sleep(1)  # Brief pause between questions
        set_question(community_questions[next_index], False)
        st.rerun()
    else:
        # All questions processed
        with st.chat_message(RoleType.ASSISTANT.value):
            st.success(f"✅ Completed processing all {len(community_questions)} community questions!")
        st.session_state.processing_community_questions = False
        st.session_state.community_question_index = 0

######### Handle new chat input #########
