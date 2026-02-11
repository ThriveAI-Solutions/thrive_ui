import logging
import time

import pandas as pd
import streamlit as st

from orm.functions import get_recent_messages, save_user_settings, set_user_preferences_in_session_state
from utils.chat_bot_helper import (
    get_message_group_css,
    get_unique_messages,
    get_vn,
    group_messages_by_id,
    normal_message_flow,
    render_message_group,
    set_question,
    _render_save_report_dialog,
    _render_report_history_popover,
    _render_report_trends_popover,
)
from utils.communicate import listen
from utils.enums import RoleType, ThemeType
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
        if "question" not in df.columns:
            st.error("CSV must have a 'question' column")
            return

        # Get the list of questions
        questions = df["question"].dropna().tolist()

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
if "messages" not in st.session_state or st.session_state.messages is None or st.session_state.messages == []:
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


st.logo(image=get_themed_asset_path("logo.png"), size="large", icon_image="assets/icon.jpg")

# LLM Selection UI
with st.sidebar.expander("🤖 LLM Selection", expanded=False):
    from utils.llm_registry.registry import get_registry
    import json

    registry = get_registry()
    secrets = {
        "ai_keys": dict(st.secrets["ai_keys"]),
        "rag_model": dict(st.secrets["rag_model"]),
    }

    # Get available providers
    providers = registry.get_available_providers(secrets)
    enabled_providers = [p for p in providers if p.enabled]

    if not enabled_providers:
        st.error("No LLM providers configured. Check secrets.toml")
        st.caption("Configure at least one provider in `.streamlit/secrets.toml`")
    else:
        # Provider selection
        provider_options = {p.provider_id: p.display_name for p in enabled_providers}
        current_provider = st.session_state.get("selected_llm_provider")

        # Find index of current provider, default to 0
        provider_index = 0
        if current_provider and current_provider in provider_options:
            provider_index = list(provider_options.keys()).index(current_provider)

        selected_provider_id = st.selectbox(
            "Provider",
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            index=provider_index,
            key="temp_llm_provider",
        )

        # Model selection (dynamic based on provider)
        models = registry.get_models_for_provider(selected_provider_id, secrets)

        if not models:
            st.warning(f"No models available for {provider_options[selected_provider_id]}")
            st.caption(f"Check your {selected_provider_id} configuration")
        else:
            model_options = {m.model_id: m.display_name for m in models}
            current_model = st.session_state.get("selected_llm_model")

            # Find default model index
            default_idx = 0
            if current_model and current_model in model_options:
                default_idx = list(model_options.keys()).index(current_model)
            else:
                # Use provider's default from secrets if available
                provider = registry.get_provider(selected_provider_id)
                if provider:
                    default_model = provider.get_default_model(secrets)
                    if default_model and default_model in model_options:
                        default_idx = list(model_options.keys()).index(default_model)

            selected_model_id = st.selectbox(
                "Model",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                index=default_idx,
                key="temp_llm_model",
            )

            # Apply button
            if st.button("Apply LLM Selection", type="primary", use_container_width=True):
                # Update session state
                st.session_state.selected_llm_provider = selected_provider_id
                st.session_state.selected_llm_model = selected_model_id

                # Save to database
                from orm.functions import save_user_settings

                save_user_settings()

                # Invalidate VannaService cache
                from utils.vanna_calls import VannaService
                import logging

                logger = logging.getLogger(__name__)
                user_id = st.session_state.cookies.get("user_id")
                user_role = st.session_state.get("user_role")
                if user_id and user_role is not None:
                    user_id = str(json.loads(user_id))
                    VannaService.invalidate_cache_for_user(user_id, user_role)
                else:
                    logger.warning(f"Could not invalidate cache: user_id={user_id}, user_role={user_role}")

                # Clear session state vanna instance
                if "_vn_instance" in st.session_state:
                    st.session_state._vn_instance = None

                st.success(f"✅ LLM changed to {model_options[selected_model_id]}")
                st.rerun()

# Display current LLM (read-only info)
try:
    vn_instance = get_vn()
    if vn_instance:
        llm_name = vn_instance.get_llm_name()
        st.sidebar.info(f"**Active LLM:** {llm_name}")
except Exception:
    pass

with st.sidebar.expander("Settings"):
    # Use st.form to group settings and eliminate temp_* state pattern
    with st.form("settings_form"):
        form_show_sql = st.checkbox(
            "Show SQL",
            value=st.session_state.get("show_sql", True),
            help="Display the generated SQL query for each question.",
        )
        form_show_table = st.checkbox(
            "Show Table",
            value=st.session_state.get("show_table", True),
            help="Display query results in a table format.",
        )
        form_show_chart = st.checkbox(
            "Show AI Chart",
            value=st.session_state.get("show_chart", False),
            help="Generate and display AI-powered visualizations for query results.",
        )
        form_show_elapsed_time = st.checkbox(
            "Show Elapsed Time",
            value=st.session_state.get("show_elapsed_time", True),
            help="Display how long each query took to execute.",
        )
        form_show_question_history = st.checkbox(
            "Show Question History",
            value=st.session_state.get("show_question_history", True),
            help="Display your previously asked questions in the sidebar for quick re-use.",
        )
        form_voice_input = st.checkbox(
            "Voice Input",
            value=st.session_state.get("voice_input", False),
            help="Enable microphone input to ask questions by speaking.",
        )
        form_speak_summary = st.checkbox(
            "Speak Summary",
            value=st.session_state.get("speak_summary", False),
            help="Read query result summaries aloud using text-to-speech.",
        )
        form_show_suggested = st.checkbox(
            "Show Suggested Questions",
            value=st.session_state.get("show_suggested", False),
            help="Display AI-generated question suggestions based on your data.",
        )
        form_show_followup = st.checkbox(
            "Show Follow-up Questions",
            value=st.session_state.get("show_followup", False),
            help="Display suggested follow-up questions after each query.",
        )
        form_llm_fallback = st.checkbox(
            "LLM Fallback on Error",
            value=st.session_state.get("llm_fallback", False),
            help="When SQL execution fails, use the LLM to provide a helpful response instead of showing an error.",
        )
        form_confirm_magic = st.checkbox(
            "Confirm Magic Commands",
            value=st.session_state.get("confirm_magic_commands", True),
            help="When enabled, shows a confirmation popup before executing detected magic commands.",
        )

        # Form submit button - updates all settings at once
        if st.form_submit_button("Save", use_container_width=True):
            # Update session state directly from form values
            st.session_state.show_sql = form_show_sql
            st.session_state.show_table = form_show_table
            st.session_state.show_chart = form_show_chart
            st.session_state.show_elapsed_time = form_show_elapsed_time
            st.session_state.show_question_history = form_show_question_history
            st.session_state.voice_input = form_voice_input
            st.session_state.speak_summary = form_speak_summary
            st.session_state.show_suggested = form_show_suggested
            st.session_state.show_followup = form_show_followup
            st.session_state.llm_fallback = form_llm_fallback
            st.session_state.confirm_magic_commands = form_confirm_magic

            # Save to database
            save_user_settings()
            st.toast("Settings saved!")

st.sidebar.button("Clear", on_click=lambda: set_question(None), use_container_width=True, type="primary")

# Community Engagement CSV Upload
with st.sidebar.popover("📊 Community Engagement", use_container_width=True):
    st.markdown("**Upload a CSV file with questions**")
    st.caption("CSV should have a 'question' column")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="community_csv_uploader",
        help="Upload a CSV file with a 'question' column",
    )

    if uploaded_file is not None:
        if st.button("Load Questions", use_container_width=True, type="primary"):
            load_community_questions(uploaded_file)
            st.rerun()

    # Show sample format
    with st.expander("View Sample Format"):
        st.code(
            """question
What is the average patient age?
How many patients visited last month?
What are the top diagnoses?""",
            language="csv",
        )

# Display Community Questions in sidebar
if st.session_state.get("community_questions"):
    with st.sidebar:
        st.title("Community Questions")
        community_questions = st.session_state.get("community_questions", [])

        # Add "Ask All Questions" button at the top
        if st.button(
            "🚀 Ask All Questions", use_container_width=True, type="primary", key="ask_all_community_questions"
        ):
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
            st.button(
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
    # Inject global CSS for message group styling (once at the top)
    st.markdown(get_message_group_css(), unsafe_allow_html=True)

    # Group messages by group_id for visual grouping
    message_groups = group_messages_by_id(st.session_state.messages)

    # Track the overall message index for callbacks
    message_index = 0
    total_groups = len(message_groups)
    for group_index, (group_id, group_messages) in enumerate(message_groups):
        is_last_group = group_index == total_groups - 1
        render_message_group(group_messages, group_index, message_index, is_last_group=is_last_group)
        message_index += len(group_messages)

# Render report-related dialogs if active
_render_save_report_dialog()
_render_report_history_popover()
_render_report_trends_popover()

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
if not my_question and st.session_state.get("pending_sql_error", False) and st.session_state.get("last_run_sql_error"):
    pending_question = st.session_state.get("pending_question")
    error_msg = st.session_state.get("last_run_sql_error")
    failed_sql = st.session_state.get("last_failed_sql")

    def handle_retry_click():
        """Callback to handle retry button click - sets up retry context."""
        user_feedback = st.session_state.get("retry_feedback_persist", "")
        st.session_state["use_retry_context"] = True
        st.session_state["retry_failed_sql"] = failed_sql
        st.session_state["retry_error_msg"] = error_msg
        st.session_state["retry_user_feedback"] = user_feedback if user_feedback else None
        st.session_state["my_question"] = pending_question
        st.session_state["pending_sql_error"] = False
        st.session_state["show_failed_sql_open"] = False

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
        # Feedback input for user hints
        st.text_input(
            "Optional feedback to help improve the query",
            key="retry_feedback_persist",
            placeholder="e.g., 'try using patient_id instead of id' or 'use a LEFT JOIN'",
        )
        # Action button with on_click callback - more reliable than checking return value
        st.button("Retry", type="primary", key="retry_persist", on_click=handle_retry_click)

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
