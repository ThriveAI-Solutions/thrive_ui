import time

import pandas as pd
import streamlit as st

from agent.session_reset import reset_agent_session
from orm.functions import (
    get_recent_messages,
    save_user_settings,
    set_user_preferences_in_session_state,
    update_user_preferences,
)
from utils.chat_bot_helper import (
    get_last_assistant_dataframe,
    get_message_group_css,
    get_unique_messages,
    get_vn,
    group_messages_by_id,
    normal_message_flow,
    render_friendly_error,
    render_message_group,
    set_question,
)
from utils.communicate import listen
from utils.enums import RoleType, ThemeType
from utils.magic_functions import is_magic_do_magic


def get_themed_asset_path(asset_name):
    theme = st.session_state.get("user_theme", ThemeType.THRIVEAI.value).lower()
    return f"assets/themes/{theme}/{asset_name}"


def load_community_questions(csv_file):
    """Load questions from a CSV file and store them in session state."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Validate the CSV has a 'question' column
        if "question" not in df.columns:
            render_friendly_error("CSV must have a 'question' column")
            return

        # Get the list of questions
        questions = df["question"].dropna().tolist()

        if not questions:
            render_friendly_error("No questions found in CSV")
            return

        # Store questions in session state
        st.session_state.community_questions = questions
        st.success(f"✅ Loaded {len(questions)} community questions")

    except Exception as e:
        render_friendly_error(f"Error loading CSV: {str(e)}")


from utils.quick_logger import get_logger

logger = get_logger(__name__)


def _clear_selected_patient() -> None:
    previous_source_id = st.session_state.get("selected_patient_source_id")
    try:
        if previous_source_id:
            from orm.agent_logging_functions import log_patient_selection

            log_patient_selection(
                session_id=st.session_state.get("agent_session_id", ""),
                user_id=int(st.session_state.get("user_id") or 0),
                source_id=None,
                display_name=None,
                selection_origin="clear_button",
                action="cleared",
                previous_source_id=previous_source_id,
                run_id=st.session_state.get("agent_current_run_id"),
            )
    except Exception:
        pass
    for k in (
        "selected_patient_source_id",
        "selected_patient_display_name",
        "selected_patient_dob",
        "selection_origin",
        "selected_at",
    ):
        st.session_state.pop(k, None)


def _render_selected_patient_sidebar() -> None:
    src = st.session_state.get("selected_patient_source_id")
    if not src:
        return
    name = st.session_state.get("selected_patient_display_name", "Unknown")
    dob = st.session_state.get("selected_patient_dob", "")

    with st.sidebar.container(border=True):
        st.markdown(f"📋 **{name}**" + (f"  \n_b. {dob}_" if dob else ""))
        if st.button("Clear patient", key="clear_patient_sidebar_btn", width="stretch"):
            _clear_selected_patient()
            st.rerun()


_RESET_CONFIRM_WINDOW_SECONDS = 5.0


def _render_reset_agent_sidebar() -> None:
    """Sidebar container that clears all agent-side conversation state.

    Gated on ``agentic_mode`` (Epic #171 / Feature #172): the container
    is only rendered when the user has agentic mode enabled. Legacy
    Vanna users see nothing here — clearing agent state doesn't affect
    the Vanna pipeline, so showing the button would imply an
    interaction that has no effect.

    The gate reads ``st.session_state["agentic_mode"]`` directly. This is
    populated from the persisted user preference by
    :func:`orm.functions.set_user_preferences_in_session_state`, which
    runs at the top of every script rerun. Commit-on-close semantics:
    toggling the dialog checkbox writes to session_state inside the
    dialog body, but the sidebar already rendered higher in the script
    on the toggle's rerun, so the sidebar does NOT update live while
    the dialog is open. When the dialog closes, the next rerun re-reads
    the (now-saved) DB value and the sidebar updates.

    Two-state UX:
      - Idle: a single "Reset agent" button. Clicking it arms the
        confirmation by stamping st.session_state['_pending_agent_reset_at'].
      - Armed (within 5s of arming): "Confirm reset" and "Cancel" buttons.
        Confirm calls reset_agent_session(); Cancel pops the arming flag.

    The arming flag is checked lazily on each render and on click — no
    background timers. Expired arming silently returns to Idle.
    """
    # Default to True so admins / fresh sessions with no persisted
    # preference still see the button.
    if not st.session_state.get("agentic_mode", True):
        return

    armed_at = st.session_state.get("_pending_agent_reset_at")
    now = time.time()
    is_armed = isinstance(armed_at, (int, float)) and (now - armed_at) <= _RESET_CONFIRM_WINDOW_SECONDS

    with st.sidebar.container(border=True):
        if is_armed:
            st.caption("Click again to confirm — this clears your current chat.")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button(
                    "🧹 Confirm reset",
                    key="confirm_agent_reset_btn",
                    type="primary",
                    width="stretch",
                ):
                    reset_agent_session(st.session_state)
                    st.toast("Agent session reset.")
                    st.rerun()
            with cancel_col:
                if st.button(
                    "Cancel",
                    key="cancel_agent_reset_btn",
                    width="stretch",
                ):
                    st.session_state.pop("_pending_agent_reset_at", None)
                    st.rerun()
        else:
            if st.button(
                "🧹 Reset agent",
                key="reset_agent_btn",
                help=(
                    "Clear the current chat and agent memory. Does not log you out or delete history from the database."
                ),
                width="stretch",
            ):
                st.session_state["_pending_agent_reset_at"] = now
                st.rerun()


set_user_preferences_in_session_state()

# Initialize session state variables
if "_messages_loaded" not in st.session_state:
    st.session_state.messages = get_recent_messages() or []
    st.session_state._messages_loaded = True

# Manage session state memory by limiting messages for performance
from utils.config_helper import get_max_session_messages

max_messages = get_max_session_messages()
if st.session_state.messages and len(st.session_state.messages) > max_messages:
    messages_to_remove = len(st.session_state.messages) - max_messages
    st.session_state.messages = st.session_state.messages[messages_to_remove:]
    logger.info(
        f"Session startup: Trimmed {messages_to_remove} messages from session state. Kept most recent {max_messages} messages."
    )


######### Settings dialog (#141) #########


def _render_llm_section():
    """LLM provider/model selection + Apply button (was inline sidebar expander)."""
    import json

    from utils.llm_registry.registry import get_registry

    registry = get_registry()
    secrets = {
        "ai_keys": dict(st.secrets["ai_keys"]),
        "rag_model": dict(st.secrets["rag_model"]),
    }

    providers = registry.get_available_providers(secrets)
    enabled_providers = [p for p in providers if p.enabled]

    if not enabled_providers:
        st.error("No LLM providers configured. Check secrets.toml")
        st.caption("Configure at least one provider in `.streamlit/secrets.toml`")
        return

    provider_options = {p.provider_id: p.display_name for p in enabled_providers}
    current_provider = st.session_state.get("selected_llm_provider")

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

    models = registry.get_models_for_provider(selected_provider_id, secrets)

    if not models:
        st.warning(f"No models available for {provider_options[selected_provider_id]}")
        st.caption(f"Check your {selected_provider_id} configuration")
        return

    model_options = {m.model_id: m.display_name for m in models}
    current_model = st.session_state.get("selected_llm_model")

    default_idx = 0
    if current_model and current_model in model_options:
        default_idx = list(model_options.keys()).index(current_model)
    else:
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

    if st.button("Apply LLM Selection", type="primary", width="stretch"):
        st.session_state.selected_llm_provider = selected_provider_id
        st.session_state.selected_llm_model = selected_model_id

        from orm.functions import save_user_settings

        save_user_settings()

        from utils.vanna_calls import VannaService

        user_id = st.session_state.cookies.get("user_id")
        user_role = st.session_state.get("user_role")
        if user_id and user_role is not None:
            user_id = str(json.loads(user_id))
            VannaService.invalidate_cache_for_user(user_id, user_role)
        else:
            logger.warning(f"Could not invalidate cache: user_id={user_id}, user_role={user_role}")

        if "_vn_instance" in st.session_state:
            st.session_state._vn_instance = None

        st.success(f"✅ LLM changed to {model_options[selected_model_id]}")
        st.rerun()


def _render_display_form():
    """Display preferences form (was inline sidebar expander).

    Extends the previous 11 checkboxes with `Show Community Engagement`
    (default OFF) — the persistence column is added by #142.
    """
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
        form_show_community_engagement = st.checkbox(
            "Show Community Engagement",
            value=st.session_state.get("show_community_engagement", False),
            help="Reveal the 📊 Community Engagement bulk CSV uploader in the sidebar. Power-user feature.",
        )

        if st.form_submit_button("Save", width="stretch"):
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
            st.session_state.show_community_engagement = form_show_community_engagement

            save_user_settings()
            st.toast("Settings saved!")


def _render_agentic_section():
    """Agentic mode toggle (was top-level sidebar checkbox)."""
    agentic_mode_initial = bool(
        getattr(st.session_state.get("user"), "agentic_mode", True)
        if st.session_state.get("user")
        else st.session_state.get("agentic_mode", True)
    )
    agentic_mode = st.checkbox(
        "Agentic mode (beta)",
        value=agentic_mode_initial,
        help="Use the new tool-driven agent for clinical questions. Vanna remains for users with this off.",
        key="agentic_mode_dialog_checkbox",
    )
    st.session_state.agentic_mode = agentic_mode
    if agentic_mode != agentic_mode_initial:
        try:
            import json as _json

            _uid_str = st.session_state.cookies.get("user_id")
            if _uid_str:
                _uid = _json.loads(_uid_str)
                update_user_preferences(user_id=_uid, agentic_mode=agentic_mode)
        except Exception:
            pass


def _settings_dialog_body():
    """Pure body of the Settings dialog. Split from the ``@st.dialog``-
    decorated wrapper (Epic #171 / Feature #172) so unit tests can drive
    the section-order logic without a Streamlit script context.
    Production unchanged — :func:`settings_dialog` delegates here.
    """
    try:
        vn_instance = get_vn()
        if vn_instance:
            st.markdown(f"**Active LLM:** {vn_instance.get_llm_name()}")
    except Exception:
        pass

    st.subheader("LLM")
    _render_llm_section()

    st.divider()

    st.subheader("Agentic")
    _render_agentic_section()

    st.divider()

    st.subheader("Display")
    _render_display_form()

    st.divider()
    if st.button("Close", key="settings_dialog_close"):
        st.rerun()


@st.dialog("Settings")
def settings_dialog():
    _settings_dialog_body()


######### Sidebar settings #########


st.logo(image=get_themed_asset_path("logo.png"), size="large", icon_image="assets/icon.jpg")

_render_selected_patient_sidebar()
_render_reset_agent_sidebar()


if st.sidebar.button("⚙️ Settings…", key="open_settings_dialog", width="stretch"):
    settings_dialog()

st.sidebar.button("Clear", on_click=lambda: set_question(None), width="stretch", type="primary")

# Community Engagement CSV Upload — gated behind the new show_community_engagement preference (#142).
if st.session_state.get("show_community_engagement", False):
    with st.sidebar.popover("📊 Community Engagement", width="stretch"):
        st.markdown("**Upload a CSV file with questions**")
        st.caption("CSV should have a 'question' column")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            key="community_csv_uploader",
            help="Upload a CSV file with a 'question' column",
        )

        if uploaded_file is not None:
            if st.button("Load Questions", width="stretch", type="primary"):
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
        if st.button("🚀 Ask All Questions", width="stretch", type="primary", key="ask_all_community_questions"):
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
                width="stretch",
            )

# Question History accordion — capped at the 10 most recent, with a View all in audit → deep-link
# into Epic #133's Audit Trail tab (graceful degrade if the tab isn't available).
if st.session_state.get("show_question_history", True):
    filtered_messages = get_unique_messages()
    total = len(filtered_messages)
    with st.sidebar.expander(f"Question History ({total})", expanded=False):
        if total == 0:
            st.caption("No questions asked yet.")
        else:
            for past_question in filtered_messages[:10]:
                st.button(
                    past_question.content,
                    on_click=set_question,
                    args=(past_question.content,),
                    key=f"qh_{past_question.id}",
                    width="stretch",
                )
            try:
                from views import admin_audit  # noqa: F401

                if st.button("View all in audit →", key="qh_view_all_in_audit"):
                    import json as _json

                    _uid_str = st.session_state.cookies.get("user_id")
                    if _uid_str:
                        st.session_state["audit_trail_pref_user_id"] = int(_json.loads(_uid_str))
                    st.switch_page("views/admin.py")
            except ImportError:
                pass

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

# Footer placeholder that always stays at the end
tail_placeholder = st.empty()

# Always show chat input — with optional 🎤 and 💡 icon-popovers above it (#143).
tool_col_voice, tool_col_suggested, _input_col = st.columns([1, 1, 20])
if st.session_state.get("voice_input", False):
    with tool_col_voice:
        with st.popover("🎤", help="Speak your question"):
            if st.button("Listen", width="stretch", key="voice_listen_btn"):
                text = listen()
                if text:
                    st.toast(f"Recognized text: {text}")
                    set_question(text, False)
                else:
                    st.error("No input detected.")
if st.session_state.get("show_suggested", False):
    with tool_col_suggested:
        with st.popover("💡", help="Show suggested questions"):
            questions = get_vn().generate_questions()
            for i, question in enumerate(questions):
                st.button(
                    question,
                    on_click=set_question,
                    args=(question, False),
                    key=f"suggested_question_chat_{i}",
                    width="stretch",
                )

chat_input = st.chat_input("Ask me a question about your data")

######### Handle new chat input #########
if chat_input:
    set_question(chat_input)

# Get question from session state
my_question = st.session_state.get("my_question", None)

# Note: pending SQL errors render in-line as MessageType.ERROR via _render_error,
# which embeds the Retry button when the message matches the active failed SQL.
# No persistent panel / st.stop() needed — users can also just type a new question.

if my_question:
    # Surface the previous result for follow-up magic commands (`head 3`,
    # `distribution glucose`, …). Agentic mode keeps its session-state df;
    # legacy mode pulls the latest assistant message's dataframe from history.
    if st.session_state.get("agentic_mode", False):
        previous_df = st.session_state.get("df")
    else:
        previous_df = get_last_assistant_dataframe()
    magic_response = is_magic_do_magic(my_question, previous_df=previous_df)
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
