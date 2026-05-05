"""Conversation thread sidebar for managing multiple chat threads.

Renders the thread list, new/rename/delete/archive actions in the Streamlit sidebar.
Uses st.fragment to minimize re-renders of the main chat area.
"""

import json
import logging

import streamlit as st

from orm.functions import (
    archive_conversation,
    create_conversation,
    delete_conversation,
    get_user_conversations,
    load_messages_for_conversation,
    rename_conversation,
)

logger = logging.getLogger(__name__)

# Session state keys that are scoped to a conversation thread and must be
# cleared/reset when the user switches threads.
THREAD_SCOPED_SESSION_KEYS = [
    "messages",
    "my_question",
    "current_group_id",
    "df",
    "pending_sql_error",
    "pending_question",
    "last_run_sql_error",
    "last_failed_sql",
    "show_failed_sql_open",
    "use_retry_context",
    "retry_failed_sql",
    "retry_error_msg",
    "retry_user_feedback",
    "manual_sql_cache",
    "manual_summary_cache",
    "streamed_summary",
    "streamed_summary_for_question",
    "streamed_summary_elapsed_time",
    "streamed_sql",
    "streamed_sql_elapsed_time",
    "streamed_thinking",
    "current_sql_for_summary",
    "processing_community_questions",
    "community_question_index",
]


def _get_current_user_id() -> int | None:
    """Get the current user ID from session state cookies."""
    try:
        user_id_str = st.session_state.cookies.get("user_id")
        if user_id_str:
            return json.loads(user_id_str)
    except Exception:
        pass
    return None


def _generate_title_from_question(question: str) -> str:
    """Generate a conversation title from the first user message, truncated to ~50 chars."""
    if not question:
        return "New Conversation"
    title = question.strip()
    if len(title) > 50:
        # Truncate at word boundary
        title = title[:50].rsplit(" ", 1)[0]
        if not title:
            title = question[:50]
        title += "..."
    return title


def ensure_active_conversation() -> str | None:
    """Ensure the user has an active conversation, creating a default one if needed.

    Returns the active conversation_id, or None if no user is logged in.
    """
    user_id = _get_current_user_id()
    if user_id is None:
        return None

    conversation_id = st.session_state.get("conversation_id")

    if conversation_id:
        return conversation_id

    # Check if user has any conversations
    conversations = get_user_conversations(user_id)
    if conversations:
        # Use the most recently updated conversation
        active = conversations[0]
        st.session_state["conversation_id"] = active.id
        return active.id

    # Create a default "General" conversation for the user
    conv = create_conversation(user_id, "General")
    if conv:
        st.session_state["conversation_id"] = conv.id
        return conv.id

    return None


def switch_conversation(conversation_id: str):
    """Switch to a different conversation thread, resetting all thread-scoped state."""
    current_id = st.session_state.get("conversation_id")
    if current_id == conversation_id:
        return  # Already on this thread

    # Clear all thread-scoped session state keys
    for key in THREAD_SCOPED_SESSION_KEYS:
        if key in st.session_state:
            try:
                del st.session_state[key]
            except Exception:
                pass

    # Set the new conversation
    st.session_state["conversation_id"] = conversation_id

    # Load messages for the new conversation
    user_id = _get_current_user_id()
    if user_id:
        messages = load_messages_for_conversation(conversation_id, user_id)
        st.session_state["messages"] = messages if messages else []


def auto_title_conversation(conversation_id: str, first_question: str):
    """Auto-generate a title for a conversation from its first user message."""
    title = _generate_title_from_question(first_question)
    rename_conversation(conversation_id, title)


def render_conversation_sidebar():
    """Render the conversation management sidebar."""
    user_id = _get_current_user_id()
    if user_id is None:
        return

    active_conversation_id = st.session_state.get("conversation_id")

    with st.sidebar:
        # New Conversation button
        if st.button("New Conversation", use_container_width=True, type="primary", key="new_conversation_btn"):
            conv = create_conversation(user_id, "New Conversation")
            if conv:
                switch_conversation(conv.id)
                st.rerun()

        st.divider()

        # Get conversations list
        conversations = get_user_conversations(user_id)

        if not conversations:
            st.caption("No conversations yet. Start a new one above.")
            return

        # Render each conversation thread
        for conv in conversations:
            is_active = conv.id == active_conversation_id

            # Use a container for each conversation entry
            col1, col2 = st.columns([0.8, 0.2])

            with col1:
                # Conversation title button (clickable to switch)
                button_type = "primary" if is_active else "secondary"
                if st.button(
                    conv.title,
                    key=f"conv_select_{conv.id}",
                    use_container_width=True,
                    type=button_type,
                ):
                    if not is_active:
                        switch_conversation(conv.id)
                        st.rerun()

            with col2:
                # Actions popover for rename/archive/delete
                with st.popover("...", use_container_width=True):
                    # Rename
                    new_title = st.text_input(
                        "Rename",
                        value=conv.title,
                        key=f"rename_input_{conv.id}",
                        max_chars=255,
                    )
                    if st.button("Save", key=f"rename_save_{conv.id}", use_container_width=True):
                        if new_title and new_title != conv.title:
                            rename_conversation(conv.id, new_title)
                            st.rerun()

                    st.divider()

                    # Archive
                    if st.button("Archive", key=f"archive_{conv.id}", use_container_width=True):
                        archive_conversation(conv.id)
                        # If archiving the active conversation, switch to the next one
                        if is_active:
                            remaining = [c for c in conversations if c.id != conv.id]
                            if remaining:
                                switch_conversation(remaining[0].id)
                            else:
                                # Create a new default conversation
                                new_conv = create_conversation(user_id, "New Conversation")
                                if new_conv:
                                    switch_conversation(new_conv.id)
                        st.rerun()

                    # Delete with confirmation
                    if st.button(
                        "Delete",
                        key=f"delete_{conv.id}",
                        use_container_width=True,
                        type="primary",
                    ):
                        st.session_state[f"confirm_delete_{conv.id}"] = True

                    if st.session_state.get(f"confirm_delete_{conv.id}"):
                        st.warning("This will permanently delete this conversation and all its messages.")
                        confirm_cols = st.columns(2)
                        with confirm_cols[0]:
                            if st.button("Confirm", key=f"confirm_yes_{conv.id}", type="primary"):
                                delete_conversation(conv.id)
                                if is_active:
                                    remaining = [c for c in conversations if c.id != conv.id]
                                    if remaining:
                                        switch_conversation(remaining[0].id)
                                    else:
                                        new_conv = create_conversation(user_id, "New Conversation")
                                        if new_conv:
                                            switch_conversation(new_conv.id)
                                st.session_state.pop(f"confirm_delete_{conv.id}", None)
                                st.rerun()
                        with confirm_cols[1]:
                            if st.button("Cancel", key=f"confirm_no_{conv.id}"):
                                st.session_state.pop(f"confirm_delete_{conv.id}", None)
                                st.rerun()
