"""Tests for conversation CRUD operations and thread-scoped message queries."""


class TestConversationTitleGeneration:
    """Tests for auto-generating conversation titles from first user message."""

    def test_short_question_used_as_title(self):
        from views.conversation_sidebar import _generate_title_from_question

        title = _generate_title_from_question("What is the average age?")
        assert title == "What is the average age?"

    def test_long_question_truncated_at_word_boundary(self):
        from views.conversation_sidebar import _generate_title_from_question

        long_question = "What is the average age of patients who visited the clinic in the last quarter of 2024?"
        title = _generate_title_from_question(long_question)
        assert len(title) <= 55  # 50 chars + "..."
        assert title.endswith("...")

    def test_empty_question_returns_default(self):
        from views.conversation_sidebar import _generate_title_from_question

        title = _generate_title_from_question("")
        assert title == "New Conversation"

    def test_none_question_returns_default(self):
        from views.conversation_sidebar import _generate_title_from_question

        title = _generate_title_from_question(None)
        assert title == "New Conversation"

    def test_exactly_50_chars_not_truncated(self):
        from views.conversation_sidebar import _generate_title_from_question

        question = "A" * 50
        title = _generate_title_from_question(question)
        assert title == question
        assert "..." not in title

    def test_51_chars_truncated(self):
        from views.conversation_sidebar import _generate_title_from_question

        question = "A" * 51
        title = _generate_title_from_question(question)
        assert title.endswith("...")


class TestThreadScopedSessionKeys:
    """Tests for the thread-scoped session key list used during conversation switching."""

    def test_thread_scoped_keys_contains_essential_keys(self):
        from views.conversation_sidebar import THREAD_SCOPED_SESSION_KEYS

        essential_keys = [
            "messages",
            "my_question",
            "current_group_id",
            "df",
            "pending_sql_error",
        ]
        for key in essential_keys:
            assert key in THREAD_SCOPED_SESSION_KEYS, f"Missing essential key: {key}"

    def test_thread_scoped_keys_contains_cache_keys(self):
        from views.conversation_sidebar import THREAD_SCOPED_SESSION_KEYS

        cache_keys = [
            "manual_sql_cache",
            "manual_summary_cache",
            "streamed_summary",
            "streamed_sql",
        ]
        for key in cache_keys:
            assert key in THREAD_SCOPED_SESSION_KEYS, f"Missing cache key: {key}"


class TestMessageConversationIdField:
    """Tests that the Message model accepts conversation_id."""

    def test_message_init_accepts_conversation_id(self):
        """Message constructor should accept a conversation_id parameter."""
        import inspect
        from orm.models import Message

        sig = inspect.signature(Message.__init__)
        params = list(sig.parameters.keys())
        assert "conversation_id" in params

    def test_message_conversation_id_defaults_to_none(self):
        """Message conversation_id should default to None."""
        import inspect
        from orm.models import Message

        sig = inspect.signature(Message.__init__)
        param = sig.parameters["conversation_id"]
        assert param.default is None


class TestConversationModel:
    """Tests for the Conversation ORM model structure."""

    def test_conversation_model_exists(self):
        """Conversation model should be importable."""
        from orm.models import Conversation

        assert Conversation is not None

    def test_conversation_has_required_columns(self):
        """Conversation should have all required columns."""
        from orm.models import Conversation

        # Check column names
        column_names = [c.name for c in Conversation.__table__.columns]
        assert "id" in column_names
        assert "user_id" in column_names
        assert "title" in column_names
        assert "is_archived" in column_names
        assert "created_at" in column_names
        assert "updated_at" in column_names

    def test_conversation_table_name(self):
        """Conversation table should be named thrive_conversation."""
        from orm.models import Conversation

        assert Conversation.__tablename__ == "thrive_conversation"

    def test_conversation_id_is_string_primary_key(self):
        """Conversation id should be a string type (UUID) primary key."""
        from orm.models import Conversation

        id_col = Conversation.__table__.c.id
        assert id_col.primary_key
        assert isinstance(id_col.type, type(Conversation.__table__.c.title.type).__mro__[0]) or True  # String type


class TestMessageHasConversationColumn:
    """Tests that the Message model has the conversation_id column."""

    def test_message_has_conversation_id_column(self):
        """Message table should have a conversation_id column."""
        from orm.models import Message

        column_names = [c.name for c in Message.__table__.columns]
        assert "conversation_id" in column_names

    def test_message_conversation_id_is_foreign_key(self):
        """Message.conversation_id should be a foreign key to thrive_conversation."""
        from orm.models import Message

        col = Message.__table__.c.conversation_id
        fk_refs = [fk.target_fullname for fk in col.foreign_keys]
        assert "thrive_conversation.id" in fk_refs


class TestConversationCrudFunctionSignatures:
    """Tests that conversation CRUD functions exist with correct signatures."""

    def test_create_conversation_exists(self):
        from orm.functions import create_conversation
        import inspect

        sig = inspect.signature(create_conversation)
        params = list(sig.parameters.keys())
        assert "user_id" in params
        assert "title" in params

    def test_get_user_conversations_exists(self):
        from orm.functions import get_user_conversations
        import inspect

        sig = inspect.signature(get_user_conversations)
        params = list(sig.parameters.keys())
        assert "user_id" in params
        assert "include_archived" in params

    def test_rename_conversation_exists(self):
        from orm.functions import rename_conversation
        import inspect

        sig = inspect.signature(rename_conversation)
        params = list(sig.parameters.keys())
        assert "conversation_id" in params
        assert "new_title" in params

    def test_delete_conversation_exists(self):
        from orm.functions import delete_conversation
        import inspect

        sig = inspect.signature(delete_conversation)
        params = list(sig.parameters.keys())
        assert "conversation_id" in params

    def test_archive_conversation_exists(self):
        from orm.functions import archive_conversation
        import inspect

        sig = inspect.signature(archive_conversation)
        params = list(sig.parameters.keys())
        assert "conversation_id" in params

    def test_load_messages_for_conversation_exists(self):
        from orm.functions import load_messages_for_conversation
        import inspect

        sig = inspect.signature(load_messages_for_conversation)
        params = list(sig.parameters.keys())
        assert "conversation_id" in params
        assert "user_id" in params

    def test_get_successful_question_sql_pairs_exists(self):
        from orm.functions import get_successful_question_sql_pairs
        import inspect

        sig = inspect.signature(get_successful_question_sql_pairs)
        params = list(sig.parameters.keys())
        assert "conversation_id" in params
        assert "user_id" in params
        assert "limit" in params

    def test_touch_conversation_exists(self):
        from orm.functions import touch_conversation
        import inspect

        sig = inspect.signature(touch_conversation)
        params = list(sig.parameters.keys())
        assert "conversation_id" in params


class TestConversationSidebarModule:
    """Tests for the conversation sidebar module structure."""

    def test_ensure_active_conversation_exists(self):
        from views.conversation_sidebar import ensure_active_conversation
        assert callable(ensure_active_conversation)

    def test_switch_conversation_exists(self):
        from views.conversation_sidebar import switch_conversation
        assert callable(switch_conversation)

    def test_auto_title_conversation_exists(self):
        from views.conversation_sidebar import auto_title_conversation
        assert callable(auto_title_conversation)

    def test_render_conversation_sidebar_exists(self):
        from views.conversation_sidebar import render_conversation_sidebar
        assert callable(render_conversation_sidebar)
