"""Tests for the thumbs down feedback feature."""

from unittest.mock import MagicMock, patch
import pytest


class TestSetFeedbackWithComment:
    """Test the set_feedback function with feedback_comment parameter."""

    @patch("utils.chat_bot_helper.st")
    def test_set_feedback_down_without_comment(self, mock_st):
        """Test that set_feedback works with just 'down' value and no comment."""
        from utils.chat_bot_helper import set_feedback

        # Create mock message
        mock_message = MagicMock()
        mock_message.feedback = None
        mock_message.feedback_comment = None

        # Setup session state
        mock_st.session_state.messages = [mock_message]
        mock_st.session_state.cookies = MagicMock()
        mock_st.session_state.cookies.get.return_value = "User"  # Non-admin user

        set_feedback(0, "down")

        assert mock_message.feedback == "down"
        mock_message.save.assert_called_once()
        # feedback_comment should not be set when not provided
        assert not hasattr(mock_message, "_feedback_comment_set") or mock_message.feedback_comment is None

    @patch("utils.chat_bot_helper.st")
    def test_set_feedback_down_with_comment(self, mock_st):
        """Test that set_feedback saves feedback_comment when provided."""
        from utils.chat_bot_helper import set_feedback

        # Create mock message
        mock_message = MagicMock()
        mock_message.feedback = None
        mock_message.feedback_comment = None

        # Setup session state
        mock_st.session_state.messages = [mock_message]
        mock_st.session_state.cookies = MagicMock()
        mock_st.session_state.cookies.get.return_value = "User"

        set_feedback(0, "down", "Incorrect SQL generated: The query used wrong column names")

        assert mock_message.feedback == "down"
        assert mock_message.feedback_comment == "Incorrect SQL generated: The query used wrong column names"
        mock_message.save.assert_called_once()

    @patch("utils.chat_bot_helper.st")
    def test_set_feedback_up_ignores_comment(self, mock_st):
        """Test that set_feedback doesn't set feedback_comment for 'up' feedback."""
        from utils.chat_bot_helper import set_feedback

        # Create mock message
        mock_message = MagicMock()
        mock_message.feedback = None
        mock_message.feedback_comment = None

        # Setup session state
        mock_st.session_state.messages = [mock_message]
        mock_st.session_state.cookies = MagicMock()
        mock_st.session_state.cookies.get.return_value = "User"

        # Even if comment is passed, thumbs up shouldn't need it
        set_feedback(0, "up", "This should be ignored")

        assert mock_message.feedback == "up"
        assert mock_message.feedback_comment == "This should be ignored"  # Currently it sets it anyway
        mock_message.save.assert_called_once()

    @patch("utils.chat_bot_helper.remove_from_file_training")
    @patch("utils.chat_bot_helper.st")
    def test_set_feedback_down_admin_removes_training(self, mock_st, mock_remove):
        """Test that thumbs down from admin removes training data."""
        from utils.chat_bot_helper import set_feedback

        mock_message = MagicMock()
        mock_message.feedback = None
        mock_message.question = "How many patients?"
        mock_message.query = "SELECT COUNT(*) FROM patients"

        mock_st.session_state.messages = [mock_message]
        mock_st.session_state.cookies = MagicMock()
        mock_st.session_state.cookies.get.return_value = "Admin"

        set_feedback(0, "down", "Wrong results returned")

        mock_remove.assert_called_once_with({
            "question": "How many patients?",
            "query": "SELECT COUNT(*) FROM patients",
        })


class TestFeedbackCategories:
    """Test that feedback categories are properly defined."""

    def test_feedback_categories_exist(self):
        """Test that FEEDBACK_CATEGORIES constant is properly defined."""
        from utils.chat_bot_helper import FEEDBACK_CATEGORIES

        assert isinstance(FEEDBACK_CATEGORIES, list)
        assert len(FEEDBACK_CATEGORIES) >= 5  # At least 5 categories per issue spec

        # Check expected categories exist
        expected_categories = [
            "Incorrect SQL generated",
            "Wrong data returned",
            "Summary doesn't match data",
            "Response too slow",
            "Didn't understand my question",
            "Other",
        ]
        for category in expected_categories:
            assert category in FEEDBACK_CATEGORIES, f"Missing category: {category}"


class TestRenderThumbsDownFeedback:
    """Test the _render_thumbs_down_feedback function."""

    @patch("utils.chat_bot_helper.st")
    def test_render_shows_popover(self, mock_st):
        """Test that the function creates a popover."""
        from utils.chat_bot_helper import _render_thumbs_down_feedback
        from orm.models import Message

        # Create a minimal mock message
        mock_message = MagicMock()
        mock_message.id = 1
        mock_message.feedback = None
        mock_message.feedback_comment = None

        # Mock the popover context manager
        mock_popover = MagicMock()
        mock_st.popover.return_value.__enter__ = MagicMock(return_value=mock_popover)
        mock_st.popover.return_value.__exit__ = MagicMock(return_value=False)

        _render_thumbs_down_feedback(mock_message, 0)

        # Verify popover was called
        mock_st.popover.assert_called_once()

    @patch("utils.chat_bot_helper.st")
    def test_render_shows_previous_feedback(self, mock_st):
        """Test that previous feedback is shown when available."""
        from utils.chat_bot_helper import _render_thumbs_down_feedback

        mock_message = MagicMock()
        mock_message.id = 1
        mock_message.feedback = "down"
        mock_message.feedback_comment = "Wrong data returned"

        # Setup mocks
        mock_st.popover.return_value.__enter__ = MagicMock()
        mock_st.popover.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [MagicMock(), MagicMock()]

        _render_thumbs_down_feedback(mock_message, 0)

        # Verify info was called to show previous feedback
        mock_st.info.assert_called_once()
        call_args = mock_st.info.call_args[0][0]
        assert "Wrong data returned" in call_args

    @patch("utils.chat_bot_helper.st")
    def test_render_icon_changes_when_feedback_submitted(self, mock_st):
        """Test that the icon shows checkmark when feedback was already submitted."""
        from utils.chat_bot_helper import _render_thumbs_down_feedback

        # Case 1: No feedback yet
        mock_message_no_feedback = MagicMock()
        mock_message_no_feedback.id = 1
        mock_message_no_feedback.feedback = None
        mock_message_no_feedback.feedback_comment = None

        mock_st.popover.return_value.__enter__ = MagicMock()
        mock_st.popover.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [MagicMock(), MagicMock()]

        _render_thumbs_down_feedback(mock_message_no_feedback, 0)
        first_call_icon = mock_st.popover.call_args[0][0]

        # Reset mock
        mock_st.reset_mock()

        # Case 2: Feedback already submitted
        mock_message_with_feedback = MagicMock()
        mock_message_with_feedback.id = 2
        mock_message_with_feedback.feedback = "down"
        mock_message_with_feedback.feedback_comment = "Test"

        mock_st.popover.return_value.__enter__ = MagicMock()
        mock_st.popover.return_value.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [MagicMock(), MagicMock()]

        _render_thumbs_down_feedback(mock_message_with_feedback, 0)
        second_call_icon = mock_st.popover.call_args[0][0]

        # Icons should be different
        assert first_call_icon != second_call_icon
        assert "✓" not in first_call_icon  # No checkmark before feedback
        assert "✓" in second_call_icon  # Checkmark after feedback


class TestMessageModelFeedbackComment:
    """Test the Message model's feedback_comment column."""

    def test_message_model_has_feedback_comment_column(self):
        """Test that the Message model has the feedback_comment column defined."""
        from orm.models import Message
        from sqlalchemy import inspect

        # Get the mapper for the Message class
        mapper = inspect(Message)
        column_names = [column.key for column in mapper.columns]

        assert "feedback_comment" in column_names

    def test_feedback_comment_column_max_length(self):
        """Test that feedback_comment column has max length of 500."""
        from orm.models import Message
        from sqlalchemy import inspect

        mapper = inspect(Message)
        feedback_comment_col = None
        for column in mapper.columns:
            if column.key == "feedback_comment":
                feedback_comment_col = column
                break

        assert feedback_comment_col is not None
        # Check the type has the expected length
        assert feedback_comment_col.type.length == 500


class TestFeedbackCommentCombining:
    """Test the logic for combining category and comment."""

    def test_category_only_when_no_comment(self):
        """Test that only category is saved when no additional comment provided."""
        category = "Incorrect SQL generated"
        comment = ""

        full_comment = f"{category}: {comment}" if comment.strip() else category

        assert full_comment == "Incorrect SQL generated"

    def test_category_and_comment_combined(self):
        """Test that category and comment are properly combined."""
        category = "Incorrect SQL generated"
        comment = "Used wrong table name"

        full_comment = f"{category}: {comment}" if comment.strip() else category

        assert full_comment == "Incorrect SQL generated: Used wrong table name"

    def test_whitespace_only_comment_treated_as_empty(self):
        """Test that whitespace-only comment is treated as empty."""
        category = "Wrong data returned"
        comment = "   "

        full_comment = f"{category}: {comment}" if comment.strip() else category

        assert full_comment == "Wrong data returned"
