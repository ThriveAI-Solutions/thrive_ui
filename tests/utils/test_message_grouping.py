"""Tests for message grouping functionality."""

import pytest
from unittest.mock import MagicMock, patch

from utils.chat_bot_helper import group_messages_by_id, get_message_group_css


class MockMessage:
    """Simple mock for Message objects."""

    def __init__(self, content: str, group_id: str = None):
        self.content = content
        self.group_id = group_id


class TestGroupMessagesByID:
    """Tests for the group_messages_by_id function."""

    def test_empty_messages_returns_empty_list(self):
        """Empty input should return empty output."""
        result = group_messages_by_id([])
        assert result == []

    def test_single_message_without_group_id(self):
        """Single message without group_id becomes its own group."""
        msg = MockMessage("Hello", group_id=None)
        result = group_messages_by_id([msg])
        assert len(result) == 1
        assert result[0] == (None, [msg])

    def test_single_message_with_group_id(self):
        """Single message with group_id becomes its own group."""
        msg = MockMessage("Hello", group_id="group-1")
        result = group_messages_by_id([msg])
        assert len(result) == 1
        assert result[0] == ("group-1", [msg])

    def test_consecutive_messages_same_group_id(self):
        """Consecutive messages with same group_id are grouped together."""
        msg1 = MockMessage("Question", group_id="group-1")
        msg2 = MockMessage("SQL", group_id="group-1")
        msg3 = MockMessage("Summary", group_id="group-1")

        result = group_messages_by_id([msg1, msg2, msg3])
        assert len(result) == 1
        assert result[0] == ("group-1", [msg1, msg2, msg3])

    def test_different_group_ids_create_separate_groups(self):
        """Messages with different group_ids create separate groups."""
        msg1 = MockMessage("Q1", group_id="group-1")
        msg2 = MockMessage("A1", group_id="group-1")
        msg3 = MockMessage("Q2", group_id="group-2")
        msg4 = MockMessage("A2", group_id="group-2")

        result = group_messages_by_id([msg1, msg2, msg3, msg4])
        assert len(result) == 2
        assert result[0] == ("group-1", [msg1, msg2])
        assert result[1] == ("group-2", [msg3, msg4])

    def test_messages_without_group_id_are_individual_groups(self):
        """Messages without group_id become individual groups."""
        msg1 = MockMessage("M1", group_id=None)
        msg2 = MockMessage("M2", group_id=None)

        result = group_messages_by_id([msg1, msg2])
        assert len(result) == 2
        assert result[0] == (None, [msg1])
        assert result[1] == (None, [msg2])

    def test_mixed_grouped_and_ungrouped_messages(self):
        """Mix of grouped and ungrouped messages."""
        msg1 = MockMessage("Old", group_id=None)  # Legacy message
        msg2 = MockMessage("Q1", group_id="group-1")
        msg3 = MockMessage("A1", group_id="group-1")
        msg4 = MockMessage("Legacy", group_id=None)  # Another legacy
        msg5 = MockMessage("Q2", group_id="group-2")

        result = group_messages_by_id([msg1, msg2, msg3, msg4, msg5])
        assert len(result) == 4
        assert result[0] == (None, [msg1])
        assert result[1] == ("group-1", [msg2, msg3])
        assert result[2] == (None, [msg4])
        assert result[3] == ("group-2", [msg5])

    def test_interleaved_groups(self):
        """Groups that appear, get interrupted, and resume."""
        # This shouldn't happen in practice, but tests edge case
        msg1 = MockMessage("Q1", group_id="group-1")
        msg2 = MockMessage("Legacy", group_id=None)
        msg3 = MockMessage("Q1-cont", group_id="group-1")  # Same group_id again

        result = group_messages_by_id([msg1, msg2, msg3])
        # Should create three separate groups since they're not consecutive
        assert len(result) == 3
        assert result[0] == ("group-1", [msg1])
        assert result[1] == (None, [msg2])
        assert result[2] == ("group-1", [msg3])


class TestGetMessageGroupCSS:
    """Tests for the get_message_group_css function."""

    def test_css_contains_border_styling(self):
        """CSS should include left border styling."""
        css = get_message_group_css()
        assert "border-left:" in css
        assert "#0b5258" in css  # Theme color

    def test_css_contains_background_color(self):
        """CSS should include background color."""
        css = get_message_group_css()
        assert "background-color:" in css
        assert "rgba" in css

    def test_css_includes_alternating_selector(self):
        """CSS should include selector for alternating backgrounds."""
        css = get_message_group_css()
        # Should have nth-of-type selector for alternating backgrounds
        assert "nth-of-type" in css

    def test_css_targets_streamlit_containers(self):
        """CSS should target Streamlit's container elements."""
        css = get_message_group_css()
        # Should target Streamlit's data-testid elements
        assert "stVerticalBlock" in css
        assert "stChatMessage" in css

    def test_css_includes_padding(self):
        """CSS should include padding for spacing."""
        css = get_message_group_css()
        assert "padding" in css

    def test_css_includes_border_radius(self):
        """CSS should include border radius for rounded corners."""
        css = get_message_group_css()
        assert "border-radius:" in css
