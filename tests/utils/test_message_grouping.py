"""Tests for message grouping functionality."""

import pytest
from unittest.mock import MagicMock, patch

from utils.chat_bot_helper import (
    group_messages_by_id,
    get_message_group_css,
    group_has_data_results,
    get_followup_command_suggestions,
    is_followup_command,
)
from utils.enums import MessageType


class MockMessage:
    """Simple mock for Message objects."""

    def __init__(self, content: str, group_id: str = None, msg_type: str = None):
        self.content = content
        self.group_id = group_id
        self.type = msg_type


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

    def test_css_includes_alternating_background(self):
        """CSS should include alternating background for visual distinction."""
        css = get_message_group_css()
        assert "nth-of-type(even)" in css
        assert "0.06" in css  # Slightly darker alternating background


class TestGroupHasDataResults:
    """Tests for the group_has_data_results function."""

    def test_empty_messages_returns_false(self):
        """Empty input should return False."""
        result = group_has_data_results([])
        assert result is False

    def test_messages_without_data_returns_false(self):
        """Messages without DATAFRAME or SUMMARY type return False."""
        msg1 = MockMessage("Hello", msg_type=MessageType.TEXT.value)
        msg2 = MockMessage("SQL", msg_type=MessageType.SQL.value)
        result = group_has_data_results([msg1, msg2])
        assert result is False

    def test_messages_with_dataframe_returns_true(self):
        """Messages containing DATAFRAME type return True."""
        msg1 = MockMessage("Question", msg_type=MessageType.TEXT.value)
        msg2 = MockMessage("Data", msg_type=MessageType.DATAFRAME.value)
        result = group_has_data_results([msg1, msg2])
        assert result is True

    def test_messages_with_summary_returns_true(self):
        """Messages containing SUMMARY type return True."""
        msg1 = MockMessage("Question", msg_type=MessageType.TEXT.value)
        msg2 = MockMessage("Summary", msg_type=MessageType.SUMMARY.value)
        result = group_has_data_results([msg1, msg2])
        assert result is True

    def test_messages_with_both_data_types_returns_true(self):
        """Messages containing both DATAFRAME and SUMMARY return True."""
        msg1 = MockMessage("Data", msg_type=MessageType.DATAFRAME.value)
        msg2 = MockMessage("Summary", msg_type=MessageType.SUMMARY.value)
        result = group_has_data_results([msg1, msg2])
        assert result is True

    def test_error_messages_return_false(self):
        """Error messages should not trigger follow-up button."""
        msg1 = MockMessage("Error", msg_type=MessageType.ERROR.value)
        result = group_has_data_results([msg1])
        assert result is False


class TestGetFollowupCommandSuggestions:
    """Tests for the get_followup_command_suggestions function."""

    def test_returns_dict_with_categories(self):
        """Should return a dict with category names as keys."""
        suggestions = get_followup_command_suggestions()
        assert isinstance(suggestions, dict)
        assert len(suggestions) > 0
        # Check that all keys are strings (category names)
        for key in suggestions.keys():
            assert isinstance(key, str)

    def test_each_category_has_commands(self):
        """Each category should have a list of command tuples."""
        suggestions = get_followup_command_suggestions()
        for category, commands in suggestions.items():
            assert isinstance(commands, list)
            assert len(commands) > 0
            for item in commands:
                assert isinstance(item, tuple)
                assert len(item) == 3  # (command, label, description)

    def test_includes_common_commands(self):
        """Should include commonly useful follow-up commands."""
        suggestions = get_followup_command_suggestions()
        # Flatten all commands from all categories
        all_commands = []
        for commands in suggestions.values():
            all_commands.extend([cmd[0] for cmd in commands])
        # Check for some expected commands
        assert "describe" in all_commands
        assert "heatmap" in all_commands
        assert "profile" in all_commands

    def test_each_suggestion_has_required_fields(self):
        """Each suggestion should have command, label, and description."""
        suggestions = get_followup_command_suggestions()
        for category, commands in suggestions.items():
            for cmd, label, description in commands:
                assert isinstance(cmd, str) and len(cmd) > 0
                assert isinstance(label, str) and len(label) > 0
                assert isinstance(description, str) and len(description) > 0

    def test_has_expected_categories(self):
        """Should have the expected category structure."""
        suggestions = get_followup_command_suggestions()
        # Check for at least some expected categories (partial match due to emojis)
        category_names = list(suggestions.keys())
        assert any("Data Exploration" in cat for cat in category_names)
        assert any("Data Quality" in cat for cat in category_names)
        assert any("Visualizations" in cat for cat in category_names)


class TestIsFollowupCommand:
    """Tests for the is_followup_command function."""

    def test_followup_command_lowercase(self):
        """Lowercase /followup should be detected."""
        assert is_followup_command("/followup describe") is True

    def test_followup_command_uppercase(self):
        """Uppercase /FOLLOWUP should be detected."""
        assert is_followup_command("/FOLLOWUP describe") is True

    def test_followup_command_mixed_case(self):
        """Mixed case /FollowUp should be detected."""
        assert is_followup_command("/FollowUp describe") is True

    def test_followup_command_with_whitespace(self):
        """Command with leading whitespace should be detected."""
        assert is_followup_command("  /followup describe") is True

    def test_followup_without_args(self):
        """Bare /followup should be detected (even if invalid command)."""
        assert is_followup_command("/followup") is True

    def test_non_followup_command(self):
        """Other commands should not be detected as followup."""
        assert is_followup_command("/help") is False
        assert is_followup_command("/describe table") is False
        assert is_followup_command("/tables") is False

    def test_regular_question(self):
        """Regular questions should not be detected as followup."""
        assert is_followup_command("What is the average age?") is False

    def test_empty_string(self):
        """Empty string should return False."""
        assert is_followup_command("") is False

    def test_none_input(self):
        """None input should return False."""
        assert is_followup_command(None) is False

    def test_followup_in_middle_of_text(self):
        """Text containing /followup but not at start should return False."""
        assert is_followup_command("Please run /followup describe") is False
