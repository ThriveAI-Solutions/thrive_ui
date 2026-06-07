import pytest
from utils.enums import MessageType, ThemeType, user_selectable_themes


def test_tool_call_message_type_exists():
    assert MessageType.TOOL_CALL.value == "tool_call"


def test_patient_chooser_message_type_exists():
    assert MessageType.PATIENT_CHOOSER.value == "patient_chooser"


def test_user_selectable_themes_excludes_welltellai():
    """#108 — WellTellAI must not appear in user-facing dropdowns."""
    assert "WellTellAI" not in user_selectable_themes()


def test_user_selectable_themes_includes_all_other_themes():
    """Every ThemeType value other than WellTellAI must remain selectable."""
    expected = [t.value for t in ThemeType if t.value != "WellTellAI"]
    assert user_selectable_themes() == expected


def test_welltellai_enum_member_still_parses():
    """#108 — ThemeType.WELLTELLAI must stay in the enum so persisted DB
    rows with theme='WellTellAI' continue to round-trip through the enum."""
    assert ThemeType.WELLTELLAI.value == "WellTellAI"
    assert ThemeType("WellTellAI") is ThemeType.WELLTELLAI
