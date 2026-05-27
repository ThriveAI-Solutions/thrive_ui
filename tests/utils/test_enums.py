import pytest
from utils.enums import MessageType


def test_tool_call_message_type_exists():
    assert MessageType.TOOL_CALL.value == "tool_call"


def test_patient_chooser_message_type_exists():
    assert MessageType.PATIENT_CHOOSER.value == "patient_chooser"
