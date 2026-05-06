from utils.chat_bot_helper import MESSAGE_RENDERERS
from utils.enums import MessageType


def test_tool_call_renderer_registered():
    assert MessageType.TOOL_CALL.value in MESSAGE_RENDERERS


def test_patient_chooser_renderer_registered():
    assert MessageType.PATIENT_CHOOSER.value in MESSAGE_RENDERERS
