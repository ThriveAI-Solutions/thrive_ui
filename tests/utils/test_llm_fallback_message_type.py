from unittest.mock import MagicMock, patch

from utils.chat_bot_helper import call_llm
from utils.enums import MessageType


def test_call_llm_uses_text_message_type():
    with patch("utils.chat_bot_helper.get_vn") as mock_get_vn, patch("utils.chat_bot_helper.add_message") as mock_add:
        # Mock vn.submit_prompt to return a string
        mock_vn = MagicMock()
        mock_vn.submit_prompt.return_value = "Response"
        mock_get_vn.return_value = mock_vn

        # Call
        call_llm("hello")

        # Verify add_message called with Message having TEXT type
        args, kwargs = mock_add.call_args
        message_obj = args[0]
        assert message_obj.type == MessageType.TEXT.value

