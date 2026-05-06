import json
from unittest.mock import MagicMock, patch
from utils.renderers.tool_call import render_tool_call_message


def test_render_passes_args_summary_to_streamlit():
    msg = MagicMock()
    msg.content = json.dumps(
        {
            "tool_name": "find_patient",
            "arguments": {"last_name": "Smith"},
            "result_summary": "match_count=3",
            "success": True,
            "elapsed_ms": 120,
        }
    )

    with patch("utils.renderers.tool_call.st") as st:
        render_tool_call_message(msg, index=0)
        st.expander.assert_called()  # used for collapsible card
