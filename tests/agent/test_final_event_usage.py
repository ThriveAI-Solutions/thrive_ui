from agent.state import FinalResponseEvent, AgentResponse


def test_final_event_carries_optional_usage():
    ev = FinalResponseEvent(
        response=AgentResponse(text="hi"), usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    )
    assert ev.usage["total_tokens"] == 5


def test_final_event_usage_defaults_none():
    ev = FinalResponseEvent(response=AgentResponse(text="hi"))
    assert ev.usage is None
