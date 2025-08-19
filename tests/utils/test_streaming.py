import types

import pytest


class _DummyUnderlying:
    def system_message(self, content):
        return {"role": "system", "content": content}

    def user_message(self, content):
        return {"role": "user", "content": content}

    def stream_submit_prompt(self, prompt):
        # assert that roles are present for sanity
        assert isinstance(prompt, list)
        assert any(msg.get("role") == "system" for msg in prompt)
        assert any(msg.get("role") == "user" for msg in prompt)
        for chunk in ["Hello", " ", "world", "!"]:
            yield chunk


class _DummyVannaService:
    def __init__(self, has_stream=True):
        self.vn = _DummyUnderlying() if has_stream else None

    def submit_prompt(self, system_message, user_message):
        return "One-shot response"


@pytest.fixture(autouse=True)
def clear_session(monkeypatch):
    # Avoid relying on Streamlit session in unit tests
    import utils.chat_bot_helper as cbh

    monkeypatch.setattr(cbh, "st", types.SimpleNamespace(session_state=types.SimpleNamespace()))
    yield


def test_get_llm_stream_generator_stream(monkeypatch):
    from utils import chat_bot_helper as cbh

    # Patch get_vn to return a service whose `.vn` supports streaming
    monkeypatch.setattr(cbh, "get_vn", lambda: _DummyVannaService(has_stream=True))

    gen = cbh.get_llm_stream_generator("Test question")
    assert list(gen) == ["Hello", " ", "world", "!"]


def test_get_llm_stream_generator_fallback(monkeypatch):
    from utils import chat_bot_helper as cbh

    # Patch get_vn to return a service without `.vn` streaming capability
    # The fallback should call submit_prompt and yield a single chunk
    service = _DummyVannaService(has_stream=False)
    monkeypatch.setattr(cbh, "get_vn", lambda: service)

    gen = cbh.get_llm_stream_generator("Another question")
    chunks = list(gen)
    assert chunks == ["One-shot response"]
