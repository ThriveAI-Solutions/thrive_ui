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

    def generate_summary(self, question, df):
        return (f"Summary for: {question}", 0.42)


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


def test_summary_cache_short_circuits_streaming(monkeypatch):
    import types

    import pandas as pd

    from utils import chat_bot_helper as cbh

    # Prepare a fake DF
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    # Patch get_vn to our dummy service
    service = _DummyVannaService(has_stream=True)
    monkeypatch.setattr(cbh, "get_vn", lambda: service)

    # Patch st and session_state
    ss = types.SimpleNamespace()
    ss.session_state = types.SimpleNamespace()
    ss.session_state.manual_summary_cache = {}
    monkeypatch.setattr(cbh, "st", ss)

    # First call should populate cache via fallback/stream (since no summary was cached yet)
    gen1 = cbh.get_summary_stream_generator("Q1", df)
    out1 = "".join(list(gen1))
    assert "Summary for: Q1" in out1
    key = cbh.create_summary_cache_key("Q1", df)
    assert key in ss.session_state.manual_summary_cache

    # Now simulate streaming path and ensure we short-circuit to cached version
    # We replace the service with a version whose stream would yield a different text,
    # but cache should be used instead
    underlying = _DummyUnderlying()
    underlying.stream_submit_prompt = lambda prompt: iter(["SHOULD_NOT_BE_USED"])
    service2 = types.SimpleNamespace(vn=underlying)
    monkeypatch.setattr(cbh, "get_vn", lambda: service2)

    gen2 = cbh.get_summary_event_stream("Q1", df, think=True)
    # The cached short-circuit returns a noop generator; ensure it doesn't produce streamed tokens
    assert list(gen2) == []
    # And session state should still reflect the cached summary
    assert ss.session_state.streamed_summary == ss.session_state.manual_summary_cache[key][0]


def test_stream_summary_wrap_persists_cache(monkeypatch):
    import types

    import pandas as pd

    from utils import chat_bot_helper as cbh

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    class _Underlying:
        def system_message(self, c):
            return {"role": "system", "content": c}

        def user_message(self, c):
            return {"role": "user", "content": c}

        def chat(self, **kwargs):
            # Simulate two chunks and ensure we set final session state afterward
            yield {"message": {"content": "part1"}}
            yield {"message": {"content": "part2"}}

    class _Svc:
        def __init__(self):
            # Provide required helpers for prompt build
            underlying = types.SimpleNamespace(
                ollama_client=_Underlying(),
                model="m",
                ollama_options={},
                keep_alive=None,
                system_message=lambda c: {"role": "system", "content": c},
                user_message=lambda c: {"role": "user", "content": c},
            )
            self.vn = underlying

        def stream_generate_summary(self, question, df):
            # Use the actual helper path: rely on vn in cbh
            underlying = self.vn
            prompt = [underlying.system_message("s"), underlying.user_message("u")]
            start = []
            for ev in underlying.ollama_client.chat(
                model="m", messages=prompt, stream=True, options={}, keep_alive=None, think=False
            ):
                msg = ev.get("message", {})
                c = msg.get("content")
                if c:
                    start.append(c)
                    yield c
            # Simulate that the service sets streamed values at end like real implementation
            cbh.st.session_state.streamed_summary = "".join(start)
            cbh.st.session_state.streamed_summary_elapsed_time = 0.5

    svc = _Svc()
    monkeypatch.setattr(cbh, "get_vn", lambda: svc)
    ss = types.SimpleNamespace()
    ss.session_state = types.SimpleNamespace()
    monkeypatch.setattr(cbh, "st", ss)

    gen = cbh.get_summary_stream_generator("Q2", df)
    _ = "".join(list(gen))

    # The wrapper updates cache on finally, so ensure access via getattr/styled set
    key = cbh.create_summary_cache_key("Q2", df)
    cache = getattr(ss.session_state, "manual_summary_cache", {})
    assert key in cache


def test_summary_event_stream_non_streaming_fallback(monkeypatch):
    import types

    import pandas as pd

    from utils import chat_bot_helper as cbh
    from utils import vanna_calls as vc

    # Prepare a fake DF
    df = pd.DataFrame({"a": [1, 2]})

    # Build a VannaService with no streaming client but working generate_summary
    class _UnderlyingNoStream:
        def system_message(self, c):
            return {"role": "system", "content": c}

        def user_message(self, c):
            return {"role": "user", "content": c}

        # No ollama_client present → triggers fallback
        def generate_summary(self, question, df):
            return (f"Sum: {question}", 0.1)

    class _Svc:
        def __init__(self):
            self.vn = _UnderlyingNoStream()

    # Patch get_vn to return our service
    service = _Svc()
    monkeypatch.setattr(cbh, "get_vn", lambda: service)

    # Monkeypatch st in vanna_calls for session_state storage
    ss = types.SimpleNamespace()
    ss.session_state = types.SimpleNamespace()
    monkeypatch.setattr(vc, "st", ss)

    # Invoke the service event stream directly to validate fallback
    gen = vc.VannaService.summary_event_stream(service, "Q-ns", df, think=False)
    chunks = list(gen)

    # Should yield exactly one content chunk with the full summary
    assert chunks == [("content", "Sum: Q-ns")]
    # And session_state should reflect final summary text
    assert getattr(ss.session_state, "streamed_summary", "") == "Sum: Q-ns"
