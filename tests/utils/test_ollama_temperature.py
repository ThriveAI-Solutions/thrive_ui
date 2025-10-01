import sys

import streamlit as st


def test_thriveai_ollama_uses_temperature_in_options(monkeypatch):
    # Arrange: fake ollama module and client
    calls = {"chats": []}

    class FakeClient:
        def __init__(self, host, timeout=None):
            self.host = host
            self.timeout = timeout

        def chat(self, model, messages, stream, options, keep_alive):
            calls["chats"].append({
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": options,
                "keep_alive": keep_alive,
            })
            return {"message": {"content": "ok"}}

    class FakeOllamaModule:
        Client = FakeClient

    monkeypatch.setitem(sys.modules, "ollama", FakeOllamaModule)

    # Lazy import under test to use patched module
    from utils.thriveai_ollama import ThriveAI_Ollama

    # Act: instantiate with temperature in options and call submit_prompt
    inst = ThriveAI_Ollama(config={
        "model": "llama3",
        "ollama_host": "http://localhost:11434",
        "options": {"temperature": 0.5}
    })
    prompt = [inst.system_message("sys"), inst.user_message("hello")] 
    inst.submit_prompt(prompt)

    # Assert: the options passed to chat contain the temperature
    assert len(calls["chats"]) == 1
    assert calls["chats"][0]["options"].get("temperature") == 0.5


def test_vanna_calls_injects_temperature_from_secrets(monkeypatch):
    # Arrange streamlit secrets
    monkeypatch.setattr(
        st,
        "secrets",
        {
            "ai_keys": {
                "ollama_model": "llama3",
                "ollama_temperature": 0.7,
            },
            "postgres": {"schema_name": "public", "dialect": "postgresql"},
        },
        raising=False,
    )

    captured = {"config": None}

    # No-op base initializers that would otherwise require external services
    monkeypatch.setattr("utils.vanna_calls.VannaDB_VectorStore.__init__", lambda *a, **k: None)
    monkeypatch.setattr("utils.vanna_calls.ThriveAI_ChromaDB.__init__", lambda *a, **k: None)
    monkeypatch.setattr("utils.vanna_calls.ThriveAI_Milvus.__init__", lambda *a, **k: None)
    monkeypatch.setattr("utils.vanna_calls.Ollama.__init__", lambda *a, **k: None)

    # Capture the config passed to ThriveAI_Ollama
    def _capture_cfg(self, config=None, **kwargs):
        captured["config"] = config

    monkeypatch.setattr("utils.vanna_calls.ThriveAI_Ollama.__init__", _capture_cfg)

    # Act: instantiate classes that should pass temperature via options
    from utils.vanna_calls import MyVannaOllama, MyVannaOllamaChromaDB, MyVannaOllamaMilvus

    _ = MyVannaOllama()
    assert captured["config"] is not None
    assert captured["config"].get("options", {}).get("temperature") == 0.7

    captured["config"] = None
    _ = MyVannaOllamaChromaDB(user_role=0, config={})
    assert captured["config"] is not None
    assert captured["config"].get("options", {}).get("temperature") == 0.7

    captured["config"] = None
    _ = MyVannaOllamaMilvus(user_role=0, config={})
    assert captured["config"] is not None
    assert captured["config"].get("options", {}).get("temperature") == 0.7


