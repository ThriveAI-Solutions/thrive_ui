import json

import httpx
import pytest

from agent.models import _sanitize_null_content, build_model


def test_build_model_ollama(monkeypatch):
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: {
            "ai_keys": {"provider": "ollama", "ollama_host": "http://localhost:11434", "ollama_model": "qwen3.6:27b"}
        },
    )
    model = build_model()
    assert model is not None


def test_build_model_anthropic(monkeypatch):
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: {
            "ai_keys": {"provider": "anthropic", "anthropic_api_key": "sk-test", "anthropic_model": "claude-sonnet-4-6"}
        },
    )
    model = build_model()
    assert model is not None


def test_build_model_bedrock(monkeypatch):
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: {
            "ai_keys": {
                "provider": "bedrock",
                "bedrock_model_id": "anthropic.claude-sonnet-4-6-v1:0",
                "aws_region": "us-east-1",
            }
        },
    )
    model = build_model()
    assert model is not None


def test_build_model_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: {"ai_keys": {"provider": "unsupported"}},
    )
    with pytest.raises(ValueError, match="Unknown provider"):
        build_model()


def _ollama_secrets(model_name: str, agent_cfg: dict) -> dict:
    return {
        "ai_keys": {"provider": "ollama", "ollama_model": model_name},
        "agent": agent_cfg,
    }


def test_build_model_ollama_thinking_on_by_default(monkeypatch):
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: _ollama_secrets("qwen3.6:27b", {}),
    )
    model = build_model()
    assert model.settings == {"extra_body": {"reasoning_effort": "high"}}


def test_build_model_ollama_global_think_off(monkeypatch):
    """Disabled path must send `reasoning_effort: "none"` explicitly —
    qwen3 hybrid models default to thinking-ON when no reasoning control
    is sent, so omitting it is not equivalent to disabling."""
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: _ollama_secrets("qwen3.6:27b", {"ollama_think": False}),
    )
    model = build_model()
    assert model.settings == {"extra_body": {"reasoning_effort": "none"}}


def test_build_model_ollama_per_model_disables_thinking(monkeypatch):
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: _ollama_secrets(
            "qwen3.6:27b",
            {"ollama_think": True, "ollama_think_per_model": {"qwen3.6:27b": False}},
        ),
    )
    model = build_model()
    assert model.settings == {"extra_body": {"reasoning_effort": "none"}}


def test_build_model_ollama_per_model_overrides_global_off(monkeypatch):
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: _ollama_secrets(
            "gpt-oss:20b",
            {"ollama_think": False, "ollama_think_per_model": {"gpt-oss:20b": True}},
        ),
    )
    model = build_model()
    assert model.settings == {"extra_body": {"reasoning_effort": "high"}}


def _build_chat_request(body: dict, *, path: str = "/v1/chat/completions") -> httpx.Request:
    return httpx.Request(
        "POST",
        f"http://aillm01:11434{path}",
        content=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )


@pytest.mark.asyncio
async def test_sanitize_rewrites_null_content_to_empty_string():
    """Ollama 0.23.x rejects content:null; ensure the hook rewrites
    every null-content message in-place before the request goes out."""
    body = {
        "model": "qwen3.6:27b",
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        ],
    }
    req = _build_chat_request(body)
    await _sanitize_null_content(req)
    rewritten = json.loads(req.content)
    assert [m["content"] for m in rewritten["messages"]] == [
        "you are helpful",
        "hi",
        "",
    ]
    # content-length header must be updated so httpx doesn't truncate.
    assert int(req.headers["content-length"]) == len(req.content)


@pytest.mark.asyncio
async def test_sanitize_skips_when_no_null_content():
    """No null content => body untouched, no needless re-serialization."""
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    req = _build_chat_request(body)
    original = bytes(req.content)
    await _sanitize_null_content(req)
    assert req.content == original


@pytest.mark.asyncio
async def test_sanitize_ignores_non_chat_completion_paths():
    """Embeddings, tags, etc. must pass through unchanged."""
    body = {"messages": [{"role": "user", "content": None}]}
    req = _build_chat_request(body, path="/api/tags")
    original = bytes(req.content)
    await _sanitize_null_content(req)
    assert req.content == original


@pytest.mark.asyncio
async def test_sanitize_tolerates_non_json_body():
    """Non-JSON body must not raise — hook is purely defensive."""
    req = httpx.Request(
        "POST",
        "http://aillm01:11434/v1/chat/completions",
        content=b"not json at all",
        headers={"content-type": "text/plain"},
    )
    await _sanitize_null_content(req)
    assert req.content == b"not json at all"


def test_build_model_ollama_per_model_miss_falls_back_to_global(monkeypatch):
    """Model not present in per-model map should use the global default."""
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: _ollama_secrets(
            "gemma4:26b",
            {"ollama_think": False, "ollama_think_per_model": {"qwen3.6:27b": True}},
        ),
    )
    model = build_model()
    assert model.settings == {"extra_body": {"reasoning_effort": "none"}}
