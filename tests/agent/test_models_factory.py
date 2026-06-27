import json

import pytest

from agent.models import _sanitize_null_content_body, build_model


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
    # Bedrock is a dormant provider: the AWS SDK (botocore/boto3) is an
    # optional extra (pydantic-ai-slim[bedrock]) that we don't install by
    # default. Skip rather than fail when it's absent; the import is lazy
    # in build_model so ollama/anthropic deployments don't need it.
    pytest.importorskip("botocore")
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


def test_sanitize_rewrites_null_content_to_empty_string():
    """Every message with content:null gets rewritten to empty string."""
    body = {
        "model": "qwen3.6:27b",
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        ],
    }
    out = _sanitize_null_content_body(json.dumps(body).encode())
    assert out is not None
    rewritten = json.loads(out)
    assert [m["content"] for m in rewritten["messages"]] == [
        "you are helpful",
        "hi",
        "",
    ]


def test_sanitize_returns_none_when_no_null_content():
    """No-op signal so the transport can skip Request rebuild."""
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    assert _sanitize_null_content_body(json.dumps(body).encode()) is None


def test_sanitize_returns_none_for_non_json_body():
    """Defensive: malformed body must not raise; just signal no rewrite."""
    assert _sanitize_null_content_body(b"not json at all") is None


def test_sanitize_returns_none_for_empty_body():
    """Empty body (GET, etc.) must not raise."""
    assert _sanitize_null_content_body(b"") is None


def test_sanitize_returns_none_when_messages_missing():
    """Body without a messages array — leave it alone."""
    body = {"model": "x", "input": "hi"}
    assert _sanitize_null_content_body(json.dumps(body).encode()) is None


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


# --- Issue #198: model client retry config -----------------------------------


def _ollama_underlying_openai_client(model):
    """Drill into the OpenAIChatModel returned by build_model and surface the
    AsyncOpenAI instance underneath. Keeps the retry-config tests independent
    of how pydantic-ai wires its `_client` attribute on a given version."""
    # OpenAIChatModel -> Provider -> AsyncOpenAI via `.client`
    return model.client


def test_build_model_ollama_sets_default_max_retries(monkeypatch):
    """Default raises the SDK floor from 2 to 3 — absorbs one transient
    Ollama hiccup without runaway latency. See agent/models._DEFAULT_MODEL_MAX_RETRIES."""
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: {
            "ai_keys": {"provider": "ollama", "ollama_model": "qwen3.6:27b"},
        },
    )
    model = build_model()
    client = _ollama_underlying_openai_client(model)
    assert client.max_retries == 3


def test_build_model_ollama_max_retries_overridable_via_secrets(monkeypatch):
    """Ops can override the default via `[agent].model_max_retries` without
    touching code — matches the existing `max_tool_calls` / `max_wall_clock_s` pattern."""
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: {
            "ai_keys": {"provider": "ollama", "ollama_model": "qwen3.6:27b"},
            "agent": {"model_max_retries": 5},
        },
    )
    model = build_model()
    client = _ollama_underlying_openai_client(model)
    assert client.max_retries == 5


def test_build_model_ollama_max_retries_invalid_falls_back_to_default(monkeypatch):
    """A misconfigured secret (e.g. a string) must not crash startup."""
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: {
            "ai_keys": {"provider": "ollama", "ollama_model": "qwen3.6:27b"},
            "agent": {"model_max_retries": "not a number"},
        },
    )
    model = build_model()
    assert _ollama_underlying_openai_client(model).max_retries == 3


def test_build_model_ollama_max_retries_negative_clamped_to_zero(monkeypatch):
    """Negative is meaningless; clamp to 0 (no retries) rather than letting
    the underlying SDK barf on a contract violation."""
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: {
            "ai_keys": {"provider": "ollama", "ollama_model": "qwen3.6:27b"},
            "agent": {"model_max_retries": -1},
        },
    )
    model = build_model()
    assert _ollama_underlying_openai_client(model).max_retries == 0


def test_build_model_anthropic_sets_default_max_retries(monkeypatch):
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: {
            "ai_keys": {
                "provider": "anthropic",
                "anthropic_api_key": "sk-test",
                "anthropic_model": "claude-sonnet-4-6",
            },
        },
    )
    model = build_model()
    # AnthropicModel -> Provider.client is an AsyncAnthropic instance.
    assert model.client.max_retries == 3


def test_build_model_anthropic_max_retries_overridable(monkeypatch):
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: {
            "ai_keys": {
                "provider": "anthropic",
                "anthropic_api_key": "sk-test",
                "anthropic_model": "claude-sonnet-4-6",
            },
            "agent": {"model_max_retries": 1},
        },
    )
    model = build_model()
    assert model.client.max_retries == 1
