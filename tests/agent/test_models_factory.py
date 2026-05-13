import pytest
from agent.models import build_model


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
    assert model.settings == {"extra_body": {"think": True}}


def test_build_model_ollama_global_think_off(monkeypatch):
    """Disabled path must send `think: false` explicitly — qwen3 hybrid
    models default to thinking-ON when the field is absent, so omitting
    it is not equivalent to disabling."""
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: _ollama_secrets("qwen3.6:27b", {"ollama_think": False}),
    )
    model = build_model()
    assert model.settings == {"extra_body": {"think": False}}


def test_build_model_ollama_per_model_disables_thinking(monkeypatch):
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: _ollama_secrets(
            "qwen3.6:27b",
            {"ollama_think": True, "ollama_think_per_model": {"qwen3.6:27b": False}},
        ),
    )
    model = build_model()
    assert model.settings == {"extra_body": {"think": False}}


def test_build_model_ollama_per_model_overrides_global_off(monkeypatch):
    monkeypatch.setattr(
        "agent.models._read_secrets",
        lambda: _ollama_secrets(
            "gpt-oss:20b",
            {"ollama_think": False, "ollama_think_per_model": {"gpt-oss:20b": True}},
        ),
    )
    model = build_model()
    assert model.settings == {"extra_body": {"think": True}}


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
    assert model.settings == {"extra_body": {"think": False}}
