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
