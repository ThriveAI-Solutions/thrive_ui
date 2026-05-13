"""Pydantic AI Model factory.

Reads st.secrets and returns the Pydantic AI Model instance configured
for the deployment's provider. Same agent code, different model line.

Import notes (pydantic-ai 1.0.8):
  - OpenAIChatModel lives in pydantic_ai.models.openai (spec-compliant)
  - OpenAIProvider in pydantic_ai.providers.openai (spec-compliant)
  - AnthropicModel in pydantic_ai.models.anthropic (spec-compliant)
  - BedrockConverseModel in pydantic_ai.models.bedrock (spec-compliant)
  - Dedicated AnthropicProvider / BedrockProvider exist in their respective
    providers modules, so we use them to pass api_key / region_name cleanly
    instead of relying on env-var fallbacks.
"""

from __future__ import annotations
from typing import Any, Literal

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider


ModelProvider = Literal["ollama", "anthropic", "bedrock"]


def _read_secrets() -> dict:
    import streamlit as st

    return dict(st.secrets)


def build_model() -> Any:
    """Return a configured Pydantic AI Model for the active provider.

    Provider is determined by ``ai_keys.provider`` in secrets.toml.
    Supported values: ``ollama``, ``anthropic``, ``bedrock``.
    """
    secrets = _read_secrets()
    ai_keys = secrets.get("ai_keys", {})
    provider = ai_keys.get("provider")

    if provider == "ollama":
        host = ai_keys.get("ollama_host", "http://localhost:11434")
        model_name = ai_keys.get("ollama_model", "qwen3.6:27b")
        # Ollama's OpenAI-compat layer surfaces reasoning as ThinkingPart only
        # when `think=True` is forwarded in the request body. Non-thinking
        # models silently ignore it, so it's safe to leave on by default.
        # Two-level config so a deployment can rotate models with different
        # thinking semantics (e.g. qwen3.6 off for tool-call accuracy,
        # gpt-oss on for chain-of-thought):
        #   [agent].ollama_think                — global default (default True)
        #   [agent.ollama_think_per_model]      — map of model_name -> bool,
        #                                         overrides the global default
        agent_cfg = secrets.get("agent", {})
        per_model = agent_cfg.get("ollama_think_per_model", {}) or {}
        if model_name in per_model:
            think_enabled = bool(per_model[model_name])
        else:
            think_enabled = bool(agent_cfg.get("ollama_think", True))
        settings = OpenAIChatModelSettings(extra_body={"think": True}) if think_enabled else None
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(base_url=f"{host.rstrip('/')}/v1"),
            settings=settings,
        )

    if provider == "anthropic":
        api_key = ai_keys.get("anthropic_api_key") or ai_keys.get("anthropic_api")
        model_name = ai_keys.get("anthropic_model", "claude-sonnet-4-6")
        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(api_key=api_key),
        )

    if provider == "bedrock":
        model_id = ai_keys.get("bedrock_model_id", "anthropic.claude-sonnet-4-6-v1:0")
        region = ai_keys.get("aws_region", "us-east-1")
        return BedrockConverseModel(
            model_id,
            provider=BedrockProvider(region_name=region),
        )

    raise ValueError(f"Unknown provider {provider!r}. Set ai_keys.provider to one of: ollama, anthropic, bedrock.")
