"""OpenAI LLM provider with static model list."""

import logging

from utils.llm_registry.base import LLMProvider
from utils.llm_registry.models import LLMModelInfo, LLMProviderHealth

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider with static model list."""

    # Well-known OpenAI models (static list for reliability)
    KNOWN_MODELS = [
        ("gpt-5-nano", "GPT-5 nano"),
        ("gpt-5-mini", "GPT-5 mini"),
    ]

    @property
    def provider_id(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def is_configured(self, secrets: dict) -> bool:
        """OpenAI is configured if API key is present in secrets."""
        ai_keys = secrets.get("ai_keys", {})
        return "openai_api" in ai_keys

    def get_default_model(self, secrets: dict) -> str | None:
        """Get the default OpenAI model from secrets."""
        return secrets.get("ai_keys", {}).get("openai_model")

    def health_check(self, secrets: dict) -> LLMProviderHealth:
        """Verify OpenAI API key is configured.

        Note: We don't validate the key by calling the API here to avoid
        unnecessary API calls and latency. Invalid keys will fail when used.
        """
        ai_keys = secrets.get("ai_keys", {})
        api_key = ai_keys.get("openai_api")

        result = LLMProviderHealth(
            healthy=False,
            provider_id=self.provider_id,
            error=None,
            available_models=[m[0] for m in self.KNOWN_MODELS],
        )

        if not api_key:
            result.error = "OpenAI API key not configured in secrets.toml"
            return result

        # Basic validation: API key should start with 'sk-'
        if not api_key.startswith("sk-"):
            result.error = "OpenAI API key format appears invalid (should start with 'sk-')"
            return result

        # If we have a properly formatted key, consider it healthy
        # Actual validation happens when the key is used
        result.healthy = True

        return result

    def get_available_models(self, secrets: dict) -> list[LLMModelInfo]:
        """Return static list of OpenAI models."""
        models = []
        for model_id, display_name in self.KNOWN_MODELS:
            models.append(
                LLMModelInfo(
                    model_id=model_id,
                    display_name=display_name,
                    provider_id=self.provider_id,
                    available=True,
                    tags=[],
                )
            )
        return models
