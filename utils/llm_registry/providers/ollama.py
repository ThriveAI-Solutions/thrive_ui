"""Ollama LLM provider with dynamic model discovery."""

import logging
import ollama
from utils.llm_registry.base import LLMProvider
from utils.llm_registry.models import LLMModelInfo, LLMProviderHealth

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama LLM provider with dynamic model discovery via API."""

    @property
    def provider_id(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return "Ollama"

    def is_configured(self, secrets: dict) -> bool:
        """Ollama is configured if ollama_model is in secrets or host is specified.

        Note: Ollama doesn't require API keys, just needs to be running.
        """
        ai_keys = secrets.get("ai_keys", {})
        # Consider configured if ollama_model is set OR ollama_host is set
        # (default host is localhost:11434, so host being set indicates intent to use Ollama)
        return "ollama_model" in ai_keys or "ollama_host" in ai_keys

    def get_default_model(self, secrets: dict) -> str | None:
        """Get the default Ollama model from secrets."""
        return secrets.get("ai_keys", {}).get("ollama_model")

    def health_check(self, secrets: dict) -> LLMProviderHealth:
        """Check Ollama connectivity and list available models.

        Based on the health check pattern from poc/utils.py.
        """
        ai_keys = secrets.get("ai_keys", {})
        host = ai_keys.get("ollama_host", "http://localhost:11434")

        result = LLMProviderHealth(healthy=False, provider_id=self.provider_id, error=None, available_models=[])

        try:
            client = ollama.Client(host)
            models_response = client.list()

            # Extract model names - handle both dict and object response formats
            for model in models_response.get("models", []):
                if hasattr(model, "model"):
                    # Object format
                    result.available_models.append(model.model)
                elif isinstance(model, dict) and "model" in model:
                    # Dict format
                    result.available_models.append(model["model"])

            result.healthy = len(result.available_models) > 0
            if not result.healthy:
                result.error = f"No models found on Ollama at {host}. Run 'ollama pull <model>' to download models."

        except Exception as e:
            logger.exception("Ollama health check failed")
            result.error = f"Cannot connect to Ollama at {host}: {type(e).__name__}. Is Ollama running?"

        return result

    def get_available_models(self, secrets: dict) -> list[LLMModelInfo]:
        """Dynamically discover Ollama models via API."""
        health = self.health_check(secrets)

        models = []
        for model_name in health.available_models:
            # Extract base name and tag
            parts = model_name.split(":")
            base_name = parts[0]
            tag = parts[1] if len(parts) > 1 else "latest"

            models.append(
                LLMModelInfo(
                    model_id=model_name,
                    display_name=f"{base_name} ({tag})",
                    provider_id=self.provider_id,
                    available=True,
                    tags=[tag],
                )
            )

        return sorted(models, key=lambda m: m.model_id)
