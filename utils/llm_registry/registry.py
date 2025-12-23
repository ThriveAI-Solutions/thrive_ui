"""Central registry for LLM providers."""

import logging
from functools import lru_cache
from typing import Dict
from utils.llm_registry.base import LLMProvider
from utils.llm_registry.providers.ollama import OllamaProvider
from utils.llm_registry.providers.openai import OpenAIProvider
from utils.llm_registry.models import LLMProviderConfig, LLMModelInfo

logger = logging.getLogger(__name__)


class LLMProviderRegistry:
    """Central registry for managing LLM providers.

    This singleton registry manages all available LLM providers and provides
    discovery, health checking, and model listing capabilities.
    """

    def __init__(self):
        self._providers: Dict[str, LLMProvider] = {}
        self._register_builtin_providers()

    def _register_builtin_providers(self):
        """Register built-in providers (Ollama, OpenAI)."""
        self.register(OllamaProvider())
        self.register(OpenAIProvider())
        # Future providers can be added here:
        # self.register(AnthropicProvider())
        # self.register(GeminiProvider())

    def register(self, provider: LLMProvider):
        """Register a provider in the registry.

        Args:
            provider: LLMProvider instance to register
        """
        self._providers[provider.provider_id] = provider
        logger.info(f"Registered LLM provider: {provider.display_name} ({provider.provider_id})")

    def get_provider(self, provider_id: str) -> LLMProvider | None:
        """Get a provider by its ID.

        Args:
            provider_id: Unique provider identifier (e.g., 'ollama', 'openai')

        Returns:
            Provider instance or None if not found
        """
        return self._providers.get(provider_id)

    def get_all_providers(self) -> list[LLMProvider]:
        """Get all registered providers.

        Returns:
            List of all registered provider instances
        """
        return list(self._providers.values())

    def get_available_providers(self, secrets: dict) -> list[LLMProviderConfig]:
        """Get all providers with their configuration and health status.

        Args:
            secrets: Dictionary containing ai_keys and other config from secrets.toml

        Returns:
            List of provider configurations with availability status
        """
        configs = []
        for provider in self._providers.values():
            try:
                config = provider.get_config(secrets)
                configs.append(config)
            except Exception as e:
                logger.exception(f"Error getting config for {provider.display_name}")
                # Return error config
                configs.append(
                    LLMProviderConfig(
                        provider_id=provider.provider_id,
                        display_name=provider.display_name,
                        enabled=False,
                        error_message=f"Error checking provider: {str(e)}",
                    )
                )
        return configs

    def get_models_for_provider(self, provider_id: str, secrets: dict) -> list[LLMModelInfo]:
        """Get available models for a specific provider.

        Args:
            provider_id: Unique provider identifier
            secrets: Dictionary containing ai_keys and other config

        Returns:
            List of available model information, or empty list if provider not found
        """
        provider = self.get_provider(provider_id)
        if not provider:
            logger.warning(f"Provider not found: {provider_id}")
            return []

        try:
            return provider.get_available_models(secrets)
        except Exception:
            logger.exception(f"Failed to get models for {provider_id}")
            return []


@lru_cache(maxsize=1)
def get_registry() -> LLMProviderRegistry:
    """Get the singleton registry instance.

    Uses lru_cache to ensure only one instance exists.

    Returns:
        The singleton LLMProviderRegistry instance
    """
    return LLMProviderRegistry()
