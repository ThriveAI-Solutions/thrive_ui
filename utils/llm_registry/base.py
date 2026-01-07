"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from utils.llm_registry.models import LLMProviderConfig, LLMModelInfo, LLMProviderHealth


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses implement specific providers (Ollama, OpenAI, Anthropic, etc.)
    with their own model discovery and health check logic.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for this provider (e.g., 'ollama', 'openai')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for this provider (e.g., 'Ollama', 'OpenAI')."""
        pass

    @abstractmethod
    def is_configured(self, secrets: dict) -> bool:
        """Check if required secrets/configuration are present for this provider.

        Args:
            secrets: Dictionary containing ai_keys and other config from secrets.toml

        Returns:
            True if provider is configured, False otherwise
        """
        pass

    @abstractmethod
    def get_available_models(self, secrets: dict) -> list[LLMModelInfo]:
        """Discover available models for this provider.

        For dynamic providers (Ollama), queries the API.
        For static providers (OpenAI, Anthropic), returns predefined list.

        Args:
            secrets: Dictionary containing ai_keys and other config

        Returns:
            List of available model information
        """
        pass

    @abstractmethod
    def health_check(self, secrets: dict) -> LLMProviderHealth:
        """Validate provider connectivity and configuration.

        Args:
            secrets: Dictionary containing ai_keys and other config

        Returns:
            Health check result with status and available models
        """
        pass

    @abstractmethod
    def get_default_model(self, secrets: dict) -> str | None:
        """Get the default model from secrets if configured.

        Args:
            secrets: Dictionary containing ai_keys and other config

        Returns:
            Default model ID from secrets, or None if not configured
        """
        pass

    def get_config(self, secrets: dict) -> LLMProviderConfig:
        """Get provider configuration including health status.

        This is a concrete method that uses the abstract methods above.

        Args:
            secrets: Dictionary containing ai_keys and other config

        Returns:
            Provider configuration with availability status
        """
        is_configured = self.is_configured(secrets)
        health = None
        error_message = None

        if is_configured:
            health = self.health_check(secrets)
            if not health.healthy:
                error_message = health.error
        else:
            error_message = f"{self.display_name} not configured in secrets.toml"

        return LLMProviderConfig(
            provider_id=self.provider_id,
            display_name=self.display_name,
            enabled=is_configured and (health.healthy if health else False),
            error_message=error_message,
        )
