"""Pydantic models for LLM registry configuration and metadata."""

from pydantic import BaseModel, Field


class LLMProviderConfig(BaseModel):
    """Configuration status for an LLM provider."""

    provider_id: str
    display_name: str
    enabled: bool = False
    error_message: str | None = None


class LLMModelInfo(BaseModel):
    """Information about a specific LLM model."""

    model_id: str
    display_name: str
    provider_id: str
    available: bool = True
    tags: list[str] = Field(default_factory=list)


class LLMProviderHealth(BaseModel):
    """Health check result for a provider."""

    healthy: bool
    provider_id: str
    error: str | None = None
    available_models: list[str] = Field(default_factory=list)
