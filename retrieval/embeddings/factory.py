"""Environment-driven embedding-provider selection."""

from os import getenv

from retrieval.embeddings.base import EmbeddingService
from retrieval.embeddings.gemini import (
    EmbeddingConfigurationError as GeminiEmbeddingConfigurationError,
    GeminiEmbeddingService,
)
from retrieval.embeddings.local import (
    LocalEmbeddingConfigurationError,
    LocalEmbeddingService,
)
from retrieval.embeddings.openai import (
    OpenAIEmbeddingConfigurationError,
    OpenAIEmbeddingService,
)

SUPPORTED_EMBEDDING_PROVIDERS = frozenset({"gemini", "local", "openai"})


class EmbeddingProviderConfigurationError(ValueError):
    """Raised when EMBEDDING_PROVIDER names an unsupported provider."""


def configured_embedding_provider() -> str:
    """Return the normalized configured provider, preserving Gemini by default."""

    provider = getenv("EMBEDDING_PROVIDER", "gemini").strip().casefold()
    if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise EmbeddingProviderConfigurationError(
            "EMBEDDING_PROVIDER must be 'gemini', 'local', or 'openai'"
        )
    return provider


def create_embedding_service(provider: str | None = None) -> EmbeddingService:
    """Create the selected provider without making an external request."""

    selected = (provider or configured_embedding_provider()).strip().casefold()
    try:
        if selected == "local":
            return LocalEmbeddingService()
        if selected == "openai":
            return OpenAIEmbeddingService()
        if selected == "gemini":
            return GeminiEmbeddingService()
        raise EmbeddingProviderConfigurationError(
            "EMBEDDING_PROVIDER must be 'gemini', 'local', or 'openai'"
        )
    except (
        GeminiEmbeddingConfigurationError,
        LocalEmbeddingConfigurationError,
        OpenAIEmbeddingConfigurationError,
    ) as exc:
        raise EmbeddingProviderConfigurationError(
            f"{selected} embedding configuration is invalid: {exc}"
        ) from exc


def validate_embedding_service_configuration(service: EmbeddingService) -> None:
    """Validate semantic-build configuration without making a provider request."""

    provider = service.provider.strip().casefold()
    if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise EmbeddingProviderConfigurationError(
            "embedding provider must be 'gemini', 'local', or 'openai'"
        )
    if not service.model.strip():
        raise EmbeddingProviderConfigurationError(
            f"{provider} embedding model must not be empty"
        )
    if service.dimension <= 0:
        raise EmbeddingProviderConfigurationError(
            f"{provider} embedding dimension must be a positive integer"
        )

    if provider == "local":
        return

    settings = getattr(service, "settings", None)
    api_key = getattr(settings, "api_key", "")
    if not isinstance(api_key, str) or not api_key.strip():
        variable = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
        raise EmbeddingProviderConfigurationError(
            f"{variable} is required for semantic indexing"
        )
