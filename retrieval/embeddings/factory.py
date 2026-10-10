"""Environment-driven embedding-provider selection."""

from os import getenv

from dotenv import load_dotenv

from retrieval.embeddings.base import EmbeddingService
from retrieval.embeddings.gemini import GeminiEmbeddingService
from retrieval.embeddings.openai import OpenAIEmbeddingService

SUPPORTED_EMBEDDING_PROVIDERS = frozenset({"gemini", "openai"})


class EmbeddingProviderConfigurationError(ValueError):
    """Raised when EMBEDDING_PROVIDER names an unsupported provider."""


def configured_embedding_provider() -> str:
    """Return the normalized configured provider, preserving Gemini by default."""

    load_dotenv()
    provider = getenv("EMBEDDING_PROVIDER", "gemini").strip().casefold()
    if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise EmbeddingProviderConfigurationError(
            "EMBEDDING_PROVIDER must be 'gemini' or 'openai'"
        )
    return provider


def create_embedding_service(provider: str | None = None) -> EmbeddingService:
    """Create the selected provider without making an external request."""

    selected = (provider or configured_embedding_provider()).strip().casefold()
    if selected == "openai":
        return OpenAIEmbeddingService()
    if selected == "gemini":
        return GeminiEmbeddingService()
    raise EmbeddingProviderConfigurationError(
        "EMBEDDING_PROVIDER must be 'gemini' or 'openai'"
    )
