"""Google Gemini embeddings for asymmetric legal-document retrieval."""

from dataclasses import dataclass
from os import getenv
from typing import Any

from google import genai
from google.genai import types

from retrieval.embeddings.common import (
    content_hash,
    execute_with_retries,
    is_transient_embedding_error,
    validate_embedding_values,
)

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_EMBEDDING_BATCH_SIZE = 100
DEFAULT_EMBEDDING_MAX_RETRIES = 4
DEFAULT_EMBEDDING_RETRY_BASE_SECONDS = 1.0


class GeminiEmbeddingError(RuntimeError):
    """Base error for retrieval-layer Gemini embedding failures."""


class EmbeddingConfigurationError(GeminiEmbeddingError):
    """Raised when embedding configuration is invalid."""


class MissingGeminiAPIKeyError(EmbeddingConfigurationError):
    """Raised only when an API request is attempted without a key."""


class EmbeddingResponseError(GeminiEmbeddingError):
    """Raised when Gemini returns a missing or invalid embedding."""


def format_query_for_embedding(query: str) -> str:
    """Format a query for Gemini Embedding 2 search retrieval."""

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query cannot be empty")
    return f"task: search result | query: {query.strip()}"


def format_document_for_embedding(text: str, title: str | None = None) -> str:
    """Format a document chunk without modifying its stored display text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None")
    if not text.strip():
        raise ValueError("document text cannot be empty")
    safe_title = title.strip() if title and title.strip() else "none"
    return f"title: {safe_title} | text: {text.strip()}"


@dataclass(frozen=True, slots=True)
class GeminiEmbeddingSettings:
    """Environment-backed Gemini embedding configuration."""

    api_key: str = ""
    model: str = DEFAULT_EMBEDDING_MODEL
    dimension: int = DEFAULT_EMBEDDING_DIMENSION

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str):
            raise EmbeddingConfigurationError("GEMINI_API_KEY must be a string")
        if not isinstance(self.model, str):
            raise EmbeddingConfigurationError(
                "GEMINI_EMBEDDING_MODEL must be a string"
            )
        if not self.model.strip():
            raise EmbeddingConfigurationError(
                "GEMINI_EMBEDDING_MODEL cannot be empty"
            )
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise EmbeddingConfigurationError(
                "GEMINI_EMBEDDING_DIMENSION must be an integer"
            )
        if self.dimension <= 0:
            raise EmbeddingConfigurationError(
                "GEMINI_EMBEDDING_DIMENSION must be greater than zero"
            )

    @classmethod
    def from_environment(cls) -> "GeminiEmbeddingSettings":
        """Load settings without requiring an API key at import time."""

        raw_dimension = getenv(
            "GEMINI_EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION)
        )
        try:
            dimension = int(raw_dimension)
        except ValueError as exc:
            raise EmbeddingConfigurationError(
                "GEMINI_EMBEDDING_DIMENSION must be an integer"
            ) from exc
        return cls(
            api_key=getenv("GEMINI_API_KEY", ""),
            model=getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            dimension=dimension,
        )


class GeminiEmbeddingService:
    """Generate validated Gemini embeddings through the official GenAI SDK.

    Client construction is lazy, so importing CaseLens and constructing this
    service are safe when ``GEMINI_API_KEY`` is absent. A client can be injected
    for deterministic tests without making an external request.
    """

    provider = "gemini"

    def __init__(
        self,
        settings: GeminiEmbeddingSettings | None = None,
        *,
        client: Any | None = None,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        max_retries: int = DEFAULT_EMBEDDING_MAX_RETRIES,
        retry_base_seconds: float = DEFAULT_EMBEDDING_RETRY_BASE_SECONDS,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise EmbeddingConfigurationError("batch_size must be an integer")
        if batch_size <= 0:
            raise EmbeddingConfigurationError("batch_size must be greater than zero")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise EmbeddingConfigurationError("max_retries must be an integer")
        if max_retries < 0:
            raise EmbeddingConfigurationError("max_retries cannot be negative")
        if retry_base_seconds < 0:
            raise EmbeddingConfigurationError(
                "retry_base_seconds cannot be negative"
            )
        self.settings = settings or GeminiEmbeddingSettings.from_environment()
        self.dimension = self.settings.dimension
        self.model = self.settings.model.strip()
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_base_seconds = float(retry_base_seconds)
        self._client = client
        self.request_count = 0
        self.retry_count = 0
        self.successful_embedding_count = 0
        self.failed_embedding_count = 0

    def document_content_hash(self, text: str, title: str | None = None) -> str:
        """Hash the exact provider-formatted document input for checkpointing."""

        return content_hash(format_document_for_embedding(text, title))

    def embed_document(self, text: str, title: str | None = None) -> list[float]:
        """Embed one retrieval document using its optional case title."""

        return self._embed(format_document_for_embedding(text, title))

    def embed_documents(
        self,
        documents: list[tuple[str, str | None]],
    ) -> list[list[float]]:
        """Embed one bounded document batch in a single provider request."""

        if len(documents) > self.batch_size:
            raise ValueError(
                f"Document batch exceeds configured batch_size of {self.batch_size}"
            )
        prepared = [
            format_document_for_embedding(text, title) for text, title in documents
        ]
        return self._embed_many(prepared)

    def embed_query(self, query: str) -> list[float]:
        """Embed one search query using the asymmetric retrieval format."""

        return self._embed(format_query_for_embedding(query))

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = self.settings.api_key.strip()
        if not api_key:
            raise MissingGeminiAPIKeyError(
                "GEMINI_API_KEY is required to request Gemini embeddings"
            )
        try:
            self._client = genai.Client(api_key=api_key)
        except Exception as exc:
            raise EmbeddingConfigurationError(
                "Could not initialize the Gemini embedding client"
            ) from exc
        return self._client

    def _embed(self, prepared_text: str) -> list[float]:
        return self._embed_many([prepared_text])[0]

    def _embed_many(self, prepared_texts: list[str]) -> list[list[float]]:
        if not prepared_texts:
            return []
        client = self._get_client()

        def request() -> Any:
            return client.models.embed_content(
                    model=self.model,
                    contents=(
                        prepared_texts[0]
                        if len(prepared_texts) == 1
                        else [
                            types.UserContent(
                                parts=[types.Part.from_text(text=text)]
                            )
                            for text in prepared_texts
                        ]
                    ),
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimension
                    ),
                )

        try:
            response = execute_with_retries(
                request,
                max_retries=self.max_retries,
                retry_base_seconds=self.retry_base_seconds,
                on_attempt=self._record_attempt,
                on_retry=self._record_retry,
            )
        except Exception as exc:
            self.failed_embedding_count += len(prepared_texts)
            raise GeminiEmbeddingError("Gemini embedding request failed") from exc

        embeddings = getattr(response, "embeddings", None)
        if not embeddings or len(embeddings) != len(prepared_texts):
            self.failed_embedding_count += len(prepared_texts)
            raise EmbeddingResponseError(
                "Gemini returned an unexpected number of embeddings"
            )
        try:
            vectors = [self._validate_values(embedding) for embedding in embeddings]
        except EmbeddingResponseError:
            self.failed_embedding_count += len(prepared_texts)
            raise
        self.successful_embedding_count += len(vectors)
        return vectors

    def _validate_values(self, embedding: Any) -> list[float]:
        return validate_embedding_values(
            getattr(embedding, "values", None),
            self.dimension,
            provider_name="Gemini",
            error_type=EmbeddingResponseError,
        )

    def _record_attempt(self) -> None:
        self.request_count += 1

    def _record_retry(self) -> None:
        self.retry_count += 1

    @staticmethod
    def _is_transient_error(error: Exception) -> bool:
        return is_transient_embedding_error(error)
