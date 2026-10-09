"""Google Gemini embeddings for asymmetric legal-document retrieval."""

from dataclasses import dataclass
import math
from numbers import Real
from os import getenv
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSION = 768


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

        load_dotenv()
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

    def __init__(
        self,
        settings: GeminiEmbeddingSettings | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or GeminiEmbeddingSettings.from_environment()
        self.dimension = self.settings.dimension
        self.model = self.settings.model.strip()
        self._client = client

    def embed_document(self, text: str, title: str | None = None) -> list[float]:
        """Embed one retrieval document using its optional case title."""

        return self._embed(format_document_for_embedding(text, title))

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
        client = self._get_client()
        try:
            response = client.models.embed_content(
                model=self.model,
                contents=prepared_text,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimension
                ),
            )
        except Exception as exc:
            raise GeminiEmbeddingError("Gemini embedding request failed") from exc

        embeddings = getattr(response, "embeddings", None)
        if not embeddings or len(embeddings) != 1:
            raise EmbeddingResponseError(
                "Gemini returned an unexpected number of embeddings"
            )
        values = getattr(embeddings[0], "values", None)
        if values is None:
            raise EmbeddingResponseError("Gemini returned no embedding values")
        if len(values) != self.dimension:
            raise EmbeddingResponseError(
                "Gemini embedding dimension does not match configured dimension"
            )

        vector: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise EmbeddingResponseError(
                    "Gemini embedding contains a non-numeric value"
                )
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise EmbeddingResponseError(
                    "Gemini embedding contains a non-finite value"
                )
            vector.append(numeric_value)
        return vector
