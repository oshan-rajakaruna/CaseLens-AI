"""OpenAI embeddings for CaseLens semantic legal-document retrieval."""

from dataclasses import dataclass
from os import getenv
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from retrieval.embeddings.common import (
    content_hash,
    execute_with_retries,
    validate_embedding_values,
)

DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_EMBEDDING_DIMENSION = 1536
DEFAULT_OPENAI_EMBEDDING_BATCH_SIZE = 100
DEFAULT_OPENAI_EMBEDDING_MAX_RETRIES = 4
DEFAULT_OPENAI_EMBEDDING_RETRY_BASE_SECONDS = 1.0
_NATIVE_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class OpenAIEmbeddingError(RuntimeError):
    """Base error for retrieval-layer OpenAI embedding failures."""


class OpenAIEmbeddingConfigurationError(OpenAIEmbeddingError):
    """Raised when OpenAI embedding configuration is invalid."""


class MissingOpenAIAPIKeyError(OpenAIEmbeddingConfigurationError):
    """Raised only when an OpenAI request is attempted without a key."""


class OpenAIEmbeddingResponseError(OpenAIEmbeddingError):
    """Raised when OpenAI returns missing, unordered, or invalid vectors."""


def format_openai_document(text: str, title: str | None = None) -> str:
    """Format a document chunk while leaving stored display text unchanged."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None")
    if not text.strip():
        raise ValueError("document text cannot be empty")
    safe_title = title.strip() if title and title.strip() else "none"
    return f"title: {safe_title} | text: {text.strip()}"


def format_openai_query(query: str) -> str:
    """Validate and normalize a semantic query for OpenAI embeddings."""

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query cannot be empty")
    return query.strip()


@dataclass(frozen=True, slots=True)
class OpenAIEmbeddingSettings:
    """Environment-backed OpenAI embedding configuration."""

    api_key: str = ""
    model: str = DEFAULT_OPENAI_EMBEDDING_MODEL
    dimension: int = DEFAULT_OPENAI_EMBEDDING_DIMENSION

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str):
            raise OpenAIEmbeddingConfigurationError(
                "OPENAI_API_KEY must be a string"
            )
        if not isinstance(self.model, str) or not self.model.strip():
            raise OpenAIEmbeddingConfigurationError(
                "OPENAI_EMBEDDING_MODEL cannot be empty"
            )
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise OpenAIEmbeddingConfigurationError(
                "OPENAI_EMBEDDING_DIMENSION must be an integer"
            )
        if self.dimension <= 0:
            raise OpenAIEmbeddingConfigurationError(
                "OPENAI_EMBEDDING_DIMENSION must be greater than zero"
            )

    @classmethod
    def from_environment(cls) -> "OpenAIEmbeddingSettings":
        """Load settings and derive the native model dimension when omitted."""

        load_dotenv()
        model = getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL)
        raw_dimension = getenv("OPENAI_EMBEDDING_DIMENSION", "").strip()
        if raw_dimension:
            try:
                dimension = int(raw_dimension)
            except ValueError as exc:
                raise OpenAIEmbeddingConfigurationError(
                    "OPENAI_EMBEDDING_DIMENSION must be an integer"
                ) from exc
        else:
            dimension = _NATIVE_DIMENSIONS.get(model.strip(), 0)
            if not dimension:
                raise OpenAIEmbeddingConfigurationError(
                    "OPENAI_EMBEDDING_DIMENSION is required for an unknown model"
                )
        return cls(
            api_key=getenv("OPENAI_API_KEY", ""),
            model=model,
            dimension=dimension,
        )


class OpenAIEmbeddingService:
    """Generate validated, ordered embeddings through the official OpenAI SDK."""

    provider = "openai"

    def __init__(
        self,
        settings: OpenAIEmbeddingSettings | None = None,
        *,
        client: Any | None = None,
        batch_size: int = DEFAULT_OPENAI_EMBEDDING_BATCH_SIZE,
        max_retries: int = DEFAULT_OPENAI_EMBEDDING_MAX_RETRIES,
        retry_base_seconds: float = DEFAULT_OPENAI_EMBEDDING_RETRY_BASE_SECONDS,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise OpenAIEmbeddingConfigurationError("batch_size must be an integer")
        if batch_size <= 0:
            raise OpenAIEmbeddingConfigurationError(
                "batch_size must be greater than zero"
            )
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise OpenAIEmbeddingConfigurationError("max_retries must be an integer")
        if max_retries < 0:
            raise OpenAIEmbeddingConfigurationError(
                "max_retries cannot be negative"
            )
        if retry_base_seconds < 0:
            raise OpenAIEmbeddingConfigurationError(
                "retry_base_seconds cannot be negative"
            )
        self.settings = settings or OpenAIEmbeddingSettings.from_environment()
        self.model = self.settings.model.strip()
        self.dimension = self.settings.dimension
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

        return content_hash(format_openai_document(text, title))

    def embed_document(self, text: str, title: str | None = None) -> list[float]:
        return self.embed_documents([(text, title)])[0]

    def embed_documents(
        self,
        documents: list[tuple[str, str | None]],
    ) -> list[list[float]]:
        if len(documents) > self.batch_size:
            raise ValueError(
                f"Document batch exceeds configured batch_size of {self.batch_size}"
            )
        prepared = [format_openai_document(text, title) for text, title in documents]
        return self._embed_many(prepared)

    def embed_query(self, query: str) -> list[float]:
        return self._embed_many([format_openai_query(query)])[0]

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = self.settings.api_key.strip()
        if not api_key:
            raise MissingOpenAIAPIKeyError(
                "OPENAI_API_KEY is required to request OpenAI embeddings"
            )
        try:
            self._client = OpenAI(api_key=api_key, max_retries=0)
        except Exception as exc:
            raise OpenAIEmbeddingConfigurationError(
                "Could not initialize the OpenAI embedding client"
            ) from exc
        return self._client

    def _embed_many(self, prepared_texts: list[str]) -> list[list[float]]:
        if not prepared_texts:
            return []
        client = self._get_client()

        def request() -> Any:
            return client.embeddings.create(
                model=self.model,
                input=prepared_texts,
                dimensions=self.dimension,
                encoding_format="float",
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
            raise OpenAIEmbeddingError("OpenAI embedding request failed") from exc

        data = getattr(response, "data", None)
        if not data or len(data) != len(prepared_texts):
            self.failed_embedding_count += len(prepared_texts)
            raise OpenAIEmbeddingResponseError(
                "OpenAI returned an unexpected number of embeddings"
            )
        ordered: list[Any | None] = [None] * len(prepared_texts)
        for item in data:
            index = getattr(item, "index", None)
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(ordered)
                or ordered[index] is not None
            ):
                self.failed_embedding_count += len(prepared_texts)
                raise OpenAIEmbeddingResponseError(
                    "OpenAI returned invalid embedding ordering"
                )
            ordered[index] = item
        try:
            vectors = [
                validate_embedding_values(
                    getattr(item, "embedding", None),
                    self.dimension,
                    provider_name="OpenAI",
                    error_type=OpenAIEmbeddingResponseError,
                )
                for item in ordered
            ]
        except OpenAIEmbeddingResponseError:
            self.failed_embedding_count += len(prepared_texts)
            raise
        self.successful_embedding_count += len(vectors)
        return vectors

    def _record_attempt(self) -> None:
        self.request_count += 1

    def _record_retry(self) -> None:
        self.retry_count += 1
