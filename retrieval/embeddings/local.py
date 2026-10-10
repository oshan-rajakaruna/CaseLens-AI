"""Local FastEmbed embeddings for CaseLens semantic retrieval."""

from collections.abc import Callable
from dataclasses import dataclass
from os import getenv
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding

from retrieval.embeddings.common import content_hash, validate_embedding_values

DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LOCAL_EMBEDDING_DIMENSION = 384
DEFAULT_LOCAL_EMBEDDING_CACHE_DIR = "data/models/fastembed"
DEFAULT_LOCAL_EMBEDDING_BATCH_SIZE = 256
APPLICATION_ROOT = Path(__file__).resolve().parents[2]


class LocalEmbeddingError(RuntimeError):
    """Base error for local FastEmbed failures."""


class LocalEmbeddingConfigurationError(LocalEmbeddingError):
    """Raised when local embedding configuration is invalid."""


class LocalEmbeddingResponseError(LocalEmbeddingError):
    """Raised when FastEmbed returns missing or invalid vectors."""


def resolve_local_cache_dir(
    configured_path: str | Path,
    *,
    application_root: str | Path | None = None,
) -> Path:
    """Resolve a cache path against the application root when it is relative."""

    if not isinstance(configured_path, (str, Path)):
        raise LocalEmbeddingConfigurationError(
            "LOCAL_EMBEDDING_CACHE_DIR must be a path"
        )
    raw_path = str(configured_path).strip()
    if not raw_path:
        raise LocalEmbeddingConfigurationError(
            "LOCAL_EMBEDDING_CACHE_DIR cannot be empty"
        )
    cache_path = Path(raw_path).expanduser()
    if not cache_path.is_absolute():
        root = Path(application_root or APPLICATION_ROOT)
        cache_path = root / cache_path
    return cache_path.resolve()


def format_local_document(text: str, title: str | None = None) -> str:
    """Format a legal chunk for local passage embedding and checkpoint hashing."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if title is not None and not isinstance(title, str):
        raise TypeError("title must be a string or None")
    if not text.strip():
        raise ValueError("document text cannot be empty")
    safe_title = title.strip() if title and title.strip() else "none"
    return f"title: {safe_title} | text: {text.strip()}"


def format_local_query(query: str) -> str:
    """Validate and normalize a semantic query for local embedding."""

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query cannot be empty")
    return query.strip()


@dataclass(frozen=True, slots=True)
class LocalEmbeddingSettings:
    """Environment-backed local FastEmbed configuration."""

    model: str = DEFAULT_LOCAL_EMBEDDING_MODEL
    dimension: int = DEFAULT_LOCAL_EMBEDDING_DIMENSION
    cache_dir: Path = APPLICATION_ROOT / DEFAULT_LOCAL_EMBEDDING_CACHE_DIR

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise LocalEmbeddingConfigurationError(
                "LOCAL_EMBEDDING_MODEL cannot be empty"
            )
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise LocalEmbeddingConfigurationError(
                "LOCAL_EMBEDDING_DIMENSION must be an integer"
            )
        if self.dimension <= 0:
            raise LocalEmbeddingConfigurationError(
                "LOCAL_EMBEDDING_DIMENSION must be greater than zero"
            )
        object.__setattr__(
            self,
            "cache_dir",
            resolve_local_cache_dir(self.cache_dir),
        )

    @classmethod
    def from_environment(
        cls,
        *,
        application_root: str | Path | None = None,
    ) -> "LocalEmbeddingSettings":
        """Read local settings without loading or downloading the model."""

        raw_dimension = getenv(
            "LOCAL_EMBEDDING_DIMENSION",
            str(DEFAULT_LOCAL_EMBEDDING_DIMENSION),
        )
        try:
            dimension = int(raw_dimension)
        except ValueError as exc:
            raise LocalEmbeddingConfigurationError(
                "LOCAL_EMBEDDING_DIMENSION must be an integer"
            ) from exc
        cache_dir = resolve_local_cache_dir(
            getenv("LOCAL_EMBEDDING_CACHE_DIR", DEFAULT_LOCAL_EMBEDDING_CACHE_DIR),
            application_root=application_root,
        )
        return cls(
            model=getenv("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL),
            dimension=dimension,
            cache_dir=cache_dir,
        )


class LocalEmbeddingService:
    """Generate local, validated embeddings with a lazily loaded FastEmbed model."""

    provider = "local"

    def __init__(
        self,
        settings: LocalEmbeddingSettings | None = None,
        *,
        embedding_model: Any | None = None,
        model_factory: Callable[..., Any] | None = None,
        batch_size: int = DEFAULT_LOCAL_EMBEDDING_BATCH_SIZE,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise LocalEmbeddingConfigurationError("batch_size must be an integer")
        if batch_size <= 0:
            raise LocalEmbeddingConfigurationError(
                "batch_size must be greater than zero"
            )
        self.settings = settings or LocalEmbeddingSettings.from_environment()
        self.model = self.settings.model.strip()
        self.dimension = self.settings.dimension
        self.cache_dir = self.settings.cache_dir
        self.batch_size = batch_size
        self._embedding_model = embedding_model
        self._model_factory = model_factory or TextEmbedding
        self.request_count = 0
        self.retry_count = 0
        self.successful_embedding_count = 0
        self.failed_embedding_count = 0

    def document_content_hash(self, text: str, title: str | None = None) -> str:
        """Hash the exact text supplied to FastEmbed passage embedding."""

        return content_hash(format_local_document(text, title))

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
        prepared = [format_local_document(text, title) for text, title in documents]
        return self._embed_many(prepared, query=False)

    def embed_query(self, query: str) -> list[float]:
        return self._embed_many([format_local_query(query)], query=True)[0]

    def _get_model(self) -> Any:
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LocalEmbeddingConfigurationError(
                "LOCAL_EMBEDDING_CACHE_DIR could not be created"
            ) from exc
        try:
            self._embedding_model = self._model_factory(
                model_name=self.model,
                cache_dir=str(self.cache_dir),
                lazy_load=True,
            )
        except Exception as exc:
            raise LocalEmbeddingError(
                "FastEmbed model could not be initialized"
            ) from exc
        return self._embedding_model

    def _embed_many(self, prepared_texts: list[str], *, query: bool) -> list[list[float]]:
        if not prepared_texts:
            return []
        model = self._get_model()
        try:
            if query:
                raw_vectors = list(
                    model.query_embed(prepared_texts, batch_size=self.batch_size)
                )
            else:
                raw_vectors = list(
                    model.passage_embed(prepared_texts, batch_size=self.batch_size)
                )
        except Exception as exc:
            self.failed_embedding_count += len(prepared_texts)
            raise LocalEmbeddingError("FastEmbed embedding failed") from exc

        if len(raw_vectors) != len(prepared_texts):
            self.failed_embedding_count += len(prepared_texts)
            raise LocalEmbeddingResponseError(
                "FastEmbed returned an unexpected number of embeddings"
            )
        try:
            vectors = [
                validate_embedding_values(
                    vector,
                    self.dimension,
                    provider_name="FastEmbed",
                    error_type=LocalEmbeddingResponseError,
                )
                for vector in raw_vectors
            ]
        except LocalEmbeddingResponseError:
            self.failed_embedding_count += len(prepared_texts)
            raise
        self.successful_embedding_count += len(vectors)
        return vectors
