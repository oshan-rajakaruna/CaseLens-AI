"""Append-only, provider-safe embedding checkpoints for resumable indexing."""

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

from retrieval.embeddings.common import validate_embedding_values
from retrieval.preprocessing.metadata import LegalTextChunk


class EmbeddingCheckpointError(ValueError):
    """Raised when a checkpoint is malformed or cannot be written safely."""


class EmbeddingCheckpointStore:
    """Cache vectors under provider/model/dimension/chunk/content identities."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._vectors: dict[tuple[str, str, int, str, str], list[float]] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self.path.exists():
            self._load()

    def get(self, service: Any, chunk: LegalTextChunk) -> list[float] | None:
        key = self._key(service, chunk)
        vector = self._vectors.get(key)
        if vector is None:
            self.misses += 1
            return None
        self.hits += 1
        return list(vector)

    def put_many(
        self,
        service: Any,
        items: Iterable[tuple[LegalTextChunk, list[float]]],
    ) -> None:
        records: list[dict[str, Any]] = []
        for chunk, raw_vector in items:
            vector = validate_embedding_values(
                raw_vector,
                service.dimension,
                provider_name="Checkpoint",
                error_type=EmbeddingCheckpointError,
            )
            provider, model, dimension, chunk_id, digest = self._key(service, chunk)
            records.append(
                {
                    "provider": provider,
                    "model": model,
                    "dimension": dimension,
                    "chunk_id": chunk_id,
                    "content_hash": digest,
                    "vector": vector,
                }
            )
        if not records:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError as exc:
            raise EmbeddingCheckpointError("Embedding checkpoint could not be written") from exc
        for record in records:
            key = (
                record["provider"],
                record["model"],
                record["dimension"],
                record["chunk_id"],
                record["content_hash"],
            )
            self._vectors[key] = record["vector"]
        self.writes += len(records)

    @staticmethod
    def _key(service: Any, chunk: LegalTextChunk) -> tuple[str, str, int, str, str]:
        try:
            digest = service.document_content_hash(
                chunk.chunk_text,
                chunk.metadata.case_name,
            )
            provider = service.provider
            model = service.model
            dimension = service.dimension
        except (AttributeError, TypeError, ValueError) as exc:
            raise EmbeddingCheckpointError(
                "Embedding service does not expose checkpoint identity"
            ) from exc
        return (provider, model, dimension, chunk.chunk_id, digest)

    def _load(self) -> None:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if not line.strip():
                    continue
                record = json.loads(line)
                provider = record["provider"]
                model = record["model"]
                dimension = record["dimension"]
                chunk_id = record["chunk_id"]
                digest = record["content_hash"]
                if not all(
                    isinstance(value, str) and value
                    for value in (provider, model, chunk_id, digest)
                ):
                    raise EmbeddingCheckpointError(
                        "Embedding checkpoint identity is invalid"
                    )
                if isinstance(dimension, bool) or not isinstance(dimension, int):
                    raise EmbeddingCheckpointError(
                        "Embedding checkpoint dimension is invalid"
                    )
                vector = validate_embedding_values(
                    record.get("vector"),
                    dimension,
                    provider_name="Checkpoint",
                    error_type=EmbeddingCheckpointError,
                )
                self._vectors[(provider, model, dimension, chunk_id, digest)] = vector
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EmbeddingCheckpointError(
                "Embedding checkpoint is not readable JSONL"
            ) from exc
