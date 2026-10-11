"""Persist and restore retrieval-ready corpus chunks and semantic vectors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from retrieval.embeddings import EmbeddingCheckpointStore, EmbeddingService
from retrieval.preprocessing.metadata import LegalTextChunk


INDEX_FORMAT_VERSION = "1"
INDEX_METADATA_FILE = "index.json"
INDEX_CHUNKS_FILE = "chunks.jsonl"
INDEX_EMBEDDINGS_FILE = "semantic-embeddings.jsonl"


class RetrievalIndexStoreError(RuntimeError):
    """Raised when a persisted retrieval snapshot is absent or inconsistent."""


class RetrievalIndexMetadata(BaseModel):
    """Small descriptor used to validate a persisted retrieval snapshot."""

    model_config = ConfigDict(extra="forbid")

    format_version: str = INDEX_FORMAT_VERSION
    chunk_count: int = Field(ge=1)
    semantic: bool
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = Field(default=None, ge=1)


@dataclass(frozen=True, slots=True)
class LoadedRetrievalIndex:
    """Validated chunks and optional vectors ready for existing index classes."""

    chunks: list[LegalTextChunk]
    vectors: list[list[float]] | None


class RetrievalIndexStore:
    """Write and load a versioned snapshot without serializing Python objects."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @property
    def embedding_checkpoint_path(self) -> Path:
        return self.directory / INDEX_EMBEDDINGS_FILE

    def save(
        self,
        chunks: list[LegalTextChunk],
        *,
        embedding_service: EmbeddingService | None = None,
        semantic_indexed: bool = False,
    ) -> None:
        if not chunks:
            raise RetrievalIndexStoreError("Cannot persist an empty retrieval index")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise RetrievalIndexStoreError("Cannot persist duplicate chunk IDs")
        if semantic_indexed and embedding_service is None:
            raise RetrievalIndexStoreError(
                "Semantic snapshot requires embedding provider metadata"
            )

        self.directory.mkdir(parents=True, exist_ok=True)
        chunks_path = self.directory / INDEX_CHUNKS_FILE
        chunks_temp = chunks_path.with_suffix(".jsonl.tmp")
        metadata_path = self.directory / INDEX_METADATA_FILE
        metadata_temp = metadata_path.with_suffix(".json.tmp")
        try:
            with chunks_temp.open("w", encoding="utf-8", newline="\n") as handle:
                for chunk in chunks:
                    handle.write(chunk.model_dump_json() + "\n")
            metadata = RetrievalIndexMetadata(
                chunk_count=len(chunks),
                semantic=semantic_indexed,
                embedding_provider=(
                    str(embedding_service.provider) if embedding_service else None
                ),
                embedding_model=(
                    str(embedding_service.model) if embedding_service else None
                ),
                embedding_dimension=(
                    int(embedding_service.dimension) if embedding_service else None
                ),
            )
            metadata_temp.write_text(
                metadata.model_dump_json(indent=2), encoding="utf-8"
            )
            chunks_temp.replace(chunks_path)
            metadata_temp.replace(metadata_path)
        except OSError as exc:
            raise RetrievalIndexStoreError(
                "Retrieval index snapshot could not be written"
            ) from exc

    def load(self, embedding_service: EmbeddingService) -> LoadedRetrievalIndex:
        metadata = self._load_metadata()
        chunks = self._load_chunks(metadata.chunk_count)
        if not metadata.semantic:
            return LoadedRetrievalIndex(chunks=chunks, vectors=None)
        identity = (
            embedding_service.provider,
            embedding_service.model,
            embedding_service.dimension,
        )
        expected = (
            metadata.embedding_provider,
            metadata.embedding_model,
            metadata.embedding_dimension,
        )
        if identity != expected:
            raise RetrievalIndexStoreError(
                "Configured embedding provider does not match persisted index"
            )
        try:
            checkpoint = EmbeddingCheckpointStore(self.embedding_checkpoint_path)
            vectors = [checkpoint.get(embedding_service, chunk) for chunk in chunks]
        except (OSError, ValueError) as exc:
            raise RetrievalIndexStoreError(
                "Persisted semantic index is not readable"
            ) from exc
        if any(vector is None for vector in vectors):
            raise RetrievalIndexStoreError(
                "Persisted semantic index is incomplete"
            )
        return LoadedRetrievalIndex(
            chunks=chunks,
            vectors=[vector for vector in vectors if vector is not None],
        )

    def _load_metadata(self) -> RetrievalIndexMetadata:
        try:
            raw: Any = json.loads(
                (self.directory / INDEX_METADATA_FILE).read_text(encoding="utf-8")
            )
            metadata = RetrievalIndexMetadata.model_validate(raw)
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
            raise RetrievalIndexStoreError(
                "Retrieval index metadata is unavailable or invalid"
            ) from exc
        if metadata.format_version != INDEX_FORMAT_VERSION:
            raise RetrievalIndexStoreError("Retrieval index format is unsupported")
        return metadata

    def _load_chunks(self, expected_count: int) -> list[LegalTextChunk]:
        try:
            lines = (self.directory / INDEX_CHUNKS_FILE).read_text(
                encoding="utf-8"
            ).splitlines()
            chunks = [
                LegalTextChunk.model_validate_json(line)
                for line in lines
                if line.strip()
            ]
        except (OSError, UnicodeError, ValidationError) as exc:
            raise RetrievalIndexStoreError(
                "Persisted retrieval chunks are unavailable or invalid"
            ) from exc
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunks) != expected_count or len(chunk_ids) != len(set(chunk_ids)):
            raise RetrievalIndexStoreError(
                "Persisted retrieval chunks are incomplete or duplicated"
            )
        return chunks
