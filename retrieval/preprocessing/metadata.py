"""Typed metadata contracts for legal documents and retrieval results."""

from pydantic import BaseModel, ConfigDict, Field


class LegalDocumentMetadata(BaseModel):
    """Minimal document metadata; only the stable identifier is required."""

    model_config = ConfigDict(extra="allow")

    document_id: str = Field(min_length=1)
    case_name: str | None = None
    court: str | None = None
    date: str | None = None
    citation: str | None = None
    legal_category: str | None = None
    document_type: str | None = None
    source: str | None = None


class LegalTextChunk(BaseModel):
    """A searchable text chunk linked to its source document metadata."""

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    chunk_text: str = Field(min_length=1)
    metadata: LegalDocumentMetadata


class BM25SearchResult(BaseModel):
    """Structured, display-ready result returned by BM25 retrieval."""

    rank: int = Field(ge=1)
    document_id: str
    chunk_id: str
    case_name: str | None = None
    score: float = Field(ge=0)
    chunk_text: str
    citation: str | None = None
    source: str | None = None
    court: str | None = None
    date: str | None = None
    legal_category: str | None = None
    document_type: str | None = None


class SemanticSearchResult(BaseModel):
    """Structured, display-ready cosine-similarity retrieval result."""

    rank: int = Field(ge=1)
    document_id: str
    chunk_id: str
    case_name: str | None = None
    similarity_score: float = Field(ge=-1, le=1)
    chunk_text: str
    citation: str | None = None
    source: str | None = None
    court: str | None = None
    date: str | None = None
    legal_category: str | None = None
    document_type: str | None = None
    metadata: LegalDocumentMetadata
