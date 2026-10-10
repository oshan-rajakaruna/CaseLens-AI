"""Structured contracts for citation-preserving RAG context."""

from pydantic import BaseModel, ConfigDict, Field


class ChunkRetrievalScore(BaseModel):
    """Transparent score information retained for one source chunk."""

    model_config = ConfigDict(allow_inf_nan=False)

    chunk_id: str
    rank: int = Field(ge=1)
    final_score: float
    bm25_score: float | None = None
    normalized_bm25_score: float | None = Field(default=None, ge=0, le=1)
    semantic_score: float | None = Field(default=None, ge=-1, le=1)
    normalized_semantic_score: float | None = Field(default=None, ge=0, le=1)
    hybrid_score: float | None = Field(default=None, ge=0, le=1)


class RAGCitation(BaseModel):
    """Source identity carried alongside an assembled passage."""

    document_id: str
    citation: str | None = None
    source: str | None = None
    source_url: str | None = None
    provenance: str | None = None
    chunk_ids: list[str]
    pages: list[int]


class RAGContextPassage(BaseModel):
    """One ranked, optionally merged context passage."""

    model_config = ConfigDict(allow_inf_nan=False)

    context_id: str = Field(pattern=r"^CTX-\d{3,}$")
    text: str = Field(min_length=1)
    document_id: str
    title: str | None = None
    court: str | None = None
    source: str | None = None
    citation: RAGCitation
    chunk_ids: list[str]
    pages: list[int]
    ranks: list[int]
    score: float
    retrieval_scores: list[ChunkRetrievalScore]
    truncated: bool = False


class RAGContext(BaseModel):
    """Complete structured context ready for a future prompt builder."""

    passages: list[RAGContextPassage] = Field(default_factory=list)
    estimated_tokens: int = Field(ge=0)
    source_documents: list[str] = Field(default_factory=list)
    input_result_count: int = Field(ge=0)
    eligible_result_count: int = Field(ge=0)
    deduplicated_result_count: int = Field(ge=0)
    truncated: bool = False
