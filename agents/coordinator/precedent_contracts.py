"""Typed Coordinator Step 4 contracts built on public Summarization models."""

from typing import Literal

from pydantic import BaseModel, Field

from agents.coordinator.schemas import CoordinatorMessage, CoordinatorStage
from agents.summarization.schemas import PrecedentSummary, SourceReference, SummarizationResponse
from backend.schemas.retrieval import RetrievalMode
from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    HybridSearchResult,
    SemanticSearchResult,
)


CoordinatorPrecedentRetrievalResult = (
    BM25SearchResult | SemanticSearchResult | HybridSearchResult
)


class CoordinatorCitationProvenance(BaseModel):
    """Request-local source mapping; official citations remain separate metadata."""

    citation_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    locator: str | None = None
    page: int | None = Field(default=None, gt=0)
    page_start: int | None = Field(default=None, gt=0)
    page_end: int | None = Field(default=None, gt=0)
    title: str | None = None
    court: str | None = None
    jurisdiction: str | None = None
    official_citation: str | None = None
    source_url: str | None = None


class CoordinatorPrecedentResponse(BaseModel):
    """Retrieved material and deterministic precedent summaries, never a legal ruling."""

    case_id: str = Field(min_length=1)
    requested_task: Literal["summarize_precedents"]
    status: Literal["completed", "partial", "requires_input", "failed"]
    retrieval_mode: RetrievalMode
    top_k: int = Field(ge=1)
    stages: list[CoordinatorStage] = Field(default_factory=list)
    retrieval_results: list[CoordinatorPrecedentRetrievalResult] = Field(default_factory=list)
    summarization_responses: list[SummarizationResponse] = Field(default_factory=list)
    precedent_summaries: list[PrecedentSummary] = Field(default_factory=list)
    citation_provenance: dict[str, CoordinatorCitationProvenance] = Field(default_factory=dict)
    source_references: list[SourceReference] = Field(default_factory=list)
    warnings: list[CoordinatorMessage] = Field(default_factory=list)
    errors: list[CoordinatorMessage] = Field(default_factory=list)
    verification_limitations: list[str] = Field(default_factory=list)
    executed: bool
