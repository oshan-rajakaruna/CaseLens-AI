"""Public API and retrieval-side coordinator contracts."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RetrievalMode = Literal["bm25", "semantic", "hybrid"]


class RetrievalFilters(BaseModel):
    """Small metadata-filter surface supported by hybrid retrieval."""

    model_config = ConfigDict(extra="forbid")

    court: str | None = None
    year: int | None = Field(default=None, ge=1000, le=9999)
    date: str | None = None
    legal_category: str | None = None
    document_type: str | None = None


class RetrievalSearchRequest(BaseModel):
    """Validated request for one retrieval mode."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    query: str
    mode: RetrievalMode = "hybrid"
    top_k: int = Field(default=5, ge=1, strict=True)
    filters: RetrievalFilters | None = None
    bm25_weight: float | None = Field(default=None, ge=0)
    semantic_weight: float | None = Field(default=None, ge=0)
    diversify: bool = True
    max_chunks_per_document: int | None = Field(default=None, ge=1, strict=True)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        """Reject whitespace-only queries while preserving the submitted text."""

        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value

    @field_validator("bm25_weight", "semantic_weight", mode="before")
    @classmethod
    def weights_cannot_be_boolean(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("retrieval weights must be numeric, not boolean")
        return value

    @model_validator(mode="after")
    def hybrid_weights_must_have_positive_total(self) -> "RetrievalSearchRequest":
        """Reject a hybrid request that disables both retrieval channels."""

        if (
            self.mode == "hybrid"
            and self.bm25_weight is not None
            and self.semantic_weight is not None
            and self.bm25_weight + self.semantic_weight <= 0
        ):
            raise ValueError(
                "at least one hybrid retrieval weight must be greater than zero"
            )
        return self


class RetrievalMetadataResponse(BaseModel):
    """Serializable document metadata returned through public contracts."""

    model_config = ConfigDict(extra="allow")

    document_id: str
    case_name: str | None = None
    court: str | None = None
    date: str | None = None
    citation: str | None = None
    legal_category: str | None = None
    document_type: str | None = None
    source: str | None = None


class RetrievalResultResponse(BaseModel):
    """Mode-neutral ranked result with explicit optional score fields."""

    rank: int = Field(ge=1)
    document_id: str
    chunk_id: str
    case_name: str | None = None
    chunk_text: str
    citation: str | None = None
    source: str | None = None
    metadata: RetrievalMetadataResponse
    bm25_score: float | None = None
    normalized_bm25_score: float | None = Field(default=None, ge=0, le=1)
    semantic_score: float | None = Field(default=None, ge=-1, le=1)
    normalized_semantic_score: float | None = Field(default=None, ge=0, le=1)
    hybrid_score: float | None = Field(default=None, ge=0, le=1)


class RetrievalSearchResponse(BaseModel):
    """Structured response envelope for API and coordinator consumers."""

    query: str
    mode: RetrievalMode
    top_k: int
    result_count: int = Field(ge=0)
    results: list[RetrievalResultResponse] = Field(default_factory=list)


class RetrievalTaskContext(RetrievalSearchRequest):
    """Retrieval-specific payload carried inside a shared ``AgentTask``."""

    legal_issue: str | None = None

    @field_validator("legal_issue")
    @classmethod
    def legal_issue_must_contain_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("legal_issue must contain text when provided")
        return value


def metadata_to_public(metadata: Any) -> RetrievalMetadataResponse:
    """Validate internal metadata into the stable public representation."""

    if hasattr(metadata, "model_dump"):
        metadata = metadata.model_dump()
    return RetrievalMetadataResponse.model_validate(metadata)
