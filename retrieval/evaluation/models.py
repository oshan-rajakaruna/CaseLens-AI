"""Typed datasets and result structures for retrieval evaluation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from retrieval.preprocessing.metadata import LegalDocumentMetadata

EvaluationMode = Literal["bm25", "semantic", "hybrid"]
RelevanceUnit = Literal["document", "chunk"]
HumanRelevanceLabel = Literal["relevant", "partially_relevant", "not_relevant"]


class HumanRelevanceJudgment(BaseModel):
    """Optional manual label kept separate from binary metric ground truth."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    label: HumanRelevanceLabel
    notes: str | None = None


class EvaluationQuery(BaseModel):
    """One information need and its binary relevance ground truth."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1)
    query: str
    legal_category: str | None = None
    relevance_unit: RelevanceUnit = "document"
    relevant_ids: list[str] = Field(default_factory=list)
    notes: str | None = None
    human_judgments: list[HumanRelevanceJudgment] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evaluation query must contain text")
        return value

    @field_validator("relevant_ids")
    @classmethod
    def relevant_ids_must_be_non_empty_strings(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("relevant_ids cannot contain empty values")
        return values


class EvaluationDocument(BaseModel):
    """Synthetic or curated document chunk used by an offline runner."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    chunk_text: str = Field(min_length=1)
    metadata: LegalDocumentMetadata


class EvaluationDataset(BaseModel):
    """Versionable evaluation dataset with explicit provenance classification."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str
    data_classification: Literal["synthetic", "curated_prototype", "real"]
    documents: list[EvaluationDocument] = Field(default_factory=list)
    queries: list[EvaluationQuery] = Field(min_length=1)

    @model_validator(mode="after")
    def identifiers_must_be_unique(self) -> "EvaluationDataset":
        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query_id values must be unique")
        chunk_ids = [document.chunk_id for document in self.documents]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("evaluation chunk_id values must be unique")
        return self


class QueryMetricsAtK(BaseModel):
    """Precision and recall for one query at one cutoff."""

    k: int = Field(ge=1)
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)


class PerQueryEvaluationResult(BaseModel):
    """Retrieved identifiers and metrics for one query/mode pair."""

    query_id: str
    query: str
    mode: EvaluationMode
    relevance_unit: RelevanceUnit
    relevant_ids: list[str]
    retrieved_ids: list[str]
    metrics_at_k: list[QueryMetricsAtK]
    reciprocal_rank: float = Field(ge=0, le=1)


class AggregateAtK(BaseModel):
    """Mean precision and recall across all evaluated queries at one cutoff."""

    k: int = Field(ge=1)
    mean_precision_at_k: float = Field(ge=0, le=1)
    mean_recall_at_k: float = Field(ge=0, le=1)


class ModeEvaluationResult(BaseModel):
    """Aggregate and per-query metrics for one retrieval mode."""

    mode: EvaluationMode
    query_count: int = Field(ge=0)
    retrieval_depth: int = Field(ge=1)
    k_values: list[int]
    aggregate_at_k: list[AggregateAtK]
    mrr: float = Field(ge=0, le=1)
    per_query: list[PerQueryEvaluationResult]


class EvaluationComparison(BaseModel):
    """Side-by-side results without asserting which retrieval mode is best."""

    dataset_id: str
    data_classification: str
    k_values: list[int]
    modes: list[ModeEvaluationResult]
