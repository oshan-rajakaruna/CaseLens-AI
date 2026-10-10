"""Contracts for provider-neutral, citation-aware summarization."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.coordinator.schemas import ContextCitationMapEntry
from agents.summarization.citations import CitationValidationResult


class SummarizationOutput(BaseModel):
    """LLM-independent answer contract for later verification."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def answer_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must contain non-whitespace text")
        return value

    @field_validator("citations")
    @classmethod
    def citations_must_use_context_ids(cls, values: list[str]) -> list[str]:
        for value in values:
            if not _is_context_id(value):
                raise ValueError("citations must use CTX-nnn identifiers")
        return values


def _is_context_id(value: str) -> bool:
    if not isinstance(value, str) or not value.startswith("CTX-"):
        return False
    suffix = value[4:]
    return len(suffix) >= 3 and suffix.isdigit()


class SummarizationProviderRequest(BaseModel):
    """Complete grounded input passed to an injectable generation provider."""

    query: str
    legal_issue: str | None = None
    formatted_context: str
    citation_map: dict[str, ContextCitationMapEntry]
    generation_instructions: str
    prompt: str


class SummarizationCitationValidation(BaseModel):
    """Consistency checks across inline and declared citation identities."""

    inline: CitationValidationResult
    declared_citations: list[str] = Field(default_factory=list)
    invalid_declared_citations: list[str] = Field(default_factory=list)
    declared_missing_inline: list[str] = Field(default_factory=list)
    inline_missing_declared: list[str] = Field(default_factory=list)
    all_citations_valid: bool


class SummarizationProviderMetadata(BaseModel):
    """Safe provider identity without credentials or request headers."""

    provider: str
    model: str | None = None
    provider_called: bool


class SummarizationResult(BaseModel):
    """Validated output ready for a future Verification Agent handoff."""

    output: SummarizationOutput
    citation_validation: SummarizationCitationValidation
    provider_metadata: SummarizationProviderMetadata
    used_context_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


SummarizationProviderResponse = SummarizationOutput | dict[str, Any]
