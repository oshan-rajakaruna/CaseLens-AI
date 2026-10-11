"""Public API contracts for development-only summary verification."""

from pydantic import BaseModel, Field

from agents.coordinator.schemas import ContextCitationMapEntry, SummarizationReadyPayload
from agents.summarization.schemas import SummarizationResult
from backend.schemas.verification import NonEmptyText, VerificationResponse


class SummarizationVerifyRequest(BaseModel):
    """Server-controlled generation request using preassembled legal context."""

    task_id: NonEmptyText
    payload: SummarizationReadyPayload


class SummarizationVerifyResponse(BaseModel):
    """Generated summary, verification outcome, and non-text provenance."""

    summary: SummarizationResult
    verification: VerificationResponse
    provenance: dict[str, ContextCitationMapEntry] = Field(default_factory=dict)
    bridge_warnings: list[str] = Field(default_factory=list)
