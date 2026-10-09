"""Coordinator-specific request and response envelopes."""

from pydantic import BaseModel, field_validator

from backend.schemas import AgentResult, AgentTask, VerificationResponse, VerificationResult


class CoordinatorRequest(BaseModel):
    """Request accepted by a future coordinator implementation."""

    task: AgentTask


class CoordinatorResponse(BaseModel):
    """Aggregated coordinator output and optional verification result."""

    result: AgentResult
    verification: VerificationResponse | VerificationResult | None = None

    @field_validator("verification", mode="before")
    @classmethod
    def validate_detailed_verification(cls, value: object) -> object:
        if isinstance(value, dict) and any(
            key in value for key in ("task_id", "overall_status", "claim_results", "warnings")
        ):
            return VerificationResponse.model_validate(value)
        return value
