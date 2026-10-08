"""Coordinator-specific request and response envelopes."""

from pydantic import BaseModel

from backend.schemas import AgentResult, AgentTask, VerificationResult


class CoordinatorRequest(BaseModel):
    """Request accepted by a future coordinator implementation."""

    task: AgentTask


class CoordinatorResponse(BaseModel):
    """Aggregated coordinator output and optional verification result."""

    result: AgentResult
    verification: VerificationResult | None = None
