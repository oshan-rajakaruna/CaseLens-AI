"""Coordinator-owned response envelopes for Steps 5 and 6."""

from typing import Literal

from pydantic import BaseModel, Field

from agents.analysis.schemas import AnalysisResponse
from agents.coordinator.precedent_contracts import CoordinatorPrecedentResponse
from agents.coordinator.schemas import CoordinatorMessage, CoordinatorStage
from agents.summarization.schemas import SummarizationResponse, SummarizationWorkflowResponse
from backend.schemas import VerificationResponse


class CoordinatorEvidenceSummaryResponse(BaseModel):
    case_id: str = Field(min_length=1)
    requested_task: Literal["summarize_evidence"]
    status: Literal["completed", "partial", "requires_input", "failed"]
    stages: list[CoordinatorStage] = Field(default_factory=list)
    summarization_response: SummarizationResponse | None = None
    warnings: list[CoordinatorMessage] = Field(default_factory=list)
    errors: list[CoordinatorMessage] = Field(default_factory=list)
    verification_limitations: list[str] = Field(default_factory=list)
    executed: bool


class CoordinatorVerificationResponse(BaseModel):
    case_id: str = Field(min_length=1)
    requested_task: Literal["verify_claims"]
    status: Literal["completed", "partial", "requires_input", "failed"]
    stages: list[CoordinatorStage] = Field(default_factory=list)
    verification_response: VerificationResponse | None = None
    warnings: list[CoordinatorMessage] = Field(default_factory=list)
    errors: list[CoordinatorMessage] = Field(default_factory=list)
    verification_limitations: list[str] = Field(default_factory=list)
    executed: bool


class CoordinatorFullResearchResponse(BaseModel):
    case_id: str = Field(min_length=1)
    requested_task: Literal["full_research"]
    status: Literal["completed", "partial", "requires_input", "failed"]
    stages: list[CoordinatorStage] = Field(default_factory=list)
    analysis_results: list[AnalysisResponse] = Field(default_factory=list)
    precedent_response: CoordinatorPrecedentResponse | None = None
    evidence_summary_response: SummarizationResponse | None = None
    research_workflow: SummarizationWorkflowResponse | None = None
    verification_response: VerificationResponse | None = None
    warnings: list[CoordinatorMessage] = Field(default_factory=list)
    errors: list[CoordinatorMessage] = Field(default_factory=list)
    verification_limitations: list[str] = Field(default_factory=list)
    executed: bool
