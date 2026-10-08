"""Minimal shared models for API and future agent communication."""

from typing import Any

from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    """Public metadata returned by the API root."""

    name: str
    phase: str
    message: str


class HealthResponse(BaseModel):
    """Health response for the backend process."""

    status: str
    service: str


class Case(BaseModel):
    """Minimal identifying information for a case."""

    id: str
    title: str
    description: str | None = None


class Document(BaseModel):
    """Minimal metadata for a document associated with a case."""

    id: str
    case_id: str
    name: str
    media_type: str | None = None


class AgentTask(BaseModel):
    """Structured work request that can be passed to a future agent."""

    id: str
    agent: str
    instruction: str
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Structured result returned by a future agent implementation."""

    task_id: str
    agent: str
    status: str
    output: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Reference from generated output to a source document location."""

    document_id: str
    locator: str | None = None
    excerpt: str | None = None


class VerificationResult(BaseModel):
    """Minimal representation of a future verification decision."""

    verified: bool
    notes: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
