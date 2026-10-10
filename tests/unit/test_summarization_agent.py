"""Focused tests for the Summarization Agent foundation."""

import asyncio

import pytest
from pydantic import ValidationError

from agents.summarization import (
    CaseInformation,
    DraftClaim,
    EvidenceSummary,
    ExtractedFact,
    InvalidSummarizationInputError,
    MissingEvidenceError,
    ModelUnavailableError,
    SourceReference,
    SummarizationAgent,
    SummarizationRequest,
    SummarizationResponse,
    UnsupportedSourceReferenceError,
)
from backend.schemas import AgentTask, Citation


SOURCE = SourceReference(document_id="document-1", locator="page 4, paragraph 2")


def make_request() -> SummarizationRequest:
    return SummarizationRequest(
        case=CaseInformation(case_id="case-1", title="Example case"),
        extracted_facts=[
            ExtractedFact(
                fact_id="fact-1",
                text="A fact supplied by an upstream extraction step.",
                source_references=[SOURCE],
            )
        ],
    )


class FakeGenerator:
    def __init__(self, response: SummarizationResponse) -> None:
        self.response = response
        self.received: SummarizationRequest | None = None

    async def generate(self, request: SummarizationRequest) -> SummarizationResponse:
        self.received = request
        return self.response


def test_request_rejects_invalid_source_reference() -> None:
    with pytest.raises(ValidationError):
        SourceReference(document_id="document-1", locator="   ")


def test_response_marks_generated_claim_as_draft() -> None:
    claim = DraftClaim(text="A proposed claim.", supporting_sources=[SOURCE])

    assert claim.status == "draft"
    with pytest.raises(ValidationError):
        DraftClaim(
            text="A proposed claim.",
            supporting_sources=[SOURCE],
            status="final",
        )


def test_agent_preserves_supported_source_references() -> None:
    response = SummarizationResponse(
        evidence_summaries=[
            EvidenceSummary(text="Grounded summary.", source_references=[SOURCE])
        ],
        citations=[
            Citation(
                document_id=SOURCE.document_id,
                locator=SOURCE.locator,
                excerpt="A fact supplied by an upstream extraction step.",
            )
        ],
    )
    generator = FakeGenerator(response)
    request = make_request()

    result = asyncio.run(SummarizationAgent(generator).summarize(request))

    assert generator.received is request
    assert result.evidence_summaries[0].source_references == [SOURCE]
    assert result.citations[0].document_id == "document-1"
    assert result.citations[0].locator == "page 4, paragraph 2"


def test_agent_rejects_generated_unsupported_reference() -> None:
    invented = SourceReference(document_id="invented", locator="page 99")
    response = SummarizationResponse(
        evidence_summaries=[
            EvidenceSummary(text="Unsupported summary.", source_references=[invented])
        ]
    )

    with pytest.raises(UnsupportedSourceReferenceError):
        asyncio.run(SummarizationAgent(FakeGenerator(response)).summarize(make_request()))


def test_agent_rejects_fabricated_citation_excerpt() -> None:
    response = SummarizationResponse(
        citations=[
            Citation(
                document_id=SOURCE.document_id,
                locator=SOURCE.locator,
                excerpt="Text that was not supplied.",
            )
        ]
    )

    with pytest.raises(UnsupportedSourceReferenceError):
        asyncio.run(SummarizationAgent(FakeGenerator(response)).summarize(make_request()))


def test_agent_rejects_missing_evidence_without_calling_generator() -> None:
    request = SummarizationRequest(
        case=CaseInformation(case_id="case-1", title="Example case")
    )
    generator = FakeGenerator(SummarizationResponse())

    with pytest.raises(MissingEvidenceError):
        asyncio.run(SummarizationAgent(generator).summarize(request))

    assert generator.received is None


def test_agent_reports_unavailable_model_configuration() -> None:
    with pytest.raises(ModelUnavailableError):
        asyncio.run(SummarizationAgent().summarize(make_request()))


def test_handle_returns_shared_agent_result() -> None:
    response = SummarizationResponse(
        evidence_summaries=[
            EvidenceSummary(text="Grounded summary.", source_references=[SOURCE])
        ]
    )
    task = AgentTask(
        id="task-1",
        agent="summarization",
        instruction="Summarize the supplied evidence.",
        context=make_request().model_dump(mode="json"),
    )

    result = asyncio.run(SummarizationAgent(FakeGenerator(response)).handle(task))

    assert result.task_id == "task-1"
    assert result.agent == "summarization"
    assert result.status == "completed"
    assert result.output == response.model_dump(mode="json")


def test_handle_rejects_invalid_task_context() -> None:
    task = AgentTask(
        id="task-1",
        agent="summarization",
        instruction="Summarize.",
        context={"case": {"case_id": "case-1"}},
    )

    with pytest.raises(InvalidSummarizationInputError):
        asyncio.run(SummarizationAgent().handle(task))
