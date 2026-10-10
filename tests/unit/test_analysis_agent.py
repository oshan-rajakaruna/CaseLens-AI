"""Unit tests for the local Analysis Agent foundation."""

import pytest
from pydantic import ValidationError

from agents.analysis import AnalysisAgent, AnalysisRequest, AnalysisResponse
from agents.analysis.schemas import EvidenceSource, ExtractedEntity
from backend.schemas import Case, Document


def make_request(text: str = "The agreement was signed on 1 January 2024.") -> AnalysisRequest:
    """Build a valid local request without network or service dependencies."""

    return AnalysisRequest(
        case=Case(id="case-001", title="Example case"),
        document=Document(id="doc-001", case_id="case-001", name="agreement.txt"),
        text=text,
        document_metadata={"language": "en"},
    )


def test_valid_request_construction() -> None:
    request = make_request()

    assert request.case.id == "case-001"
    assert request.document.id == "doc-001"
    assert request.document_metadata == {"language": "en"}


def test_valid_typed_response() -> None:
    response = AnalysisAgent().analyze(make_request())

    assert isinstance(response, AnalysisResponse)
    assert response.status == "completed"
    assert response.stage_status["entities"] == "completed"


@pytest.mark.parametrize("text", ["", " \t\n "])
def test_empty_or_whitespace_text_requires_input(text: str) -> None:
    response = AnalysisAgent().analyze(make_request(text))

    assert response.status == "requires_input"
    assert response.warnings[0].code == "empty_text"
    assert response.facts == []


@pytest.mark.parametrize(
    ("case_id", "document_id"), [("", "doc-001"), ("case-001", "   ")]
)
def test_invalid_identifiers_are_rejected(case_id: str, document_id: str) -> None:
    with pytest.raises(ValidationError):
        AnalysisRequest(
            case=Case(id=case_id, title="Example case"),
            document=Document(id=document_id, case_id=case_id, name="evidence.txt"),
            text="Evidence",
        )


def test_mismatched_case_identifier_is_rejected() -> None:
    with pytest.raises(ValidationError, match="document.case_id must match case.id"):
        AnalysisRequest(
            case=Case(id="case-001", title="Example case"),
            document=Document(id="doc-001", case_id="case-002", name="evidence.txt"),
            text="Evidence",
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_invalid_confidence_values_are_rejected(confidence: float) -> None:
    source = EvidenceSource(document_id="doc-001", excerpt="Signed agreement")

    with pytest.raises(ValidationError):
        ExtractedEntity(
            text="Agreement",
            label="DOCUMENT",
            source=source,
            method="rule",
            confidence=confidence,
        )


def test_source_reference_validation() -> None:
    with pytest.raises(ValidationError, match="start_char is required"):
        EvidenceSource(document_id="doc-001", end_char=10)
    with pytest.raises(ValidationError, match="end_char must not precede"):
        EvidenceSource(document_id="doc-001", start_char=10, end_char=4)


def test_response_serializes_to_json() -> None:
    response = AnalysisAgent().analyze(make_request())
    payload = response.model_dump_json()

    assert '"case_id":"case-001"' in payload
    assert '"status":"completed"' in payload


def test_agent_has_no_external_api_dependency() -> None:
    agent = AnalysisAgent()

    assert not hasattr(agent, "client")
    assert agent.analyze(make_request()).errors == []
