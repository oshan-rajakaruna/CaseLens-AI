"""Tests for the deterministic, source-grounded summarization foundation."""

import pytest
from pydantic import ValidationError

from agents.summarization import (
    AnalysisFinding,
    EvidenceItem,
    RetrievedPassage,
    SourceReference,
    SummarizationAgent,
    SummarizationRequest,
)


def _source() -> SourceReference:
    return SourceReference(
        document_id="DOC-001",
        locator="paragraph-2",
        excerpt="Ms Perera alleges that the payment was not made.",
        page=2,
        paragraph=1,
        start_char=14,
        end_char=61,
        citation_id="SRC-001",
    )


def _request(**changes: object) -> SummarizationRequest:
    values: dict[str, object] = {
        "case_id": "CASE-001",
        "document_id": "DOC-001",
        "query": "What evidence is available?",
        "evidence_items": [
            EvidenceItem(
                text="Ms Perera alleges that the payment was not made.",
                source=_source(),
                is_negated=True,
                is_attributed=True,
                epistemic_status="alleged",
            )
        ],
    }
    values.update(changes)
    return SummarizationRequest(**values)


def test_foundation_returns_extractively_summarized_provenance_preserving_response() -> None:
    request = _request()

    response = SummarizationAgent().summarize(request)

    assert response.status == "completed"
    assert response.evidence_summary is not None
    assert response.evidence_summary.summary_text == (
        "Ms Perera alleges that the payment was not made."
    )
    assert response.precedent_summaries == []
    assert response.case_comparisons == []
    assert response.report_sections == []
    assert response.source_references == [_source()]
    assert response.citations == ["SRC-001"]
    assert response.evidence_summary.points[0].is_negated is True
    assert response.evidence_summary.points[0].is_attributed is True
    assert response.stage_status == {
        "validation": "completed",
        "evidence_summarization": "completed",
    }


def test_empty_evidence_requires_input_without_fabricating_a_summary() -> None:
    response = SummarizationAgent().summarize(
        _request(evidence_items=[], evidence_text=None)
    )

    assert response.status == "requires_input"
    assert response.evidence_summary is None
    assert response.citations == []
    assert [warning.code for warning in response.warnings] == [
        "NO_EVIDENCE",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", " "),
        ("document_id", ""),
        ("query", "\t"),
    ],
)
def test_blank_required_identifiers_and_query_are_rejected(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValidationError):
        _request(**{field: value})


def test_request_requires_a_document_identifier() -> None:
    with pytest.raises(ValidationError, match="document_id or document_ids"):
        _request(document_id=None, document_ids=[])


@pytest.mark.parametrize("citation_id", ["CTX-001", "SRC-001", "CIT-Case.12"])
def test_source_citation_formats_are_supported(citation_id: str) -> None:
    assert SourceReference(document_id="DOC-001", citation_id=citation_id).citation_id == citation_id


@pytest.mark.parametrize("citation_id", ["CTX-1", "unknown", "[CTX-001]"])
def test_invalid_source_citation_formats_are_rejected(citation_id: str) -> None:
    with pytest.raises(ValidationError, match="citation_id"):
        SourceReference(document_id="DOC-001", citation_id=citation_id)


def test_source_document_must_be_declared_by_request() -> None:
    with pytest.raises(ValidationError, match="source document_id"):
        _request(
            evidence_items=[
                EvidenceItem(
                    text="Synthetic evidence.",
                    source=SourceReference(document_id="DOC-OTHER"),
                )
            ]
        )


def test_analysis_semantics_and_retrieval_metadata_are_preserved() -> None:
    source = _source()
    request = _request(
        evidence_items=[],
        analysis_findings=[
            AnalysisFinding(
                finding_type="fact",
                statement="The witness did not confirm payment.",
                sources=[source],
                is_negated=True,
                is_attributed=True,
                epistemic_status="alleged",
            )
        ],
        retrieved_passages=[
            RetrievedPassage(
                text="A fictional precedent discusses payment evidence.",
                source=source,
                rank=1,
                relevance_score=0.71,
            )
        ],
    )

    response = SummarizationAgent().summarize(request)

    assert response.source_references == [source]
    assert response.citations == ["SRC-001"]
    assert request.analysis_findings[0].is_negated is True
    assert request.analysis_findings[0].is_attributed is True
    assert request.model_dump(mode="json")["retrieved_passages"][0]["rank"] == 1


def test_response_has_pydantic_json_serialization() -> None:
    response = SummarizationAgent().summarize(_request())

    encoded = response.model_dump_json()

    assert '"case_id":"CASE-001"' in encoded
    assert '"status":"completed"' in encoded


def test_foundation_path_never_calls_an_injected_legacy_provider() -> None:
    class FailingProvider:
        provider_name = "failing"
        model_name = None

        def generate(self, request: object) -> object:
            raise AssertionError("foundation path must not call a provider")

    response = SummarizationAgent(FailingProvider()).summarize(_request())

    assert response.status == "completed"

