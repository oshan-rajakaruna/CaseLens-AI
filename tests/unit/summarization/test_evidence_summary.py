"""Deterministic source-grounded evidence summarization tests."""

import pytest
from pydantic import ValidationError

from agents.summarization import (
    AnalysisFinding,
    EvidenceItem,
    EvidenceSummarizer,
    RetrievedPassage,
    SourceReference,
    SummarizationAgent,
    SummarizationRequest,
)


def _source(document_id: str = "DOC-001", citation_id: str | None = "SRC-001") -> SourceReference:
    return SourceReference(
        document_id=document_id,
        locator="paragraph-1",
        excerpt="Synthetic legal evidence.",
        page=1,
        paragraph=0,
        start_char=0,
        end_char=25,
        citation_id=citation_id,
    )


def _request(items: list[EvidenceItem], **changes: object) -> SummarizationRequest:
    values: dict[str, object] = {
        "case_id": "CASE-100",
        "document_ids": ["DOC-001", "DOC-002"],
        "query": "What does the supplied evidence state?",
        "evidence_items": items,
    }
    values.update(changes)
    return SummarizationRequest(**values)


def test_single_grounded_payment_statement_is_extracted_without_inference() -> None:
    text = "ABC Holdings agreed to pay LKR 500,000 by 15 March 2025."
    result = EvidenceSummarizer().summarize(
        _request([EvidenceItem(text=text, source=_source())])
    )

    assert result.summary_text == text
    assert result.key_points == [text]
    assert result.points[0].source_references == [_source()]
    assert result.points[0].citations == ["SRC-001"]
    assert "breach" not in result.summary_text.casefold()


def test_multiple_documents_and_actual_citations_are_preserved() -> None:
    first = "Ms Perera sent a notice on 12 April 2025."
    second = "Lanka Trading Ltd denied receiving the notice."
    result = EvidenceSummarizer().summarize(_request([
        EvidenceItem(text=first, source=_source("DOC-001", "SRC-001")),
        EvidenceItem(text=second, source=_source("DOC-002", "SRC-002"), is_negated=True),
    ]))

    assert result.key_points == [first, second]
    assert [point.citations for point in result.points] == [["SRC-001"], ["SRC-002"]]
    assert [source.document_id for source in result.source_references] == ["DOC-001", "DOC-002"]


@pytest.mark.parametrize(
    ("text", "field", "value"),
    [
        ("XYZ Ltd alleged that payment was not made.", "is_attributed", True),
        ("The defendant did not receive the notice.", "is_negated", True),
        ("If approval is granted, the agreement will commence.", "is_conditional", True),
        ("Payment may be due in April.", "is_attributed", False),
    ],
)
def test_semantic_qualifiers_are_never_removed(
    text: str,
    field: str,
    value: bool,
) -> None:
    item = EvidenceItem(
        text=text,
        source=_source(),
        is_attributed="alleged" in text.casefold(),
        is_negated="not" in text.casefold() or "did not" in text.casefold(),
        is_conditional=text.startswith("If"),
        epistemic_status="alleged" if "alleged" in text.casefold() else "stated",
    )
    point = EvidenceSummarizer().summarize(_request([item])).points[0]

    assert point.text == text
    assert getattr(point, field) is value
    for token in ("alleged", "not", "may", "If"):
        if token.casefold() in text.casefold():
            assert token in point.text


def test_analysis_finding_preserves_allegation_and_uses_its_actual_source() -> None:
    source = _source()
    request = _request([], analysis_findings=[AnalysisFinding(
        finding_type="fact",
        statement="The claimant stated that the goods were defective.",
        sources=[source],
        is_attributed=True,
        epistemic_status="alleged",
    )])

    point = EvidenceSummarizer().summarize(request).points[0]

    assert point.text == "The claimant stated that the goods were defective."
    assert point.is_attributed is True
    assert point.epistemic_status == "alleged"
    assert point.source_references == [source]


def test_raw_evidence_is_extractable_but_explicitly_provenance_limited() -> None:
    request = _request([], evidence_text="The contract required payment of Rs. 25,000.")

    result = EvidenceSummarizer().summarize(request)

    assert result.points[0].text == "The contract required payment of Rs. 25,000."
    assert result.points[0].source_references == []
    assert result.points[0].citations == []
    assert result.points[0].provenance_limited is True
    assert "limited_provenance" in result.uncertainty_flags


def test_duplicate_text_from_two_actual_sources_merges_provenance() -> None:
    text = "ABC Holdings must pay LKR 100,000 by 1 May 2025."
    result = EvidenceSummarizer().summarize(_request([
        EvidenceItem(text=text, source=_source("DOC-001", "SRC-001")),
        EvidenceItem(text=text, source=_source("DOC-002", "SRC-002")),
    ]))

    assert result.key_points == [text]
    assert [source.citation_id for source in result.points[0].source_references] == ["SRC-001", "SRC-002"]
    assert result.points[0].citations == ["SRC-001", "SRC-002"]


def test_irrelevant_evidence_is_not_promoted_to_a_summary_point() -> None:
    result = EvidenceSummarizer().summarize(_request([
        EvidenceItem(text="The sky was clear.", source=_source()),
    ]))

    assert result.points == []
    assert result.summary_text == ""
    assert "no_significant_evidence_selected" in result.uncertainty_flags


def test_empty_evidence_returns_requires_input_from_agent() -> None:
    response = SummarizationAgent().summarize(_request([]))

    assert response.status == "requires_input"
    assert response.evidence_summary is None
    assert response.warnings[0].code == "NO_EVIDENCE"


def test_malformed_evidence_and_invalid_source_are_rejected_by_models() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(text=" ", source=_source())
    with pytest.raises(ValidationError):
        SourceReference(document_id="DOC-001", start_char=5, end_char=2)


def test_retrieved_passage_is_used_only_with_its_supplied_source() -> None:
    source = _source("DOC-002", "CTX-002")
    request = _request([], retrieved_passages=[RetrievedPassage(
        text="The court recorded that a notice was filed.", source=source, rank=1
    )])

    point = EvidenceSummarizer().summarize(request).points[0]

    assert point.source_references == [source]
    assert point.citations == ["CTX-002"]


def test_long_input_stops_at_sentence_boundary_and_warns() -> None:
    sentence = "ABC Holdings must pay LKR 1,000 by 1 May 2025."
    result = EvidenceSummarizer().summarize(_request([
        EvidenceItem(text=" ".join([sentence] * 500), source=_source()),
    ]))

    assert "input_truncated_at_safe_sentence_boundary" in result.uncertainty_flags
    assert all(point.text.endswith(".") for point in result.points)
    assert len(result.points) <= 6


def test_output_is_json_serializable_and_deterministic() -> None:
    request = _request([EvidenceItem(
        text="ABC Holdings shall pay LKR 9,000 on 5 June 2025.", source=_source()
    )])
    first = EvidenceSummarizer().summarize(request)
    second = EvidenceSummarizer().summarize(request)

    assert first == second
    assert '"SRC-001"' in first.model_dump_json()


def test_agent_does_not_call_a_legacy_provider_for_evidence_requests() -> None:
    class FailingProvider:
        provider_name = "test"
        model_name = None

        def generate(self, request: object) -> object:
            raise AssertionError("deterministic evidence path must not use a provider")

    response = SummarizationAgent(FailingProvider()).summarize(_request([
        EvidenceItem(text="ABC Ltd must pay LKR 1,000.", source=_source()),
    ]))
    assert response.status == "completed"

