"""Tests for conservative, source-grounded case-to-precedent comparison."""

import pytest
from pydantic import ValidationError

from agents.summarization import (
    CaseComparisonRequest,
    CasePrecedentComparator,
    EvidenceSummary,
    EvidenceSummaryPoint,
    PrecedentSummary,
    PrecedentSummaryPoint,
    SourceReference,
    SummarizationAgent,
    SummarizationRequest,
)


def _source(doc: str, citation: str | None) -> SourceReference:
    return SourceReference(document_id=doc, locator="paragraph-3", page=3, citation_id=citation)


def _current(text: str, *, source: SourceReference | None = None, **flags: bool) -> EvidenceSummaryPoint:
    return EvidenceSummaryPoint(
        text=text,
        source_references=[source] if source else [],
        citations=[source.citation_id] if source and source.citation_id else [],
        provenance_limited=source is None,
        **flags,
    )


def _precedent(text: str, *, source: SourceReference | None = None, **flags: bool) -> PrecedentSummaryPoint:
    return PrecedentSummaryPoint(
        text=text,
        category="holding",
        source_references=[source] if source else [],
        citations=[source.citation_id] if source and source.citation_id else [],
        provenance_limited=source is None,
        **flags,
    )


def _request(
    current_points: list[EvidenceSummaryPoint],
    precedent_points: list[PrecedentSummaryPoint],
    **changes: object,
) -> CaseComparisonRequest:
    precedent = PrecedentSummary(
        case_identifier="Oceanic Traders v. Perera",
        court="Fictional Commercial Court",
        jurisdiction="Singapore",
        points=precedent_points,
    )
    values: dict[str, object] = {
        "case_id": "SL-CASE-001",
        "current_case_points": current_points,
        "precedents": [precedent],
        "current_case_jurisdiction": "Sri Lanka",
    }
    values.update(changes)
    return CaseComparisonRequest(**values)


def test_same_payment_obligation_is_a_source_grounded_similarity() -> None:
    current_source = _source("SL-DOC-1", "SRC-001")
    precedent_source = _source("SG-JUDG-1", "CIT-001")
    comparison = CasePrecedentComparator().compare(_request(
        [_current("ABC Holdings must pay LKR 500,000 by 15 March 2025.", source=current_source)],
        [_precedent("The agreement required payment by 15 March 2025.", source=precedent_source)],
    ))[0]

    point = next(item for item in comparison.points if item.comparison_type == "similarity")
    assert point.topic == "payment_obligation"
    assert point.current_case_sources == [current_source]
    assert point.precedent_sources == [precedent_source]
    assert comparison.authority_status == "not_determined"


@pytest.mark.parametrize(
    ("current", "precedent", "phrase"),
    [
        ("ABC Ltd must pay LKR 500,000.", "The contract required LKR 700,000 payment.", "payment amount or currency"),
        ("ABC Ltd must pay USD 500.", "The contract required LKR 500 payment.", "payment amount or currency"),
        ("Payment is due 15 March 2025.", "Payment is due 16 March 2025.", "stated date or deadline"),
    ],
)
def test_explicit_amount_currency_and_deadline_differences_are_reported(
    current: str, precedent: str, phrase: str
) -> None:
    comparison = CasePrecedentComparator().compare(_request(
        [_current(current, source=_source("SL-DOC-1", "SRC-001"))],
        [_precedent(precedent, source=_source("SG-JUDG-1", "CIT-001"))],
    ))[0]

    assert comparison.supported_differences
    assert phrase in comparison.supported_differences[0]


def test_missing_information_is_not_reported_as_a_contradiction() -> None:
    comparison = CasePrecedentComparator().compare(_request(
        [_current("The agreement requires a written notice.", source=_source("SL-DOC-1", "SRC-001"))],
        [_precedent("The Court dismisses the appeal.", source=_source("SG-JUDG-1", "CIT-001"))],
    ))[0]

    point = next(item for item in comparison.points if item.comparison_type == "missing_information")
    assert "precedent information for that topic was not supplied" in point.label
    assert "missing_information_is_not_a_contradiction" in point.limitations


def test_allegation_is_not_equated_with_judicial_finding() -> None:
    comparison = CasePrecedentComparator().compare(_request(
        [_current(
            "The claimant alleged that payment was not made.",
            source=_source("SL-DOC-1", "SRC-001"), is_negated=True, is_attributed=True,
        )],
        [_precedent(
            "The court found that payment was not made.",
            source=_source("SG-JUDG-1", "CIT-001"), is_negated=True,
        )],
    ))[0]

    point = next(item for item in comparison.points if item.topic == "payment_obligation")
    assert "different_attribution_or_allegation_status" in point.limitations
    assert point.current_case_statement.startswith("The claimant alleged")
    assert point.precedent_statement.startswith("The court found")


def test_conditionality_and_negation_are_preserved_as_review_limits() -> None:
    comparison = CasePrecedentComparator().compare(_request(
        [_current(
            "If notice is not served, the agreement may terminate.",
            source=_source("SL-DOC-1", "SRC-001"), is_negated=True, is_conditional=True,
        )],
        [_precedent(
            "The agreement terminated after notice was served.",
            source=_source("SG-JUDG-1", "CIT-001"),
        )],
    ))[0]

    point = next(item for item in comparison.points if item.topic == "termination_condition")
    assert "different_negation_status" in point.limitations
    assert "different_conditional_status" in point.limitations


def test_cross_jurisdiction_is_flagged_without_binding_authority_claim() -> None:
    comparison = CasePrecedentComparator().compare(_request(
        [_current("The agreement requires notice.", source=_source("SL-DOC-1", "SRC-001"))],
        [_precedent("The court considered a notice requirement.", source=_source("SG-JUDG-1", "CIT-001"))],
    ))[0]

    jurisdiction = next(item for item in comparison.points if item.comparison_type == "jurisdiction")
    assert "authority is not determined" in jurisdiction.label
    assert comparison.authority_status == "not_determined"
    assert "human_legal_review_required" in comparison.limitations


def test_same_keyword_without_shared_controlled_topic_is_not_a_similarity() -> None:
    comparison = CasePrecedentComparator().compare(_request(
        [_current("Payment was mentioned during a social event.", source=_source("SL-DOC-1", "SRC-001"))],
        [_precedent("The Court dismisses the appeal.", source=_source("SG-JUDG-1", "CIT-001"))],
    ))[0]

    assert not comparison.supported_similarities
    assert "no_supported_topic_alignment" in comparison.limitations


def test_limited_provenance_does_not_fabricate_citations() -> None:
    comparison = CasePrecedentComparator().compare(_request(
        [_current("The contract requires payment.")],
        [_precedent("The contract required payment.")],
    ))[0]

    point = next(item for item in comparison.points if item.topic == "payment_obligation")
    assert point.provenance_limited is True
    assert point.current_case_sources == []
    assert point.precedent_sources == []
    assert comparison.supporting_sources == []


def test_multiple_precedents_remain_separate() -> None:
    first = PrecedentSummary(case_identifier="First v. Party", points=[
        _precedent("The contract required payment.", source=_source("J-1", "CIT-001"))
    ])
    second = PrecedentSummary(case_identifier="Second v. Party", points=[
        _precedent("The agreement requires notice.", source=_source("J-2", "CIT-002"))
    ])
    result = CasePrecedentComparator().compare(CaseComparisonRequest(
        case_id="SL-CASE-001",
        current_case_points=[_current("The contract requires payment.", source=_source("SL-DOC-1", "SRC-001"))],
        precedents=[first, second],
    ))

    assert [item.precedent_identifier for item in result] == ["First v. Party", "Second v. Party"]
    assert result[0].points != result[1].points


def test_request_validation_and_agent_dispatch_are_safe_and_json_serializable() -> None:
    with pytest.raises(ValidationError, match="current_case_summary or current_case_points"):
        CaseComparisonRequest(case_id="SL-CASE-001")
    with pytest.raises(ValidationError, match="comparison_request"):
        SummarizationRequest(
            case_id="SL-CASE-001", document_id="SL-DOC-1", query="Compare", summary_type="case_comparison"
        )

    compare_request = _request(
        [_current("The contract requires payment.", source=_source("SL-DOC-1", "SRC-001"))],
        [_precedent("The contract required payment.", source=_source("SG-JUDG-1", "CIT-001"))],
    )
    response = SummarizationAgent().summarize(SummarizationRequest(
        case_id="SL-CASE-001",
        document_id="SL-DOC-1",
        query="Compare supplied materials.",
        summary_type="case_comparison",
        comparison_request=compare_request,
    ))
    assert response.status == "completed"
    assert response.case_comparisons[0].precedent_identifier == "Oceanic Traders v. Perera"
    assert '"authority_status":"not_determined"' in response.model_dump_json()


def test_no_legacy_provider_or_network_is_used_for_comparison() -> None:
    class FailingProvider:
        provider_name = "failing"
        model_name = None

        def generate(self, request: object) -> object:
            raise AssertionError("comparison must not call a provider")

    compare_request = _request(
        [_current("The contract requires payment.", source=_source("SL-DOC-1", "SRC-001"))],
        [_precedent("The contract required payment.", source=_source("SG-JUDG-1", "CIT-001"))],
    )
    response = SummarizationAgent(FailingProvider()).summarize(SummarizationRequest(
        case_id="SL-CASE-001", document_id="SL-DOC-1", query="Compare",
        summary_type="case_comparison", comparison_request=compare_request,
    ))
    assert response.case_comparisons

