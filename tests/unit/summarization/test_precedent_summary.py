"""Source-grounded deterministic judgment and precedent extraction tests."""

import pytest

from agents.summarization import (
    EvidenceItem,
    JudgmentMetadata,
    PrecedentSummarizer,
    RetrievedPassage,
    SourceReference,
    SummarizationAgent,
    SummarizationRequest,
)


def _source(document_id: str = "JUDG-001", citation_id: str | None = "CIT-001") -> SourceReference:
    return SourceReference(
        document_id=document_id,
        locator="judgment-paragraph-4",
        excerpt="Fictional judgment text.",
        page=4,
        paragraph=3,
        start_char=120,
        end_char=180,
        citation_id=citation_id,
    )


def _request(*, text: str | None = None, passages: list[RetrievedPassage] | None = None, **changes: object) -> SummarizationRequest:
    values: dict[str, object] = {
        "case_id": "CASE-200",
        "document_id": "JUDG-001",
        "query": "Summarize the fictional judgment.",
        "summary_type": "precedent_summary",
        "retrieved_passages": passages or [],
        "evidence_text": text,
        "judgment_metadata": JudgmentMetadata(
            case_title="Perera v. Oceanic Traders",
            case_number="CA-17/2025",
            court="Fictional Court of Appeal",
            jurisdiction="Fictional jurisdiction",
            judgment_date="2025-06-12",
            official_citation="[2025] FCA 17",
        ),
    }
    values.update(changes)
    return SummarizationRequest(**values)


def _judgment_text() -> str:
    return (
        "The claimant alleged that the respondent failed to deliver the goods. "
        "The issue before the court was whether the notice satisfied section 12 of the Fictional Contracts Act. "
        "The court reasoned that the notice was served within the contractual period. "
        "Accordingly, the appeal is dismissed. "
        "The court considered Silva v. Fernando."
    )


def test_basic_judgment_extracts_explicit_metadata_and_categories() -> None:
    source = _source()
    summary = PrecedentSummarizer().summarize(_request(passages=[
        RetrievedPassage(text=_judgment_text(), source=source, rank=1),
    ]))

    assert summary.case_identifier == "Perera v. Oceanic Traders (CA-17/2025)"
    assert summary.court == "Fictional Court of Appeal"
    assert summary.jurisdiction == "Fictional jurisdiction"
    assert summary.judgment_date == "2025-06-12"
    assert summary.background == "The claimant alleged that the respondent failed to deliver the goods."
    assert summary.legal_issues == [
        "The issue before the court was whether the notice satisfied section 12 of the Fictional Contracts Act."
    ]
    assert summary.reasoning == "The court reasoned that the notice was served within the contractual period."
    assert summary.holding == "Accordingly, the appeal is dismissed."
    assert summary.statutory_provisions == ["section 12"]
    assert summary.cited_authorities == ["Silva v. Fernando"]
    assert summary.citations == ["CIT-001"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The Court allows the appeal.", "The Court allows the appeal."),
        ("The Court dismisses the appeal.", "The Court dismisses the appeal."),
        ("Accordingly, the application is granted.", "Accordingly, the application is granted."),
    ],
)
def test_explicit_current_court_holdings_are_extracted(text: str, expected: str) -> None:
    summary = PrecedentSummarizer().summarize(_request(passages=[
        RetrievedPassage(text=text, source=_source(), rank=1),
    ]))

    assert summary.holding == expected
    point = next(point for point in summary.points if point.category == "holding")
    assert point.source_references == [_source()]


@pytest.mark.parametrize(
    "text",
    [
        "The claimant requested that the appeal be allowed.",
        "Counsel submitted that the appeal should be allowed.",
        "The High Court previously dismissed the claim.",
        "The judgment quoted Silva v. Fernando, where the appeal was dismissed.",
    ],
)
def test_non_current_or_party_statements_are_not_current_holdings(text: str) -> None:
    summary = PrecedentSummarizer().summarize(_request(passages=[
        RetrievedPassage(text=text, source=_source(), rank=1),
    ]))

    assert summary.holding is None
    assert "current_court_holding_not_explicitly_identified" in summary.uncertainty_flags


def test_party_submission_is_not_classified_as_judicial_reasoning() -> None:
    text = "Counsel submitted that section 12 required dismissal of the appeal."
    summary = PrecedentSummarizer().summarize(_request(passages=[
        RetrievedPassage(text=text, source=_source(), rank=1),
    ]))

    assert summary.reasoning is None
    assert all(point.category != "reasoning" for point in summary.points)


def test_explicit_refused_application_is_a_holding_but_generic_consideration_is_not_reasoning() -> None:
    source = _source()
    holding = PrecedentSummarizer().summarize(_request(passages=[
        RetrievedPassage(text="The application is refused.", source=source, rank=1),
    ]))
    generic = PrecedentSummarizer().summarize(_request(passages=[
        RetrievedPassage(text="The court considered section 8.", source=source, rank=1),
    ]))

    assert holding.holding == "The application is refused."
    assert generic.reasoning is None


def test_allegation_condition_and_negation_are_preserved_in_background() -> None:
    text = "The claimant alleged that if notice was not served, payment may not be due."
    summary = PrecedentSummarizer().summarize(_request(passages=[
        RetrievedPassage(text=text, source=_source(), rank=1),
    ]))

    point = next(point for point in summary.points if point.category == "background")
    assert point.text == text
    assert point.is_attributed is True
    assert point.is_conditional is True
    assert point.is_negated is True


def test_raw_judgment_is_source_limited_not_fabricated() -> None:
    summary = PrecedentSummarizer().summarize(_request(
        text="The Court dismisses the appeal.",
    ))

    point = next(point for point in summary.points if point.category == "holding")
    assert point.source_references == []
    assert point.citations == []
    assert point.provenance_limited is True
    assert "limited_provenance" in summary.uncertainty_flags


def test_multiple_sources_and_citations_are_preserved() -> None:
    first = RetrievedPassage(
        text="The issue before the court was whether notice was valid.",
        source=_source("JUDG-001", "CIT-001"), rank=1,
    )
    second = RetrievedPassage(
        text="The court reasoned that notice was valid.",
        source=_source("JUDG-002", "CIT-002"), rank=2,
    )
    request = _request(
        passages=[first, second],
        document_id=None,
        document_ids=["JUDG-001", "JUDG-002"],
    )
    summary = PrecedentSummarizer().summarize(request)

    assert [source.document_id for source in summary.source_references] == ["JUDG-001", "JUDG-002"]
    assert summary.citations == ["CIT-001", "CIT-002"]


def test_missing_metadata_and_missing_sections_are_explicit() -> None:
    request = _request(
        passages=[RetrievedPassage(
            text="The court considered section 8.", source=_source(), rank=1,
        )],
        judgment_metadata=None,
    )
    summary = PrecedentSummarizer().summarize(request)

    assert summary.case_identifier == "unknown"
    assert summary.court is None
    assert summary.holding is None
    assert "legal_issue_not_explicitly_identified" in summary.uncertainty_flags
    assert "current_court_holding_not_explicitly_identified" in summary.uncertainty_flags


def test_long_judgment_is_bounded_and_warned() -> None:
    sentence = "The Court dismisses the appeal."
    summary = PrecedentSummarizer().summarize(_request(passages=[
        RetrievedPassage(text=" ".join([sentence] * 1_000), source=_source(), rank=1),
    ]))

    assert "judgment_input_truncated_at_safe_sentence_boundary" in summary.uncertainty_flags
    assert len(summary.points) <= 3


def test_duplicate_judgment_statement_is_stable_and_json_serializable() -> None:
    source = _source()
    request = _request(passages=[
        RetrievedPassage(text="The Court dismisses the appeal.", source=source, rank=1),
        RetrievedPassage(text="The Court dismisses the appeal.", source=source, rank=2),
    ])
    first = PrecedentSummarizer().summarize(request)
    second = PrecedentSummarizer().summarize(request)

    assert first == second
    assert len([point for point in first.points if point.category == "holding"]) == 1
    assert '"CIT-001"' in first.model_dump_json()


def test_agent_dispatches_precedent_without_calling_legacy_provider() -> None:
    class FailingProvider:
        provider_name = "failing"
        model_name = None

        def generate(self, request: object) -> object:
            raise AssertionError("precedent extraction must remain offline")

    response = SummarizationAgent(FailingProvider()).summarize(_request(passages=[
        RetrievedPassage(text="The Court dismisses the appeal.", source=_source(), rank=1),
    ]))

    assert response.status == "partial"
    assert response.precedent_summaries[0].holding == "The Court dismisses the appeal."


def test_evidence_summary_default_and_legacy_path_remain_compatible() -> None:
    evidence_request = SummarizationRequest(
        case_id="CASE-201",
        document_id="DOC-201",
        query="What happened?",
        evidence_items=[EvidenceItem(
            text="ABC Ltd must pay LKR 4,000.", source=SourceReference(document_id="DOC-201")
        )],
    )

    response = SummarizationAgent().summarize(evidence_request)

    assert response.status == "completed"
    assert response.evidence_summary is not None
    assert response.precedent_summaries == []

