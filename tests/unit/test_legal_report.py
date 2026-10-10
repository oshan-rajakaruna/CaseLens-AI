"""Offline tests for deterministic legal research report assembly."""

import pytest

from agents.summarization import (
    CaseInformation,
    Comparison,
    DraftClaim,
    EvidenceSummary,
    ExtractedFact,
    LegalPassage,
    PrecedentSummary,
    SourceReference,
    SummarizationRequest,
    SummarizationResponse,
)
from backend.schemas import Citation
from backend.services import (
    InconsistentCaseInformationError,
    InvalidReportInputError,
    LegalReportAssembler,
    MarkdownLegalReportRenderer,
    UnsupportedReportSourceError,
)


EVIDENCE = SourceReference(document_id="evidence-1", locator="page 2")
PRECEDENT = SourceReference(document_id="precedent-1", locator="paragraph 18")
FACT_TEXT = "The supplied witness statement records the event."
PASSAGE_TEXT = "The supplied judgment discusses the relevant legal principle."


def make_source_request(
    case: CaseInformation | None = None,
) -> SummarizationRequest:
    return SummarizationRequest(
        case=case or CaseInformation(
            case_id="case-1",
            title="Synthetic research matter",
            description="A synthetic case used only for unit testing.",
        ),
        extracted_facts=[
            ExtractedFact(
                fact_id="fact-1",
                text=FACT_TEXT,
                source_references=[EVIDENCE],
            )
        ],
        retrieved_passages=[
            LegalPassage(
                text=PASSAGE_TEXT,
                source=PRECEDENT,
                authority="Synthetic authority",
            )
        ],
    )


def make_complete_summary() -> SummarizationResponse:
    citation = Citation(
        document_id=EVIDENCE.document_id,
        locator=EVIDENCE.locator,
        excerpt=FACT_TEXT,
    )
    return SummarizationResponse(
        evidence_summaries=[
            EvidenceSummary(
                text="Evidence summary.",
                source_references=[EVIDENCE, EVIDENCE],
            )
        ],
        precedent_summaries=[
            PrecedentSummary(
                text="Precedent summary.", source_references=[PRECEDENT]
            )
        ],
        comparisons=[
            Comparison(
                subject="Factual comparison",
                analysis="Draft comparison.",
                source_references=[EVIDENCE, PRECEDENT],
            )
        ],
        draft_claims=[
            DraftClaim(text="Provisional finding.", supporting_sources=[EVIDENCE])
        ],
        citations=[citation, citation.model_copy(deep=True)],
        warnings=["Requires independent verification."],
    )


def test_complete_report_assembly_preserves_sources_and_status() -> None:
    request = make_source_request()

    report = LegalReportAssembler().assemble(
        request.case, request, make_complete_summary()
    )

    assert report.case.case_id == "case-1"
    assert report.verification_status == "pending_verification"
    assert report.draft_findings[0].status == "draft"
    assert report.evidence_summaries[0].source_references == [EVIDENCE]
    assert report.comparisons[0].source_references == [EVIDENCE, PRECEDENT]
    assert len(report.citations) == 1
    assert report.limitations == []


def test_evidence_only_report_flags_missing_sections() -> None:
    case = CaseInformation(case_id="case-1", title="Evidence-only matter")
    request = SummarizationRequest(
        case=case,
        extracted_facts=[
            ExtractedFact(
                fact_id="fact-1",
                text=FACT_TEXT,
                source_references=[EVIDENCE],
            )
        ],
    )
    summary = SummarizationResponse(
        evidence_summaries=[
            EvidenceSummary(text="Evidence summary.", source_references=[EVIDENCE])
        ]
    )

    report = LegalReportAssembler().assemble(case, request, summary)

    assert report.precedent_summaries == []
    assert report.comparisons == []
    assert "No legal precedent summaries were supplied." in report.limitations
    assert "No case description was supplied." in report.limitations


def test_direct_assembly_rejects_unsupported_reference_and_citation() -> None:
    request = make_source_request()
    unsupported = SourceReference(document_id="unknown", locator="page 99")
    unsupported_summary = SummarizationResponse(
        evidence_summaries=[
            EvidenceSummary(text="Unsupported.", source_references=[unsupported])
        ]
    )
    unsupported_citation = SummarizationResponse(
        citations=[Citation(document_id="unknown", locator="page 99")]
    )
    fabricated_excerpt = SummarizationResponse(
        citations=[
            Citation(
                document_id=EVIDENCE.document_id,
                locator=EVIDENCE.locator,
                excerpt="Text absent from the supplied evidence.",
            )
        ]
    )

    with pytest.raises(UnsupportedReportSourceError):
        LegalReportAssembler().assemble(
            request.case, request, unsupported_summary
        )
    with pytest.raises(UnsupportedReportSourceError):
        LegalReportAssembler().assemble(
            request.case, request, unsupported_citation
        )
    with pytest.raises(UnsupportedReportSourceError):
        LegalReportAssembler().assemble(
            request.case, request, fabricated_excerpt
        )


def test_inconsistent_or_invalid_case_input_is_rejected() -> None:
    request = make_source_request()
    different_case = CaseInformation(case_id="case-2", title="Different matter")

    with pytest.raises(InconsistentCaseInformationError):
        LegalReportAssembler().assemble(
            different_case, request, SummarizationResponse()
        )
    with pytest.raises(InvalidReportInputError):
        LegalReportAssembler().assemble(  # type: ignore[arg-type]
            {"case_id": "case-1"}, request, SummarizationResponse()
        )


def test_markdown_renderer_escapes_untrusted_structures() -> None:
    case = CaseInformation(
        case_id="case-1",
        title="Matter\n# Injected heading",
        description="**untrusted emphasis**",
    )
    request = make_source_request(case)
    summary = SummarizationResponse(
        evidence_summaries=[
            EvidenceSummary(
                text="Evidence\n## False section", source_references=[EVIDENCE]
            )
        ],
        warnings=["[Misleading link](https://example.invalid)"],
    )
    report = LegalReportAssembler().assemble(case, request, summary)

    markdown = MarkdownLegalReportRenderer().render(report)

    assert "# Draft Legal Research Report" in markdown
    assert "## Evidence Summaries" in markdown
    assert "## Relevant Legal Precedents\n\nNot provided." in markdown
    assert "pending_verification" in markdown
    assert "\\# Injected heading" in markdown
    assert "\\#\\# False section" in markdown
    assert "\\*\\*untrusted emphasis\\*\\*" in markdown
    assert "\\[Misleading link\\]\\(https://example\\.invalid\\)" in markdown
    assert "## Disclaimer" in markdown


def test_markdown_renderer_rejects_invalid_report_data() -> None:
    with pytest.raises(InvalidReportInputError):
        MarkdownLegalReportRenderer().render(  # type: ignore[arg-type]
            {"verification_status": "verified"}
        )


def test_report_assembly_is_deterministic_and_has_no_generator_dependency() -> None:
    request = make_source_request()
    summary = make_complete_summary()
    assembler = LegalReportAssembler()

    first = assembler.assemble(request.case, request, summary)
    second = assembler.assemble(request.case, request, summary)

    assert first == second
