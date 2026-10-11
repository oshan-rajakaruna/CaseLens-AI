"""Offline tests for deterministic structured legal report assembly."""

from copy import deepcopy

import pytest

from agents.analysis import AnalysisAgent
from agents.analysis.schemas import (
    AnalysisResponse,
    AnalysisWarning,
    ContradictionCandidate,
    EvidenceSource,
    ExtractedClause,
    ExtractedFact,
    LegalIssue,
    TimelineEvent,
)
from agents.summarization import GeminiSummarizationProvider
from agents.verification import VerificationAgent
from backend.schemas import (
    Case,
    Citation,
    CitationValidationResult,
    ClaimVerificationResult,
    LegalClaim,
    ResponsibleAIWarning,
    SourceMetadata,
    VerificationResponse,
)
from backend.schemas.legal_report import LegalResearchReport
from backend.schemas.legal_comparison import FactAuthoritySelection
from backend.schemas.summarization import SummarizationVerifyResponse
from backend.services.legal_report import (
    LegalResearchReportAssembler,
    LegalResearchReportValidationError,
)
from tests.unit.summarization.test_verification_bridge import (
    _passage,
    _payload,
    _result,
)


def _analysis(status: str = "completed", *, findings: bool = True) -> AnalysisResponse:
    source = EvidenceSource(
        document_id="PRIVATE-DOC-1",
        page=2,
        locator="page:2",
        start_char=0,
        end_char=24,
        excerpt="The tenant did not pay.",
    )
    if not findings:
        return AnalysisResponse(
            case_id="case-1",
            document_id="PRIVATE-DOC-1",
            status=status,
        )
    return AnalysisResponse(
        case_id="case-1",
        document_id="PRIVATE-DOC-1",
        status=status,
        facts=[
            ExtractedFact(
                statement="The tenant did not pay.",
                source=source,
                method="rule",
                confidence=0.8,
                epistemic_status="alleged",
                verification_status="not_independently_verified",
            )
        ],
        clauses=[
            ExtractedClause(
                text="The tenant shall pay rent.",
                clause_type="obligation",
                source=source,
                method="rule",
                confidence=0.7,
                epistemic_status="stated",
            )
        ],
        legal_issues=[
            LegalIssue(
                label="Payment dispute",
                sources=[source],
                method="rule",
                confidence=0.6,
                review_status="needs_review",
                epistemic_status="uncertain",
            )
        ],
        timeline=[
            TimelineEvent(
                description="Payment was allegedly missed.",
                date_text="12 January 2025",
                normalized_date="2025-01-12",
                source=source,
                method="rule",
                confidence=0.7,
                epistemic_status="alleged",
            )
        ],
        contradiction_candidates=[
            ContradictionCandidate(
                description="Payment and non-payment statements conflict.",
                sources=[source, source.model_copy(update={"page": 3})],
                confidence=0.5,
            )
        ],
        stage_status={"facts": "completed"},
    )


def _verification(status: str = "supported") -> VerificationResponse:
    citation = Citation(document_id="DOC-A", locator="CTX-001")
    claim = LegalClaim(
        id="summary-claim-001",
        text="The appeal was allowed.",
        citations=[citation],
    )
    source = SourceMetadata(document_id="DOC-A", title="Decision A")
    if status == "invalid_citations":
        result = ClaimVerificationResult(
            claim=claim,
            status="uncertain",
            citation_results=[
                CitationValidationResult(
                    citation=citation,
                    status="invalid",
                    issues=["Invalid citation."],
                )
            ],
            issues=["A citation is invalid."],
        )
    else:
        result = ClaimVerificationResult(
            claim=claim,
            status=status,
            evidence_ids=["CTX-001"],
            citation_results=[
                CitationValidationResult(
                    citation=citation,
                    status="valid",
                    evidence_id="CTX-001",
                    source=source,
                )
            ],
            issues=[] if status == "supported" else ["Review required."],
        )
    return VerificationResponse(
        verified=status == "supported",
        task_id="task-1",
        overall_status=status,
        claim_results=[result],
        citations=[citation],
        notes=list(result.issues),
        warnings=[
            ResponsibleAIWarning(
                code="legal_information_only",
                message="Human legal review is required.",
                severity="info",
            )
        ],
    )


def _response(payload=None, *, status: str = "supported") -> SummarizationVerifyResponse:
    payload = payload or _payload(_passage())
    summary = _result(
        "The appeal was allowed. [CTX-001]",
        ["CTX-001"],
    ).model_copy(update={"warnings": ["Summary warning"]})
    summary.output.limitations.append("Limited to retrieved material")
    return SummarizationVerifyResponse(
        summary=summary,
        verification=_verification(status),
        provenance={"CTX-001": payload.citation_map["CTX-001"]},
        bridge_warnings=["Whole summary was verified as one claim."],
    )


def _assemble(*, analyses=None, payload=None, status: str = "supported"):
    payload = payload or _payload(_passage())
    return LegalResearchReportAssembler().assemble(
        Case(id="case-1", title="Synthetic case", description="Research draft"),
        payload,
        _response(payload, status=status),
        analyses=analyses,
    )


def test_complete_report_preserves_all_analysis_findings_and_sources() -> None:
    report = _assemble(analyses=[_analysis()])

    assert isinstance(report, LegalResearchReport)
    assert report.analysis.availability == "completed"
    result = report.analysis.results[0]
    assert result.facts[0].source.locator == "page:2"
    assert result.facts[0].epistemic_status == "alleged"
    assert result.facts[0].verification_status == "not_independently_verified"
    assert result.clauses and result.legal_issues and result.timeline
    assert result.contradiction_candidates[0].review_status == "needs_review"


def test_analysis_not_supplied_is_distinct_from_empty_supplied_results() -> None:
    absent = _assemble()
    empty = _assemble(analyses=[])

    assert absent.analysis.availability == "not_supplied"
    assert empty.analysis.availability == "unavailable"
    assert absent.analysis.supplied is False
    assert empty.analysis.supplied is True


@pytest.mark.parametrize(
    ("status", "availability"),
    [("requires_input", "requires_input"), ("partial", "partial")],
)
def test_incomplete_analysis_status_and_diagnostics_are_preserved(
    status: str,
    availability: str,
) -> None:
    analysis = _analysis(status, findings=False).model_copy(
        update={
            "warnings": [AnalysisWarning(code="warning", message="Needs input")],
            "errors": [AnalysisWarning(code="facts_failed", message="Stage failed")],
            "stage_status": {"facts": "failed"},
        }
    )

    report = _assemble(analyses=[analysis])

    assert report.analysis.availability == availability
    assert report.analysis.results[0].warnings == analysis.warnings
    assert report.analysis.results[0].errors == analysis.errors
    assert "One or more Analysis stages reported errors." in report.limitations


def test_completed_analysis_with_zero_findings_remains_completed() -> None:
    report = _assemble(analyses=[_analysis(findings=False)])

    assert report.analysis.availability == "completed"
    assert report.analysis.results[0].facts == []


def test_analysis_case_mismatch_is_rejected_without_private_details() -> None:
    analysis = _analysis().model_copy(update={"case_id": "private-other-case"})

    with pytest.raises(
        LegalResearchReportValidationError,
        match="does not belong",
    ) as caught:
        _assemble(analyses=[analysis])

    assert "private-other-case" not in str(caught.value)


def test_authority_provenance_scores_and_summary_fields_are_preserved() -> None:
    payload = _payload(_passage())
    report = _assemble(payload=payload)
    authority = report.authorities[0]

    assert authority.context_id == "CTX-001"
    assert authority.document_id == "DOC-A"
    assert authority.chunk_ids == payload.citation_map["CTX-001"].chunk_ids
    assert authority.pages == [3]
    assert authority.retrieval_score == 1.0
    assert authority.provenance == "official_court_source"
    assert not hasattr(authority, "text")
    assert report.summary.output.limitations == ["Limited to retrieved material"]
    assert report.summary.warnings == ["Summary warning"]


def test_same_document_ctx_entries_are_retained_and_repeated_use_is_deduplicated() -> None:
    first = _passage("CTX-001", "DOC-A", page=3)
    second = _passage("CTX-002", "DOC-A", "Costs were awarded.", page=4)
    payload = _payload(first, second)
    response = _response(payload)
    response.summary = _result(
        "The appeal was allowed [CTX-001] and costs were awarded [CTX-002].",
        ["CTX-001", "CTX-002"],
    )
    response.summary.used_context_ids = ["CTX-001", "CTX-001", "CTX-002"]
    response.provenance["CTX-002"] = payload.citation_map["CTX-002"]
    response.verification.citations.append(
        Citation(document_id="DOC-A", locator="CTX-002")
    )

    report = LegalResearchReportAssembler().assemble(
        Case(id="case-1", title="Synthetic case"), payload, response
    )

    assert [item.context_id for item in report.authorities] == ["CTX-001", "CTX-002"]
    assert [item.document_id for item in report.authorities] == ["DOC-A", "DOC-A"]


def test_missing_optional_authority_metadata_is_not_invented() -> None:
    payload = _payload(_passage(title=None))
    report = _assemble(payload=payload)

    assert report.authorities[0].title is None
    assert any("incomplete optional metadata" in item for item in report.limitations)


@pytest.mark.parametrize(
    "mismatch",
    ["unknown", "response", "passage", "verification"],
)
def test_unknown_or_mismatched_ctx_identity_is_rejected(mismatch: str) -> None:
    payload = _payload(_passage())
    response = _response(payload)
    if mismatch == "unknown":
        response.summary.used_context_ids = ["CTX-999"]
    elif mismatch == "response":
        response.provenance["CTX-001"] = response.provenance["CTX-001"].model_copy(
            update={"document_id": "DOC-OTHER"}
        )
    elif mismatch == "passage":
        payload.context_passages[0].document_id = "DOC-OTHER"
    else:
        response.verification.citations[0].document_id = "DOC-OTHER"

    with pytest.raises(LegalResearchReportValidationError):
        LegalResearchReportAssembler().assemble(
            Case(id="case-1", title="Synthetic case"), payload, response
        )


@pytest.mark.parametrize(
    "status",
    [
        "supported",
        "partially_supported",
        "unsupported",
        "uncertain",
        "invalid_citations",
    ],
)
def test_verification_outcomes_are_preserved_without_report_wide_verification(
    status: str,
) -> None:
    report = _assemble(status=status)

    assert report.verification.response.overall_status == status
    assert report.verification.response.verified is (status == "supported")
    assert "verified" not in LegalResearchReport.model_fields
    assert "complete generated summary as one claim" in report.verification.scope


def test_disclaimer_and_unassessed_comparison_are_mandatory() -> None:
    report = _assemble()

    assert report.legal_comparison_status == "not_assessed"
    assert "AI-assisted legal research drafts" in report.human_review_disclaimer
    assert "Citation-ID validity" in report.human_review_disclaimer
    assert "professional legal advice" in report.human_review_disclaimer


def test_explicit_comparison_link_integrates_without_changing_summary_or_verification() -> None:
    payload = _payload(
        _passage("CTX-001", "DOC-A", page=3),
        _passage("CTX-002", "DOC-B", "A separate authority.", page=4),
    )
    response = _response(payload)
    report = LegalResearchReportAssembler().assemble(
        Case(id="case-1", title="Synthetic case"),
        payload,
        response,
        analyses=[_analysis()],
        selections=[
            FactAuthoritySelection(
                analysis_document_id="PRIVATE-DOC-1",
                fact_index=0,
                context_id="CTX-002",
            )
        ],
    )

    assert report.legal_comparison_status == "links_prepared_for_review"
    assert report.comparisons[0].authority.context_id == "CTX-002"
    assert [item.context_id for item in report.authorities] == ["CTX-001"]
    assert report.summary == response.summary
    assert report.verification.response == response.verification


def test_existing_report_call_without_selections_remains_backward_compatible() -> None:
    report = _assemble(analyses=[_analysis()])

    assert report.comparisons == []
    assert report.legal_comparison_status == "not_assessed"


def test_identical_inputs_are_deterministic_and_not_mutated() -> None:
    case = Case(id="case-1", title="Synthetic case")
    payload = _payload(_passage())
    response = _response(payload)
    analysis = _analysis()
    before = deepcopy(
        (case.model_dump(), payload.model_dump(), response.model_dump(), analysis.model_dump())
    )

    first = LegalResearchReportAssembler().assemble(case, payload, response, [analysis])
    second = LegalResearchReportAssembler().assemble(case, payload, response, [analysis])

    assert first.model_dump() == second.model_dump()
    assert before == (
        case.model_dump(),
        payload.model_dump(),
        response.model_dump(),
        analysis.model_dump(),
    )


def test_assembler_does_not_execute_agents_or_gemini(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("agent or provider execution is forbidden")

    monkeypatch.setattr(AnalysisAgent, "analyze", forbidden)
    monkeypatch.setattr(VerificationAgent, "verify", forbidden)
    monkeypatch.setattr(GeminiSummarizationProvider, "generate", forbidden)

    report = _assemble(analyses=[_analysis()])

    assert report.case_information.case.id == "case-1"
