"""Offline end-to-end coverage for the internal research workflow."""

from typing import Any

import pytest

from agents.analysis.schemas import (
    AnalysisResponse,
    AnalysisWarning,
    EvidenceSource,
    ExtractedFact,
)
from agents.coordinator import LegalRetrievalContextService
from agents.summarization import SummarizationAgent, SummarizationVerificationBridge
from agents.verification import VerificationAgent
from backend.schemas.common import Case, Document
from backend.schemas.legal_comparison import (
    FactAuthoritySelection,
    LegalComparisonProviderOutput,
)
from backend.schemas.research_workflow import (
    EvidenceDocumentInput,
    ResearchWorkflowRequest,
)
from backend.services.legal_comparison_generation import LegalComparisonDraftService
from backend.services.research_workflow import ResearchWorkflow
from backend.services.summarization_service import SummarizationService
from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    LegalDocumentMetadata,
)


class _Analysis:
    def __init__(self, status: str = "completed") -> None:
        self.status = status
        self.calls = 0

    def analyze(self, request) -> AnalysisResponse:
        self.calls += 1
        return AnalysisResponse(
            case_id=request.case.id,
            document_id=request.document.id,
            status=self.status,
            facts=[
                ExtractedFact(
                    statement="The tenant received notice.",
                    source=EvidenceSource(
                        document_id=request.document.id,
                        page=2,
                        locator="page:2",
                        excerpt="PRIVATE EVIDENCE",
                    ),
                    method="test-rule",
                    verification_status="not_independently_verified",
                )
            ],
            errors=(
                [AnalysisWarning(code="facts_failed", message="Facts were partial.")]
                if self.status == "partial"
                else []
            ),
            stage_status={"facts": "failed" if self.status == "partial" else "completed"},
        )


class _RetrievalAgent:
    def __init__(self, results: list[BM25SearchResult] | None = None) -> None:
        self.results = _authority_results() if results is None else results
        self.calls = 0

    def search_bm25(self, query: str, **kwargs: Any) -> list[BM25SearchResult]:
        self.calls += 1
        return self.results


class _FailingRetrieval:
    def build(self, request):
        raise RuntimeError("D:/private/evidence/secret.pdf")


class _SummaryProvider:
    provider_name = "offline-summary"
    model_name = "fake-v1"

    def __init__(self, answer: str | None = None, failure: Exception | None = None) -> None:
        self.answer = answer or "Notice is required. [CTX-001]"
        self.failure = failure
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return {
            "answer": self.answer,
            "citations": ["CTX-001"],
            "limitations": ["Limited to the retrieved authority."],
        }


class _CountingSummaryService:
    def __init__(self, delegate: SummarizationService) -> None:
        self.delegate = delegate
        self.calls = 0

    def summarize_and_verify(self, request):
        self.calls += 1
        return self.delegate.summarize_and_verify(request)


class _ComparisonProvider:
    provider_name = "offline-comparison"
    model_name = "fake-comparison-v1"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return LegalComparisonProviderOutput(
            comparison_id=request.comparison_id,
            context_id=request.context_id,
            explanation="The selected sources may relate. [CTX-001]",
            uncertainties=["Legal applicability was not assessed."],
            ai_generated=True,
            human_review_required=True,
        )


def _authority_results() -> list[BM25SearchResult]:
    metadata = LegalDocumentMetadata(
        document_id="ICJ-001",
        case_name="Synthetic Authority",
        court="International Court of Justice",
        citation="I.C.J. Reports 2021, p. 206",
        source="Official reports",
        source_url="https://example.test/icj-001",
        provenance="official_court_source",
        page_number=7,
        page_start=7,
        page_end=7,
    )
    return [
        BM25SearchResult(
            rank=1,
            document_id=metadata.document_id,
            chunk_id="ICJ-001-chunk-0001",
            case_name=metadata.case_name,
            score=4.2,
            chunk_text="Notice is required.",
            citation=metadata.citation,
            source=metadata.source,
            court=metadata.court,
            metadata=metadata,
        )
    ]


def _request(**updates: Any) -> ResearchWorkflowRequest:
    values: dict[str, Any] = {
        "task_id": "task-001",
        "case": Case(id="case-001", title="Authorized synthetic matter"),
        "query": "notice requirement",
        "evidence_documents": [
            EvidenceDocumentInput(
                document=Document(
                    id="evidence-001",
                    case_id="case-001",
                    name="authorized.txt",
                    media_type="text/plain",
                ),
                text="The tenant received notice.",
            )
        ],
        "retrieval_mode": "bm25",
    }
    values.update(updates)
    return ResearchWorkflowRequest(**values)


def _workflow(
    *,
    analysis: _Analysis | None = None,
    retrieval=None,
    summary_provider: _SummaryProvider | None = None,
    comparison_provider: _ComparisonProvider | None = None,
):
    provider = summary_provider or _SummaryProvider()
    summary = _CountingSummaryService(
        SummarizationService(
            SummarizationAgent(provider),
            SummarizationVerificationBridge(VerificationAgent()),
        )
    )
    comparison = (
        LegalComparisonDraftService(comparison_provider)
        if comparison_provider is not None
        else None
    )
    workflow = ResearchWorkflow(
        analysis=analysis or _Analysis(),
        retrieval=retrieval or LegalRetrievalContextService(_RetrievalAgent()),
        summarization=summary,
        comparison_drafts=comparison,
    )
    return workflow, provider, summary


def test_complete_pipeline_preserves_ctx_and_real_report_contract() -> None:
    workflow, provider, summary = _workflow()

    result = workflow.run(_request())

    assert result.status == "completed"
    assert provider.calls == 1
    assert summary.calls == 1
    assert result.report is not None
    assert result.report.verification.response.overall_status == "supported"
    authority = result.report.authorities[0]
    assert authority.context_id == "CTX-001"
    assert authority.document_id == "ICJ-001"
    assert authority.pages == [7]
    assert authority.chunk_ids == ["ICJ-001-chunk-0001"]
    assert authority.official_citation == "I.C.J. Reports 2021, p. 206"
    assert "professional legal advice" in result.report.human_review_disclaimer


@pytest.mark.parametrize(
    ("evidence_documents", "analysis", "expected_availability"),
    [
        ([], _Analysis(), "unavailable"),
        (None, _Analysis("partial"), "partial"),
    ],
)
def test_missing_and_partial_analysis_are_preserved(
    evidence_documents, analysis, expected_availability
) -> None:
    workflow, _, _ = _workflow(analysis=analysis)
    request = _request()
    if evidence_documents == []:
        request = request.model_copy(update={"evidence_documents": []})

    result = workflow.run(request)

    assert result.status == "partial"
    assert result.report is not None
    assert result.report.analysis.availability == expected_availability
    assert any("Analysis" in item for item in result.report.limitations)


def test_retrieval_unavailable_is_sanitized_and_stops_generation() -> None:
    workflow, provider, summary = _workflow(retrieval=_FailingRetrieval())

    result = workflow.run(_request())

    assert result.status == "failed"
    assert result.stages.retrieval == "failed"
    assert provider.calls == 0
    assert summary.calls == 0
    sanitized_errors = " ".join(
        error.model_dump_json() for error in result.errors
    ).casefold()
    assert "private evidence" not in sanitized_errors


def test_empty_retrieval_calls_service_once_without_network_and_returns_controlled_failure() -> None:
    retrieval = LegalRetrievalContextService(_RetrievalAgent([]))
    workflow, provider, summary = _workflow(retrieval=retrieval)

    result = workflow.run(_request())

    assert result.status == "failed"
    assert summary.calls == 1
    assert provider.calls == 0
    assert result.errors[-1].code == "empty_retrieval_context"


@pytest.mark.parametrize("failure", [RuntimeError("provider secret"), TimeoutError("slow")])
def test_generation_failure_and_timeout_are_sanitized(failure: Exception) -> None:
    provider = _SummaryProvider(failure=failure)
    workflow, _, summary = _workflow(summary_provider=provider)

    result = workflow.run(_request())

    assert result.status == "failed"
    assert summary.calls == 1
    assert provider.calls == 1
    serialized = result.model_dump_json().casefold()
    assert "provider secret" not in serialized
    assert "slow" not in serialized


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Notice is not required. [CTX-001]", "unsupported"),
        ("Damages are available. [CTX-001]", "uncertain"),
    ],
)
def test_verification_findings_are_preserved_without_promoting_report(
    answer: str, expected: str
) -> None:
    workflow, _, _ = _workflow(summary_provider=_SummaryProvider(answer=answer))

    result = workflow.run(_request())

    assert result.report is not None
    assert result.report.verification.response.overall_status == expected
    assert result.report.verification.response.verified is False
    assert any(expected in item for item in result.report.limitations)


def test_comparisons_disabled_make_no_provider_call() -> None:
    comparison_provider = _ComparisonProvider()
    workflow, _, _ = _workflow(comparison_provider=comparison_provider)

    result = workflow.run(_request())

    assert result.report is not None
    assert result.report.comparisons == []
    assert comparison_provider.calls == 0


def test_enabled_comparison_draft_is_unverified_and_rendered_in_markdown() -> None:
    comparison_provider = _ComparisonProvider()
    workflow, _, _ = _workflow(comparison_provider=comparison_provider)
    selection = FactAuthoritySelection(
        analysis_document_id="evidence-001",
        fact_index=0,
        context_id="CTX-001",
    )

    result = workflow.run(
        _request(
            comparisons=[selection],
            generate_comparison_drafts=True,
            render_markdown=True,
        )
    )

    assert result.status == "completed"
    assert comparison_provider.calls == 1
    assert result.report is not None
    record = result.report.comparisons[0]
    assert record.review_status == "requires_human_review"
    assert record.ai_draft is not None
    assert record.ai_draft.verification_status == "not_verified"
    assert result.markdown is not None
    assert "AI-generated comparative explanation" in result.markdown
    assert "Not independently verified" in result.markdown
    assert "Human Review Disclaimer" in result.markdown
