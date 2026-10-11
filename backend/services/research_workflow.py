"""Internal orchestration of existing CaseLens research components."""

from __future__ import annotations

from typing import Protocol

from agents.analysis.schemas import AnalysisRequest, AnalysisResponse
from agents.coordinator import LegalRetrievalContextRequest
from agents.coordinator.schemas import SummarizationReadyPayload
from backend.schemas.legal_report import LegalResearchReport
from backend.schemas.research_workflow import (
    ResearchWorkflowError,
    ResearchWorkflowRequest,
    ResearchWorkflowResult,
    ResearchWorkflowStages,
)
from backend.schemas.summarization import (
    SummarizationVerifyRequest,
    SummarizationVerifyResponse,
)
from backend.services.legal_comparison_generation import LegalComparisonDraftService
from backend.services.legal_report import LegalResearchReportAssembler
from backend.services.legal_report_markdown import LegalResearchReportMarkdownRenderer


class AnalysisService(Protocol):
    def analyze(self, request: AnalysisRequest) -> AnalysisResponse: ...


class RetrievalContextService(Protocol):
    def build(
        self, request: LegalRetrievalContextRequest
    ) -> SummarizationReadyPayload: ...


class SummaryVerificationService(Protocol):
    def summarize_and_verify(
        self, request: SummarizationVerifyRequest
    ) -> SummarizationVerifyResponse: ...


class ResearchWorkflow:
    """Execute one private, non-persistent research workflow."""

    def __init__(
        self,
        *,
        analysis: AnalysisService,
        retrieval: RetrievalContextService,
        summarization: SummaryVerificationService,
        report_assembler: LegalResearchReportAssembler | None = None,
        comparison_drafts: LegalComparisonDraftService | None = None,
        markdown_renderer: LegalResearchReportMarkdownRenderer | None = None,
    ) -> None:
        self.analysis = analysis
        self.retrieval = retrieval
        self.summarization = summarization
        self.report_assembler = report_assembler or LegalResearchReportAssembler()
        self.comparison_drafts = comparison_drafts
        self.markdown_renderer = markdown_renderer or LegalResearchReportMarkdownRenderer()

    def run(self, request: ResearchWorkflowRequest | dict) -> ResearchWorkflowResult:
        validated = ResearchWorkflowRequest.model_validate(request)
        stages = ResearchWorkflowStages()
        errors: list[ResearchWorkflowError] = []
        analyses = self._analyze(validated, stages, errors)

        try:
            retrieval_values = {
                "query": validated.query,
                "legal_issue": validated.legal_issue,
                "top_k": validated.top_k,
                "filters": validated.filters,
                "retrieval_mode": validated.retrieval_mode,
                "diversify": validated.diversify,
                "max_chunks_per_document": validated.max_chunks_per_document,
                "rag_max_context_tokens": validated.rag_max_context_tokens,
                "rag_max_context_passages": validated.rag_max_context_passages,
            }
            if "rag_min_retrieval_score" in validated.model_fields_set:
                retrieval_values["rag_min_retrieval_score"] = (
                    validated.rag_min_retrieval_score
                )
            payload = self.retrieval.build(
                LegalRetrievalContextRequest.model_validate(retrieval_values)
            )
            stages.retrieval = "completed"
        except Exception:
            stages.retrieval = "failed"
            errors.append(_error("retrieval", "retrieval_unavailable", "Legal authority retrieval could not complete."))
            return self._result(validated, stages, analyses, errors)

        try:
            summary = self.summarization.summarize_and_verify(
                SummarizationVerifyRequest(task_id=validated.task_id, payload=payload)
            )
            stages.summarization_verification = "completed"
        except Exception:
            stages.summarization_verification = "failed"
            no_context = payload.diagnostics.no_context
            errors.append(
                _error(
                    "summarization_verification",
                    "empty_retrieval_context" if no_context else "summarization_verification_failed",
                    (
                        "No retrieved legal context was available for a verifiable summary."
                        if no_context
                        else "Summary generation or verification could not complete."
                    ),
                )
            )
            return self._result(validated, stages, analyses, errors, retrieval_payload=payload)

        try:
            report = self.report_assembler.assemble(
                validated.case,
                payload,
                summary,
                analyses,
                validated.comparisons,
            )
            stages.report = "completed"
            stages.comparisons = "completed" if validated.comparisons else "skipped"
        except Exception:
            stages.report = "failed"
            if validated.comparisons:
                stages.comparisons = "failed"
            errors.append(_error("report", "report_assembly_failed", "The legal research report could not be assembled safely."))
            return self._result(
                validated,
                stages,
                analyses,
                errors,
                retrieval_payload=payload,
                summary_verification=summary,
            )

        if validated.generate_comparison_drafts:
            if self.comparison_drafts is None:
                stages.comparisons = "partial"
                errors.append(_error("comparisons", "comparison_provider_unavailable", "Comparison drafts were requested but no provider was configured."))
            else:
                try:
                    comparisons = self.comparison_drafts.generate(report.comparisons, analyses, payload)
                    report = report.model_copy(deep=True, update={"comparisons": comparisons})
                    stages.comparisons = "completed"
                except Exception:
                    stages.comparisons = "partial"
                    errors.append(_error("comparisons", "comparison_generation_failed", "Optional comparison drafts could not be generated."))

        markdown: str | None = None
        if validated.render_markdown:
            try:
                markdown = self.markdown_renderer.render(report)
                stages.markdown = "completed"
            except Exception:
                stages.markdown = "failed"
                errors.append(_error("markdown", "markdown_render_failed", "The report could not be rendered as Markdown."))
        else:
            stages.markdown = "skipped"

        return self._result(
            validated,
            stages,
            analyses,
            errors,
            retrieval_payload=payload,
            summary_verification=summary,
            report=report,
            markdown=markdown,
        )

    def _analyze(
        self,
        request: ResearchWorkflowRequest,
        stages: ResearchWorkflowStages,
        errors: list[ResearchWorkflowError],
    ) -> list[AnalysisResponse]:
        if not request.evidence_documents:
            stages.analysis = "skipped"
            errors.append(_error("analysis", "missing_evidence", "No evidence documents were supplied for analysis."))
            return []

        analyses: list[AnalysisResponse] = []
        failures = 0
        for item in request.evidence_documents:
            try:
                analyses.append(
                    self.analysis.analyze(
                        AnalysisRequest(
                            case=request.case,
                            document=item.document,
                            text=item.text,
                            document_metadata=item.document_metadata,
                            segments=item.segments,
                        )
                    )
                )
            except Exception:
                failures += 1
        partial_results = any(item.status != "completed" for item in analyses)
        if failures:
            errors.append(_error("analysis", "analysis_document_failed", "One or more evidence documents could not be analyzed."))
        stages.analysis = "partial" if failures or partial_results else "completed"
        return analyses

    @staticmethod
    def _result(
        request: ResearchWorkflowRequest,
        stages: ResearchWorkflowStages,
        analyses: list[AnalysisResponse],
        errors: list[ResearchWorkflowError],
        *,
        retrieval_payload: SummarizationReadyPayload | None = None,
        summary_verification: SummarizationVerifyResponse | None = None,
        report: LegalResearchReport | None = None,
        markdown: str | None = None,
    ) -> ResearchWorkflowResult:
        failed_core = any(
            getattr(stages, name) == "failed"
            for name in ("retrieval", "summarization_verification", "report")
        )
        has_partial = bool(errors) or any(
            getattr(stages, name) == "partial"
            for name in ("analysis", "comparisons", "markdown")
        )
        status = "failed" if failed_core and report is None else ("partial" if has_partial else "completed")
        return ResearchWorkflowResult(
            task_id=request.task_id,
            status=status,
            stages=stages,
            analyses=analyses,
            retrieval_payload=retrieval_payload,
            summary_verification=summary_verification,
            report=report,
            markdown=markdown,
            errors=errors,
        )


def _error(stage: str, code: str, message: str) -> ResearchWorkflowError:
    return ResearchWorkflowError(stage=stage, code=code, message=message)
