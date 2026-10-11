"""Provider-neutral Summarization Agent with strict citation validation."""

from collections.abc import Mapping
import re
from typing import Any, overload

from pydantic import ValidationError

from agents.coordinator.schemas import SummarizationReadyPayload
from agents.summarization.citations import (
    validate_context_citations,
    validate_source_citation_ids,
)
from agents.summarization.comparison import CasePrecedentComparator
from agents.summarization.evidence_summary import EvidenceSummarizer
from agents.summarization.precedent_summary import PrecedentSummarizer
from agents.summarization.prompt import (
    build_llm_evidence_provider_request,
    build_summarization_provider_request,
)
from agents.summarization.provider import SummarizationProvider
from agents.summarization.report import LegalResearchReportGenerator, render_report_markdown
from agents.summarization.schemas import (
    SummarizationCitationValidation,
    SummarizationOutput,
    SummarizationProviderMetadata,
    SummarizationRequest,
    SummarizationResult,
    SummarizationResponse,
    SourceReference,
    GroundingWarning,
    EvidenceSummary,
    EvidenceSummaryPoint,
    LLMSummaryDraft,
    LLMSourceContext,
    CaseComparisonRequest,
    LegalResearchReport,
    LegalResearchReportRequest,
    SummarizationWorkflowRequest,
    SummarizationWorkflowResponse,
    WorkflowStageResult,
)

INSUFFICIENT_CONTEXT_ANSWER = (
    "Insufficient retrieved legal context to answer reliably."
)


class SummarizationAgentError(RuntimeError):
    """Base class for controlled summarization failures."""


class SummarizationProviderError(SummarizationAgentError):
    """Raised when an injected provider cannot produce a usable response."""


class SummarizationValidationError(SummarizationAgentError):
    """Raised when generated output violates grounding or citation rules."""

    def __init__(
        self,
        message: str,
        *,
        citation_validation: SummarizationCitationValidation | None = None,
    ) -> None:
        super().__init__(message)
        self.citation_validation = citation_validation


class SummarizationAgent:
    """Generate and validate evidence-grounded legal summaries."""

    def __init__(self, provider: SummarizationProvider | None = None) -> None:
        self.provider = provider

    @overload
    def summarize(self, request: SummarizationRequest) -> SummarizationResponse:
        ...

    @overload
    def summarize(
        self,
        payload: SummarizationReadyPayload | Mapping[str, Any],
    ) -> SummarizationResult:
        ...

    def summarize(
        self,
        payload: SummarizationRequest | SummarizationReadyPayload | Mapping[str, Any],
    ) -> SummarizationResponse | SummarizationResult:
        """Handle deterministic foundation requests or legacy retrieval payloads.

        ``SummarizationRequest`` is deliberately non-generative in Step 1.
        The legacy Coordinator payload retains its existing provider-backed
        behavior so established consumers are not broken during migration.
        """

        if isinstance(payload, SummarizationRequest):
            return self._summarize_foundation(payload)
        if isinstance(payload, Mapping) and "case_id" in payload:
            return self._summarize_foundation(
                SummarizationRequest.model_validate(payload)
            )

        return self._summarize_retrieval_payload(payload)

    @staticmethod
    def generate_report(request: LegalResearchReportRequest) -> LegalResearchReport:
        """Assemble a deterministic report without invoking a provider."""
        return LegalResearchReportGenerator().generate(request)

    def run_research_workflow(
        self,
        request: SummarizationWorkflowRequest,
    ) -> SummarizationWorkflowResponse:
        """Run supplied evidence through the internal deterministic workflow.

        The workflow never retrieves documents or upgrades a precedent holding
        into a current-case conclusion. Each judgment is processed separately;
        an isolated expected validation failure leaves other stages intact.
        """
        stages: list[WorkflowStageResult] = []
        warnings: list[GroundingWarning] = []
        evidence_response = self.summarize(request.current_case_request)
        assert isinstance(evidence_response, SummarizationResponse)
        evidence = evidence_response.evidence_summary
        stages.append(WorkflowStageResult(
            name="evidence_summarization", status=evidence_response.status,
            warnings=evidence_response.warnings, errors=evidence_response.errors,
        ))
        warnings.extend(evidence_response.warnings)

        precedents: list[Any] = []
        for index, precedent_request in enumerate(request.precedent_requests, start=1):
            try:
                precedent_response = self.summarize(precedent_request)
                assert isinstance(precedent_response, SummarizationResponse)
            except (ValidationError, ValueError) as exc:
                stage = WorkflowStageResult(
                    name=f"precedent_summarization_{index}", status="failed",
                    errors=[GroundingWarning(code="PRECEDENT_INPUT_REJECTED", description="A supplied precedent input could not be processed safely")],
                )
                stages.append(stage); warnings.extend(stage.errors)
                continue
            precedents.extend(precedent_response.precedent_summaries)
            stages.append(WorkflowStageResult(
                name=f"precedent_summarization_{index}", status=precedent_response.status,
                warnings=precedent_response.warnings, errors=precedent_response.errors,
            ))
            warnings.extend(precedent_response.warnings)

        comparisons: list[Any] = []
        if evidence is None:
            stage = WorkflowStageResult(
                name="case_precedent_comparison", status="requires_input",
                warnings=[GroundingWarning(code="NO_EVIDENCE_FOR_COMPARISON", description="No source-grounded current-case evidence summary was available")],
            )
            stages.append(stage); warnings.extend(stage.warnings)
        elif not precedents:
            stage = WorkflowStageResult(
                name="case_precedent_comparison", status="requires_input",
                warnings=[GroundingWarning(code="NO_PRECEDENT_FOR_COMPARISON", description="No usable precedent summary was available")],
            )
            stages.append(stage); warnings.extend(stage.warnings)
        else:
            comparison_request = CaseComparisonRequest(
                case_id=request.case_id, current_case_summary=evidence,
                precedents=precedents, current_case_jurisdiction=request.current_case_jurisdiction,
                research_question=request.research_question,
            )
            comparisons = CasePrecedentComparator().compare(comparison_request)
            comparison_status = "completed" if all(item.status == "completed" for item in comparisons) else "partial"
            stages.append(WorkflowStageResult(name="case_precedent_comparison", status=comparison_status))

        report: LegalResearchReport | None = None
        markdown: str | None = None
        try:
            report = self.generate_report(LegalResearchReportRequest(
                case_id=request.case_id, research_question=request.research_question,
                report_title=request.report_title, current_case_jurisdiction=request.current_case_jurisdiction,
                evidence_summary=evidence, precedent_summaries=precedents,
                case_comparisons=comparisons, analysis_findings=request.analysis_findings,
                clauses=request.clauses, timeline_events=request.timeline_events,
                inconsistency_candidates=request.inconsistency_candidates, warnings=warnings,
            ))
            if request.render_markdown:
                markdown = render_report_markdown(report)
            stages.append(WorkflowStageResult(name="report_assembly", status=report.status, warnings=report.warnings))
        except (ValidationError, ValueError) as exc:
            stages.append(WorkflowStageResult(
                name="report_assembly", status="failed",
                errors=[GroundingWarning(code="REPORT_ASSEMBLY_FAILED", description="The structured report could not be assembled safely")],
            ))

        sources = _unique_sources([
            *(evidence.source_references if evidence else []),
            *(source for precedent in precedents for source in precedent.source_references),
            *(source for comparison in comparisons for source in comparison.supporting_sources),
            *(report.source_references if report else []),
        ])
        citations = _source_citations(sources)
        failed = any(stage.status == "failed" for stage in stages)
        incomplete = any(stage.status in {"partial", "requires_input", "failed"} for stage in stages)
        status = "failed" if report is None else ("partial" if incomplete or failed else "completed")
        if evidence is None and not precedents:
            status = "requires_input" if report is not None else "failed"
        return SummarizationWorkflowResponse(
            case_id=request.case_id, status=status, evidence_summary=evidence,
            precedent_summaries=precedents, case_comparisons=comparisons, report=report,
            markdown=markdown, stages=stages, source_references=sources,
            citations=citations, warnings=_unique_warnings(warnings),
        )

    def _summarize_retrieval_payload(
        self,
        payload: SummarizationReadyPayload | Mapping[str, Any],
    ) -> SummarizationResult:
        """Return validated legacy Coordinator output or a controlled error."""

        validated_payload = SummarizationReadyPayload.model_validate(payload)
        if not validated_payload.context_passages or not validated_payload.citation_map:
            return self._no_context_result(validated_payload)
        if self.provider is None:
            raise SummarizationProviderError(
                "A provider is required for legacy retrieval summarization"
            )

        provider_request = build_summarization_provider_request(validated_payload)
        try:
            raw_output = self.provider.generate(provider_request)
        except Exception as exc:
            raise SummarizationProviderError(
                "Summarization provider failed"
            ) from exc
        try:
            output = SummarizationOutput.model_validate(raw_output)
        except ValidationError as exc:
            raise SummarizationProviderError(
                "Summarization provider returned invalid structured output"
            ) from exc

        citation_validation = _validate_output_citations(
            output,
            validated_payload.citation_map,
        )
        if not citation_validation.inline.has_citations:
            raise SummarizationValidationError(
                "Summarization output must cite retrieved context inline",
                citation_validation=citation_validation,
            )
        if not citation_validation.all_citations_valid:
            raise SummarizationValidationError(
                "Summarization output contains invalid or inconsistent citations",
                citation_validation=citation_validation,
            )
        if validated_payload.diagnostics.warnings and not _mentions_uncertainty(
            output.limitations
        ):
            raise SummarizationValidationError(
                "Summarization limitations must acknowledge weak context",
                citation_validation=citation_validation,
            )

        return SummarizationResult(
            output=output,
            citation_validation=citation_validation,
            provider_metadata=self._provider_metadata(provider_called=True),
            used_context_ids=citation_validation.inline.valid_citations,
            warnings=list(validated_payload.diagnostics.warnings),
        )

    def _summarize_foundation(
        self,
        request: SummarizationRequest,
    ) -> SummarizationResponse:
        """Return a provenance-preserving, non-generative foundation response."""

        if request.summary_type == "case_comparison":
            return self._summarize_comparison(request)

        has_evidence = bool(
            request.evidence_text is not None
            or request.evidence_items
            or request.analysis_findings
            or request.retrieved_passages
        )
        if not has_evidence:
            return SummarizationResponse(
                case_id=request.case_id,
                status="requires_input",
                warnings=[GroundingWarning(
                    code="NO_EVIDENCE",
                    description=(
                        "No evidence, analysis findings, or retrieved passages were supplied"
                    ),
                    affected_field="evidence_text",
                )],
                stage_status={
                    "validation": "completed",
                    "evidence_summarization": "requires_input",
                },
            )

        if request.mode == "llm_assisted":
            return self._summarize_llm_assisted_evidence(request)

        if request.summary_type == "precedent_summary":
            precedent = PrecedentSummarizer().summarize(request)
            warnings = _summary_warnings(precedent.uncertainty_flags)
            if precedent.points:
                return SummarizationResponse(
                    case_id=request.case_id,
                    status="completed" if not precedent.uncertainty_flags else "partial",
                    precedent_summaries=[precedent],
                    source_references=precedent.source_references,
                    citations=precedent.citations,
                    warnings=warnings,
                    stage_status={
                        "validation": "completed",
                        "precedent_summarization": "completed",
                    },
                )
            return SummarizationResponse(
                case_id=request.case_id,
                status="partial",
                warnings=warnings,
                stage_status={
                    "validation": "completed",
                    "precedent_summarization": "no_safe_judgment_content",
                },
            )

        summary = EvidenceSummarizer().summarize(request)
        warnings = _summary_warnings(summary.uncertainty_flags)
        if summary.points:
            return SummarizationResponse(
                case_id=request.case_id,
                status="completed",
                evidence_summary=summary,
                source_references=summary.source_references,
                citations=_source_citations(summary.source_references),
                warnings=warnings,
                stage_status={
                    "validation": "completed",
                    "evidence_summarization": "completed",
                },
            )

        return SummarizationResponse(
            case_id=request.case_id,
            status="partial",
            warnings=warnings,
            stage_status={
                "validation": "completed",
                "evidence_summarization": "no_safe_significant_statement",
            },
        )

    def _summarize_llm_assisted_evidence(
        self,
        request: SummarizationRequest,
    ) -> SummarizationResponse:
        """Draft only selected evidence through an explicitly injected provider."""

        deterministic = EvidenceSummarizer().summarize(request)
        if not deterministic.points:
            return SummarizationResponse(
                case_id=request.case_id,
                status="partial",
                warnings=_summary_warnings(deterministic.uncertainty_flags),
                stage_status={
                    "validation": "completed",
                    "evidence_selection": "no_safe_significant_statement",
                    "llm_assisted_drafting": "not_called",
                },
            )
        if self.provider is None:
            return self._llm_failure(
                request,
                "LLM_PROVIDER_NOT_CONFIGURED",
                "LLM-assisted drafting requires an explicitly injected provider",
            )

        provider_request, context_warnings = build_llm_evidence_provider_request(
            query=request.query,
            points=deterministic.points,
        )
        if not provider_request.source_context:
            return self._llm_failure(
                request,
                "LLM_INSUFFICIENT_CITABLE_CONTEXT",
                "No citation-backed evidence statement was available for LLM drafting",
                warnings=context_warnings,
            )
        try:
            raw_draft = self.provider.generate(provider_request)
        except TimeoutError:
            return self._llm_failure(
                request,
                "LLM_PROVIDER_TIMEOUT",
                "The LLM provider did not respond within the configured limit",
                warnings=context_warnings,
            )
        except Exception:
            return self._llm_failure(
                request,
                "LLM_PROVIDER_FAILURE",
                "The LLM provider failed without a usable draft",
                warnings=context_warnings,
            )
        try:
            draft = LLMSummaryDraft.model_validate(raw_draft)
        except ValidationError:
            return self._llm_failure(
                request,
                "LLM_INVALID_STRUCTURED_OUTPUT",
                "The LLM provider returned an invalid or empty structured draft",
                warnings=context_warnings,
            )

        try:
            summary, semantic_warnings = _validate_llm_draft(
                draft,
                provider_request.source_context,
                provider_request.allowed_citation_ids,
            )
        except SummarizationValidationError as exc:
            return self._llm_failure(
                request,
                "LLM_DRAFT_REJECTED",
                str(exc),
                warnings=context_warnings,
            )

        warnings = [
            *_summary_warnings(deterministic.uncertainty_flags),
            *_summary_warnings(context_warnings),
            *_summary_warnings(semantic_warnings),
        ]
        return SummarizationResponse(
            case_id=request.case_id,
            status="completed",
            evidence_summary=summary,
            source_references=summary.source_references,
            citations=_source_citations(summary.source_references),
            warnings=warnings,
            stage_status={
                "validation": "completed",
                "evidence_selection": "completed",
                "llm_assisted_drafting": "completed_with_human_review_required",
            },
        )

    @staticmethod
    def _llm_failure(
        request: SummarizationRequest,
        code: str,
        description: str,
        *,
        warnings: list[str] | None = None,
    ) -> SummarizationResponse:
        return SummarizationResponse(
            case_id=request.case_id,
            status="failed",
            warnings=_summary_warnings(warnings or []),
            errors=[GroundingWarning(code=code, description=description)],
            stage_status={
                "validation": "completed",
                "llm_assisted_drafting": "failed",
            },
        )

    @staticmethod
    def _summarize_comparison(request: SummarizationRequest) -> SummarizationResponse:
        """Compare only the explicitly supplied, source-grounded summaries."""

        comparison_request = request.comparison_request
        if comparison_request is None:  # Defensive: schema validation normally prevents this.
            return SummarizationResponse(
                case_id=request.case_id,
                status="requires_input",
                warnings=[GroundingWarning(
                    code="NO_COMPARISON_INPUT",
                    description="No current-case and precedent comparison input was supplied",
                )],
                stage_status={"validation": "requires_input", "case_comparison": "requires_input"},
            )
        if not comparison_request.precedents:
            return SummarizationResponse(
                case_id=request.case_id,
                status="requires_input",
                warnings=[GroundingWarning(
                    code="NO_PRECEDENT_INPUT",
                    description="No precedent summary was supplied for comparison",
                    affected_field="comparison_request.precedents",
                )],
                stage_status={"validation": "completed", "case_comparison": "requires_input"},
            )
        comparisons = CasePrecedentComparator().compare(comparison_request)
        flags = [flag for comparison in comparisons for flag in comparison.limitations]
        return SummarizationResponse(
            case_id=request.case_id,
            status="completed" if all(item.status == "completed" for item in comparisons) else "partial",
            case_comparisons=comparisons,
            source_references=_unique_sources_from_comparisons(comparisons),
            citations=_source_citations(_unique_sources_from_comparisons(comparisons)),
            warnings=_summary_warnings(flags),
            stage_status={"validation": "completed", "case_comparison": "completed"},
        )

    def _no_context_result(
        self,
        payload: SummarizationReadyPayload,
    ) -> SummarizationResult:
        output = SummarizationOutput(
            answer=INSUFFICIENT_CONTEXT_ANSWER,
            citations=[],
            limitations=["No retrieved legal context was available"],
        )
        citation_validation = _validate_output_citations(output, {})
        return SummarizationResult(
            output=output,
            citation_validation=citation_validation,
            provider_metadata=self._provider_metadata(provider_called=False),
            used_context_ids=[],
            warnings=[*payload.diagnostics.warnings, "Provider was not called"],
        )

    def _provider_metadata(
        self,
        *,
        provider_called: bool,
    ) -> SummarizationProviderMetadata:
        if self.provider is None:
            return SummarizationProviderMetadata(
                provider="not_configured",
                model=None,
                provider_called=provider_called,
            )
        provider_name = getattr(
            self.provider,
            "provider_name",
            type(self.provider).__name__,
        )
        model_name = getattr(self.provider, "model_name", None)
        return SummarizationProviderMetadata(
            provider=str(provider_name),
            model=str(model_name) if model_name is not None else None,
            provider_called=provider_called,
        )


def _validate_output_citations(
    output: SummarizationOutput,
    citation_map: Mapping[str, Any],
) -> SummarizationCitationValidation:
    allowed = set(citation_map)
    inline = validate_context_citations(output.answer, allowed)
    declared_unique = list(dict.fromkeys(output.citations))
    invalid_declared = [item for item in declared_unique if item not in allowed]
    inline_unique = list(dict.fromkeys(inline.cited_context_ids))
    declared_missing_inline = [
        item for item in declared_unique if item not in inline_unique
    ]
    inline_missing_declared = [
        item for item in inline_unique if item not in declared_unique
    ]
    all_valid = (
        inline.all_citations_valid
        and not invalid_declared
        and not declared_missing_inline
        and not inline_missing_declared
    )
    return SummarizationCitationValidation(
        inline=inline,
        declared_citations=output.citations,
        invalid_declared_citations=invalid_declared,
        declared_missing_inline=declared_missing_inline,
        inline_missing_declared=inline_missing_declared,
        all_citations_valid=all_valid,
    )


def _mentions_uncertainty(limitations: list[str]) -> bool:
    markers = ("weak", "uncertain", "limited", "insufficient", "relevance")
    return any(
        marker in limitation.casefold()
        for limitation in limitations
        for marker in markers
    )


def _request_sources(request: SummarizationRequest) -> list[SourceReference]:
    """Collect supplied sources in order without creating or modifying them."""

    sources = [
        item.source
        for item in request.evidence_items
        if item.source is not None
    ]
    sources.extend(
        source
        for finding in request.analysis_findings
        for source in finding.sources
    )
    sources.extend(passage.source for passage in request.retrieved_passages)
    return sources


def _source_citations(sources: list[SourceReference]) -> list[str]:
    """Return only supplied citation identifiers, retaining first-seen order."""

    return list(dict.fromkeys(
        source.citation_id for source in sources if source.citation_id is not None
    ))


def _summary_warnings(flags: list[str]) -> list[GroundingWarning]:
    """Translate deterministic selection limitations into neutral warnings."""

    descriptions = {
        "no_significant_evidence_selected": (
            "Evidence was supplied, but no safely significant statement was selected"
        ),
        "input_truncated_at_safe_sentence_boundary": (
            "Evidence processing stopped at a sentence boundary due to safety limits"
        ),
        "limited_provenance": (
            "One or more selected statements have no supplied source locator"
        ),
        "legal_issue_not_explicitly_identified": (
            "No legal issue was explicitly identified in the supplied judgment material"
        ),
        "judicial_reasoning_not_explicitly_identified": (
            "No judicial reasoning was explicitly identified in the supplied judgment material"
        ),
        "current_court_holding_not_explicitly_identified": (
            "No current-court holding was explicitly identified in the supplied judgment material"
        ),
        "judgment_input_truncated_at_safe_sentence_boundary": (
            "Judgment processing stopped at a sentence boundary due to safety limits"
        ),
        "no_safely_identifiable_judgment_content": (
            "Judgment material was supplied but no safe structured content was identified"
        ),
        "no_source_locator_or_citation_supplied": (
            "Selected judgment material has no supplied source locator or citation"
        ),
        "authority_not_determined": (
            "The authority or applicability of the precedent has not been determined"
        ),
        "human_legal_review_required": (
            "Comparison results require human legal review"
        ),
        "no_supported_topic_alignment": (
            "No supported factual or procedural topic alignment was identified"
        ),
        "limited_provenance": (
            "One or more comparison inputs have limited supplied provenance"
        ),
        "llm_context_omitted_at_complete_statement_boundary": (
            "Some complete evidence statements were omitted from the bounded LLM context"
        ),
        "llm_context_statement_exceeds_safe_limit": (
            "A complete evidence statement exceeded the safe LLM context limit and was omitted"
        ),
        "llm_generated_draft_not_independently_verified": (
            "The LLM-assisted draft passed basic format checks but is not independently verified"
        ),
        "human_review_required": (
            "LLM-assisted drafting requires independent human legal review"
        ),
        "conflicting_evidence_requires_human_review": (
            "Supplied evidence contains differing semantic positions and requires human review"
        ),
        "provider_human_review_flag_overridden": (
            "Provider output did not request human review; CaseLens requires it regardless"
        ),
    }
    return [
        GroundingWarning(code=flag.upper(), description=descriptions[flag])
        for flag in flags
        if flag in descriptions
    ]


def _unique_sources_from_comparisons(comparisons: list[Any]) -> list[SourceReference]:
    sources: list[SourceReference] = []
    for comparison in comparisons:
        for source in comparison.supporting_sources:
            if source not in sources:
                sources.append(source)
    return sources


_NEGATION_MARKERS = re.compile(r"\b(?:not|no|never|neither|nor|den(?:y|ied|ies))\b", re.IGNORECASE)
_ATTRIBUTION_MARKERS = re.compile(r"\b(?:alleg(?:e|ed|es)|claim(?:ed|s)?|stat(?:e|ed|es)|said|reported|submitted|den(?:y|ied|ies))\b", re.IGNORECASE)
_CONDITIONAL_MARKERS = re.compile(
    r"\b(?:if|unless|subject to|conditional(?:ly)?|might|could)\b|(?<!\d\s)\bmay\b",
    re.IGNORECASE,
)
_MONEY = re.compile(r"\b(?:LKR|Rs\.?|USD|EUR|GBP)\s*[0-9][0-9,]*(?:\.\d{1,2})?", re.IGNORECASE)
_DATE = re.compile(r"\b(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_WORDS = re.compile(r"[A-Za-z]{3,}|\d+[\d,]*")
_STOP_WORDS = frozenset({
    "that", "this", "with", "from", "were", "have", "been", "would", "could",
    "should", "their", "there", "which", "about", "into", "after", "before",
    "under", "where", "when", "only", "said", "stated", "alleged", "claimant",
    "respondent", "evidence", "summary", "according",
})


def _validate_llm_draft(
    draft: LLMSummaryDraft,
    contexts: list[LLMSourceContext],
    allowed_citation_ids: list[str],
) -> tuple[EvidenceSummary, list[str]]:
    """Reject easily detectable unsupported rewrites; retain review limits.

    This deliberately bounded check cannot establish legal or factual truth.
    It only prevents a provider response from bypassing known source semantics,
    request-local citation identities, or obvious value changes.
    """

    by_citation: dict[str, list[LLMSourceContext]] = {}
    for context in contexts:
        for citation in context.citations:
            by_citation.setdefault(citation, []).append(context)

    validated_points: list[EvidenceSummaryPoint] = []
    semantic_warnings = ["llm_generated_draft_not_independently_verified", "human_review_required"]
    for point in draft.points:
        citation_result = validate_source_citation_ids(point.citations, allowed_citation_ids)
        if not citation_result.all_citations_valid:
            raise SummarizationValidationError(
                "LLM draft contains missing, malformed, or unsupported citation identifiers"
            )
        citations = list(dict.fromkeys(citation_result.valid_citations))
        supporting_contexts = [context for citation in citations for context in by_citation[citation]]
        _validate_point_semantics(point.text, supporting_contexts)
        sources = _unique_sources([
            source for context in supporting_contexts for source in context.sources
        ])
        validated_points.append(EvidenceSummaryPoint(
            text=point.text,
            citations=citations,
            source_references=sources,
            is_negated=any(context.is_negated for context in supporting_contexts),
            is_conditional=any(context.is_conditional for context in supporting_contexts),
            is_attributed=any(context.is_attributed for context in supporting_contexts),
            epistemic_status=_combined_status(supporting_contexts, "epistemic_status"),
            verification_status="not_independently_verified",
            provenance_limited=not sources,
        ))

    _validate_summary_text(draft.summary_text, contexts)
    if _has_conflicting_semantics(contexts):
        semantic_warnings.append("conflicting_evidence_requires_human_review")
    if not draft.human_review_required:
        semantic_warnings.append("provider_human_review_flag_overridden")
    sources = _unique_sources([
        source for point in validated_points for source in point.source_references
    ])
    return EvidenceSummary(
        summary_text=draft.summary_text,
        key_points=[point.text for point in validated_points],
        points=validated_points,
        source_references=sources,
        uncertainty_flags=semantic_warnings,
    ), semantic_warnings


def _validate_point_semantics(text: str, contexts: list[LLMSourceContext]) -> None:
    if not _has_material_overlap(text, " ".join(context.text for context in contexts)):
        raise SummarizationValidationError(
            "LLM draft point lacks sufficient lexical support from its cited evidence"
        )
    if any(context.is_negated for context in contexts) and not _NEGATION_MARKERS.search(text):
        raise SummarizationValidationError("LLM draft dropped explicit negation from cited evidence")
    if any(context.is_attributed for context in contexts) and not _ATTRIBUTION_MARKERS.search(text):
        raise SummarizationValidationError("LLM draft dropped allegation or attribution from cited evidence")
    if any(context.is_conditional for context in contexts) and not _CONDITIONAL_MARKERS.search(text):
        raise SummarizationValidationError("LLM draft dropped conditional or uncertain language from cited evidence")
    _reject_changed_values(text, contexts)


def _validate_summary_text(text: str, contexts: list[LLMSourceContext]) -> None:
    if not _has_material_overlap(text, " ".join(context.text for context in contexts)):
        raise SummarizationValidationError(
            "LLM draft summary lacks sufficient lexical support from supplied evidence"
        )
    if any(context.is_negated for context in contexts) and not _NEGATION_MARKERS.search(text):
        raise SummarizationValidationError("LLM draft summary dropped explicit negation from supplied evidence")
    if any(context.is_attributed for context in contexts) and not _ATTRIBUTION_MARKERS.search(text):
        raise SummarizationValidationError("LLM draft summary dropped allegation or attribution from supplied evidence")
    if any(context.is_conditional for context in contexts) and not _CONDITIONAL_MARKERS.search(text):
        raise SummarizationValidationError("LLM draft summary dropped conditional or uncertain language from supplied evidence")
    _reject_changed_values(text, contexts)


def _reject_changed_values(text: str, contexts: list[LLMSourceContext]) -> None:
    source_text = " ".join(context.text for context in contexts)
    for pattern, label in ((_MONEY, "monetary value"), (_DATE, "date")):
        source_values = {match.group(0).lower().replace(" ", "") for match in pattern.finditer(source_text)}
        generated_values = {match.group(0).lower().replace(" ", "") for match in pattern.finditer(text)}
        if generated_values.difference(source_values):
            raise SummarizationValidationError(
                f"LLM draft introduced or changed a {label}"
            )


def _has_material_overlap(generated: str, source: str) -> bool:
    generated_words = {
        word.lower() for word in _WORDS.findall(generated)
        if word.lower() not in _STOP_WORDS
    }
    source_words = {
        word.lower() for word in _WORDS.findall(source)
        if word.lower() not in _STOP_WORDS
    }
    return len(generated_words.intersection(source_words)) >= min(2, len(generated_words))


def _combined_status(contexts: list[LLMSourceContext], attribute: str) -> str:
    values = list(dict.fromkeys(getattr(context, attribute) for context in contexts))
    return values[0] if len(values) == 1 else "mixed_source_semantics"


def _has_conflicting_semantics(contexts: list[LLMSourceContext]) -> bool:
    return any(context.is_negated for context in contexts) and any(
        not context.is_negated for context in contexts
    )


def _unique_sources(sources: list[SourceReference]) -> list[SourceReference]:
    unique: list[SourceReference] = []
    for source in sources:
        if source not in unique:
            unique.append(source)
    return unique


def _unique_warnings(warnings: list[GroundingWarning]) -> list[GroundingWarning]:
    """Keep first-seen non-sensitive warnings without relying on hashability."""
    unique: list[GroundingWarning] = []
    for warning in warnings:
        if warning not in unique:
            unique.append(warning)
    return unique
