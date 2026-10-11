"""Deterministic planning foundation for future CaseLens orchestration."""

from __future__ import annotations

from typing import Literal

from agents.coordinator.schemas import (
    CoordinatorAnalysisResponse,
    CoordinatorMessage,
    CoordinatorPlanResponse,
    CoordinatorRequest,
    CoordinatorStage,
    CoordinatorTask,
    CoordinatorRetrievalResponse,
)
from agents.coordinator.analysis_adapter import AnalysisCoordinatorAdapter
from agents.coordinator.retrieval_adapter import (
    CoordinatorRetrievalAdapter,
    RetrievalUnavailableError,
)


_ROUTES: dict[CoordinatorTask, tuple[tuple[str, str], ...]] = {
    "analyze_evidence": (("analysis", "analyze_evidence"),),
    "retrieve_precedents": (("retrieval", "retrieve_precedents"),),
    "summarize_evidence": (("summarization", "summarize_evidence"),),
    "summarize_precedents": (
        ("retrieval", "retrieve_precedents"),
        ("summarization", "summarize_precedents"),
    ),
    "verify_claims": (("verification", "verify_claims"),),
    "full_research": (
        ("analysis", "analyze_evidence"),
        ("retrieval", "retrieve_precedents"),
        ("summarization", "summarize_precedents"),
        ("verification", "verify_claims"),
    ),
}

_EXECUTION_LIMITATION = "This Coordinator route is planning-only; specialized agents were not invoked."
_VERIFICATION_LIMITATION = (
    "No agent output is independently verified as a legal conclusion."
)


class Coordinator:
    """Plan routes and execute only explicitly integrated specialized-agent paths."""

    def __init__(
        self,
        analysis_adapter: AnalysisCoordinatorAdapter | None = None,
        retrieval_adapter: CoordinatorRetrievalAdapter | None = None,
        summarization_adapter: "CoordinatorSummarizationAdapter | None" = None,
        evidence_summarization_adapter: "CoordinatorEvidenceSummarizationAdapter | None" = None,
        verification_adapter: "CoordinatorVerificationAdapter | None" = None,
    ) -> None:
        self._analysis_adapter = analysis_adapter or AnalysisCoordinatorAdapter()
        self._retrieval_adapter = retrieval_adapter
        self._summarization_adapter = summarization_adapter
        self._evidence_summarization_adapter = evidence_summarization_adapter
        self._verification_adapter = verification_adapter

    def plan(self, request: CoordinatorRequest) -> CoordinatorPlanResponse:
        """Return a deterministic route and any known request prerequisites."""

        prerequisites = self._missing_prerequisites(request)
        stages = [
            CoordinatorStage(
                agent=agent,
                task_name=task_name,
                status="not_run" if prerequisites else "planned",
            )
            for agent, task_name in _ROUTES[request.requested_task]
        ]
        warnings = [CoordinatorMessage(code="execution_not_implemented", message=_EXECUTION_LIMITATION)]
        return CoordinatorPlanResponse(
            case_id=request.case_id,
            requested_task=request.requested_task,
            status="requires_input" if prerequisites else "planned",
            document_ids=request.document_ids,
            stages=stages,
            warnings=warnings,
            errors=prerequisites,
            verification_limitations=[_VERIFICATION_LIMITATION],
        )

    async def handle(
        self, request: CoordinatorRequest
    ) -> (
        CoordinatorPlanResponse
        | CoordinatorAnalysisResponse
        | CoordinatorRetrievalResponse
        | CoordinatorPrecedentResponse
        | CoordinatorEvidenceSummaryResponse
        | CoordinatorVerificationResponse
        | CoordinatorFullResearchResponse
    ):
        """Execute explicitly integrated tasks; retain planning-only behavior otherwise."""

        if request.requested_task == "analyze_evidence":
            return self.execute_analysis(request)
        if request.requested_task == "retrieve_precedents":
            return self.execute_retrieval(request)
        if request.requested_task == "summarize_precedents":
            return self.execute_summarize_precedents(request)
        if request.requested_task == "summarize_evidence":
            return self.execute_summarize_evidence(request)
        if request.requested_task == "verify_claims":
            return self.execute_verify_claims(request)
        if request.requested_task == "full_research":
            return self.execute_full_research(request)

        return self.plan(request)

    def execute_analysis(self, request: CoordinatorRequest) -> CoordinatorAnalysisResponse:
        """Run the existing Analysis Agent without transforming its output."""

        if request.requested_task != "analyze_evidence":
            raise ValueError("execute_analysis supports only the analyze_evidence task")
        prerequisites = self._analysis_prerequisites(request)
        if prerequisites:
            return CoordinatorAnalysisResponse(
                case_id=request.case_id,
                requested_task="analyze_evidence",
                status="requires_input",
                document_ids=request.document_ids,
                stages=[
                    CoordinatorStage(
                        agent="analysis",
                        task_name="analyze_evidence",
                        status="not_run",
                        errors=prerequisites,
                    )
                ],
                errors=prerequisites,
                verification_limitations=[_VERIFICATION_LIMITATION],
                executed=False,
            )

        results = list(self._analysis_adapter.analyze_documents(request))
        stages = [
            CoordinatorStage(
                agent="analysis",
                task_name="analyze_evidence",
                status=result.status,
                result_reference=result.document_id,
                warnings=[
                    CoordinatorMessage(code=warning.code, message=warning.message, stage="analysis")
                    for warning in result.warnings
                ],
                errors=[
                    CoordinatorMessage(code=error.code, message=error.message, stage="analysis")
                    for error in result.errors
                ],
            )
            for result in results
        ]
        status = _aggregate_analysis_status(results)
        return CoordinatorAnalysisResponse(
            case_id=request.case_id,
            requested_task="analyze_evidence",
            status=status,
            document_ids=request.document_ids,
            stages=stages,
            analysis_results=results,
            warnings=[message for stage in stages for message in stage.warnings],
            errors=[message for stage in stages for message in stage.errors],
            verification_limitations=[_VERIFICATION_LIMITATION],
            executed=True,
        )

    def execute_retrieval(self, request: CoordinatorRequest) -> CoordinatorRetrievalResponse:
        """Run a caller-injected prepared Retrieval Agent for precedent search."""

        if request.requested_task != "retrieve_precedents":
            raise ValueError("execute_retrieval supports only the retrieve_precedents task")
        if self._retrieval_adapter is None:
            return self._retrieval_requires_input(
                request,
                "missing_retrieval_service",
                "A prepared Retrieval Agent must be supplied before retrieval can run.",
            )
        try:
            results = self._retrieval_adapter.search(request)
        except RetrievalUnavailableError as error:
            return self._retrieval_requires_input(
                request, "missing_retrieval_index", str(error)
            )
        except (RuntimeError, ValueError, NotImplementedError):
            return CoordinatorRetrievalResponse(
                case_id=request.case_id,
                requested_task="retrieve_precedents",
                status="failed",
                retrieval_mode=request.retrieval_mode,
                top_k=request.top_k,
                stages=[
                    CoordinatorStage(
                        agent="retrieval",
                        task_name="retrieve_precedents",
                        status="failed",
                        errors=[
                            CoordinatorMessage(
                                code="retrieval_failed",
                                message="Retrieval could not complete safely.",
                                stage="retrieval",
                            )
                        ],
                    )
                ],
                errors=[
                    CoordinatorMessage(
                        code="retrieval_failed",
                        message="Retrieval could not complete safely.",
                        stage="retrieval",
                    )
                ],
                verification_limitations=[_VERIFICATION_LIMITATION],
                executed=True,
            )

        warnings = []
        if not results:
            warnings.append(
                CoordinatorMessage(
                    code="no_retrieval_results",
                    message="Retrieval completed but returned no matching passages.",
                    stage="retrieval",
                )
            )
        return CoordinatorRetrievalResponse(
            case_id=request.case_id,
            requested_task="retrieve_precedents",
            status="completed",
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            stages=[
                CoordinatorStage(
                    agent="retrieval",
                    task_name="retrieve_precedents",
                    status="completed",
                    warnings=warnings,
                )
            ],
            retrieval_results=results,
            warnings=warnings,
            verification_limitations=[
                _VERIFICATION_LIMITATION,
                "Retrieved relevance scores do not establish authority, applicability, or binding status.",
            ],
            executed=True,
        )

    def execute_summarize_evidence(
        self, request: CoordinatorRequest
    ) -> "CoordinatorEvidenceSummaryResponse":
        """Run only deterministic evidence summarization over supplied documents."""

        from agents.coordinator.evidence_summarization_adapter import (
            CoordinatorEvidenceSummarizationAdapter,
        )
        from agents.coordinator.workflow_contracts import CoordinatorEvidenceSummaryResponse

        if request.requested_task != "summarize_evidence":
            raise ValueError("execute_summarize_evidence supports only summarize_evidence")
        if not request.evidence_documents:
            error = CoordinatorMessage(
                code="missing_evidence_documents",
                message="Supplied evidence documents are required for evidence summarization.",
                stage="summarization",
            )
            return CoordinatorEvidenceSummaryResponse(
                case_id=request.case_id, requested_task="summarize_evidence", status="requires_input",
                stages=[CoordinatorStage(agent="summarization", task_name="summarize_evidence", status="not_run", errors=[error])],
                errors=[error], verification_limitations=[_VERIFICATION_LIMITATION], executed=False,
            )
        adapter = self._evidence_summarization_adapter or CoordinatorEvidenceSummarizationAdapter()
        try:
            response = adapter.summarize(request)
        except (RuntimeError, ValueError, NotImplementedError):
            error = CoordinatorMessage(
                code="evidence_summarization_failed",
                message="Deterministic evidence summarization could not complete safely.",
                stage="summarization",
            )
            return CoordinatorEvidenceSummaryResponse(
                case_id=request.case_id, requested_task="summarize_evidence", status="failed",
                stages=[CoordinatorStage(agent="summarization", task_name="summarize_evidence", status="failed", errors=[error])],
                errors=[error], verification_limitations=[_VERIFICATION_LIMITATION], executed=True,
            )
        warnings = [_summary_message(item) for item in response.warnings]
        errors = [_summary_message(item) for item in response.errors]
        status = "completed" if response.status == "completed" else "partial"
        return CoordinatorEvidenceSummaryResponse(
            case_id=request.case_id, requested_task="summarize_evidence", status=status,
            stages=[CoordinatorStage(agent="summarization", task_name="summarize_evidence", status=status, warnings=warnings, errors=errors)],
            summarization_response=response, warnings=warnings, errors=errors,
            verification_limitations=[_VERIFICATION_LIMITATION], executed=True,
        )

    def execute_verify_claims(self, request: CoordinatorRequest) -> "CoordinatorVerificationResponse":
        """Run the actual narrow deterministic verifier over caller-supplied records."""

        from agents.coordinator.verification_adapter import CoordinatorVerificationAdapter
        from agents.coordinator.workflow_contracts import CoordinatorVerificationResponse

        if request.requested_task != "verify_claims":
            raise ValueError("execute_verify_claims supports only verify_claims")
        if not request.verification_claims:
            error = CoordinatorMessage(
                code="missing_verification_claims",
                message="Structured verification claims are required for verification execution.",
                stage="verification",
            )
            return CoordinatorVerificationResponse(
                case_id=request.case_id, requested_task="verify_claims", status="requires_input",
                stages=[CoordinatorStage(agent="verification", task_name="verify_claims", status="not_run", errors=[error])],
                errors=[error], verification_limitations=[_VERIFICATION_LIMITATION], executed=False,
            )
        adapter = self._verification_adapter or CoordinatorVerificationAdapter()
        try:
            verified = adapter.verify(request.verification_claims, request.verification_evidence, f"verify-{request.case_id}")
        except (RuntimeError, ValueError, NotImplementedError):
            error = CoordinatorMessage(
                code="verification_failed", message="Verification could not complete safely.", stage="verification"
            )
            return CoordinatorVerificationResponse(
                case_id=request.case_id, requested_task="verify_claims", status="failed",
                stages=[CoordinatorStage(agent="verification", task_name="verify_claims", status="failed", errors=[error])],
                errors=[error], verification_limitations=[_VERIFICATION_LIMITATION], executed=True,
            )
        status = "completed" if verified.overall_status == "supported" else "partial"
        warnings = [
            CoordinatorMessage(code=warning.code, message=warning.message, stage="verification")
            for warning in verified.warnings
        ]
        return CoordinatorVerificationResponse(
            case_id=request.case_id, requested_task="verify_claims", status=status,
            stages=[CoordinatorStage(agent="verification", task_name="verify_claims", status=status, warnings=warnings)],
            verification_response=verified, warnings=warnings,
            verification_limitations=[
                _VERIFICATION_LIMITATION,
                "Verification checks supplied wording and citations; it does not establish legal truth or authority.",
            ], executed=True,
        )

    def execute_full_research(self, request: CoordinatorRequest) -> "CoordinatorFullResearchResponse":
        """Run the currently integrated agents, retaining every stage's real output."""

        from agents.coordinator.verification_adapter import CoordinatorVerificationAdapter, VerificationMappingError
        from agents.coordinator.workflow_contracts import CoordinatorFullResearchResponse
        from agents.summarization import SummarizationAgent
        from agents.summarization.schemas import LegalResearchReportRequest

        if request.requested_task != "full_research":
            raise ValueError("execute_full_research supports only full_research")
        prerequisites = self._analysis_prerequisites(request)
        if self._retrieval_adapter is None:
            prerequisites.append(CoordinatorMessage(
                code="missing_retrieval_service",
                message="A prepared Retrieval Agent is required for full research.",
                stage="retrieval",
            ))
        if prerequisites:
            return CoordinatorFullResearchResponse(
                case_id=request.case_id, requested_task="full_research", status="requires_input",
                stages=[CoordinatorStage(agent=agent, task_name=task, status="not_run") for agent, task in _ROUTES["full_research"]],
                errors=prerequisites, verification_limitations=[_VERIFICATION_LIMITATION], executed=False,
            )

        analysis = self.execute_analysis(request.model_copy(update={"requested_task": "analyze_evidence"}))
        evidence = self.execute_summarize_evidence(request.model_copy(update={"requested_task": "summarize_evidence"}))
        precedent = self.execute_summarize_precedents(request.model_copy(update={"requested_task": "summarize_precedents"}))
        stages = [*analysis.stages, *evidence.stages, *precedent.stages]
        warnings = [*analysis.warnings, *evidence.warnings, *precedent.warnings]
        errors = [*analysis.errors, *evidence.errors, *precedent.errors]

        research_workflow = None
        if evidence.summarization_response and precedent.summarization_responses:
            try:
                research_workflow = SummarizationAgent().run_research_workflow(
                    _workflow_request(request, evidence.summarization_response, precedent.summarization_responses, analysis.analysis_results)
                )
                stages.append(CoordinatorStage(
                    agent="summarization", task_name="research_report", status=research_workflow.status,
                ))
            except (RuntimeError, ValueError, NotImplementedError):
                message = CoordinatorMessage(
                    code="research_report_failed", message="The structured research report could not be assembled safely.", stage="summarization"
                )
                stages.append(CoordinatorStage(agent="summarization", task_name="research_report", status="failed", errors=[message]))
                errors.append(message)

        verification_response = None
        if evidence.summarization_response and evidence.summarization_response.evidence_summary:
            try:
                titles = {document.document_id: document.name for document in request.evidence_documents}
                adapter = self._verification_adapter or CoordinatorVerificationAdapter()
                claims, verification_evidence = adapter.from_evidence_summary(
                    evidence.summarization_response.evidence_summary,
                    task_id=f"full-research-{request.case_id}", source_titles=titles,
                )
                if claims:
                    verification_response = adapter.verify(claims, verification_evidence, f"full-research-{request.case_id}")
                    verification_status = "completed" if verification_response.overall_status == "supported" else "partial"
                    stages.append(CoordinatorStage(agent="verification", task_name="verify_claims", status=verification_status))
                else:
                    stages.append(CoordinatorStage(agent="verification", task_name="verify_claims", status="requires_input"))
                    warnings.append(CoordinatorMessage(
                        code="no_traceable_summary_claims",
                        message="No extractive summary point had a sufficient traceable source for verification.",
                        stage="verification",
                    ))
            except (VerificationMappingError, RuntimeError, ValueError, NotImplementedError):
                message = CoordinatorMessage(
                    code="verification_failed", message="Verification could not complete safely.", stage="verification"
                )
                stages.append(CoordinatorStage(agent="verification", task_name="verify_claims", status="failed", errors=[message]))
                errors.append(message)
        else:
            stages.append(CoordinatorStage(agent="verification", task_name="verify_claims", status="requires_input"))

        statuses = [stage.status for stage in stages]
        status = "completed" if statuses and all(item == "completed" for item in statuses) else "partial"
        if all(item == "failed" for item in statuses):
            status = "failed"
        return CoordinatorFullResearchResponse(
            case_id=request.case_id, requested_task="full_research", status=status,
            stages=stages, analysis_results=analysis.analysis_results, precedent_response=precedent,
            evidence_summary_response=evidence.summarization_response, research_workflow=research_workflow,
            verification_response=verification_response, warnings=warnings, errors=errors,
            verification_limitations=[
                _VERIFICATION_LIMITATION,
                "Verification checks supplied wording and citations; it does not establish legal truth or authority.",
                "Retrieved relevance scores do not establish authority, applicability, or binding status.",
            ], executed=True,
        )

    def execute_summarize_precedents(
        self, request: CoordinatorRequest
    ) -> "CoordinatorPrecedentResponse":
        """Retrieve and deterministically summarize explicitly typed judgment material."""

        if request.requested_task != "summarize_precedents":
            raise ValueError(
                "execute_summarize_precedents supports only the summarize_precedents task"
            )
        from agents.coordinator.precedent_contracts import CoordinatorPrecedentResponse
        from agents.coordinator.summarization_adapter import (
            CitationCollisionError,
            CoordinatorSummarizationAdapter,
        )

        summarization_adapter = (
            self._summarization_adapter or CoordinatorSummarizationAdapter()
        )
        if self._retrieval_adapter is None:
            return self._precedent_requires_input(
                request,
                "missing_retrieval_service",
                "A prepared Retrieval Agent must be supplied before precedent summarization can run.",
            )
        try:
            results = self._retrieval_adapter.search(request)
        except RetrievalUnavailableError as error:
            return self._precedent_requires_input(
                request, "missing_retrieval_index", str(error)
            )
        except (RuntimeError, ValueError, NotImplementedError):
            error = CoordinatorMessage(
                code="retrieval_failed",
                message="Retrieval could not complete safely.",
                stage="retrieval",
            )
            return CoordinatorPrecedentResponse(
                case_id=request.case_id,
                requested_task="summarize_precedents",
                status="failed",
                retrieval_mode=request.retrieval_mode,
                top_k=request.top_k,
                stages=[
                    CoordinatorStage(
                        agent="retrieval", task_name="retrieve_precedents", status="failed", errors=[error]
                    ),
                    CoordinatorStage(
                        agent="summarization", task_name="summarize_precedents", status="not_run"
                    ),
                ],
                errors=[error],
                verification_limitations=[_VERIFICATION_LIMITATION],
                executed=True,
            )

        retrieval_stage = CoordinatorStage(
            agent="retrieval", task_name="retrieve_precedents", status="completed"
        )
        if not results:
            warning = CoordinatorMessage(
                code="no_retrieval_results",
                message="Retrieval completed but returned no matching passages.",
                stage="retrieval",
            )
            retrieval_stage.warnings.append(warning)
            return CoordinatorPrecedentResponse(
                case_id=request.case_id,
                requested_task="summarize_precedents",
                status="partial",
                retrieval_mode=request.retrieval_mode,
                top_k=request.top_k,
                stages=[
                    retrieval_stage,
                    CoordinatorStage(
                        agent="summarization", task_name="summarize_precedents", status="not_run"
                    ),
                ],
                retrieval_results=results,
                warnings=[warning],
                verification_limitations=[_VERIFICATION_LIMITATION],
                executed=True,
            )

        try:
            summaries, provenance, adapter_warnings = (
                summarization_adapter.summarize_precedents(request, results)
            )
        except (CitationCollisionError, RuntimeError, ValueError, NotImplementedError):
            error = CoordinatorMessage(
                code="summarization_failed",
                message="Deterministic precedent summarization could not complete safely.",
                stage="summarization",
            )
            return CoordinatorPrecedentResponse(
                case_id=request.case_id,
                requested_task="summarize_precedents",
                status="partial",
                retrieval_mode=request.retrieval_mode,
                top_k=request.top_k,
                stages=[
                    retrieval_stage,
                    CoordinatorStage(
                        agent="summarization", task_name="summarize_precedents", status="failed", errors=[error]
                    ),
                ],
                retrieval_results=results,
                errors=[error],
                verification_limitations=[_VERIFICATION_LIMITATION],
                executed=True,
            )

        warnings = list(adapter_warnings)
        errors: list[CoordinatorMessage] = []
        for summary in summaries:
            warnings.extend(
                CoordinatorMessage(
                    code=warning.code, message=warning.description, stage="summarization"
                )
                for warning in summary.warnings
            )
            errors.extend(
                CoordinatorMessage(
                    code=error.code, message=error.description, stage="summarization"
                )
                for error in summary.errors
            )
        if not summaries:
            warnings.append(
                CoordinatorMessage(
                    code="no_supported_judgment_material",
                    message="Retrieved material contained no explicitly classified judgment or precedent passages.",
                    stage="summarization",
                )
            )
        summary_status = _aggregate_summary_status(summaries, warnings, errors)
        source_references = _unique_source_references(summaries)
        summarization_stage = CoordinatorStage(
            agent="summarization",
            task_name="summarize_precedents",
            status=summary_status,
            warnings=warnings,
            errors=errors,
        )
        return CoordinatorPrecedentResponse(
            case_id=request.case_id,
            requested_task="summarize_precedents",
            status=summary_status,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            stages=[retrieval_stage, summarization_stage],
            retrieval_results=results,
            summarization_responses=summaries,
            precedent_summaries=[item for summary in summaries for item in summary.precedent_summaries],
            citation_provenance=provenance,
            source_references=source_references,
            warnings=warnings,
            errors=errors,
            verification_limitations=[
                _VERIFICATION_LIMITATION,
                "Retrieved relevance scores do not establish authority, applicability, or binding status.",
            ],
            executed=True,
        )

    @staticmethod
    def _precedent_requires_input(
        request: CoordinatorRequest, code: str, message: str
    ) -> "CoordinatorPrecedentResponse":
        from agents.coordinator.precedent_contracts import CoordinatorPrecedentResponse
        error = CoordinatorMessage(code=code, message=message, stage="retrieval")
        return CoordinatorPrecedentResponse(
            case_id=request.case_id,
            requested_task="summarize_precedents",
            status="requires_input",
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            stages=[
                CoordinatorStage(
                    agent="retrieval", task_name="retrieve_precedents", status="not_run", errors=[error]
                ),
                CoordinatorStage(
                    agent="summarization", task_name="summarize_precedents", status="not_run"
                ),
            ],
            errors=[error],
            verification_limitations=[_VERIFICATION_LIMITATION],
            executed=False,
        )

    @staticmethod
    def _retrieval_requires_input(
        request: CoordinatorRequest, code: str, message: str
    ) -> CoordinatorRetrievalResponse:
        error = CoordinatorMessage(code=code, message=message, stage="retrieval")
        return CoordinatorRetrievalResponse(
            case_id=request.case_id,
            requested_task="retrieve_precedents",
            status="requires_input",
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            stages=[
                CoordinatorStage(
                    agent="retrieval",
                    task_name="retrieve_precedents",
                    status="not_run",
                    errors=[error],
                )
            ],
            errors=[error],
            verification_limitations=[_VERIFICATION_LIMITATION],
            executed=False,
        )

    @staticmethod
    def _missing_prerequisites(request: CoordinatorRequest) -> list[CoordinatorMessage]:
        needs_documents = request.requested_task in {
            "analyze_evidence",
            "summarize_evidence",
            "full_research",
        }
        needs_claims = request.requested_task == "verify_claims"
        errors: list[CoordinatorMessage] = []
        if needs_documents and not (request.document_ids or request.evidence_ids):
            errors.append(
                CoordinatorMessage(
                    code="missing_evidence_input",
                    message="Document or evidence identifiers are required for this planned task.",
                )
            )
        if needs_claims and not request.claim_ids:
            errors.append(
                CoordinatorMessage(
                    code="missing_claim_input",
                    message="Claim identifiers are required for verification planning.",
                )
            )
        return errors

    @staticmethod
    def _analysis_prerequisites(request: CoordinatorRequest) -> list[CoordinatorMessage]:
        errors = Coordinator._missing_prerequisites(request)
        if request.case_title is None:
            errors.append(
                CoordinatorMessage(
                    code="missing_case_title",
                    message="A case title is required to construct the Analysis request.",
                )
            )
        if not request.evidence_documents:
            errors.append(
                CoordinatorMessage(
                    code="missing_evidence_documents",
                    message="Supplied evidence documents are required for Analysis execution.",
                )
            )
        return errors


def _aggregate_analysis_status(
    results: list["AnalysisResponse"],
) -> "Literal['completed', 'partial', 'requires_input', 'failed']":
    """Preserve stage outcomes without representing partial work as completed."""

    statuses = [result.status for result in results]
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status in {"partial", "requires_input"} for status in statuses):
        return "partial"
    return "completed"


def _aggregate_summary_status(
    summaries: list["SummarizationResponse"],
    warnings: list[CoordinatorMessage],
    errors: list[CoordinatorMessage],
) -> "Literal['completed', 'partial', 'requires_input', 'failed']":
    """Avoid representing incomplete source coverage as a completed summary."""

    if errors or any(summary.status == "failed" for summary in summaries):
        return "partial" if summaries else "failed"
    if not summaries or warnings or any(
        summary.status != "completed" for summary in summaries
    ):
        return "partial"
    return "completed"


def _unique_source_references(
    summaries: list["SummarizationResponse"],
) -> list["SourceReference"]:
    """Preserve first retrieval order and never merge unequal sources under one ID."""

    references: list["SourceReference"] = []
    seen: set[str] = set()
    for summary in summaries:
        for source in summary.source_references:
            if source.citation_id not in seen:
                references.append(source)
                seen.add(source.citation_id)
    return references


def _summary_message(item: object) -> CoordinatorMessage:
    """Translate public Summarization warnings without exposing source text."""

    return CoordinatorMessage(
        code=str(getattr(item, "code")),
        message=str(getattr(item, "description")),
        stage="summarization",
    )


def _workflow_request(
    request: CoordinatorRequest,
    evidence_response: "SummarizationResponse",
    precedent_responses: list["SummarizationResponse"],
    analysis_results: list["AnalysisResponse"],
) -> "SummarizationWorkflowRequest":
    """Build a public deterministic workflow request from already grounded outputs."""

    from agents.summarization.schemas import (
        EvidenceItem,
        RetrievedPassage,
        SummarizationRequest,
        SummarizationWorkflowRequest,
    )

    evidence_summary = evidence_response.evidence_summary
    assert evidence_summary is not None
    current_sources = evidence_summary.source_references
    current = SummarizationRequest(
        case_id=request.case_id,
        document_ids=list(dict.fromkeys(source.document_id for source in current_sources)),
        query=request.user_query,
        summary_type="evidence_summary",
        evidence_items=[
            EvidenceItem(
                text=point.text,
                source=point.source_references[0] if len(point.source_references) == 1 else None,
                is_negated=point.is_negated,
                is_conditional=point.is_conditional,
                is_attributed=point.is_attributed,
                epistemic_status=point.epistemic_status,
                verification_status=point.verification_status,
            )
            for point in evidence_summary.points
        ],
    )
    precedent_requests = []
    for response in precedent_responses:
        sources = response.source_references
        if not sources:
            continue
        precedent_requests.append(SummarizationRequest(
            case_id=request.case_id,
            document_ids=list(dict.fromkeys(source.document_id for source in sources)),
            query=request.user_query,
            summary_type="precedent_summary",
            retrieved_passages=[
                RetrievedPassage(text=source.excerpt, source=source)
                for source in sources
                if source.excerpt and source.excerpt.strip()
            ],
        ))
    return SummarizationWorkflowRequest(
        case_id=request.case_id,
        research_question=request.user_query,
        current_case_request=current,
        precedent_requests=precedent_requests,
        analysis_findings=[],
    )
