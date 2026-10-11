"""Contracts for provider-neutral and foundation-stage summarization."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agents.coordinator.schemas import ContextCitationMapEntry
from agents.summarization.citations import CitationValidationResult


def _require_text(value: str, field_name: str) -> str:
    """Reject blank identifiers without altering supplied evidence text."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must contain non-whitespace text")
    return value


class GroundingWarning(BaseModel):
    """Non-sensitive warning or error about grounding and processing."""

    model_config = ConfigDict(extra="forbid")

    code: str
    description: str
    affected_field: str | None = None

    @field_validator("code", "description")
    @classmethod
    def required_text(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)


class SourceReference(BaseModel):
    """A supplied evidence location; no provenance values are inferred."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    locator: str | None = None
    excerpt: str | None = None
    page: int | None = Field(default=None, gt=0)
    paragraph: int | None = Field(default=None, ge=0)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    citation_id: str | None = None

    @field_validator("document_id")
    @classmethod
    def document_id_required(cls, value: str) -> str:
        return _require_text(value, "document_id")

    @field_validator("citation_id")
    @classmethod
    def citation_id_is_supported_format(cls, value: str | None) -> str | None:
        if value is None:
            return value
        import re

        if not re.fullmatch(r"(?:CTX|SRC|CIT)-[A-Za-z0-9][A-Za-z0-9_.-]{1,63}", value):
            raise ValueError("citation_id must use CTX-, SRC-, or CIT- format")
        return value

    @model_validator(mode="after")
    def offset_range_is_valid(self) -> "SourceReference":
        if self.end_char is not None and self.start_char is None:
            raise ValueError("start_char is required when end_char is supplied")
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char < self.start_char
        ):
            raise ValueError("end_char must not precede start_char")
        return self


class EvidenceItem(BaseModel):
    """Original supplied evidence and its optional immutable provenance."""

    model_config = ConfigDict(extra="forbid")

    text: str
    source: SourceReference | None = None
    is_negated: bool = False
    is_conditional: bool = False
    is_attributed: bool = False
    epistemic_status: str = "stated"
    verification_status: str = "not_independently_verified"

    @field_validator("text", "epistemic_status", "verification_status")
    @classmethod
    def required_text(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)


class AnalysisFinding(BaseModel):
    """Future-facing analysis input with explicit uncertainty semantics."""

    model_config = ConfigDict(extra="forbid")

    finding_type: str
    statement: str
    sources: list[SourceReference] = Field(default_factory=list)
    is_negated: bool = False
    is_conditional: bool = False
    is_attributed: bool = False
    epistemic_status: str = "stated"
    verification_status: str = "not_independently_verified"

    @field_validator("finding_type", "statement", "epistemic_status", "verification_status")
    @classmethod
    def required_text(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)


class RetrievedPassage(BaseModel):
    """A retrieval result retained for a future summary stage."""

    model_config = ConfigDict(extra="forbid")

    text: str
    source: SourceReference
    rank: int | None = Field(default=None, ge=1)
    relevance_score: float | None = None

    @field_validator("text")
    @classmethod
    def text_required(cls, value: str) -> str:
        return _require_text(value, "text")


class EvidenceSummary(BaseModel):
    """Deterministic extractive summary with point-level provenance."""

    model_config = ConfigDict(extra="forbid")

    summary_text: str
    key_points: list[str] = Field(default_factory=list)
    points: list["EvidenceSummaryPoint"] = Field(default_factory=list)
    source_references: list[SourceReference] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)


class EvidenceSummaryPoint(BaseModel):
    """One extractive statement and only the sources that support it."""

    model_config = ConfigDict(extra="forbid")

    text: str
    source_references: list[SourceReference] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    is_negated: bool = False
    is_conditional: bool = False
    is_attributed: bool = False
    epistemic_status: str = "stated"
    verification_status: str = "not_independently_verified"
    provenance_limited: bool = False

    @field_validator("text", "epistemic_status", "verification_status")
    @classmethod
    def required_text(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)


class LLMSummaryDraftPoint(BaseModel):
    """Provider-produced point awaiting source and semantic validation."""

    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[str] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_required(cls, value: str) -> str:
        return _require_text(value, "text")

    @field_validator("citations")
    @classmethod
    def citations_required(cls, values: list[str]) -> list[str]:
        return [_require_text(value, "citations") for value in values]


class LLMSummaryDraft(BaseModel):
    """Strict, provider-neutral contract for an assisted evidence draft.

    Passing this structural contract does not verify that a claim follows from
    cited evidence. The agent performs bounded citation and semantic checks
    before exposing the draft as an ``EvidenceSummary``.
    """

    model_config = ConfigDict(extra="forbid")

    summary_text: str
    points: list[LLMSummaryDraftPoint] = Field(min_length=1, max_length=6)
    limitations: list[str] = Field(default_factory=list)
    human_review_required: bool = True

    @field_validator("summary_text")
    @classmethod
    def summary_text_required(cls, value: str) -> str:
        value = _require_text(value, "summary_text")
        if len(value) > 1800:
            raise ValueError("summary_text exceeds the safe display limit")
        return value


class JudgmentMetadata(BaseModel):
    """Explicit judgment metadata supplied by a trusted upstream source."""

    model_config = ConfigDict(extra="forbid")

    case_title: str | None = None
    case_number: str | None = None
    court: str | None = None
    jurisdiction: str | None = None
    judgment_date: str | None = None
    official_citation: str | None = None
    source_url: str | None = None


class PrecedentSummaryPoint(BaseModel):
    """An exact judgment excerpt grouped by its safe extraction category."""

    model_config = ConfigDict(extra="forbid")

    text: str
    category: Literal[
        "background", "legal_issue", "reasoning", "holding", "statute", "authority"
    ]
    source_references: list[SourceReference] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    provenance_limited: bool = False
    is_negated: bool = False
    is_conditional: bool = False
    is_attributed: bool = False

    @field_validator("text")
    @classmethod
    def text_required(cls, value: str) -> str:
        return _require_text(value, "text")


class PrecedentSummary(BaseModel):
    """Deterministic judgment extraction with point-level provenance."""

    model_config = ConfigDict(extra="forbid")

    case_identifier: str = "unknown"
    case_number: str | None = None
    court: str | None = None
    jurisdiction: str | None = None
    judgment_date: str | None = None
    legal_issues: list[str] = Field(default_factory=list)
    background: str | None = None
    reasoning: str | None = None
    holding: str | None = None
    statutory_provisions: list[str] = Field(default_factory=list)
    cited_authorities: list[str] = Field(default_factory=list)
    points: list[PrecedentSummaryPoint] = Field(default_factory=list)
    source_references: list[SourceReference] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)


class CaseComparison(BaseModel):
    """A reviewable, source-grounded current-case to precedent comparison."""

    model_config = ConfigDict(extra="forbid")

    current_case_context: str = ""
    precedent_context: str = ""
    supported_similarities: list[str] = Field(default_factory=list)
    supported_differences: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    supporting_sources: list[SourceReference] = Field(default_factory=list)
    current_case_id: str | None = None
    precedent_identifier: str | None = None
    precedent_court: str | None = None
    precedent_jurisdiction: str | None = None
    current_case_jurisdiction: str | None = None
    comparable_legal_issues: list[str] = Field(default_factory=list)
    points: list["ComparisonPoint"] = Field(default_factory=list)
    authority_status: Literal["not_determined"] = "not_determined"
    status: Literal["completed", "partial", "requires_input"] = "partial"


class ComparisonPoint(BaseModel):
    """One conservative comparison claim with evidence from each available side."""

    model_config = ConfigDict(extra="forbid")

    comparison_type: Literal["similarity", "difference", "missing_information", "jurisdiction"]
    topic: str
    current_case_statement: str | None = None
    precedent_statement: str | None = None
    label: str
    current_case_sources: list[SourceReference] = Field(default_factory=list)
    precedent_sources: list[SourceReference] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provenance_limited: bool = False


class CaseComparisonRequest(BaseModel):
    """Explicit, source-grounded inputs for one current case and precedents."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    current_case_summary: EvidenceSummary | None = None
    current_case_points: list[EvidenceSummaryPoint] = Field(default_factory=list)
    precedents: list[PrecedentSummary] = Field(default_factory=list)
    current_case_jurisdiction: str | None = None
    research_question: str | None = None

    @field_validator("case_id")
    @classmethod
    def case_id_required(cls, value: str) -> str:
        return _require_text(value, "case_id")

    @model_validator(mode="after")
    def requires_current_case_material(self) -> "CaseComparisonRequest":
        if self.current_case_summary is None and not self.current_case_points:
            raise ValueError("current_case_summary or current_case_points is required")
        return self


class ReportSection(BaseModel):
    """A future structured report section with direct source references."""

    model_config = ConfigDict(extra="forbid")

    title: str
    content: str
    citations: list[str] = Field(default_factory=list)
    points: list["ReportPoint"] = Field(default_factory=list)
    warnings: list[GroundingWarning] = Field(default_factory=list)


class ReportPoint(BaseModel):
    """A report statement retaining supplied source semantics."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    category: str
    source_references: list[SourceReference] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    is_negated: bool = False
    is_conditional: bool = False
    is_attributed: bool = False
    epistemic_status: str = "stated"
    verification_status: str = "not_independently_verified"
    provenance_limited: bool = False

    @field_validator("statement", "category", "epistemic_status", "verification_status")
    @classmethod
    def report_point_text_required(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)


class LegalResearchReportRequest(BaseModel):
    """Optional validated inputs for deterministic legal research assembly."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    research_question: str
    report_title: str | None = None
    current_case_jurisdiction: str | None = None
    evidence_summary: EvidenceSummary | None = None
    precedent_summaries: list[PrecedentSummary] = Field(default_factory=list)
    case_comparisons: list[CaseComparison] = Field(default_factory=list)
    analysis_findings: list[AnalysisFinding] = Field(default_factory=list)
    clauses: list[ReportPoint] = Field(default_factory=list)
    timeline_events: list[ReportPoint] = Field(default_factory=list)
    inconsistency_candidates: list[ReportPoint] = Field(default_factory=list)
    warnings: list[GroundingWarning] = Field(default_factory=list)
    document_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("case_id", "research_question")
    @classmethod
    def report_request_text_required(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)


class LegalResearchReport(BaseModel):
    """Primary structured output for deterministic, review-oriented reports."""

    model_config = ConfigDict(extra="forbid")

    report_id: str
    case_id: str
    research_question: str
    title: str
    status: Literal["completed", "partial", "requires_input", "failed"]
    sections: list[ReportSection] = Field(default_factory=list)
    source_references: list[SourceReference] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[GroundingWarning] = Field(default_factory=list)
    generation_method: Literal["deterministic"] = "deterministic"
    verification_status: str = "not_independently_verified"
    human_review_required: bool = True


class SummarizationWorkflowRequest(BaseModel):
    """Caller-supplied materials for the standalone, non-retrieving workflow."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    research_question: str
    current_case_request: "SummarizationRequest"
    precedent_requests: list["SummarizationRequest"] = Field(default_factory=list, max_length=12)
    current_case_jurisdiction: str | None = None
    analysis_findings: list[AnalysisFinding] = Field(default_factory=list)
    clauses: list[ReportPoint] = Field(default_factory=list, max_length=100)
    timeline_events: list[ReportPoint] = Field(default_factory=list, max_length=100)
    inconsistency_candidates: list[ReportPoint] = Field(default_factory=list, max_length=100)
    report_title: str | None = None
    render_markdown: bool = False

    @field_validator("case_id", "research_question")
    @classmethod
    def workflow_text_required(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)

    @model_validator(mode="after")
    def workflow_material_types_match(self) -> "SummarizationWorkflowRequest":
        if self.current_case_request.case_id != self.case_id:
            raise ValueError("current_case_request.case_id must match case_id")
        if self.current_case_request.summary_type != "evidence_summary":
            raise ValueError("current_case_request must be an evidence_summary request")
        for request in self.precedent_requests:
            if request.summary_type != "precedent_summary":
                raise ValueError("precedent_requests must use precedent_summary")
        citation_documents: dict[str, str] = {}
        for nested in [self.current_case_request, *self.precedent_requests]:
            sources = [
                *(item.source for item in nested.evidence_items if item.source),
                *(source for finding in nested.analysis_findings for source in finding.sources),
                *(passage.source for passage in nested.retrieved_passages),
            ]
            for source in sources:
                if source.citation_id is None:
                    continue
                existing = citation_documents.setdefault(source.citation_id, source.document_id)
                if existing != source.document_id:
                    raise ValueError(
                        "citation_id must not identify different documents within one workflow"
                    )
        return self


class WorkflowStageResult(BaseModel):
    """Safe stage status; messages intentionally contain no evidence text."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: Literal["completed", "partial", "requires_input", "failed", "not_run"]
    warnings: list[GroundingWarning] = Field(default_factory=list)
    errors: list[GroundingWarning] = Field(default_factory=list)


class SummarizationWorkflowResponse(BaseModel):
    """Complete standalone workflow output with preserved intermediate results."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    status: Literal["completed", "partial", "requires_input", "failed"]
    evidence_summary: EvidenceSummary | None = None
    precedent_summaries: list[PrecedentSummary] = Field(default_factory=list)
    case_comparisons: list[CaseComparison] = Field(default_factory=list)
    report: LegalResearchReport | None = None
    markdown: str | None = None
    stages: list[WorkflowStageResult] = Field(default_factory=list)
    source_references: list[SourceReference] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    warnings: list[GroundingWarning] = Field(default_factory=list)
    verification_status: str = "not_independently_verified"
    human_review_required: bool = True


class SummarizationRequest(BaseModel):
    """Independent, source-grounded input for the Step 1 foundation."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    document_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    query: str
    summary_type: Literal["evidence_summary", "precedent_summary", "case_comparison"] = "evidence_summary"
    mode: Literal["deterministic", "llm_assisted"] = "deterministic"
    evidence_text: str | None = None
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    analysis_findings: list[AnalysisFinding] = Field(default_factory=list)
    retrieved_passages: list[RetrievedPassage] = Field(default_factory=list)
    judgment_metadata: JudgmentMetadata | None = None
    comparison_request: CaseComparisonRequest | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("case_id", "query", "document_id")
    @classmethod
    def identifiers_and_query_required(cls, value: str | None, info: Any) -> str | None:
        if value is None and info.field_name == "document_id":
            return value
        return _require_text(value, info.field_name)  # type: ignore[arg-type]

    @field_validator("document_ids")
    @classmethod
    def document_ids_required(cls, values: list[str]) -> list[str]:
        return [_require_text(value, "document_ids") for value in values]

    @model_validator(mode="after")
    def validate_document_sources(self) -> "SummarizationRequest":
        document_ids = list(dict.fromkeys([
            *([self.document_id] if self.document_id else []),
            *self.document_ids,
        ]))
        if not document_ids:
            raise ValueError("document_id or document_ids is required")
        source_ids = [
            source.document_id
            for item in self.evidence_items
            if item.source is not None
            for source in [item.source]
        ]
        source_ids.extend(
            source.document_id
            for finding in self.analysis_findings
            for source in finding.sources
        )
        source_ids.extend(passage.source.document_id for passage in self.retrieved_passages)
        unexpected = sorted(set(source_ids).difference(document_ids))
        if unexpected:
            raise ValueError("all source document_id values must be declared")
        if self.summary_type == "case_comparison":
            if self.comparison_request is None:
                raise ValueError("comparison_request is required for case_comparison")
            if self.comparison_request.case_id != self.case_id:
                raise ValueError("comparison_request.case_id must match case_id")
        if self.mode == "llm_assisted" and self.summary_type != "evidence_summary":
            raise ValueError("llm_assisted mode currently supports evidence_summary only")
        return self


class SummarizationResponse(BaseModel):
    """Deterministic Step 1 result; generated summary fields remain empty."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    status: Literal["not_summarized", "completed", "partial", "requires_input", "failed"]
    evidence_summary: EvidenceSummary | None = None
    precedent_summaries: list[PrecedentSummary] = Field(default_factory=list)
    case_comparisons: list[CaseComparison] = Field(default_factory=list)
    report_sections: list[ReportSection] = Field(default_factory=list)
    source_references: list[SourceReference] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    warnings: list[GroundingWarning] = Field(default_factory=list)
    errors: list[GroundingWarning] = Field(default_factory=list)
    stage_status: dict[str, str] = Field(default_factory=dict)


class SummarizationOutput(BaseModel):
    """LLM-independent answer contract for later verification."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def answer_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must contain non-whitespace text")
        return value

    @field_validator("citations")
    @classmethod
    def citations_must_use_context_ids(cls, values: list[str]) -> list[str]:
        for value in values:
            if not _is_context_id(value):
                raise ValueError("citations must use CTX-nnn identifiers")
        return values


def _is_context_id(value: str) -> bool:
    if not isinstance(value, str) or not value.startswith("CTX-"):
        return False
    suffix = value[4:]
    return len(suffix) >= 3 and suffix.isdigit()


class SummarizationProviderRequest(BaseModel):
    """Complete grounded input passed to an injectable generation provider."""

    query: str
    legal_issue: str | None = None
    formatted_context: str
    citation_map: dict[str, ContextCitationMapEntry]
    generation_instructions: str
    prompt: str
    task: Literal["legacy_retrieval", "llm_assisted_evidence"] = "legacy_retrieval"
    allowed_citation_ids: list[str] = Field(default_factory=list)
    source_context: list["LLMSourceContext"] = Field(default_factory=list)


class LLMSourceContext(BaseModel):
    """One complete, caller-supplied source statement for a provider request."""

    model_config = ConfigDict(extra="forbid")

    text: str
    citations: list[str] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    is_negated: bool = False
    is_conditional: bool = False
    is_attributed: bool = False
    epistemic_status: str = "stated"
    verification_status: str = "not_independently_verified"

    @field_validator("text", "epistemic_status", "verification_status")
    @classmethod
    def context_text_required(cls, value: str, info: Any) -> str:
        return _require_text(value, info.field_name)


class SummarizationCitationValidation(BaseModel):
    """Consistency checks across inline and declared citation identities."""

    inline: CitationValidationResult
    declared_citations: list[str] = Field(default_factory=list)
    invalid_declared_citations: list[str] = Field(default_factory=list)
    declared_missing_inline: list[str] = Field(default_factory=list)
    inline_missing_declared: list[str] = Field(default_factory=list)
    all_citations_valid: bool


class SummarizationProviderMetadata(BaseModel):
    """Safe provider identity without credentials or request headers."""

    provider: str
    model: str | None = None
    provider_called: bool


class SummarizationResult(BaseModel):
    """Validated output ready for a future Verification Agent handoff."""

    output: SummarizationOutput
    citation_validation: SummarizationCitationValidation
    provider_metadata: SummarizationProviderMetadata
    used_context_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


SummarizationProviderResponse = SummarizationOutput | LLMSummaryDraft | dict[str, Any]
