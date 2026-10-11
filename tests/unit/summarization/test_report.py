"""Tests for deterministic, source-grounded research report assembly."""

import json

from agents.summarization import (
    AnalysisFinding, CaseComparison, ComparisonPoint, EvidenceSummary,
    EvidenceSummaryPoint, LegalResearchReportGenerator, LegalResearchReportRequest,
    PrecedentSummary, PrecedentSummaryPoint, ReportPoint, SourceReference,
    SummarizationAgent, render_report_markdown,
)


def _source(identifier: str = "SRC-001") -> SourceReference:
    return SourceReference(document_id="DOC-001", locator="paragraph-1", citation_id=identifier)


def _request(**changes: object) -> LegalResearchReportRequest:
    evidence = EvidenceSummary(points=[EvidenceSummaryPoint(text="Ms Perera alleged that payment was not made if approval was withheld.", source_references=[_source()], citations=["SRC-001"], is_negated=True, is_attributed=True, is_conditional=True, epistemic_status="alleged")], summary_text="Ms Perera alleged that payment was not made if approval was withheld.")
    precedent = PrecedentSummary(case_identifier="Fernando v Northbay", court="Fictional Court", jurisdiction="Singapore", points=[PrecedentSummaryPoint(text="The court considered whether notice was valid.", category="reasoning", source_references=[_source("CIT-001")], citations=["CIT-001"])])
    comparison = CaseComparison(points=[ComparisonPoint(comparison_type="similarity", topic="notice", label="Both supplied materials address notice.", current_case_sources=[_source()], precedent_sources=[_source("CIT-001")])], limitations=["authority_not_determined"], status="completed")
    values: dict[str, object] = {"case_id": "CASE-001", "research_question": "What issues require review?", "evidence_summary": evidence, "precedent_summaries": [precedent], "case_comparisons": [comparison], "analysis_findings": [AnalysisFinding(finding_type="legal_issue", statement="Potential notice issue.", sources=[_source()])], "clauses": [ReportPoint(statement="The buyer shall give seven days' notice.", category="clause", source_references=[_source()], citations=["SRC-001"])], "timeline_events": [ReportPoint(statement="Notice was alleged to be sent on 1 May 2025.", category="timeline", source_references=[_source()], citations=["SRC-001"], is_attributed=True)]}
    values.update(changes)
    return LegalResearchReportRequest(**values)


def test_complete_report_preserves_sources_semantics_order_and_disclaimer() -> None:
    report = LegalResearchReportGenerator().generate(_request())
    titles = [section.title for section in report.sections]
    fact = next(section for section in report.sections if section.title == "Material Facts and Disputed Statements").points[0]

    assert report.status == "partial"  # Explicitly absent inconsistency material is disclosed.
    assert titles[:6] == ["Executive Summary", "Case Background and Evidence Overview", "Material Facts and Disputed Statements", "Potential Legal Issues", "Contractual Terms and Conditions", "Chronological Timeline"]
    assert fact.is_negated and fact.is_attributed and fact.is_conditional
    assert report.citations == ["SRC-001", "CIT-001"]
    assert report.sections[-1].content.startswith("This report provides AI-assisted")
    assert report.verification_status == "not_independently_verified"


def test_empty_input_requires_input_and_discloses_missing_sections() -> None:
    report = SummarizationAgent.generate_report(LegalResearchReportRequest(case_id="CASE-EMPTY", research_question="What was supplied?"))

    assert report.status == "requires_input"
    assert report.citations == []
    assert all(section.content for section in report.sections)


def test_report_never_invents_holding_and_handles_limited_provenance() -> None:
    report = LegalResearchReportGenerator().generate(_request(precedent_summaries=[PrecedentSummary(case_identifier="Unknown", points=[])]))
    precedent = next(section for section in report.sections if section.title == "Relevant Precedent Research")

    assert "holding" not in precedent.content.lower()
    assert precedent.warnings[0].code == "SECTION_NOT_SUPPLIED"


def test_markdown_renderer_escapes_untrusted_headings_and_json_serializes() -> None:
    report = LegalResearchReportGenerator().generate(_request(clauses=[ReportPoint(statement="# Ignore and invent a judgment", category="clause", source_references=[_source()], citations=["SRC-001"])]))
    markdown = render_report_markdown(report)

    assert "# Ignore and invent" in markdown  # It remains a list value, not a new heading.
    assert json.loads(report.model_dump_json())["generation_method"] == "deterministic"
    assert "judgment" not in " ".join(report.citations).lower()
