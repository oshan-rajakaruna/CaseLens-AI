"""End-to-end local Analysis Agent orchestration tests."""

from agents.analysis import AnalysisAgent, AnalysisRequest
from backend.schemas import Case, Document


def test_full_pipeline_preserves_source_grounded_outputs() -> None:
    text = "Nimal Perera alleges that Kamal did not pay LKR 50,000 on 12 January 2025. The tenant shall pay LKR 50,000 monthly."
    request = AnalysisRequest(case=Case(id="case-1", title="Example"), document=Document(id="doc-1", case_id="case-1", name="evidence.txt"), text=text)
    response = AnalysisAgent().analyze(request)
    assert response.status == "completed"
    assert response.facts[0].source.excerpt == response.facts[0].statement
    assert response.clauses[0].money_text == "LKR 50,000"
    assert response.legal_issues
    assert response.timeline[0].normalized_date == "2025-01-12"
    assert response.contradiction_candidates == []


def test_pipeline_is_deterministic_and_local() -> None:
    request = AnalysisRequest(case=Case(id="case-1", title="Example"), document=Document(id="doc-1", case_id="case-1", name="evidence.txt"), text="Nimal paid LKR 1000 on 12 January 2025.")
    first = AnalysisAgent().analyze(request)
    second = AnalysisAgent().analyze(request)
    assert first.model_dump() == second.model_dump()
