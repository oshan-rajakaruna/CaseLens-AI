"""Local orchestration for the independently executable Analysis Agent."""

from agents.analysis.schemas import AnalysisRequest, AnalysisResponse, AnalysisWarning
from nlp.clauses import extract_clauses
from nlp.contradictions import detect_contradictions
from nlp.entities import extract_entities
from nlp.facts import extract_facts
from nlp.issues import identify_legal_issues
from nlp.timeline import build_timeline


class AnalysisAgent:
    """Validate supplied evidence and expose a stable future analysis boundary."""

    def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Run local evidence analysis while retaining partial-stage outcomes."""

        if not request.text.strip():
            return AnalysisResponse(
                case_id=request.case.id,
                document_id=request.document.id,
                status="requires_input",
                warnings=[
                    AnalysisWarning(
                        code="empty_text",
                        message="Document text is empty or whitespace-only; no analysis was performed.",
                        field="text",
                    )
                ],
            )

        results: dict[str, object] = {"entities": [], "facts": [], "clauses": [], "legal_issues": [], "timeline": [], "contradiction_candidates": []}
        stages: dict[str, str] = {}
        errors: list[AnalysisWarning] = []
        def run(name: str, call):
            try:
                results[name] = call()
                stages[name] = "completed"
            except Exception:
                stages[name] = "failed"
                errors.append(AnalysisWarning(code=f"{name}_failed", message=f"{name} stage could not complete."))
        run("entities", lambda: extract_entities(request.text, request.document.id, request.segments or None))
        run("facts", lambda: extract_facts(request.text, request.document.id, request.segments or None, results["entities"]))
        run("clauses", lambda: extract_clauses(request.text, request.document.id, request.segments or None, results["entities"]))
        run("legal_issues", lambda: identify_legal_issues(results["facts"], results["clauses"]))
        run("timeline", lambda: build_timeline(results["facts"]))
        run("contradiction_candidates", lambda: detect_contradictions(results["facts"], case_id=request.case.id))
        return AnalysisResponse(case_id=request.case.id, document_id=request.document.id, status="partial" if errors else "completed", entities=results["entities"], facts=results["facts"], clauses=results["clauses"], legal_issues=results["legal_issues"], timeline=results["timeline"], contradiction_candidates=results["contradiction_candidates"], errors=errors, stage_status=stages)
