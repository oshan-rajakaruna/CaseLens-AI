"""Coordinator-owned tests for Analysis execution mapping."""

from agents.analysis.schemas import AnalysisResponse
from agents.coordinator import (
    AnalysisCoordinatorAdapter,
    Coordinator,
    CoordinatorEvidenceDocument,
    CoordinatorRequest,
)


class RecordingAnalysisAgent:
    """Offline fake used to prove Coordinator mapping and invocation order."""

    def __init__(self) -> None:
        self.requests = []

    def analyze(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return AnalysisResponse(
            case_id=request.case.id,
            document_id=request.document.id,
            status="completed",
            stage_status={"facts": "completed"},
        )


def _request(**overrides: object) -> CoordinatorRequest:
    values: dict[str, object] = {
        "case_id": "CASE-ANALYSIS-001",
        "case_title": "Fictional commercial dispute",
        "user_query": "Analyze the supplied evidence.",
        "requested_task": "analyze_evidence",
        "document_ids": ["DOC-001"],
        "evidence_documents": [
            CoordinatorEvidenceDocument(
                document_id="DOC-001",
                name="evidence.txt",
                media_type="text/plain",
                text="Nimal Perera signed the agreement.",
                document_metadata={"language": "en"},
            )
        ],
    }
    values.update(overrides)
    return CoordinatorRequest.model_validate(values)


def test_adapter_constructs_an_analysis_request_without_changing_evidence() -> None:
    agent = RecordingAnalysisAgent()
    adapter = AnalysisCoordinatorAdapter(agent)  # type: ignore[arg-type]

    results = adapter.analyze_documents(_request())

    assert len(results) == 1
    analysis_request = agent.requests[0]
    assert analysis_request.case.id == "CASE-ANALYSIS-001"
    assert analysis_request.document.id == "DOC-001"
    assert analysis_request.document.case_id == "CASE-ANALYSIS-001"
    assert analysis_request.text == "Nimal Perera signed the agreement."
    assert analysis_request.document_metadata == {"language": "en"}


def test_coordinator_executes_only_analysis_and_preserves_typed_results() -> None:
    agent = RecordingAnalysisAgent()
    coordinator = Coordinator(AnalysisCoordinatorAdapter(agent))  # type: ignore[arg-type]

    response = coordinator.execute_analysis(_request())

    assert len(agent.requests) == 1
    assert response.executed is True
    assert response.status == "completed"
    assert response.stages[0].agent == "analysis"
    assert response.stages[0].status == "completed"
    assert response.stages[0].result_reference == "DOC-001"
    assert response.analysis_results[0].stage_status == {"facts": "completed"}
    assert "No agent output" in response.verification_limitations[0]


def test_missing_execution_inputs_are_reported_without_invoking_analysis() -> None:
    agent = RecordingAnalysisAgent()
    coordinator = Coordinator(AnalysisCoordinatorAdapter(agent))  # type: ignore[arg-type]

    response = coordinator.execute_analysis(
        _request(case_title=None, evidence_documents=[])
    )

    assert agent.requests == []
    assert response.executed is False
    assert response.status == "requires_input"
    assert response.analysis_results == []
    assert response.stages[0].status == "not_run"
    assert {error.code for error in response.errors} == {
        "missing_case_title",
        "missing_evidence_documents",
    }
