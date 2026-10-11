"""Offline tests for the shared Coordinator ResearchWorkflow adapter."""

import asyncio
from types import SimpleNamespace

import pytest

from agents.analysis.schemas import AnalysisResponse, EvidenceSource, ExtractedFact
from agents.coordinator import Coordinator, ResearchWorkflowCoordinatorAdapter
from agents.coordinator.schemas import RetrievalContextDiagnostics
from backend.schemas import AgentTask
from backend.schemas.research_workflow import (
    ResearchWorkflowError,
    ResearchWorkflowStages,
)


class _Workflow:
    def __init__(self, result=None, failure: Exception | None = None) -> None:
        self.result = result or _workflow_result()
        self.failure = failure
        self.calls = []

    def run(self, request):
        self.calls.append(request)
        if self.failure is not None:
            raise self.failure
        return self.result


def _task(task_id: str = "task-001", **context_updates) -> AgentTask:
    context = {
        "case": {"id": "case-001", "title": "Authorized synthetic matter"},
        "query": "notice requirement",
        "evidence_documents": [
            {
                "document": {
                    "id": "evidence-001",
                    "case_id": "case-001",
                    "name": "authorized.txt",
                    "media_type": "text/plain",
                },
                "text": "PRIVATE EVIDENCE: the tenant received notice.",
            }
        ],
        "retrieval_mode": "bm25",
    }
    context.update(context_updates)
    return AgentTask(
        id=task_id,
        agent="coordinator",
        instruction="Run internal legal research.",
        context=context,
    )


def _workflow_result(status: str = "completed", *, error: bool = False):
    analysis = AnalysisResponse(
        case_id="case-001",
        document_id="evidence-001",
        status="partial" if status == "partial" else "completed",
        facts=[
            ExtractedFact(
                statement="PRIVATE EVIDENCE: the tenant received notice.",
                source=EvidenceSource(
                    document_id="evidence-001",
                    excerpt="PRIVATE EVIDENCE",
                ),
                method="test-rule",
            )
        ],
    )
    stages = ResearchWorkflowStages(
        analysis="partial" if status == "partial" else "completed",
        retrieval="failed" if status == "failed" else "completed",
        summarization_verification=(
            "not_started" if status == "failed" else "completed"
        ),
        report="not_started" if status == "failed" else "completed",
        comparisons="completed" if status != "failed" else "not_started",
        markdown="skipped",
    )
    errors = (
        [
            ResearchWorkflowError(
                stage="retrieval",
                code="retrieval_unavailable",
                message="Legal authority retrieval could not complete.",
            )
        ]
        if error or status == "failed"
        else []
    )
    diagnostics = RetrievalContextDiagnostics(
        retrieval_result_count=1,
        rag_passage_count=1,
        source_document_count=1,
        estimated_context_tokens=20,
        context_truncated=False,
        empty_retrieval=False,
        no_context=False,
    )
    verification = SimpleNamespace(
        overall_status="uncertain",
        verified=False,
        claim_results=[object()],
        warnings=[object()],
    )
    authority = SimpleNamespace(
        context_id="CTX-001",
        document_id="ICJ-001",
        pages=[7],
        chunk_ids=["ICJ-001-chunk-0001"],
        official_citation="I.C.J. Reports 2021, p. 206",
    )
    comparison = SimpleNamespace(
        id="CMP-001",
        authority=SimpleNamespace(context_id="CTX-001"),
        review_status="requires_human_review",
        ai_draft=SimpleNamespace(verification_status="not_verified"),
    )
    return SimpleNamespace(
        task_id="task-001",
        status=status,
        stages=stages,
        analyses=[analysis],
        retrieval_payload=(
            None if status == "failed" else SimpleNamespace(diagnostics=diagnostics)
        ),
        summary_verification=(
            None
            if status == "failed"
            else SimpleNamespace(verification=verification)
        ),
        report=(
            None
            if status == "failed"
            else SimpleNamespace(authorities=[authority], comparisons=[comparison])
        ),
        markdown=None,
        errors=errors,
    )


def test_coordinator_delegates_once_and_preserves_task_identity_and_request() -> None:
    workflow = _Workflow()
    coordinator = Coordinator(ResearchWorkflowCoordinatorAdapter(workflow))

    result = asyncio.run(coordinator.handle(_task()))

    assert result.task_id == "task-001"
    assert result.agent == "coordinator"
    assert result.status == "completed"
    assert len(workflow.calls) == 1
    request = workflow.calls[0]
    assert request.task_id == "task-001"
    assert request.case.id == "case-001"
    assert request.query == "notice requirement"
    assert request.evidence_documents[0].document.id == "evidence-001"


@pytest.mark.parametrize("status", ["completed", "partial", "failed"])
def test_workflow_status_maps_without_promotion(status: str) -> None:
    workflow = _Workflow(_workflow_result(status))
    adapter = ResearchWorkflowCoordinatorAdapter(workflow)

    result = asyncio.run(adapter.run(_task()))

    assert result.status == status
    assert result.output["workflow_status"] == status
    assert result.output["report_available"] is (status != "failed")


def test_safe_output_preserves_ctx_and_verification_without_private_evidence() -> None:
    workflow = _Workflow()
    adapter = ResearchWorkflowCoordinatorAdapter(workflow)

    result = asyncio.run(adapter.run(_task()))

    assert result.output["verification"] == {
        "overall_status": "uncertain",
        "verified": False,
        "claim_count": 1,
        "warning_count": 1,
    }
    assert result.output["authorities"] == [
        {
            "context_id": "CTX-001",
            "document_id": "ICJ-001",
            "pages": [7],
            "chunk_ids": ["ICJ-001-chunk-0001"],
            "official_citation": "I.C.J. Reports 2021, p. 206",
        }
    ]
    assert result.output["comparisons"][0]["review_status"] == "requires_human_review"
    assert result.output["comparisons"][0]["draft_verification_status"] == "not_verified"
    assert "private evidence" not in result.model_dump_json().casefold()
    assert adapter.get_private_result("task-001").analyses[0].facts[0].source.excerpt == "PRIVATE EVIDENCE"


def test_repeated_identical_task_returns_cached_result_without_second_execution() -> None:
    workflow = _Workflow()
    adapter = ResearchWorkflowCoordinatorAdapter(workflow)
    task = _task()

    async def execute_twice():
        return await adapter.run(task), await adapter.run(task)

    first, second = asyncio.run(execute_twice())

    assert first == second
    assert len(workflow.calls) == 1


def test_reused_task_id_with_different_request_is_rejected() -> None:
    workflow = _Workflow()
    adapter = ResearchWorkflowCoordinatorAdapter(workflow)

    async def execute_conflict():
        await adapter.run(_task())
        return await adapter.run(_task(query="different query"))

    result = asyncio.run(execute_conflict())

    assert result.status == "failed"
    assert result.output["errors"][0]["code"] == "task_id_conflict"
    assert len(workflow.calls) == 1


@pytest.mark.parametrize(
    "task",
    [
        AgentTask(id="bad-1", agent="retrieval", instruction="x", context={}),
        AgentTask(id="bad-2", agent="coordinator", instruction="x", context={}),
        _task(task_id="bad-3", task_id_from_context="not-used"),
        _task(
            task_id="bad-4",
            evidence_documents=[
                {
                    "document": {
                        "id": "doc-1",
                        "case_id": "another-case",
                        "name": "bad.txt",
                    },
                    "text": "text",
                }
            ],
        ),
    ],
)
def test_invalid_tasks_are_rejected_without_workflow_calls(task: AgentTask) -> None:
    if task.id == "bad-3":
        task.context["task_id"] = task.context.pop("task_id_from_context")
    workflow = _Workflow()
    adapter = ResearchWorkflowCoordinatorAdapter(workflow)

    result = asyncio.run(adapter.run(task))

    assert result.status == "failed"
    assert result.output["report_available"] is False
    assert workflow.calls == []


def test_missing_evidence_is_adapted_for_workflow_partial_handling() -> None:
    workflow = _Workflow(_workflow_result("partial"))
    adapter = ResearchWorkflowCoordinatorAdapter(workflow)
    task = _task(evidence_documents=[])

    result = asyncio.run(adapter.run(task))

    assert result.status == "partial"
    assert workflow.calls[0].evidence_documents == []


def test_unexpected_exception_is_sanitized_and_not_retried() -> None:
    workflow = _Workflow(failure=RuntimeError("PRIVATE EVIDENCE provider secret"))
    adapter = ResearchWorkflowCoordinatorAdapter(workflow)
    task = _task()

    async def execute_twice():
        return await adapter.run(task), await adapter.run(task)

    first, second = asyncio.run(execute_twice())

    assert first == second
    assert first.status == "failed"
    assert first.output["errors"][0]["code"] == "workflow_execution_failed"
    assert "private evidence" not in first.model_dump_json().casefold()
    assert len(workflow.calls) == 1
