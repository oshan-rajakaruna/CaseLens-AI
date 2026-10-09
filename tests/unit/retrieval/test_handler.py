"""Tests for the retrieval-side Coordinator contract adapter."""

import asyncio

from agents.retrieval.handler import RetrievalTaskHandler
from backend.schemas import AgentTask
from backend.services.retrieval_service import RetrievalService
from tests.unit.retrieval._service_fakes import FakeRetrievalAgent


def test_handler_runs_structured_task_and_preserves_contract_fields() -> None:
    agent = FakeRetrievalAgent()
    handler = RetrievalTaskHandler(
        RetrievalService(agent)  # type: ignore[arg-type]
    )
    task = AgentTask(
        id="task-001",
        agent="retrieval",
        instruction="Retrieve relevant legal chunks",
        context={
            "query": "unlawful termination",
            "legal_issue": "Whether dismissal lacked procedural fairness",
            "mode": "hybrid",
            "top_k": 4,
            "filters": {"legal_category": "employment", "year": 2025},
            "bm25_weight": 0.6,
            "semantic_weight": 0.4,
        },
    )

    result = asyncio.run(handler.handle(task))

    assert result.task_id == "task-001"
    assert result.agent == "retrieval"
    assert result.status == "success"
    assert result.output["query"] == "unlawful termination"
    assert result.output["legal_issue"] == (
        "Whether dismissal lacked procedural fairness"
    )
    assert result.output["mode"] == "hybrid"
    assert result.output["top_k"] == 4
    assert result.output["filters"] == {
        "year": 2025,
        "legal_category": "employment",
    }
    assert result.output["results"][0]["rank"] == 1
    assert result.output["results"][0]["document_id"] == "employment-001"
    assert agent.calls[0]["query"] == "unlawful termination"
    assert agent.calls[0]["top_k"] == 4


def test_handler_returns_structured_error_for_invalid_context() -> None:
    handler = RetrievalTaskHandler(
        RetrievalService(FakeRetrievalAgent())  # type: ignore[arg-type]
    )
    task = AgentTask(
        id="task-invalid",
        agent="retrieval",
        instruction="Retrieve",
        context={"query": "   ", "mode": "hybrid"},
    )

    result = asyncio.run(handler.handle(task))

    assert result.status == "error"
    assert result.output["error"] == {
        "code": "invalid_retrieval_task",
        "message": "Retrieval task context is invalid",
    }


def test_handler_rejects_task_for_another_agent() -> None:
    handler = RetrievalTaskHandler(
        RetrievalService(FakeRetrievalAgent())  # type: ignore[arg-type]
    )
    task = AgentTask(
        id="task-analysis",
        agent="analysis",
        instruction="Retrieve",
        context={"query": "termination"},
    )

    result = asyncio.run(handler.handle(task))

    assert result.status == "error"
    assert result.output["error"]["code"] == "invalid_agent"


def test_handler_returns_safe_error_when_retrieval_fails() -> None:
    agent = FakeRetrievalAgent(failure=RuntimeError("C:/private/secret.txt"))
    handler = RetrievalTaskHandler(
        RetrievalService(agent)  # type: ignore[arg-type]
    )
    task = AgentTask(
        id="task-failure",
        agent="retrieval_agent",
        instruction="Retrieve",
        context={
            "query": "termination",
            "legal_issue": "Employment issue",
            "mode": "bm25",
            "top_k": 2,
        },
    )

    result = asyncio.run(handler.handle(task))

    assert result.status == "error"
    assert result.output["query"] == "termination"
    assert result.output["mode"] == "bm25"
    assert result.output["top_k"] == 2
    assert result.output["error"] == {
        "code": "retrieval_failed",
        "message": "Retrieval search failed",
    }
    assert "private" not in str(result.output)
