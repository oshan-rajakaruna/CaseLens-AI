"""Offline tests for Coordinator Step 1 planning and routing."""

import asyncio

import pytest
from pydantic import ValidationError

from agents.coordinator import Coordinator, CoordinatorRequest


def _request(task: str, **overrides: object) -> CoordinatorRequest:
    values: dict[str, object] = {
        "case_id": "CASE-2026-001",
        "user_query": "What international precedents may be relevant?",
        "requested_task": task,
        "document_ids": ["DOC-001"],
    }
    values.update(overrides)
    return CoordinatorRequest.model_validate(values)


@pytest.mark.parametrize(
    ("task", "agents"),
    [
        ("analyze_evidence", ["analysis"]),
        ("retrieve_precedents", ["retrieval"]),
        ("summarize_evidence", ["summarization"]),
        ("summarize_precedents", ["retrieval", "summarization"]),
        ("verify_claims", ["verification"]),
        ("full_research", ["analysis", "retrieval", "summarization", "verification"]),
    ],
)
def test_supported_tasks_receive_deterministic_planning_routes(
    task: str, agents: list[str]
) -> None:
    response = Coordinator().plan(
        _request(task, claim_ids=["CLAIM-001"] if task == "verify_claims" else [])
    )

    assert response.case_id == "CASE-2026-001"
    assert response.requested_task == task
    assert [stage.agent for stage in response.stages] == agents
    assert all(stage.status == "planned" for stage in response.stages)
    assert response.status == "planned"
    assert response.executed is False
    assert response.errors == []


@pytest.mark.parametrize(
    "values",
    [
        {"requested_task": "unsupported"},
        {"case_id": "   "},
        {"user_query": "\t"},
        {"user_query": "x" * 2001},
        {"document_ids": ["DOC-001", "DOC-001"]},
        {"document_ids": ["  "]},
        {"metadata": []},
        {"metadata": {" ": "value"}},
    ],
)
def test_invalid_planning_input_is_rejected(values: dict[str, object]) -> None:
    base = {
        "case_id": "CASE-2026-001",
        "user_query": "Find relevant precedent",
        "requested_task": "retrieve_precedents",
    }
    base.update(values)

    with pytest.raises(ValidationError):
        CoordinatorRequest.model_validate(base)


@pytest.mark.parametrize(
    ("task", "overrides", "error_code"),
    [
        ("analyze_evidence", {"document_ids": []}, "missing_evidence_input"),
        ("summarize_evidence", {"document_ids": [], "evidence_ids": []}, "missing_evidence_input"),
        ("full_research", {"document_ids": []}, "missing_evidence_input"),
        ("verify_claims", {"claim_ids": []}, "missing_claim_input"),
    ],
)
def test_missing_prerequisites_are_explicit_and_never_marked_complete(
    task: str, overrides: dict[str, object], error_code: str
) -> None:
    response = Coordinator().plan(_request(task, **overrides))

    assert response.status == "requires_input"
    assert [error.code for error in response.errors] == [error_code]
    assert all(stage.status == "not_run" for stage in response.stages)
    assert response.executed is False
    assert "not invoked" in response.warnings[0].message


def test_evidence_identifiers_satisfy_evidence_planning_without_fabricating_documents() -> None:
    response = Coordinator().plan(
        _request("summarize_evidence", document_ids=[], evidence_ids=["EVID-001"])
    )

    assert response.status == "planned"
    assert response.document_ids == []
    assert response.stages[0].status == "planned"


def test_plan_is_json_serializable_and_repeatable() -> None:
    request = _request("full_research")
    first = Coordinator().plan(request)
    second = Coordinator().plan(request)

    assert first == second
    restored = type(first).model_validate_json(first.model_dump_json())
    assert restored == first
    assert "No agent output" in first.verification_limitations[0]


def test_async_handle_reports_missing_retrieval_service_without_claiming_execution() -> None:
    request = _request("retrieve_precedents")

    response = asyncio.run(Coordinator().handle(request))

    assert response.status == "requires_input"
    assert response.executed is False
    assert response.stages[0].agent == "retrieval"
    assert response.stages[0].status == "not_run"
    assert response.errors[0].code == "missing_retrieval_service"
