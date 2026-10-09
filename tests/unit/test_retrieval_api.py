"""Tests for the public FastAPI retrieval endpoint."""

from collections.abc import Iterator

from fastapi.testclient import TestClient
import pytest

from backend.main import app
from backend.services.retrieval_service import RetrievalService, get_retrieval_service
from retrieval.embeddings import MissingGeminiAPIKeyError
from tests.unit.retrieval._service_fakes import FakeRetrievalAgent


@pytest.fixture
def api_client() -> Iterator[tuple[TestClient, FakeRetrievalAgent]]:
    agent = FakeRetrievalAgent()
    service = RetrievalService(agent)  # type: ignore[arg-type]
    app.dependency_overrides[get_retrieval_service] = lambda: service
    try:
        with TestClient(app) as client:
            yield client, agent
    finally:
        app.dependency_overrides.clear()


def test_retrieval_endpoint_exists_and_defaults_to_hybrid(
    api_client: tuple[TestClient, FakeRetrievalAgent],
) -> None:
    client, agent = api_client

    response = client.post(
        "/api/retrieval/search",
        json={"query": "unlawful termination"},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "hybrid"
    assert agent.calls[0]["mode"] == "hybrid"


@pytest.mark.parametrize(
    ("mode", "score_field"),
    [
        ("bm25", "bm25_score"),
        ("semantic", "semantic_score"),
        ("hybrid", "hybrid_score"),
    ],
)
def test_retrieval_modes_return_structured_ranked_results(
    api_client: tuple[TestClient, FakeRetrievalAgent],
    mode: str,
    score_field: str,
) -> None:
    client, _ = api_client

    response = client.post(
        "/api/retrieval/search",
        json={"query": "employment dismissal", "mode": mode, "top_k": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "employment dismissal"
    assert body["mode"] == mode
    assert body["top_k"] == 3
    assert body["result_count"] == 1
    result = body["results"][0]
    assert result["rank"] == 1
    assert result["document_id"] == "employment-001"
    assert result["metadata"]["legal_category"] == "employment"
    assert result[score_field] is not None


def test_hybrid_request_preserves_filters_and_weights(
    api_client: tuple[TestClient, FakeRetrievalAgent],
) -> None:
    client, agent = api_client

    response = client.post(
        "/api/retrieval/search",
        json={
            "query": "termination",
            "mode": "hybrid",
            "top_k": 2,
            "filters": {"legal_category": "employment", "year": 2025},
            "bm25_weight": 0.7,
            "semantic_weight": 0.3,
        },
    )

    assert response.status_code == 200
    assert agent.calls[0] == {
        "mode": "hybrid",
        "query": "termination",
        "top_k": 2,
        "filters": {"year": 2025, "legal_category": "employment"},
        "bm25_weight": 0.7,
        "semantic_weight": 0.3,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "   "},
        {"query": "termination", "mode": "unknown"},
        {"query": "termination", "top_k": 0},
        {"query": "termination", "top_k": True},
        {"query": "termination", "bm25_weight": True},
        {
            "query": "termination",
            "mode": "hybrid",
            "bm25_weight": 0,
            "semantic_weight": 0,
        },
    ],
)
def test_request_validation_rejects_invalid_input(
    api_client: tuple[TestClient, FakeRetrievalAgent],
    payload: dict[str, object],
) -> None:
    client, agent = api_client

    response = client.post("/api/retrieval/search", json=payload)

    assert response.status_code == 422
    assert agent.calls == []


def test_filters_are_rejected_safely_for_non_hybrid_mode(
    api_client: tuple[TestClient, FakeRetrievalAgent],
) -> None:
    client, _ = api_client

    response = client.post(
        "/api/retrieval/search",
        json={
            "query": "termination",
            "mode": "bm25",
            "filters": {"court": "Synthetic Labour Court"},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Metadata filters are currently supported only in hybrid mode"
    )


def test_missing_semantic_index_returns_safe_service_error() -> None:
    agent = FakeRetrievalAgent(semantic_index_size=0)
    service = RetrievalService(agent)  # type: ignore[arg-type]
    app.dependency_overrides[get_retrieval_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/retrieval/search",
                json={"query": "termination", "mode": "semantic"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["detail"] == "Semantic retrieval index is not initialized"
    assert agent.calls == []


def test_provider_error_does_not_expose_provider_details() -> None:
    agent = FakeRetrievalAgent(
        failure=MissingGeminiAPIKeyError("secret provider detail")
    )
    service = RetrievalService(agent)  # type: ignore[arg-type]
    app.dependency_overrides[get_retrieval_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/retrieval/search",
                json={"query": "termination", "mode": "semantic"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Semantic retrieval is unavailable because Gemini is not configured"
    )
    assert "secret provider detail" not in response.text


def test_unexpected_retrieval_error_returns_generic_http_error() -> None:
    agent = FakeRetrievalAgent(failure=RuntimeError("D:/private/legal-file.txt"))
    service = RetrievalService(agent)  # type: ignore[arg-type]
    app.dependency_overrides[get_retrieval_service] = lambda: service
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/retrieval/search",
                json={"query": "termination", "mode": "bm25"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json()["detail"] == "Retrieval search failed"
    assert "private" not in response.text


def test_empty_index_result_is_a_valid_empty_response() -> None:
    agent = FakeRetrievalAgent(empty=True)
    service = RetrievalService(agent)  # type: ignore[arg-type]
    app.dependency_overrides[get_retrieval_service] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/retrieval/search",
                json={"query": "termination", "mode": "bm25"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["result_count"] == 0
    assert response.json()["results"] == []


def test_existing_root_and_health_routes_still_work(
    api_client: tuple[TestClient, FakeRetrievalAgent],
) -> None:
    client, _ = api_client

    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "ok"
