"""Public Coordinator endpoint contract checks without external identity services."""
from fastapi.testclient import TestClient
from backend.main import app


def test_coordinator_endpoint_is_documented_and_rejects_anonymous_execution() -> None:
    client = TestClient(app)
    schema = app.openapi()
    response = client.post("/api/coordinator/execute", json={
        "case_id": "CASE-001", "user_query": "Analyze", "requested_task": "analyze_evidence"
    })

    assert "/api/coordinator/execute" in schema["paths"]
    assert schema["paths"]["/api/coordinator/execute"]["post"]["tags"] == ["Coordinator — Agent Testing"]
    assert response.status_code == 401
