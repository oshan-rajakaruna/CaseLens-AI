"""Tests for the minimal FastAPI application."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.main import app


def test_application_can_be_imported() -> None:
    assert isinstance(app, FastAPI)


def test_health_returns_expected_response() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "caselens-backend",
    }
