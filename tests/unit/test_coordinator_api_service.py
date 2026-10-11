"""Backend-owned authorization preparation tests for the common Coordinator API."""
from datetime import datetime

import pytest

from agents.coordinator.schemas import CoordinatorRequest
from backend.db.models import DocumentRecord
from backend.services.coordinator_service import execute_authorized


def _document() -> DocumentRecord:
    return DocumentRecord(
        id="DOC-001", case_id="CASE-001", owner_id="user-a", original_name="fictional.txt",
        media_type="text/plain", storage_key="private-key", size_bytes=24,
        extraction_status="extracted", extracted_text="Nimal Perera paid LKR 5,000.",
    )


def test_authorized_service_uses_stored_evidence_not_client_text() -> None:
    request = CoordinatorRequest(
        case_id="CASE-001", case_title="Fictional case", user_query="Analyze evidence",
        requested_task="analyze_evidence", document_ids=["DOC-001"],
    )
    _, enriched = execute_authorized(request, [_document()])

    assert enriched.evidence_documents[0].document_id == "DOC-001"
    assert enriched.evidence_documents[0].text == "Nimal Perera paid LKR 5,000."


def test_service_rejects_document_ids_not_owned_by_the_case() -> None:
    request = CoordinatorRequest(
        case_id="CASE-001", user_query="Analyze evidence", requested_task="analyze_evidence",
        document_ids=["DOC-OTHER"],
    )
    with pytest.raises(ValueError, match="not available"):
        execute_authorized(request, [_document()])
