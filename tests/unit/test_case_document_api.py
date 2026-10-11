"""Offline tests for authenticated private case-evidence uploads."""

from collections.abc import Iterator
from io import BytesIO

import fitz
from fastapi.testclient import TestClient
import pytest

from backend.api.dependencies import get_persistence_service, get_private_document_storage
from backend.config import get_settings
from backend.main import app
from backend.services.private_documents import (
    EvidenceTextUnavailableError,
    PrivateDocumentStorage,
    PrivateStorageError,
    PrivateEvidenceService,
)
from database import PersistenceService, TrustedPrincipal, build_engine, build_session_factory
from database.errors import OwnershipError, PersistenceError
from database.models import Base

_ORIGIN = "http://127.0.0.1:5173"
_PASSWORD = "offline-development-password"
_SECRET = "offline-signing-key-with-at-least-32-characters"
_OWNER = TrustedPrincipal(subject="local|operator")


def _pdf_bytes(text: str | None) -> bytes:
    pdf = fitz.open()
    page = pdf.new_page()
    if text:
        page.insert_text((72, 72), text)
    content = pdf.tobytes()
    pdf.close()
    return content


@pytest.fixture
def upload_setup(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, PersistenceService, PrivateDocumentStorage]]:
    engine = build_engine(f"sqlite+pysqlite:///{tmp_path / 'upload.sqlite3'}")
    Base.metadata.create_all(engine)
    persistence = PersistenceService(build_session_factory(engine))
    storage = PrivateDocumentStorage(tmp_path / "private-documents", 1024 * 1024)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("ENABLE_DEV_CASE_API", "true")
    monkeypatch.setenv("DEV_AUTH_OPERATOR_SUBJECT", _OWNER.subject)
    monkeypatch.setenv("DEV_AUTH_PASSWORD", _PASSWORD)
    monkeypatch.setenv("DEV_AUTH_ALLOWED_ORIGIN", _ORIGIN)
    monkeypatch.setenv("SECRET_KEY", _SECRET)
    get_settings.cache_clear()
    app.dependency_overrides[get_persistence_service] = lambda: persistence
    app.dependency_overrides[get_private_document_storage] = lambda: storage
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, persistence, storage
    finally:
        app.dependency_overrides.clear()
        get_settings.cache_clear()
        get_private_document_storage.cache_clear()
        engine.dispose()


def _login(client: TestClient) -> str:
    response = client.post(
        "/api/auth/login",
        headers={"Origin": _ORIGIN},
        json={"password": _PASSWORD},
    )
    assert response.status_code == 200
    return response.json()["csrf_token"]


def _upload(client: TestClient, case_id: str, csrf: str, filename: str, content: bytes, media_type: str):
    return client.post(
        f"/api/cases/{case_id}/documents",
        headers={"Origin": _ORIGIN, "X-CSRF-Token": csrf},
        files={"file": (filename, content, media_type)},
    )


def test_unauthorized_upload_is_blocked(upload_setup) -> None:
    client, persistence, _ = upload_setup
    persistence.create_case(_OWNER, case_id="case-1", title="Case")

    response = client.post(
        "/api/cases/case-1/documents",
        files={"file": ("evidence.txt", b"private", "text/plain")},
    )
    assert response.status_code == 401


def test_upload_requires_owner_case_and_hides_foreign_case(upload_setup) -> None:
    client, persistence, storage = upload_setup
    other = TrustedPrincipal(subject="local|other")
    persistence.create_case(other, case_id="case-other", title="Other")
    csrf = _login(client)

    missing = _upload(client, "missing", csrf, "evidence.txt", b"text", "text/plain")
    foreign = _upload(client, "case-other", csrf, "evidence.txt", b"text", "text/plain")
    assert missing.status_code == foreign.status_code == 404
    assert client.get("/api/cases/case-other/documents").status_code == 404
    assert not list(storage.root.rglob("*.txt"))


def test_upload_requires_exact_origin_and_csrf(upload_setup) -> None:
    client, persistence, _ = upload_setup
    persistence.create_case(_OWNER, case_id="case-1", title="Case")
    csrf = _login(client)
    files = {"file": ("evidence.txt", b"text", "text/plain")}

    assert client.post("/api/cases/case-1/documents", files=files).status_code == 403
    assert client.post(
        "/api/cases/case-1/documents",
        headers={"Origin": _ORIGIN, "X-CSRF-Token": "wrong"},
        files=files,
    ).status_code == 403
    assert client.post(
        "/api/cases/case-1/documents",
        headers={"Origin": "http://localhost:9999", "X-CSRF-Token": csrf},
        files=files,
    ).status_code == 403


def test_valid_text_upload_persists_safe_metadata_and_lists_owner_scope(upload_setup) -> None:
    client, persistence, storage = upload_setup
    persistence.create_case(_OWNER, case_id="case-1", title="Case")
    csrf = _login(client)

    response = _upload(
        client, "case-1", csrf, "Evidence Notes.txt", b"First line\nSecond line", "text/plain"
    )
    assert response.status_code == 201
    body = response.json()
    assert body["case_id"] == "case-1"
    assert body["safe_filename"] == "Evidence Notes.txt"
    assert body["media_type"] == "text/plain"
    assert "storage" not in response.text.casefold()
    assert "First line" not in response.text

    record = persistence.get_document(_OWNER, "case-1", body["id"])
    assert record.storage_key.startswith("objects/")
    assert "Evidence Notes" not in record.storage_key
    assert not record.storage_key.startswith(("/", "\\"))
    assert storage.resolve(record.storage_key).read_bytes() == b"First line\nSecond line"
    assert "data/legal" not in storage.root.as_posix().casefold()
    listing = client.get("/api/cases/case-1/documents")
    assert listing.status_code == 200
    assert listing.json()["items"] == [body]


def test_valid_pdf_upload_and_duplicate_display_names_are_safe(upload_setup) -> None:
    client, persistence, _ = upload_setup
    persistence.create_case(_OWNER, case_id="case-1", title="Case")
    csrf = _login(client)
    content = _pdf_bytes("Private PDF evidence")

    first = _upload(client, "case-1", csrf, "evidence.pdf", content, "application/pdf")
    second = _upload(client, "case-1", csrf, "evidence.pdf", content, "application/pdf")
    assert first.status_code == second.status_code == 201
    records = persistence.list_documents(_OWNER, "case-1")
    assert len(records) == 2
    assert records[0].storage_key != records[1].storage_key


@pytest.mark.parametrize(
    ("filename", "content", "media_type"),
    [
        ("evidence.csv", b"a,b", "text/csv"),
        ("evidence.pdf", b"not a pdf", "application/pdf"),
        ("evidence.txt", b"\xff\xfe", "text/plain"),
        ("evidence.pdf", b"plain text", "text/plain"),
        ("../escape.txt", b"text", "text/plain"),
        ("..\\escape.txt", b"text", "text/plain"),
    ],
)
def test_invalid_unsupported_and_traversal_uploads_are_rejected(
    upload_setup, filename, content, media_type
) -> None:
    client, persistence, storage = upload_setup
    persistence.create_case(_OWNER, case_id="case-1", title="Case")
    response = _upload(client, "case-1", _login(client), filename, content, media_type)

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid document upload"}
    assert not [path for path in storage.root.rglob("*") if path.is_file()]


def test_empty_and_oversized_uploads_are_rejected_and_cleaned(upload_setup) -> None:
    client, persistence, storage = upload_setup
    persistence.create_case(_OWNER, case_id="case-1", title="Case")
    csrf = _login(client)

    empty = _upload(client, "case-1", csrf, "empty.txt", b"", "text/plain")
    oversized = _upload(
        client,
        "case-1",
        csrf,
        "large.txt",
        b"x" * (storage.max_upload_bytes + 1),
        "text/plain",
    )
    assert empty.status_code == 400
    assert oversized.status_code == 413
    assert not [path for path in storage.root.rglob("*") if path.is_file()]


def test_private_storage_rejects_public_corpus_root(tmp_path) -> None:
    with pytest.raises(ValueError, match="public corpus"):
        PrivateDocumentStorage(tmp_path / "data" / "legal" / "raw", 1024)


def test_private_storage_rejects_untrusted_storage_keys(tmp_path) -> None:
    storage = PrivateDocumentStorage(tmp_path / "private", 1024)
    with pytest.raises(PrivateStorageError):
        storage.resolve("../outside.txt")
    with pytest.raises(PrivateStorageError):
        storage.delete("objects/aa/not-an-opaque-id.txt")


def test_interrupted_private_storage_write_is_cleaned(tmp_path) -> None:
    storage = PrivateDocumentStorage(tmp_path / "private", 1024)

    class InterruptedStream(BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError("simulated private evidence detail")
            return super().read(1)

    with pytest.raises(PrivateStorageError, match="storage failed"):
        storage.store(InterruptedStream(b"private evidence"), ".txt")
    assert not [path for path in storage.root.rglob("*") if path.is_file()]


def test_trusted_evidence_service_builds_analysis_input(upload_setup) -> None:
    client, persistence, storage = upload_setup
    persistence.create_case(_OWNER, case_id="case-1", title="Case")
    response = _upload(
        client, "case-1", _login(client), "evidence.txt", b"Supplied evidence", "text/plain"
    )
    document_id = response.json()["id"]

    evidence = PrivateEvidenceService(persistence, storage).load_for_analysis(
        _OWNER, case_id="case-1", document_id=document_id
    )
    assert evidence.document.id == document_id
    assert evidence.document.case_id == "case-1"
    assert evidence.text == "Supplied evidence"
    assert evidence.segments[0].source.locator == "text:1"
    with pytest.raises(OwnershipError):
        PrivateEvidenceService(persistence, storage).load_for_analysis(
            TrustedPrincipal(subject="local|other"),
            case_id="case-1",
            document_id=document_id,
        )


def test_textless_pdf_is_labelled_and_not_promoted_to_analysis_text(upload_setup) -> None:
    client, persistence, storage = upload_setup
    persistence.create_case(_OWNER, case_id="case-1", title="Case")
    response = _upload(
        client, "case-1", _login(client), "scan.pdf", _pdf_bytes(None), "application/pdf"
    )
    assert response.status_code == 201
    record = persistence.get_document(_OWNER, "case-1", response.json()["id"])
    assert record.source_metadata["extraction_status"] == "no_text"
    assert record.source_metadata["ocr_required"] is True
    with pytest.raises(EvidenceTextUnavailableError):
        PrivateEvidenceService(persistence, storage).load_for_analysis(
            _OWNER, case_id="case-1", document_id=record.id
        )


def test_database_failure_removes_private_file_and_sanitizes_error(upload_setup) -> None:
    client, persistence, storage = upload_setup
    persistence.create_case(_OWNER, case_id="case-1", title="Case")
    csrf = _login(client)

    class FailingPersistence:
        def get_case(self, *args, **kwargs):
            return persistence.get_case(*args, **kwargs)

        def add_document(self, *args, **kwargs):
            raise PersistenceError("PRIVATE-EVIDENCE database=password")

    app.dependency_overrides[get_persistence_service] = lambda: FailingPersistence()
    response = _upload(client, "case-1", csrf, "evidence.txt", b"PRIVATE-EVIDENCE", "text/plain")

    assert response.status_code == 503
    assert response.json() == {"detail": "Database is unavailable"}
    assert "PRIVATE-EVIDENCE" not in response.text
    assert "password" not in response.text
    assert not [path for path in storage.root.rglob("*") if path.is_file()]
