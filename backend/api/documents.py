"""Authenticated private case-evidence upload and metadata listing."""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.api.dependencies import get_persistence_service, get_private_document_storage
from backend.api.persistence_errors import persistence_http_error
from backend.auth.session import require_authenticated_principal, require_csrf_principal
from backend.schemas.case_report_api import CaseDocumentListResponse, CaseDocumentResponse
from backend.services.private_documents import (
    InvalidUploadError,
    PrivateDocumentStorage,
    PrivateStorageError,
    UploadTooLargeError,
    validate_upload_identity,
)
from database import PersistenceService, TrustedPrincipal
from nlp.extraction import extract_document

router = APIRouter(prefix="/cases", tags=["case documents"])


def _response(record) -> CaseDocumentResponse:
    return CaseDocumentResponse(
        id=record.id,
        case_id=record.case_id,
        safe_filename=record.safe_filename,
        media_type=record.media_type,
        uploaded_at=record.uploaded_at,
    )


@router.post("/{case_id}/documents", response_model=CaseDocumentResponse, status_code=201)
def upload_case_document(
    case_id: str,
    file: Annotated[UploadFile, File(...)],
    principal: Annotated[TrustedPrincipal, Depends(require_csrf_principal)],
    persistence: Annotated[PersistenceService, Depends(get_persistence_service)],
    storage: Annotated[PrivateDocumentStorage, Depends(get_private_document_storage)],
) -> CaseDocumentResponse:
    stored = None
    try:
        persistence.get_case(principal, case_id)
        safe_filename, extension = validate_upload_identity(file.filename, file.content_type)
        media_type = file.content_type or ""
        stored = storage.store(file.file, extension)
        document_id = f"document-{uuid4()}"
        extraction = extract_document(
            storage.resolve(stored.storage_key),
            document_id,
            max_file_size_bytes=storage.max_upload_bytes,
        )
        if extraction.status == "failed" or (
            media_type == "text/plain"
            and (extraction.status != "extracted" or not extraction.text.strip())
        ):
            storage.delete(stored.storage_key)
            stored = None
            raise InvalidUploadError("Uploaded document content is invalid")
        record = persistence.add_document(
            principal,
            document_id=document_id,
            case_id=case_id,
            safe_filename=safe_filename,
            media_type=media_type,
            storage_key=stored.storage_key,
            source_metadata={
                "file_size_bytes": stored.size_bytes,
                "sha256": stored.sha256_hex,
                "extraction_status": extraction.status,
                "page_count": extraction.metadata.get("page_count"),
                "ocr_required": any(
                    warning.code == "ocr_required" for warning in extraction.warnings
                ),
            },
        )
        return _response(record)
    except UploadTooLargeError:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Upload is too large") from None
    except InvalidUploadError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document upload") from None
    except PrivateStorageError:
        if stored is not None:
            try:
                storage.delete(stored.storage_key)
            except PrivateStorageError:
                pass
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Private document storage is unavailable") from None
    except Exception as error:
        if stored is not None:
            try:
                storage.delete(stored.storage_key)
            except PrivateStorageError:
                pass
        raise persistence_http_error(error) from None
    finally:
        file.file.close()


@router.get("/{case_id}/documents", response_model=CaseDocumentListResponse)
def list_case_documents(
    case_id: str,
    principal: Annotated[TrustedPrincipal, Depends(require_authenticated_principal)],
    persistence: Annotated[PersistenceService, Depends(get_persistence_service)],
) -> CaseDocumentListResponse:
    try:
        return CaseDocumentListResponse(
            items=[_response(item) for item in persistence.list_documents(principal, case_id)]
        )
    except Exception as error:
        raise persistence_http_error(error) from None
