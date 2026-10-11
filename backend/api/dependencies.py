"""Internal database dependency for authenticated API routes."""

from functools import lru_cache

from fastapi import HTTPException, status

from backend.config import get_settings
from database import PersistenceService, build_engine, build_session_factory
from backend.services.private_documents import PrivateDocumentStorage


@lru_cache
def get_persistence_service() -> PersistenceService:
    database_url = get_settings().database_url.strip()
    if not database_url:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unavailable",
        )
    try:
        return PersistenceService(build_session_factory(build_engine(database_url)))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is unavailable",
        ) from None


@lru_cache
def get_private_document_storage() -> PrivateDocumentStorage:
    settings = get_settings()
    try:
        return PrivateDocumentStorage(
            settings.private_document_storage_dir,
            settings.max_private_upload_bytes,
        )
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Private document storage is unavailable",
        ) from None
