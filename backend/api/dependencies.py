"""Internal database dependency for authenticated API routes."""

from functools import lru_cache

from fastapi import HTTPException, status

from backend.config import get_settings
from database import PersistenceService, build_engine, build_session_factory


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
