"""Internal database dependency for authenticated API routes."""

from functools import lru_cache

from fastapi import HTTPException, status

from backend.config import get_settings
from database import PersistenceService, build_engine, build_session_factory
from agents.analysis import AnalysisAgent
from agents.coordinator import (
    Coordinator,
    LegalRetrievalContextService,
    ResearchWorkflowCoordinatorAdapter,
)
from backend.services.private_documents import PrivateDocumentStorage, PrivateEvidenceService
from backend.services.research_execution import ResearchExecutionService
from backend.services.research_workflow import ResearchWorkflow
from backend.services.retrieval_service import get_retrieval_service
from backend.services.summarization_service import get_summarization_service


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


@lru_cache
def get_research_execution_service() -> ResearchExecutionService:
    try:
        persistence = get_persistence_service()
        storage = get_private_document_storage()
        adapter = ResearchWorkflowCoordinatorAdapter(
            ResearchWorkflow(
                analysis=AnalysisAgent(),
                retrieval=LegalRetrievalContextService(
                    get_retrieval_service().agent
                ),
                summarization=get_summarization_service(),
            )
        )
        return ResearchExecutionService(
            persistence=persistence,
            evidence=PrivateEvidenceService(persistence, storage),
            coordinator=Coordinator(adapter),
            private_results=adapter,
            max_workers=1,
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research execution is unavailable",
        ) from None
