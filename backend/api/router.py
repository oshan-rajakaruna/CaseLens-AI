"""Root API router composing implemented feature routers."""

from fastapi import APIRouter

from backend.api.retrieval import router as retrieval_router
from backend.api.summarization import router as summarization_router
from backend.api.auth import router as auth_router
from backend.api.cases import router as cases_router
from backend.api.reports import router as reports_router
from backend.api.documents import router as documents_router

api_router = APIRouter(prefix="/api")
api_router.include_router(retrieval_router)
api_router.include_router(summarization_router)
api_router.include_router(auth_router)
api_router.include_router(cases_router)
api_router.include_router(reports_router)
api_router.include_router(documents_router)

# Future routers: coordinator, evidence, analysis, and verification.
