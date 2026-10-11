"""Root API router composing implemented feature routers."""

from fastapi import APIRouter

from backend.api.retrieval import router as retrieval_router
from backend.api.cases import router as cases_router
from backend.api.documents import router as documents_router
from backend.api.coordinator import router as coordinator_router

api_router = APIRouter(prefix="/api")
api_router.include_router(retrieval_router)
api_router.include_router(cases_router)
api_router.include_router(documents_router)
api_router.include_router(coordinator_router)

# Future routers: cases, reports, coordinator, evidence, analysis,
# authentication, and verification.
