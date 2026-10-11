"""Root API router composing implemented feature routers."""

from fastapi import APIRouter

from backend.api.retrieval import router as retrieval_router
from backend.api.summarization import router as summarization_router

api_router = APIRouter(prefix="/api")
api_router.include_router(retrieval_router)
api_router.include_router(summarization_router)

# Future routers: cases, reports, coordinator, evidence, analysis,
# authentication, and verification.
