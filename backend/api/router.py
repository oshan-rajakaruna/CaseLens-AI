"""Root API router composing implemented feature routers."""

from fastapi import APIRouter

from backend.api.retrieval import router as retrieval_router

api_router = APIRouter(prefix="/api")
api_router.include_router(retrieval_router)

# Future routers: cases, reports, coordinator, evidence, analysis,
# authentication, and verification.
