"""Root API router reserved for future feature-specific routers."""

from fastapi import APIRouter

from backend.api.drafts import draft_router

api_router = APIRouter(prefix="/api")
api_router.include_router(draft_router)
