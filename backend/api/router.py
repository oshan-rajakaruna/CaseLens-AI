"""Root API router reserved for future feature-specific routers."""

from fastapi import APIRouter

api_router = APIRouter(prefix="/api")

# Future routers: cases, reports, coordinator, search, retrieval, evidence,
# analysis, authentication, and verification. They will be included here only
# when their real contracts and behavior are implemented.
