"""FastAPI routes for structured legal information retrieval."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.schemas.retrieval import RetrievalSearchRequest, RetrievalSearchResponse
from backend.services.retrieval_service import (
    RetrievalExecutionError,
    RetrievalIndexNotReadyError,
    RetrievalProviderError,
    RetrievalRequestError,
    RetrievalService,
    get_retrieval_service,
)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/search", response_model=RetrievalSearchResponse)
def search_retrieval(
    request: RetrievalSearchRequest,
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> RetrievalSearchResponse:
    """Search the configured retrieval indexes using the requested mode."""

    try:
        return service.search(request)
    except RetrievalRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RetrievalIndexNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except RetrievalProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except RetrievalExecutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval search failed",
        ) from exc
