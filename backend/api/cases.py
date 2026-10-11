"""Authenticated owner-scoped case endpoints."""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, Query

from backend.api.dependencies import get_persistence_service
from backend.api.persistence_errors import persistence_http_error
from backend.auth.session import require_authenticated_principal, require_csrf_principal
from backend.schemas.case_report_api import CaseCreateRequest, CaseListResponse, CaseResponse
from database import PersistenceService, TrustedPrincipal

router = APIRouter(prefix="/cases", tags=["cases"])


@router.post("", response_model=CaseResponse, status_code=201)
def create_case(
    request: CaseCreateRequest,
    principal: Annotated[TrustedPrincipal, Depends(require_csrf_principal)],
    persistence: Annotated[PersistenceService, Depends(get_persistence_service)],
) -> CaseResponse:
    try:
        record = persistence.create_case(
            principal,
            case_id=f"case-{uuid4()}",
            title=request.title,
            description=request.description,
            research_question=request.research_question,
        )
        return CaseResponse.model_validate(record, from_attributes=True)
    except Exception as error:
        raise persistence_http_error(error) from None


@router.get("", response_model=CaseListResponse)
def list_cases(
    principal: Annotated[TrustedPrincipal, Depends(require_authenticated_principal)],
    persistence: Annotated[PersistenceService, Depends(get_persistence_service)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> CaseListResponse:
    try:
        records = persistence.list_cases(principal, offset=offset, limit=limit)
        return CaseListResponse(
            items=[CaseResponse.model_validate(item, from_attributes=True) for item in records],
            offset=offset,
            limit=limit,
        )
    except Exception as error:
        raise persistence_http_error(error) from None


@router.get("/{case_id}", response_model=CaseResponse)
def get_case(
    case_id: str,
    principal: Annotated[TrustedPrincipal, Depends(require_authenticated_principal)],
    persistence: Annotated[PersistenceService, Depends(get_persistence_service)],
) -> CaseResponse:
    try:
        return CaseResponse.model_validate(
            persistence.get_case(principal, case_id), from_attributes=True
        )
    except Exception as error:
        raise persistence_http_error(error) from None
