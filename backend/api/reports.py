"""Authenticated owner-scoped structured legal-report endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_persistence_service
from backend.api.persistence_errors import persistence_http_error
from backend.auth.session import require_authenticated_principal
from backend.schemas.case_report_api import LegalReportListResponse, LegalReportResponse
from database import PersistenceService, TrustedPrincipal

router = APIRouter(tags=["reports"])


def _response(record) -> LegalReportResponse:
    return LegalReportResponse(
        id=record.id,
        case_id=record.case_id,
        task_id=record.task_id,
        state=record.state,
        report=record.report,
        created_at=record.created_at,
    )


@router.get("/cases/{case_id}/reports", response_model=LegalReportListResponse)
def list_case_reports(
    case_id: str,
    principal: Annotated[TrustedPrincipal, Depends(require_authenticated_principal)],
    persistence: Annotated[PersistenceService, Depends(get_persistence_service)],
) -> LegalReportListResponse:
    try:
        return LegalReportListResponse(
            items=[_response(item) for item in persistence.find_reports_by_case(principal, case_id)]
        )
    except Exception as error:
        raise persistence_http_error(error) from None


@router.get("/reports/{report_id}", response_model=LegalReportResponse)
def get_report(
    report_id: str,
    principal: Annotated[TrustedPrincipal, Depends(require_authenticated_principal)],
    persistence: Annotated[PersistenceService, Depends(get_persistence_service)],
) -> LegalReportResponse:
    try:
        return _response(persistence.get_report(principal, report_id))
    except Exception as error:
        raise persistence_http_error(error) from None
