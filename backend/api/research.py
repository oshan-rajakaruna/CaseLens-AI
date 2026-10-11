"""Authenticated development research initiation and safe task polling."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.dependencies import get_research_execution_service
from backend.api.persistence_errors import persistence_http_error
from backend.auth.session import require_authenticated_principal, require_csrf_principal
from backend.schemas.case_report_api import ResearchStartRequest, ResearchTaskResponse
from backend.services.private_documents import (
    EvidenceTextUnavailableError,
    PrivateStorageError,
)
from backend.services.research_execution import ResearchExecutionService
from database import TrustedPrincipal

router = APIRouter(tags=["research"])


def _response(record, report_id: str | None = None) -> ResearchTaskResponse:
    return ResearchTaskResponse(
        id=record.id,
        case_id=record.case_id,
        status=record.status,
        stages=record.stages,
        error_category=record.error_category,
        report_id=report_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        completed_at=record.completed_at,
    )


@router.post(
    "/cases/{case_id}/research",
    response_model=ResearchTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_research(
    case_id: str,
    request: ResearchStartRequest,
    principal: Annotated[TrustedPrincipal, Depends(require_csrf_principal)],
    execution: Annotated[
        ResearchExecutionService, Depends(get_research_execution_service)
    ],
) -> ResearchTaskResponse:
    try:
        task = execution.start(principal, case_id=case_id, request=request)
        return _response(task, execution.report_id_for(principal, task.id))
    except EvidenceTextUnavailableError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Selected evidence has no extractable text",
        ) from None
    except PrivateStorageError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Private evidence is unavailable",
        ) from None
    except Exception as error:
        raise persistence_http_error(error) from None


@router.get("/research-tasks/{task_id}", response_model=ResearchTaskResponse)
def get_research_task(
    task_id: str,
    principal: Annotated[
        TrustedPrincipal, Depends(require_authenticated_principal)
    ],
    execution: Annotated[
        ResearchExecutionService, Depends(get_research_execution_service)
    ],
) -> ResearchTaskResponse:
    try:
        task = execution.get_task(principal, task_id)
        return _response(task, execution.report_id_for(principal, task.id))
    except Exception as error:
        raise persistence_http_error(error) from None
