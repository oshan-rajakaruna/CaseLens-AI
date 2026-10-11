"""Single authenticated Coordinator execution endpoint for academic agent tests."""
from typing import Annotated, Any
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from agents.coordinator.schemas import CoordinatorRequest
from backend.api.cases import _owned
from backend.auth import AuthenticatedPrincipal, get_current_principal
from backend.db.models import DocumentRecord
from backend.db.session import get_session
from backend.services.coordinator_service import execute_authorized

router = APIRouter(prefix="/coordinator", tags=["Coordinator — Agent Testing"])

class CoordinatorExecutionEnvelope(BaseModel):
    task: str
    case_id: str
    executed: bool
    overall_status: str
    stages: list[Any] = Field(default_factory=list)
    results: dict[str, Any]
    source_references: list[Any] = Field(default_factory=list)
    warnings: list[Any] = Field(default_factory=list)
    errors: list[Any] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

@router.post("/execute", response_model=CoordinatorExecutionEnvelope)
async def execute(
    request: Annotated[CoordinatorRequest, Body(examples=[
        {"summary": "Analysis", "value": {"case_id": "YOUR_CASE_ID", "case_title": "Fictional payment dispute", "user_query": "Analyze supplied evidence", "requested_task": "analyze_evidence"}},
        {"summary": "Retrieval", "value": {"case_id": "YOUR_CASE_ID", "user_query": "contractual notice", "requested_task": "retrieve_precedents", "retrieval_mode": "bm25", "top_k": 3}},
        {"summary": "Evidence summary", "value": {"case_id": "YOUR_CASE_ID", "user_query": "Summarize evidence", "requested_task": "summarize_evidence"}},
        {"summary": "Precedent summary", "value": {"case_id": "YOUR_CASE_ID", "user_query": "notice appeal", "requested_task": "summarize_precedents", "retrieval_mode": "bm25"}},
        {"summary": "Verification", "value": {"case_id": "YOUR_CASE_ID", "user_query": "Check claim", "requested_task": "verify_claims", "verification_claims": []}},
        {"summary": "Full research", "value": {"case_id": "YOUR_CASE_ID", "case_title": "Fictional payment dispute", "user_query": "Was notice timely?", "requested_task": "full_research", "retrieval_mode": "bm25"}},
    ])],
    principal: Annotated[AuthenticatedPrincipal, Depends(get_current_principal)],
    session: Annotated[Session, Depends(get_session)],
) -> CoordinatorExecutionEnvelope:
    _owned(session, principal, request.case_id)
    documents = list(session.scalars(select(DocumentRecord).where(
        DocumentRecord.case_id == request.case_id, DocumentRecord.owner_id == principal.subject
    )))
    try:
        coordinator, authorized_request = execute_authorized(request, documents)
        output = await coordinator.handle(authorized_request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    data = output.model_dump(mode="json")
    return CoordinatorExecutionEnvelope(
        task=authorized_request.requested_task, case_id=authorized_request.case_id,
        executed=data.get("executed", False), overall_status=data.get("status", "unknown"),
        stages=data.get("stages", []), results=data,
        source_references=data.get("source_references", []), warnings=data.get("warnings", []),
        errors=data.get("errors", []), limitations=data.get("verification_limitations", []),
    )
