"""Owner-scoped case management endpoints."""
from typing import Annotated
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.auth import AuthenticatedPrincipal, get_current_principal
from backend.db.models import CaseRecord
from backend.db.session import get_session

router = APIRouter(prefix="/cases", tags=["cases"])
class CaseCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    description: str | None = Field(default=None, max_length=10000)
    jurisdiction: str | None = Field(default=None, max_length=200)
class CaseUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=500)
    description: str | None = Field(default=None, max_length=10000)
    status: str | None = Field(default=None, pattern="^(open|archived|closed)$")
def _owned(session: Session, principal: AuthenticatedPrincipal, case_id: str) -> CaseRecord:
    item = session.scalar(select(CaseRecord).where(CaseRecord.id == case_id, CaseRecord.owner_id == principal.subject))
    if item is None: raise HTTPException(status_code=404, detail="Case not found")
    return item
def _out(item: CaseRecord) -> dict:
    return {"id": item.id, "title": item.title, "description": item.description, "jurisdiction": item.jurisdiction, "status": item.status, "created_at": item.created_at, "updated_at": item.updated_at}
@router.post("", status_code=201)
def create_case(body: CaseCreate, principal: Annotated[AuthenticatedPrincipal, Depends(get_current_principal)], session: Annotated[Session, Depends(get_session)]):
    item = CaseRecord(owner_id=principal.subject, **body.model_dump()); session.add(item); session.commit(); session.refresh(item); return _out(item)
@router.get("")
def list_cases(principal: Annotated[AuthenticatedPrincipal, Depends(get_current_principal)], session: Annotated[Session, Depends(get_session)]):
    return [_out(item) for item in session.scalars(select(CaseRecord).where(CaseRecord.owner_id == principal.subject).order_by(CaseRecord.created_at.desc()))]
@router.get("/{case_id}")
def get_case(case_id: str, principal: Annotated[AuthenticatedPrincipal, Depends(get_current_principal)], session: Annotated[Session, Depends(get_session)]): return _out(_owned(session, principal, case_id))
@router.patch("/{case_id}")
def update_case(case_id: str, body: CaseUpdate, principal: Annotated[AuthenticatedPrincipal, Depends(get_current_principal)], session: Annotated[Session, Depends(get_session)]):
    item=_owned(session, principal, case_id)
    for key, value in body.model_dump(exclude_unset=True).items(): setattr(item, key, value)
    session.commit(); session.refresh(item); return _out(item)
@router.delete("/{case_id}", status_code=204)
def delete_case(case_id: str, principal: Annotated[AuthenticatedPrincipal, Depends(get_current_principal)], session: Annotated[Session, Depends(get_session)]):
    session.delete(_owned(session, principal, case_id)); session.commit()
