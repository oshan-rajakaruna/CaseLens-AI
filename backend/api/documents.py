"""Authenticated case-document API; uploaded bytes are never public files."""
from typing import Annotated
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session
from backend.api.cases import _owned
from backend.auth import AuthenticatedPrincipal, get_current_principal
from backend.db.models import DocumentRecord
from backend.db.session import get_session
from backend.services.document_service import delete_private_file, document_output, save_upload

router = APIRouter(prefix="/cases/{case_id}/documents", tags=["documents"])
def _document(session: Session, principal: AuthenticatedPrincipal, case_id: str, document_id: str) -> DocumentRecord:
    _owned(session, principal, case_id)
    item=session.scalar(select(DocumentRecord).where(DocumentRecord.id==document_id, DocumentRecord.case_id==case_id, DocumentRecord.owner_id==principal.subject))
    if item is None: raise HTTPException(404, "Document not found")
    return item
@router.post("", status_code=201)
async def upload_document(case_id: str, file: Annotated[UploadFile, File(...)], principal: Annotated[AuthenticatedPrincipal, Depends(get_current_principal)], session: Annotated[Session, Depends(get_session)]):
    _owned(session, principal, case_id)
    return document_output(await save_upload(session, case_id=case_id, owner_id=principal.subject, upload=file))
@router.get("")
def list_documents(case_id: str, principal: Annotated[AuthenticatedPrincipal, Depends(get_current_principal)], session: Annotated[Session, Depends(get_session)]):
    _owned(session, principal, case_id)
    return [document_output(item) for item in session.scalars(select(DocumentRecord).where(DocumentRecord.case_id==case_id, DocumentRecord.owner_id==principal.subject))]
@router.get("/{document_id}")
def get_document(case_id: str, document_id: str, principal: Annotated[AuthenticatedPrincipal, Depends(get_current_principal)], session: Annotated[Session, Depends(get_session)]): return document_output(_document(session, principal, case_id, document_id))
@router.delete("/{document_id}", status_code=204)
def delete_document(case_id: str, document_id: str, principal: Annotated[AuthenticatedPrincipal, Depends(get_current_principal)], session: Annotated[Session, Depends(get_session)]):
    item=_document(session, principal, case_id, document_id); delete_private_file(item); session.delete(item); session.commit()
