"""Private, owner-checked development document storage and extraction."""
from pathlib import Path
from uuid import uuid4
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session
from backend.config import get_settings
from backend.db.models import DocumentRecord
from nlp.extraction import extract_document

_TYPES = {".txt": "text/plain", ".pdf": "application/pdf", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

def _root() -> Path:
    root = Path(get_settings().private_upload_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root

def _safe_name(name: str | None) -> tuple[str, str]:
    if not name or Path(name).name != name or len(name) > 255:
        raise HTTPException(400, "Invalid filename")
    extension = Path(name).suffix.lower()
    if extension not in _TYPES: raise HTTPException(415, "Only PDF, DOCX, and TXT files are supported")
    return name, extension

async def save_upload(session: Session, *, case_id: str, owner_id: str, upload: UploadFile) -> DocumentRecord:
    original_name, extension = _safe_name(upload.filename)
    document_id, storage_key = str(uuid4()), f"{uuid4()}{extension}"
    path = (_root() / storage_key).resolve()
    if path.parent != _root(): raise HTTPException(400, "Invalid upload path")
    total = 0
    try:
        with path.open("xb") as destination:
            while chunk := await upload.read(64 * 1024):
                total += len(chunk)
                if total > get_settings().max_upload_bytes:
                    raise HTTPException(413, "Upload exceeds configured size limit")
                destination.write(chunk)
        if total == 0: raise HTTPException(400, "Empty uploads are not supported")
        extraction = extract_document(path, document_id, max_file_size_bytes=get_settings().max_upload_bytes)
        if extraction.status == "failed":
            raise HTTPException(422, extraction.errors[0].message if extraction.errors else "Document extraction failed")
        record = DocumentRecord(
            id=document_id, case_id=case_id, owner_id=owner_id, original_name=original_name,
            media_type=_TYPES[extension], storage_key=storage_key, size_bytes=total,
            extraction_status=extraction.status, extracted_text=extraction.text or None,
        )
        session.add(record); session.commit(); session.refresh(record)
        return record
    except Exception:
        if path.exists(): path.unlink()
        session.rollback()
        raise

def document_output(item: DocumentRecord) -> dict:
    return {"id": item.id, "case_id": item.case_id, "filename": item.original_name, "media_type": item.media_type, "size_bytes": item.size_bytes, "extraction_status": item.extraction_status, "created_at": item.created_at}

def delete_private_file(item: DocumentRecord) -> None:
    path = (_root() / item.storage_key).resolve()
    if path.parent == _root() and path.exists(): path.unlink()
