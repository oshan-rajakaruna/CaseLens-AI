"""Private local evidence storage and trusted Analysis-input loading."""

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
from typing import BinaryIO
from uuid import uuid4

from backend.schemas.common import Document
from backend.schemas.research_workflow import EvidenceDocumentInput
from database import PersistenceService, TrustedPrincipal
from nlp.extraction import DocumentExtractionResult, extract_document

_STORAGE_KEY = re.compile(r"^objects/[0-9a-f]{2}/[0-9a-f]{32}\.(pdf|txt)$")
_CHUNK_SIZE = 1024 * 1024
_MEDIA_EXTENSIONS = {
    "application/pdf": ".pdf",
    "text/plain": ".txt",
}


class PrivateDocumentError(RuntimeError):
    """Safe base error for private-document operations."""


class InvalidUploadError(PrivateDocumentError):
    pass


class UploadTooLargeError(PrivateDocumentError):
    pass


class PrivateStorageError(PrivateDocumentError):
    pass


class EvidenceTextUnavailableError(PrivateDocumentError):
    pass


@dataclass(frozen=True, slots=True)
class StoredPrivateDocument:
    storage_key: str
    size_bytes: int
    sha256_hex: str


def validate_upload_identity(filename: str | None, media_type: str | None) -> tuple[str, str]:
    if not filename or len(filename) > 255:
        raise InvalidUploadError("Upload filename is invalid")
    if "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise InvalidUploadError("Upload filename is invalid")
    if any(ord(character) < 32 for character in filename):
        raise InvalidUploadError("Upload filename is invalid")
    extension = Path(filename).suffix.casefold()
    expected = _MEDIA_EXTENSIONS.get(media_type or "")
    if expected is None or extension != expected:
        raise InvalidUploadError("Unsupported document type")
    return filename, expected


class PrivateDocumentStorage:
    """Exclusive local-file storage rooted outside public corpus directories."""

    def __init__(self, root: str | Path, max_upload_bytes: int):
        if max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be positive")
        self.root = Path(root).expanduser().resolve()
        self.max_upload_bytes = max_upload_bytes
        self._reject_public_root()

    def store(self, stream: BinaryIO, extension: str) -> StoredPrivateDocument:
        if extension not in {".pdf", ".txt"}:
            raise InvalidUploadError("Unsupported document type")
        identity = uuid4().hex
        storage_key = f"objects/{identity[:2]}/{identity}{extension}"
        target = self._resolve_new(storage_key)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._assert_inside_root(target.parent.resolve())
        except OSError:
            raise PrivateStorageError("Private document storage failed") from None
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor: int | None = None
        size = 0
        digest = sha256()
        try:
            descriptor = os.open(target, flags, 0o600)
            with os.fdopen(descriptor, "wb") as destination:
                descriptor = None
                while True:
                    chunk = stream.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise UploadTooLargeError("Upload exceeds the configured size limit")
                    destination.write(chunk)
                    digest.update(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            if size == 0:
                raise InvalidUploadError("Uploaded document is empty")
            return StoredPrivateDocument(storage_key, size, digest.hexdigest())
        except (InvalidUploadError, UploadTooLargeError):
            self._unlink_if_present(target)
            raise
        except Exception:
            self._unlink_if_present(target)
            raise PrivateStorageError("Private document storage failed") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def resolve(self, storage_key: str) -> Path:
        if not _STORAGE_KEY.fullmatch(storage_key):
            raise PrivateStorageError("Private storage reference is invalid")
        try:
            candidate = (self.root / Path(storage_key)).resolve(strict=True)
        except (OSError, RuntimeError):
            raise PrivateStorageError("Private document is unavailable") from None
        self._assert_inside_root(candidate)
        if not candidate.is_file() or candidate.is_symlink():
            raise PrivateStorageError("Private document is unavailable")
        return candidate

    def delete(self, storage_key: str) -> None:
        if not _STORAGE_KEY.fullmatch(storage_key):
            raise PrivateStorageError("Private storage reference is invalid")
        try:
            candidate = (self.root / Path(storage_key)).resolve(strict=False)
        except (OSError, RuntimeError):
            raise PrivateStorageError("Private storage reference is invalid") from None
        self._assert_inside_root(candidate)
        self._unlink_if_present(candidate)

    def _resolve_new(self, storage_key: str) -> Path:
        try:
            candidate = (self.root / Path(storage_key)).resolve(strict=False)
        except (OSError, RuntimeError):
            raise PrivateStorageError("Private storage reference is invalid") from None
        self._assert_inside_root(candidate)
        return candidate

    def _assert_inside_root(self, path: Path) -> None:
        if path == self.root or self.root not in path.parents:
            raise PrivateStorageError("Private storage boundary rejected the path")

    def _reject_public_root(self) -> None:
        normalized = self.root.as_posix().casefold()
        forbidden = ("/data/legal/", "/data/raw/", "/frontend/public/")
        probe = f"/{normalized.strip('/')}/"
        if any(value in probe for value in forbidden):
            raise ValueError("private storage must not use a public corpus directory")

    @staticmethod
    def _unlink_if_present(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


class PrivateEvidenceService:
    """Resolve persisted private evidence into existing workflow input contracts."""

    def __init__(self, persistence: PersistenceService, storage: PrivateDocumentStorage):
        self._persistence = persistence
        self._storage = storage

    def extract(
        self,
        principal: TrustedPrincipal,
        *,
        case_id: str,
        document_id: str,
    ) -> DocumentExtractionResult:
        record = self._persistence.get_document(principal, case_id, document_id)
        path = self._storage.resolve(record.storage_key)
        return extract_document(
            path,
            record.id,
            max_file_size_bytes=self._storage.max_upload_bytes,
        )

    def load_for_analysis(
        self,
        principal: TrustedPrincipal,
        *,
        case_id: str,
        document_id: str,
    ) -> EvidenceDocumentInput:
        record = self._persistence.get_document(principal, case_id, document_id)
        extraction = self.extract(
            principal, case_id=case_id, document_id=document_id
        )
        if extraction.status != "extracted" or not extraction.text:
            raise EvidenceTextUnavailableError(
                "Document has no safely extractable text"
            )
        metadata = dict(extraction.metadata)
        metadata.update({"media_type": record.media_type})
        return EvidenceDocumentInput(
            document=Document(
                id=record.id,
                case_id=record.case_id,
                name=record.safe_filename,
                media_type=record.media_type,
            ),
            text=extraction.text,
            document_metadata=metadata,
            segments=extraction.segments,
        )
