"""Internal durable persistence boundary for CaseLens."""

from database.errors import (
    DuplicateRecordError,
    ImmutableRecordError,
    InvalidRecordError,
    InvalidStateTransitionError,
    MissingRecordError,
    OwnershipError,
    PersistenceError,
)
from database.schemas import (
    AgentLogRecord,
    CaseDocumentRecord,
    CaseRecord,
    LegalReportRecord,
    ResearchTaskRecord,
    TrustedPrincipal,
)
from database.service import PersistenceService
from database.session import build_engine, build_session_factory

__all__ = [
    "AgentLogRecord", "CaseDocumentRecord", "CaseRecord",
    "DuplicateRecordError", "ImmutableRecordError", "InvalidRecordError",
    "InvalidStateTransitionError", "LegalReportRecord", "MissingRecordError",
    "OwnershipError", "PersistenceError", "PersistenceService",
    "ResearchTaskRecord", "TrustedPrincipal", "build_engine",
    "build_session_factory",
]
