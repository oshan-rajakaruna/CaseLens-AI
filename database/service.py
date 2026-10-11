"""Transaction-owning internal persistence service."""

from collections.abc import Callable
from typing import Any, TypeVar

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from backend.schemas.legal_report import LegalResearchReport
from database.errors import DuplicateRecordError, PersistenceError
from database.repository import CaseLensRepository
from database.schemas import ReportState, TaskStatus, TrustedPrincipal

T = TypeVar("T")


class PersistenceService:
    """Open one transaction per operation and return detached validated records."""

    def __init__(self, session_factory: Callable[[], Session]):
        self._session_factory = session_factory

    def create_case(self, principal: TrustedPrincipal, **values: Any):
        return self._write(lambda repo: repo.create_case(principal, **values))

    def get_case(self, principal: TrustedPrincipal, case_id: str):
        return self._read(lambda repo: repo.get_case(principal, case_id))

    def list_cases(self, principal: TrustedPrincipal, *, offset: int = 0, limit: int = 25):
        return self._read(
            lambda repo: repo.list_cases(principal, offset=offset, limit=limit)
        )

    def add_document(self, principal: TrustedPrincipal, **values: Any):
        return self._write(lambda repo: repo.add_document(principal, **values))

    def create_task(self, principal: TrustedPrincipal, **values: Any):
        return self._write(lambda repo: repo.create_task(principal, **values))

    def get_task(self, principal: TrustedPrincipal, task_id: str):
        return self._read(lambda repo: repo.get_task(principal, task_id))

    def update_task(self, principal: TrustedPrincipal, task_id: str, *, status: TaskStatus, stages=None, error_category=None):
        return self._write(lambda repo: repo.update_task(principal, task_id, status=status, stages=stages, error_category=error_category))

    def save_report(self, principal: TrustedPrincipal, *, report: LegalResearchReport, state: ReportState, **values: Any):
        return self._write(lambda repo: repo.save_report(principal, report=report, state=state, **values))

    def get_report(self, principal: TrustedPrincipal, report_id: str):
        return self._read(lambda repo: repo.get_report(principal, report_id))

    def find_reports_by_case(self, principal: TrustedPrincipal, case_id: str):
        return self._read(lambda repo: repo.find_reports_by_case(principal, case_id))

    def find_report_by_task(self, principal: TrustedPrincipal, task_id: str):
        return self._read(lambda repo: repo.find_report_by_task(principal, task_id))

    def record_agent_event(self, principal: TrustedPrincipal, **values: Any):
        return self._write(lambda repo: repo.record_agent_event(principal, **values))

    def list_agent_events(self, principal: TrustedPrincipal, task_id: str):
        return self._read(lambda repo: repo.list_agent_events(principal, task_id))

    def _read(self, operation: Callable[[CaseLensRepository], T]) -> T:
        try:
            with self._session_factory() as session:
                return operation(CaseLensRepository(session))
        except PersistenceError:
            raise
        except SQLAlchemyError:
            raise PersistenceError("persistence operation failed") from None

    def _write(self, operation: Callable[[CaseLensRepository], T]) -> T:
        try:
            with self._session_factory.begin() as session:
                return operation(CaseLensRepository(session))
        except PersistenceError:
            raise
        except IntegrityError:
            raise DuplicateRecordError(
                "persistence constraint rejected the record"
            ) from None
        except SQLAlchemyError:
            raise PersistenceError("persistence operation failed") from None
