"""Development-only background execution of authorized research tasks."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from hashlib import sha256
import json
from threading import Lock
from typing import Protocol
from uuid import uuid4

from agents.coordinator import Coordinator
from backend.schemas import AgentTask, Case
from backend.schemas.case_report_api import ResearchStartRequest
from backend.schemas.legal_report import LegalResearchReport
from backend.schemas.research_workflow import ResearchWorkflowResult
from backend.services.private_documents import PrivateEvidenceService
from database import DuplicateRecordError, PersistenceError, PersistenceService, TrustedPrincipal
from database.schemas import ResearchTaskRecord

_INITIAL_STAGES = {
    "analysis": "pending",
    "retrieval": "pending",
    "summarization_verification": "pending",
    "report": "pending",
    "comparisons": "pending",
    "markdown": "pending",
}


class PrivateResultStore(Protocol):
    def get_private_result(self, task_id: str) -> ResearchWorkflowResult | None: ...


class ResearchExecutionService:
    """Persist, execute, and reconcile one bounded process-local task queue."""

    def __init__(
        self,
        *,
        persistence: PersistenceService,
        evidence: PrivateEvidenceService,
        coordinator: Coordinator,
        private_results: PrivateResultStore,
        max_workers: int = 1,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._persistence = persistence
        self._evidence = evidence
        self._coordinator = coordinator
        self._private_results = private_results
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="caselens-research",
        )
        self._active: dict[str, Future[None]] = {}
        self._active_lock = Lock()
        self._submission_lock = Lock()

    def start(
        self,
        principal: TrustedPrincipal,
        *,
        case_id: str,
        request: ResearchStartRequest,
    ) -> ResearchTaskRecord:
        with self._submission_lock:
            return self._start_locked(
                principal,
                case_id=case_id,
                request=request,
            )

    def _start_locked(
        self,
        principal: TrustedPrincipal,
        *,
        case_id: str,
        request: ResearchStartRequest,
    ) -> ResearchTaskRecord:
        case_record = self._persistence.get_case(principal, case_id)
        for document_id in request.document_ids:
            self._persistence.get_document(principal, case_id, document_id)

        fingerprint = _request_fingerprint(case_id, request)
        existing = self._persistence.find_task_by_fingerprint(
            principal, case_id, fingerprint
        )
        if existing is not None:
            return self._reconcile_if_interrupted(principal, existing)

        evidence_documents = [
            self._evidence.load_for_analysis(
                principal,
                case_id=case_id,
                document_id=document_id,
            )
            for document_id in request.document_ids
        ]
        task_id = f"task-{uuid4()}"
        try:
            task = self._persistence.create_task(
                principal,
                task_id=task_id,
                case_id=case_id,
                request_fingerprint=fingerprint,
                stages=_INITIAL_STAGES,
            )
        except DuplicateRecordError:
            existing = self._persistence.find_task_by_fingerprint(
                principal, case_id, fingerprint
            )
            if existing is None:
                raise
            return self._reconcile_if_interrupted(principal, existing)

        agent_task = AgentTask(
            id=task.id,
            agent="coordinator",
            instruction="Run authorized internal legal research.",
            context={
                "case": Case(
                    id=case_record.id,
                    title=case_record.title,
                    description=case_record.description,
                ).model_dump(mode="json"),
                "query": request.query,
                "legal_issue": request.legal_issue,
                "top_k": request.top_k,
                "evidence_documents": [
                    item.model_dump(mode="json") for item in evidence_documents
                ],
                "retrieval_mode": "bm25",
                "comparisons": [],
                "generate_comparison_drafts": False,
                "render_markdown": False,
            },
        )
        try:
            self._submit(principal, case_id, agent_task)
        except Exception:
            return self._fail_task(
                principal,
                task.id,
                error_category="execution_unavailable",
                stages=_INITIAL_STAGES,
            )
        return task

    def get_task(
        self,
        principal: TrustedPrincipal,
        task_id: str,
    ) -> ResearchTaskRecord:
        task = self._persistence.get_task(principal, task_id)
        return self._reconcile_if_interrupted(principal, task)

    def report_id_for(
        self,
        principal: TrustedPrincipal,
        task_id: str,
    ) -> str | None:
        report = self._persistence.find_report_by_task(principal, task_id)
        return None if report is None else report.id

    def _submit(
        self,
        principal: TrustedPrincipal,
        case_id: str,
        task: AgentTask,
    ) -> None:
        with self._active_lock:
            if task.id in self._active:
                return
            future = self._executor.submit(self._execute, principal, case_id, task)
            self._active[task.id] = future
        future.add_done_callback(
            lambda completed, task_id=task.id: self._discard(task_id, completed)
        )

    def _execute(
        self,
        principal: TrustedPrincipal,
        case_id: str,
        task: AgentTask,
    ) -> None:
        try:
            self._persistence.update_task(
                principal,
                task.id,
                status="running",
                stages={**_INITIAL_STAGES, "analysis": "running"},
            )
            agent_result = asyncio.run(self._coordinator.handle(task))
            private_result = self._private_results.get_private_result(task.id)
            if private_result is None or private_result.report is None:
                category = _agent_error_category(agent_result.output)
                self._fail_task(
                    principal,
                    task.id,
                    error_category=category,
                    stages=_safe_stages(agent_result.output.get("stages")),
                )
                return
            if private_result.status not in {"completed", "partial"}:
                self._fail_task(
                    principal,
                    task.id,
                    error_category="invalid_workflow_result",
                    stages=_safe_stages(
                        private_result.stages.model_dump(mode="json")
                    ),
                )
                return

            report = LegalResearchReport.model_validate(private_result.report)
            terminal_status = (
                "partial" if private_result.status == "partial" else "completed"
            )
            error_category = (
                _safe_error_category(private_result.errors[0].code)
                if private_result.errors and terminal_status == "partial"
                else None
            )
            try:
                self._persistence.finalize_task_with_report(
                    principal,
                    report_id=f"report-{uuid4()}",
                    case_id=case_id,
                    task_id=task.id,
                    report=report,
                    state=terminal_status,
                    task_status=terminal_status,
                    stages=_safe_stages(
                        private_result.stages.model_dump(mode="json")
                    ),
                    error_category=error_category,
                )
            except PersistenceError:
                self._fail_task(
                    principal,
                    task.id,
                    error_category="report_persistence_failed",
                    stages=_safe_stages(
                        private_result.stages.model_dump(mode="json")
                    ),
                )
        except Exception:
            self._fail_task(
                principal,
                task.id,
                error_category="research_execution_failed",
                stages=_INITIAL_STAGES,
            )

    def _reconcile_if_interrupted(
        self,
        principal: TrustedPrincipal,
        task: ResearchTaskRecord,
    ) -> ResearchTaskRecord:
        if task.status not in {"queued", "running"} or self._is_active(task.id):
            return task
        stages = {
            name: ("failed" if value == "running" else value)
            for name, value in task.stages.items()
        }
        return self._fail_task(
            principal,
            task.id,
            error_category="restart_interrupted",
            stages=stages,
        )

    def _fail_task(
        self,
        principal: TrustedPrincipal,
        task_id: str,
        *,
        error_category: str,
        stages: dict[str, str],
    ) -> ResearchTaskRecord:
        try:
            current = self._persistence.get_task(principal, task_id)
            if current.status in {"completed", "partial", "failed"}:
                return current
            return self._persistence.update_task(
                principal,
                task_id,
                status="failed",
                stages=_safe_stages(stages),
                error_category=error_category,
            )
        except PersistenceError:
            raise

    def _is_active(self, task_id: str) -> bool:
        with self._active_lock:
            future = self._active.get(task_id)
            return future is not None and not future.done()

    def _discard(self, task_id: str, completed: Future[None]) -> None:
        del completed
        with self._active_lock:
            self._active.pop(task_id, None)


def _request_fingerprint(case_id: str, request: ResearchStartRequest) -> str:
    payload = {"case_id": case_id, **request.model_dump(mode="json")}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _safe_stages(values: object) -> dict[str, str]:
    if not isinstance(values, dict):
        return dict(_INITIAL_STAGES)
    allowed = {"pending", "running", "completed", "partial", "failed", "skipped", "unavailable"}
    stages: dict[str, str] = {}
    for name in _INITIAL_STAGES:
        value = values.get(name, "pending")
        stages[name] = "pending" if value == "not_started" else value
        if stages[name] not in allowed:
            stages[name] = "unavailable"
    return stages


def _agent_error_category(output: dict) -> str:
    errors = output.get("errors")
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        code = errors[0].get("code")
        if isinstance(code, str):
            return _safe_error_category(code)
    return "report_unavailable"


def _safe_error_category(value: str) -> str:
    if value and len(value) <= 128 and value.replace("_", "").isalnum():
        return value
    return "research_execution_failed"
