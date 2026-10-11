"""Internal Coordinator adapter for the synchronous ResearchWorkflow."""

from __future__ import annotations

import asyncio
from hashlib import sha256
from typing import Protocol

from pydantic import ValidationError

from backend.schemas import AgentResult, AgentTask
from backend.schemas.research_workflow import (
    ResearchWorkflowRequest,
    ResearchWorkflowResult,
)


class InternalResearchWorkflow(Protocol):
    """Synchronous private workflow surface used by the Coordinator."""

    def run(self, request: ResearchWorkflowRequest) -> ResearchWorkflowResult: ...


class ResearchWorkflowCoordinatorAdapter:
    """Validate, execute once, and safely summarize a private workflow result."""

    def __init__(self, workflow: InternalResearchWorkflow) -> None:
        self.workflow = workflow
        self._lock = asyncio.Lock()
        self._fingerprints: dict[str, str] = {}
        self._agent_results: dict[str, AgentResult] = {}
        self._private_results: dict[str, ResearchWorkflowResult] = {}

    async def run(self, task: AgentTask) -> AgentResult:
        """Adapt one trusted task and run blocking work outside the event loop."""

        request_or_error = self._adapt(task)
        if isinstance(request_or_error, AgentResult):
            return request_or_error
        request = request_or_error
        fingerprint = sha256(
            request.model_dump_json().encode("utf-8")
        ).hexdigest()

        async with self._lock:
            previous = self._fingerprints.get(task.id)
            if previous is not None:
                if previous == fingerprint:
                    return self._agent_results[task.id].model_copy(deep=True)
                return _failed_result(
                    task.id,
                    code="task_id_conflict",
                    message="Task ID was already used for a different workflow request.",
                )
            try:
                workflow_result = await asyncio.to_thread(self.workflow.run, request)
            except Exception:
                agent_result = _failed_result(
                    task.id,
                    code="workflow_execution_failed",
                    message="The internal research workflow could not complete.",
                )
                self._fingerprints[task.id] = fingerprint
                self._agent_results[task.id] = agent_result.model_copy(deep=True)
                return agent_result

            agent_result = _to_agent_result(task.id, workflow_result)
            self._fingerprints[task.id] = fingerprint
            self._agent_results[task.id] = agent_result.model_copy(deep=True)
            self._private_results[task.id] = workflow_result
            return agent_result

    def get_private_result(self, task_id: str) -> ResearchWorkflowResult | None:
        """Return the full result only to a trusted in-process caller."""

        return self._private_results.get(task_id)

    @staticmethod
    def _adapt(task: AgentTask) -> ResearchWorkflowRequest | AgentResult:
        if task.agent.casefold() not in {"coordinator", "coordinator_agent"}:
            return _failed_result(
                task.id,
                code="invalid_agent",
                message="Task is not addressed to the Coordinator.",
            )
        if "task_id" in task.context:
            return _failed_result(
                task.id,
                code="invalid_task_context",
                message="Workflow task ID must come from the AgentTask envelope.",
            )
        try:
            return ResearchWorkflowRequest.model_validate(
                {"task_id": task.id, **task.context}
            )
        except ValidationError:
            return _failed_result(
                task.id,
                code="invalid_task_context",
                message="Coordinator task context is invalid.",
            )


def _to_agent_result(
    task_id: str,
    result: ResearchWorkflowResult,
) -> AgentResult:
    output: dict = {
        "workflow_status": result.status,
        "stages": result.stages.model_dump(mode="json"),
        "errors": [error.model_dump(mode="json") for error in result.errors],
        "analysis": {
            "result_count": len(result.analyses),
            "statuses": [analysis.status for analysis in result.analyses],
        },
        "report_available": result.report is not None,
        "markdown_available": result.markdown is not None,
    }
    if result.retrieval_payload is not None:
        diagnostics = result.retrieval_payload.diagnostics
        output["retrieval"] = {
            "result_count": diagnostics.retrieval_result_count,
            "context_count": diagnostics.rag_passage_count,
            "source_document_count": diagnostics.source_document_count,
            "no_context": diagnostics.no_context,
            "context_truncated": diagnostics.context_truncated,
        }
    if result.summary_verification is not None:
        verification = result.summary_verification.verification
        output["verification"] = {
            "overall_status": verification.overall_status,
            "verified": verification.verified,
            "claim_count": len(verification.claim_results),
            "warning_count": len(verification.warnings),
        }
    if result.report is not None:
        output["authorities"] = [
            {
                "context_id": authority.context_id,
                "document_id": authority.document_id,
                "pages": list(authority.pages),
                "chunk_ids": list(authority.chunk_ids),
                "official_citation": authority.official_citation,
            }
            for authority in result.report.authorities
        ]
        output["comparisons"] = [
            {
                "id": comparison.id,
                "context_id": comparison.authority.context_id,
                "review_status": comparison.review_status,
                "draft_verification_status": (
                    comparison.ai_draft.verification_status
                    if comparison.ai_draft is not None
                    else None
                ),
            }
            for comparison in result.report.comparisons
        ]
        output["human_review_required"] = True
    return AgentResult(
        task_id=task_id,
        agent="coordinator",
        status=result.status,
        output=output,
    )


def _failed_result(task_id: str, *, code: str, message: str) -> AgentResult:
    return AgentResult(
        task_id=task_id,
        agent="coordinator",
        status="failed",
        output={
            "workflow_status": "failed",
            "errors": [{"stage": "coordinator", "code": code, "message": message}],
            "report_available": False,
            "markdown_available": False,
        },
    )
