"""Retrieval-side adapter for shared Coordinator task/result envelopes."""

import asyncio

from pydantic import ValidationError

from backend.schemas import AgentResult, AgentTask
from backend.schemas.retrieval import RetrievalSearchRequest, RetrievalTaskContext
from backend.services.retrieval_service import RetrievalService, RetrievalServiceError


class RetrievalTaskHandler:
    """Handle structured retrieval tasks without implementing orchestration."""

    def __init__(self, service: RetrievalService) -> None:
        self.service = service

    async def handle(self, task: AgentTask) -> AgentResult:
        """Validate one retrieval task and return a shared ``AgentResult``."""

        if task.agent.casefold() not in {"retrieval", "retrieval_agent"}:
            return self._error_result(
                task,
                "invalid_agent",
                "Task is not addressed to the Retrieval Agent",
            )

        try:
            context = RetrievalTaskContext.model_validate(task.context)
        except ValidationError:
            return self._error_result(
                task,
                "invalid_retrieval_task",
                "Retrieval task context is invalid",
            )

        request = RetrievalSearchRequest.model_validate(
            context.model_dump(exclude={"legal_issue"})
        )
        try:
            response = await asyncio.to_thread(self.service.search, request)
        except RetrievalServiceError as exc:
            return self._error_result(
                task,
                "retrieval_failed",
                str(exc),
                context=context,
            )
        except Exception:
            return self._error_result(
                task,
                "retrieval_failed",
                "Retrieval task failed",
                context=context,
            )

        return AgentResult(
            task_id=task.id,
            agent="retrieval",
            status="success",
            output={
                "query": context.query,
                "legal_issue": context.legal_issue,
                "mode": context.mode,
                "top_k": context.top_k,
                "filters": (
                    context.filters.model_dump(exclude_none=True)
                    if context.filters is not None
                    else None
                ),
                "result_count": response.result_count,
                "results": [
                    result.model_dump(mode="json") for result in response.results
                ],
            },
        )

    @staticmethod
    def _error_result(
        task: AgentTask,
        code: str,
        message: str,
        *,
        context: RetrievalTaskContext | None = None,
    ) -> AgentResult:
        output = {
            "error": {"code": code, "message": message},
        }
        if context is not None:
            output.update(
                {
                    "query": context.query,
                    "legal_issue": context.legal_issue,
                    "mode": context.mode,
                    "top_k": context.top_k,
                    "filters": (
                        context.filters.model_dump(exclude_none=True)
                        if context.filters is not None
                        else None
                    ),
                }
            )
        return AgentResult(
            task_id=task.id,
            agent="retrieval",
            status="error",
            output=output,
        )
