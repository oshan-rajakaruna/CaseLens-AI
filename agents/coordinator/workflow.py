"""Workflow contract for internal Coordinator implementations."""

from typing import Protocol

from backend.schemas import AgentResult, AgentTask


class CoordinatorWorkflow(Protocol):
    """Async boundary implemented by the ResearchWorkflow adapter."""

    async def run(self, task: AgentTask) -> AgentResult:
        """Run a validated workflow for a task."""
        ...
