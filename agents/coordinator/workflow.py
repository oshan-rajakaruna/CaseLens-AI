"""Workflow contract for future coordinator implementations."""

from typing import Protocol

from backend.schemas import AgentResult, AgentTask


class CoordinatorWorkflow(Protocol):
    """Interface that a real coordinator workflow must implement later."""

    async def run(self, task: AgentTask) -> AgentResult:
        """Run a validated workflow for a task."""
        ...
