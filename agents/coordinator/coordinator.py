"""Internal Coordinator entry point for CaseLens research orchestration."""

from agents.coordinator.workflow import CoordinatorWorkflow
from backend.schemas import AgentResult, AgentTask


class Coordinator:
    """Delegate validated internal tasks to the configured workflow adapter."""

    def __init__(self, workflow: CoordinatorWorkflow) -> None:
        self.workflow = workflow

    async def handle(self, task: AgentTask) -> AgentResult:
        """Delegate exactly once; the adapter owns validation and execution safety."""

        return await self.workflow.run(task)
