"""Coordinator agent interface for future CaseLens orchestration.

The coordinator will receive a task, determine a workflow, pass structured
context to specialist agents, aggregate their results, and return verified
output. Phase 1 intentionally provides no executable orchestration.
"""

from backend.schemas import AgentResult, AgentTask


class Coordinator:
    """Declare the future coordinator entry point without fake behavior."""

    async def handle(self, task: AgentTask) -> AgentResult:
        """Coordinate a task after workflow behavior is implemented."""

        raise NotImplementedError(
            "Coordinator execution is planned for a later development phase."
        )
