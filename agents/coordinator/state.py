"""In-memory state shape for a single future coordinator run."""

from dataclasses import dataclass, field

from backend.schemas import AgentResult, AgentTask, VerificationResult


@dataclass(slots=True)
class CoordinatorState:
    """Structured state only; it performs no workflow or persistence logic."""

    task: AgentTask
    results: list[AgentResult] = field(default_factory=list)
    verification: VerificationResult | None = None
