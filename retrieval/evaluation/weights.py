"""Hybrid-weight experiment support without automatic winner selection."""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from retrieval.evaluation.evaluator import RetrievalEvaluator, normalize_k_values
from retrieval.evaluation.models import EvaluationDataset, ModeEvaluationResult
from retrieval.hybrid import normalize_weights

DEFAULT_WEIGHT_CONFIGURATIONS: tuple[tuple[float, float], ...] = (
    (0.2, 0.8),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.8, 0.2),
)


class HybridWeightExperimentResult(BaseModel):
    """Metrics measured for one normalized hybrid weight configuration."""

    bm25_weight: float = Field(ge=0, le=1)
    semantic_weight: float = Field(ge=0, le=1)
    evaluation: ModeEvaluationResult


def evaluate_hybrid_weights(
    evaluator: RetrievalEvaluator,
    dataset: EvaluationDataset,
    *,
    configurations: Sequence[tuple[float, float]] = DEFAULT_WEIGHT_CONFIGURATIONS,
    k_values: Sequence[int] = (1, 3, 5),
) -> list[HybridWeightExperimentResult]:
    """Measure configurations in supplied order without declaring an optimum."""

    cutoffs = normalize_k_values(k_values)
    experiments: list[HybridWeightExperimentResult] = []
    for bm25_weight, semantic_weight in configurations:
        normalized_bm25, normalized_semantic = normalize_weights(
            bm25_weight,
            semantic_weight,
        )
        experiments.append(
            HybridWeightExperimentResult(
                bm25_weight=normalized_bm25,
                semantic_weight=normalized_semantic,
                evaluation=evaluator.evaluate_mode(
                    dataset.queries,
                    "hybrid",
                    k_values=cutoffs,
                    retrieval_depth=max(max(cutoffs), len(dataset.documents)),
                    bm25_weight=normalized_bm25,
                    semantic_weight=normalized_semantic,
                ),
            )
        )
    return experiments
