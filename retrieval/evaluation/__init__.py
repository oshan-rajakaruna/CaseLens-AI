"""Reusable offline evaluation framework for CaseLens retrieval modes."""

from retrieval.evaluation.evaluator import RetrievalEvaluator
from retrieval.evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from retrieval.evaluation.models import (
    AggregateAtK,
    EvaluationComparison,
    EvaluationDataset,
    EvaluationDocument,
    EvaluationQuery,
    HumanRelevanceJudgment,
    ModeEvaluationResult,
    PerQueryEvaluationResult,
)
from retrieval.evaluation.weights import (
    DEFAULT_WEIGHT_CONFIGURATIONS,
    HybridWeightExperimentResult,
    evaluate_hybrid_weights,
)

__all__ = [
    "AggregateAtK",
    "DEFAULT_WEIGHT_CONFIGURATIONS",
    "EvaluationComparison",
    "EvaluationDataset",
    "EvaluationDocument",
    "EvaluationQuery",
    "HumanRelevanceJudgment",
    "HybridWeightExperimentResult",
    "ModeEvaluationResult",
    "PerQueryEvaluationResult",
    "RetrievalEvaluator",
    "evaluate_hybrid_weights",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
