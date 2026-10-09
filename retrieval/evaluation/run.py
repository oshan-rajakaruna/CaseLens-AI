"""Command-line runner for reproducible, offline retrieval evaluation.

The built-in deterministic embedding service exists only to exercise the
evaluation framework without network access. Its measurements are synthetic
prototype checks and are not a substitute for Gemini or a curated legal corpus.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from agents.retrieval import RetrievalAgent
from retrieval.evaluation.evaluator import RetrievalEvaluator
from retrieval.evaluation.models import EvaluationComparison, EvaluationDataset
from retrieval.evaluation.weights import (
    HybridWeightExperimentResult,
    evaluate_hybrid_weights,
)
from retrieval.preprocessing.metadata import LegalTextChunk

_DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "evaluation"
    / "retrieval"
    / "synthetic_dataset.json"
)
_TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


class DeterministicEvaluationEmbeddingService:
    """Offline feature-hashing embeddings for synthetic framework checks only."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def embed_document(self, text: str, title: str | None = None) -> list[float]:
        combined = f"{title or ''} {text}"
        return self._embed(combined)

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN_PATTERN.findall(text.casefold())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        if not any(vector):
            vector[0] = 1.0
        return vector


def load_evaluation_dataset(path: str | Path) -> EvaluationDataset:
    """Load and validate a UTF-8 JSON evaluation dataset."""

    dataset_path = Path(path)
    try:
        raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load evaluation dataset: {dataset_path}") from exc
    return EvaluationDataset.model_validate(raw_data)


def build_offline_evaluation_agent(dataset: EvaluationDataset) -> RetrievalAgent:
    """Build test-only indexes from documents explicitly stored in the dataset."""

    chunks = [
        LegalTextChunk(
            chunk_id=document.chunk_id,
            document_id=document.metadata.document_id,
            chunk_text=document.chunk_text,
            metadata=document.metadata,
        )
        for document in dataset.documents
    ]
    agent = RetrievalAgent(
        chunks,
        embedding_service=DeterministicEvaluationEmbeddingService(),
    )
    agent.index_semantic_chunks(chunks)
    return agent


def run_evaluation(
    dataset: EvaluationDataset,
    *,
    k_values: tuple[int, ...] = (1, 3, 5),
    include_weight_experiments: bool = False,
) -> tuple[EvaluationComparison, list[HybridWeightExperimentResult]]:
    """Run mode comparison and optional weight experiments offline."""

    evaluator = RetrievalEvaluator(build_offline_evaluation_agent(dataset))
    comparison = evaluator.compare_modes(dataset, k_values=k_values)
    weight_experiments = (
        evaluate_hybrid_weights(evaluator, dataset, k_values=k_values)
        if include_weight_experiments
        else []
    )
    return comparison, weight_experiments


def _json_output(
    dataset: EvaluationDataset,
    comparison: EvaluationComparison,
    weight_experiments: list[HybridWeightExperimentResult],
) -> str:
    output: dict[str, Any] = {
        "notice": (
            "Synthetic offline prototype evaluation; not real-world legal "
            "retrieval accuracy."
        ),
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "data_classification": dataset.data_classification,
            "description": dataset.description,
        },
        "comparison": comparison.model_dump(mode="json"),
    }
    if weight_experiments:
        output["weight_experiments"] = [
            result.model_dump(mode="json") for result in weight_experiments
        ]
    return json.dumps(output, indent=2)


def _text_output(
    dataset: EvaluationDataset,
    comparison: EvaluationComparison,
    weight_experiments: list[HybridWeightExperimentResult],
) -> str:
    lines = [
        f"Dataset: {dataset.name} ({dataset.data_classification})",
        "NOTICE: Synthetic offline prototype measurements; not real-world legal accuracy.",
        f"Queries: {len(dataset.queries)}",
    ]
    for mode in comparison.modes:
        lines.append(f"\n{mode.mode.upper()} | MRR={mode.mrr:.4f}")
        for aggregate in mode.aggregate_at_k:
            lines.append(
                f"  K={aggregate.k}: "
                f"P@K={aggregate.mean_precision_at_k:.4f} "
                f"R@K={aggregate.mean_recall_at_k:.4f}"
            )
    if weight_experiments:
        lines.append("\nHYBRID WEIGHT EXPERIMENTS (no automatic winner selection)")
        for experiment in weight_experiments:
            lines.append(
                f"  BM25={experiment.bm25_weight:.2f} "
                f"Semantic={experiment.semantic_weight:.2f} "
                f"MRR={experiment.evaluation.mrr:.4f}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line evaluator and print text or JSON output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help="Path to a validated evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Positive evaluation cutoffs.",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Also evaluate the predefined hybrid weight configurations.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON rather than the text summary.",
    )
    arguments = parser.parse_args(argv)

    dataset = load_evaluation_dataset(arguments.dataset)
    comparison, weight_experiments = run_evaluation(
        dataset,
        k_values=tuple(arguments.k),
        include_weight_experiments=arguments.weights,
    )
    output = (
        _json_output(dataset, comparison, weight_experiments)
        if arguments.json
        else _text_output(dataset, comparison, weight_experiments)
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
