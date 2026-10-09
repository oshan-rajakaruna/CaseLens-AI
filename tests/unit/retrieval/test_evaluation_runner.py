"""Tests for loading and running the offline synthetic evaluation fixture."""

import json
from pathlib import Path

from retrieval.evaluation.run import load_evaluation_dataset, main, run_evaluation

DATASET_PATH = (
    Path(__file__).parents[2]
    / "evaluation"
    / "retrieval"
    / "synthetic_dataset.json"
)


def test_synthetic_dataset_is_explicitly_labelled_and_valid() -> None:
    dataset = load_evaluation_dataset(DATASET_PATH)

    assert dataset.data_classification == "synthetic"
    assert "not real case law" in dataset.description
    assert len(dataset.documents) == 3
    assert len(dataset.queries) == 4


def test_offline_runner_compares_all_modes_without_live_provider() -> None:
    dataset = load_evaluation_dataset(DATASET_PATH)

    comparison, weights = run_evaluation(dataset, k_values=(1, 3))

    assert [result.mode for result in comparison.modes] == [
        "bm25",
        "semantic",
        "hybrid",
    ]
    assert all(result.query_count == 4 for result in comparison.modes)
    assert weights == []


def test_cli_json_output_is_machine_readable_and_warns_synthetic(
    capsys: object,
) -> None:
    exit_code = main(
        [
            "--dataset",
            str(DATASET_PATH),
            "--k",
            "1",
            "3",
            "--weights",
            "--json",
        ]
    )
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    payload = json.loads(output)

    assert exit_code == 0
    assert "Synthetic offline prototype" in payload["notice"]
    assert payload["dataset"]["data_classification"] == "synthetic"
    assert len(payload["comparison"]["modes"]) == 3
    assert len(payload["weight_experiments"]) == 5
