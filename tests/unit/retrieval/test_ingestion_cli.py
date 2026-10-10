"""CLI safety and reporting tests for legal dataset ingestion."""

import json
from pathlib import Path

from retrieval.ingestion.run import main

FIXTURES = Path("tests/fixtures/ingestion")


def test_cli_loads_environment_once_at_startup(
    capsys,
    monkeypatch,
) -> None:
    calls: list[None] = []

    def fake_load_environment() -> bool:
        calls.append(None)
        return True

    monkeypatch.setattr(
        "retrieval.ingestion.run.load_retrieval_environment",
        fake_load_environment,
    )

    exit_code = main(
        [
            "--manifest",
            str(FIXTURES / "valid_manifest.json"),
            "--input-dir",
            str(FIXTURES),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert calls == [None]
    assert json.loads(capsys.readouterr().out)["dry_run"] is True


def test_cli_dry_run_with_semantic_flag_makes_no_provider_request(
    capsys,
) -> None:
    exit_code = main(
        [
            "--manifest",
            str(FIXTURES / "valid_manifest.json"),
            "--input-dir",
            str(FIXTURES),
            "--dry-run",
            "--semantic",
            "--chunk-size",
            "10",
            "--chunk-overlap",
            "2",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert report["dry_run"] is True
    assert report["semantic_requested"] is True
    assert report["semantic_indexed_chunks"] == 0
    assert "dry-run makes no API calls" in captured.err


def test_cli_non_strict_report_returns_failure_after_finishing_batch(capsys) -> None:
    exit_code = main(
        [
            "--manifest",
            str(FIXTURES / "mixed_manifest.json"),
            "--input-dir",
            str(FIXTURES),
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["successfully_processed"] == 1
    assert report["failed_documents"] == 2
    assert report["skipped_documents"] == 1
