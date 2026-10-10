"""CLI for curated legal dataset validation, ingestion, and index building."""

import argparse
import json
from pathlib import Path
import sys

from retrieval.config import load_retrieval_environment
from retrieval.ingestion.manifest import ManifestLoadError
from retrieval.ingestion.pipeline import IngestionPipeline, StrictIngestionError
from retrieval.ingestion.report import IngestionReport


def _print_report(report: IngestionReport) -> None:
    print(json.dumps(report.model_dump(mode="json"), indent=2))


def main(argv: list[str] | None = None) -> int:
    """Run ingestion and emit its structured JSON report."""

    load_retrieval_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count chunks without building indexes or calling a provider.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop at the first invalid document or index failure.",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Explicitly opt in to configured-provider semantic indexing after BM25.",
    )
    parser.add_argument(
        "--embedding-checkpoint",
        type=Path,
        help="Optional resumable JSONL embedding checkpoint path.",
    )
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument(
        "--chunk-overlap",
        "--overlap",
        dest="overlap",
        type=int,
        default=50,
        help="Word overlap between adjacent chunks (default: 50).",
    )
    arguments = parser.parse_args(argv)

    try:
        pipeline = IngestionPipeline(
            arguments.input_dir,
            chunk_size=arguments.chunk_size,
            overlap=arguments.overlap,
            strict=arguments.strict,
        )
        outcome = pipeline.ingest(
            arguments.manifest,
            dry_run=arguments.dry_run,
            semantic_requested=arguments.semantic,
        )
        if arguments.semantic:
            print(
                f"Semantic indexing requested for {len(outcome.chunks)} chunks"
                + ("; dry-run makes no API calls." if arguments.dry_run else "."),
                file=sys.stderr,
            )
        pipeline.build_indexes(
            outcome,
            semantic=arguments.semantic,
            checkpoint_path=arguments.embedding_checkpoint,
        )
    except ManifestLoadError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Ingestion configuration is invalid: {exc}", file=sys.stderr)
        return 2
    except StrictIngestionError as exc:
        _print_report(exc.report)
        return 1

    _print_report(outcome.report)
    return 1 if outcome.report.failed_documents or outcome.report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
