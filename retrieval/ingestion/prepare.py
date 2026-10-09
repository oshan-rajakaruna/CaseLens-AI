"""CLI and API for curated dataset summary, dry-run, and index readiness."""

import argparse
import json
from pathlib import Path
import sys

from retrieval.ingestion.dataset import prepare_dataset
from retrieval.ingestion.manifest import ManifestLoadError
from retrieval.ingestion.readiness import ReadinessStatus


def main(argv: list[str] | None = None) -> int:
    """Print a JSON preparation report without creating indexes or embeddings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    arguments = parser.parse_args(argv)

    try:
        report = prepare_dataset(
            arguments.manifest,
            arguments.input_dir,
            chunk_size=arguments.chunk_size,
            overlap=arguments.chunk_overlap,
        )
    except ManifestLoadError as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Dataset preparation configuration is invalid: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report.model_dump(mode="json"), indent=2))
    return (
        0
        if report.readiness.bm25_status
        is ReadinessStatus.READY_FOR_BM25_INDEXING
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
