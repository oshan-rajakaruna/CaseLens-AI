"""Tests for BM25 indexing and structured top-k search."""

from pathlib import Path

import pytest

from retrieval.bm25 import BM25Index, BM25Searcher, tokenize_for_search
from retrieval.preprocessing import LegalDocumentMetadata, LegalTextChunk, load_text_document

FIXTURES = Path(__file__).parents[2] / "fixtures" / "retrieval"


def _fixture_chunk(name: str, document_id: str, case_name: str) -> LegalTextChunk:
    metadata = LegalDocumentMetadata(
        document_id=document_id,
        case_name=case_name,
        citation=f"SYNTHETIC-{document_id.upper()}",
        source=f"tests/fixtures/retrieval/{name}",
        document_type="synthetic test fixture",
    )
    return LegalTextChunk(
        chunk_id=f"{document_id}-chunk-0001",
        document_id=document_id,
        chunk_text=load_text_document(FIXTURES / name),
        metadata=metadata,
    )


def _chunks() -> list[LegalTextChunk]:
    return [
        _fixture_chunk(
            "synthetic_employment.txt", "employment", "Synthetic Employment Matter"
        ),
        _fixture_chunk("synthetic_contract.txt", "contract", "Synthetic Contract Matter"),
        _fixture_chunk("synthetic_property.txt", "property", "Synthetic Property Matter"),
    ]


def test_tokenization_is_normalized_without_changing_chunk_text() -> None:
    chunk = _chunks()[0]
    original = chunk.chunk_text

    assert tokenize_for_search("Court's ORDER 42") == ["court", "s", "order", "42"]
    BM25Index().build([chunk])
    assert chunk.chunk_text == original


def test_bm25_ranks_employment_text_above_unrelated_text() -> None:
    searcher = BM25Searcher(BM25Index().build(_chunks()))

    results = searcher.search("wrongful employment termination dismissal", top_k=3)

    assert results[0].document_id == "employment"
    assert results[0].rank == 1
    assert results[0].score > 0
    assert results[0].case_name == "Synthetic Employment Matter"
    assert results[0].citation == "SYNTHETIC-EMPLOYMENT"
    assert "wrongful dismissal" in results[0].chunk_text


def test_search_handles_empty_inputs_and_large_top_k() -> None:
    empty_searcher = BM25Searcher(BM25Index())
    populated_searcher = BM25Searcher(BM25Index().build(_chunks()))

    assert empty_searcher.search("employment", top_k=5) == []
    assert populated_searcher.search("   ", top_k=5) == []
    assert populated_searcher.search("zzznomatchzzz", top_k=5) == []
    assert len(populated_searcher.search("synthetic", top_k=50)) == 3


def test_search_uses_stable_ids_to_break_equal_score_ties() -> None:
    chunks = [
        LegalTextChunk(
            chunk_id=f"{document_id}-chunk-0001",
            document_id=document_id,
            chunk_text="shared term",
            metadata=LegalDocumentMetadata(document_id=document_id),
        )
        for document_id in ("property", "employment", "contract")
    ]
    results = BM25Searcher(BM25Index().build(chunks)).search("shared", top_k=3)

    assert [result.document_id for result in results] == [
        "contract",
        "employment",
        "property",
    ]


def test_index_rejects_duplicate_chunk_ids() -> None:
    chunk = _chunks()[0]

    with pytest.raises(ValueError, match="chunk_id values must be unique"):
        BM25Index().build([chunk, chunk])


def test_search_rejects_negative_top_k() -> None:
    searcher = BM25Searcher(BM25Index().build(_chunks()))

    with pytest.raises(ValueError, match="top_k cannot be negative"):
        searcher.search("employment", top_k=-1)
