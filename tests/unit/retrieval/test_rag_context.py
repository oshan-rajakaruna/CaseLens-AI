"""Tests for ranked, citation-preserving RAG context assembly."""

import pytest

from retrieval.preprocessing.metadata import (
    HybridSearchResult,
    LegalDocumentMetadata,
)
from retrieval.rag import (
    RAGContextAssembler,
    RAGContextSettings,
    estimate_tokens,
    format_context,
)


def _result(
    document_id: str,
    chunk_number: int,
    rank: int,
    *,
    score: float = 0.8,
    text: str | None = None,
    page: int | None = None,
) -> HybridSearchResult:
    metadata = LegalDocumentMetadata(
        document_id=document_id,
        case_name=f"Case {document_id}",
        court="Synthetic Court",
        citation=f"CIT-{document_id}",
        source="official source",
        source_url="https://example.invalid/source",
        provenance="official_court_source",
        page_number=page,
        page_start=page,
        page_end=page,
    )
    return HybridSearchResult(
        rank=rank,
        document_id=document_id,
        chunk_id=f"{document_id}-chunk-{chunk_number:04d}",
        case_name=metadata.case_name,
        chunk_text=text or f"Evidence from {document_id} chunk {chunk_number}.",
        citation=metadata.citation,
        source=metadata.source,
        court=metadata.court,
        metadata=metadata,
        raw_bm25_score=score * 10,
        normalized_bm25_score=score,
        raw_semantic_score=score,
        normalized_semantic_score=score,
        hybrid_score=score,
    )


def _assembler(
    *,
    max_tokens: int = 4000,
    max_passages: int = 8,
    min_score: float | None = None,
) -> RAGContextAssembler:
    return RAGContextAssembler(
        RAGContextSettings(
            max_tokens=max_tokens,
            max_passages=max_passages,
            min_retrieval_score=min_score,
        )
    )


def test_ranked_selection_and_context_ids_are_deterministic() -> None:
    results = [
        _result("DOC-C", 1, 3),
        _result("DOC-A", 1, 1),
        _result("DOC-B", 1, 2),
    ]

    first = _assembler().assemble(results)
    second = _assembler().assemble(results)

    assert [passage.document_id for passage in first.passages] == [
        "DOC-A",
        "DOC-B",
        "DOC-C",
    ]
    assert [passage.context_id for passage in first.passages] == [
        "CTX-001",
        "CTX-002",
        "CTX-003",
    ]
    assert first == second


def test_max_passages_is_enforced_without_filling_past_limit() -> None:
    context = _assembler(max_passages=2).assemble(
        [_result(f"DOC-{index}", 1, index) for index in range(1, 5)]
    )

    assert len(context.passages) == 2
    assert context.truncated is True


def test_token_budget_is_never_exceeded_and_text_can_be_truncated() -> None:
    context = _assembler(max_tokens=60).assemble(
        [_result("DOC-A", 1, 1, text="evidence " * 100)]
    )

    assert len(context.passages) == 1
    assert context.passages[0].truncated is True
    assert context.truncated is True
    assert context.estimated_tokens <= 60


def test_adjacent_overlapping_chunks_merge_in_document_order() -> None:
    first_text = "one two three four five six seven eight nine ten alpha"
    second_text = "four five six seven eight nine ten alpha beta gamma"
    context = _assembler().assemble(
        [
            _result("DOC-A", 2, 1, text=second_text, page=2),
            _result("DOC-A", 1, 2, text=first_text, page=1),
        ]
    )

    assert len(context.passages) == 1
    passage = context.passages[0]
    assert passage.text == f"{first_text} beta gamma"
    assert set(passage.chunk_ids) == {
        "DOC-A-chunk-0001",
        "DOC-A-chunk-0002",
    }
    assert passage.pages == [1, 2]
    assert len(passage.retrieval_scores) == 2
    assert passage.score == 0.8


def test_non_adjacent_chunks_are_not_merged() -> None:
    context = _assembler().assemble(
        [
            _result("DOC-A", 1, 1, page=1),
            _result("DOC-A", 3, 2, page=1),
        ]
    )

    assert len(context.passages) == 2


def test_provenance_and_citation_survive_merging() -> None:
    context = _assembler().assemble(
        [
            _result("DOC-A", 1, 1, page=4),
            _result("DOC-A", 2, 2, page=5),
        ]
    )

    passage = context.passages[0]
    assert passage.title == "Case DOC-A"
    assert passage.court == "Synthetic Court"
    assert passage.source == "official source"
    assert passage.citation.document_id == "DOC-A"
    assert passage.citation.citation == "CIT-DOC-A"
    assert passage.citation.provenance == "official_court_source"
    assert passage.citation.pages == [4, 5]
    assert passage.citation.chunk_ids == passage.chunk_ids


def test_exact_and_conservative_near_duplicates_are_collapsed() -> None:
    base = ("international responsibility and legal obligation " * 20).strip()
    near = f"{base[:-1]}x"
    context = _assembler().assemble(
        [
            _result("DOC-A", 1, 1, text=base),
            _result("DOC-A", 10, 2, text=base),
            _result("DOC-A", 20, 3, text=near),
        ]
    )

    assert context.deduplicated_result_count == 1
    assert len(context.passages) == 1
    assert len(context.passages[0].chunk_ids) == 3


def test_identical_text_from_different_documents_keeps_distinct_provenance() -> None:
    text = "The same quoted treaty provision appears in distinct authorities."
    context = _assembler().assemble(
        [
            _result("DOC-A", 1, 1, text=text),
            _result("DOC-B", 1, 2, text=text),
        ]
    )

    assert [passage.document_id for passage in context.passages] == [
        "DOC-A",
        "DOC-B",
    ]


def test_optional_score_threshold_excludes_lower_results() -> None:
    context = _assembler(min_score=0.5).assemble(
        [
            _result("DOC-A", 1, 1, score=0.7),
            _result("DOC-B", 1, 2, score=0.49),
        ]
    )

    assert context.input_result_count == 2
    assert context.eligible_result_count == 1
    assert [passage.document_id for passage in context.passages] == ["DOC-A"]


def test_unset_threshold_keeps_all_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MIN_RETRIEVAL_SCORE", "")
    settings = RAGContextSettings.from_environment()
    context = RAGContextAssembler(settings).assemble(
        [_result("DOC-A", 1, 1, score=0.01)]
    )

    assert settings.min_retrieval_score is None
    assert len(context.passages) == 1


@pytest.mark.parametrize("value", ["not-a-number", "nan", "inf", "-inf"])
def test_invalid_threshold_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("RAG_MIN_RETRIEVAL_SCORE", value)

    with pytest.raises(ValueError, match="RAG_MIN_RETRIEVAL_SCORE"):
        RAGContextSettings.from_environment()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_MAX_CONTEXT_TOKENS", "0"),
        ("RAG_MAX_CONTEXT_TOKENS", "many"),
        ("RAG_MAX_CONTEXT_PASSAGES", "-1"),
    ],
)
def test_invalid_budget_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        RAGContextSettings.from_environment()


def test_empty_and_fewer_than_requested_results_are_valid() -> None:
    empty = _assembler(max_passages=8).assemble([])
    partial = _assembler(max_passages=8).assemble([_result("DOC-A", 1, 1)])

    assert empty.passages == []
    assert empty.estimated_tokens == 0
    assert empty.truncated is False
    assert len(partial.passages) == 1
    assert partial.truncated is False


def test_formatter_preserves_context_ids_and_source_fields() -> None:
    context = _assembler().assemble([_result("DOC-A", 1, 1, page=7)])

    formatted = format_context(context)

    assert "[CTX-001]" in formatted
    assert "Case: Case DOC-A" in formatted
    assert "Court: Synthetic Court" in formatted
    assert "Pages: 7" in formatted
    assert "Citation: CIT-DOC-A" in formatted
    assert "Text:" in formatted
    assert context.estimated_tokens == estimate_tokens(formatted)
