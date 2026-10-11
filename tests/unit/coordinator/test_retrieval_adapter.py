"""Coordinator-owned tests for prepared Retrieval execution."""

import asyncio

import pytest
from pydantic import ValidationError

from agents.coordinator import (
    Coordinator,
    CoordinatorRequest,
    CoordinatorRetrievalAdapter,
)
from retrieval.preprocessing.metadata import (
    BM25SearchResult,
    HybridSearchResult,
    LegalDocumentMetadata,
    SemanticSearchResult,
)


def _metadata() -> LegalDocumentMetadata:
    return LegalDocumentMetadata(
        document_id="PREC-001",
        case_name="Fernando v. Atlas Trading",
        court="Fictional Commercial Court",
        citation="[2025] FCT 12",
        document_type="judgment",
        source="fictional official archive",
        source_url="https://example.invalid/prec-001",
        page_number=7,
        page_start=7,
        page_end=8,
    )


class RecordingRetrievalAgent:
    def __init__(self, *, bm25: int = 1, semantic: int = 1, failure: Exception | None = None) -> None:
        self.bm25_index_size = bm25
        self.semantic_index_size = semantic
        self.failure = failure
        self.calls: list[tuple[str, str, int, object]] = []

    def search_bm25(self, query: str, top_k: int = 5, **kwargs: object):
        self.calls.append(("bm25", query, top_k, kwargs.get("filters")))
        if self.failure:
            raise self.failure
        return [
            BM25SearchResult(
                rank=1, document_id="PREC-001", chunk_id="PREC-001-chunk-0007",
                score=2.5, chunk_text="Payment notice clause evidence.",
                metadata=_metadata(), case_name="Fernando v. Atlas Trading",
                court="Fictional Commercial Court", citation="[2025] FCT 12",
                source="fictional official archive", document_type="judgment",
            )
        ]

    def search_semantic(self, query: str, top_k: int = 5, **kwargs: object):
        self.calls.append(("semantic", query, top_k, kwargs.get("filters")))
        return [
            SemanticSearchResult(
                rank=1, document_id="PREC-001", chunk_id="PREC-001-chunk-0007",
                similarity_score=0.88, chunk_text="Payment notice clause evidence.",
                metadata=_metadata(), case_name="Fernando v. Atlas Trading",
                court="Fictional Commercial Court", citation="[2025] FCT 12",
                source="fictional official archive", document_type="judgment",
            )
        ]

    def search_hybrid(self, query: str, top_k: int = 5, **kwargs: object):
        self.calls.append(("hybrid", query, top_k, kwargs.get("filters")))
        return [
            HybridSearchResult(
                rank=1, document_id="PREC-001", chunk_id="PREC-001-chunk-0007",
                chunk_text="Payment notice clause evidence.", metadata=_metadata(),
                case_name="Fernando v. Atlas Trading", court="Fictional Commercial Court",
                citation="[2025] FCT 12", source="fictional official archive",
                document_type="judgment", raw_bm25_score=2.5,
                normalized_bm25_score=0.7, raw_semantic_score=0.88,
                normalized_semantic_score=0.88, hybrid_score=0.79,
            )
        ]


def _request(mode: str = "bm25", **overrides: object) -> CoordinatorRequest:
    values: dict[str, object] = {
        "case_id": "CASE-RET-001",
        "user_query": "contractual payment notice",
        "requested_task": "retrieve_precedents",
        "retrieval_mode": mode,
        "top_k": 3,
    }
    values.update(overrides)
    return CoordinatorRequest.model_validate(values)


@pytest.mark.parametrize("mode", ["bm25", "semantic", "hybrid"])
def test_adapter_maps_each_supported_mode_and_preserves_native_result(mode: str) -> None:
    agent = RecordingRetrievalAgent()
    response = Coordinator(
        retrieval_adapter=CoordinatorRetrievalAdapter(agent)  # type: ignore[arg-type]
    ).execute_retrieval(_request(mode))

    assert agent.calls == [(mode, "contractual payment notice", 3, None)]
    assert response.executed is True
    assert response.status == "completed"
    assert response.stages[0].status == "completed"
    result = response.retrieval_results[0]
    assert result.document_id == "PREC-001"
    assert result.chunk_id == "PREC-001-chunk-0007"
    assert result.chunk_text == "Payment notice clause evidence."
    assert result.metadata.page_start == 7
    assert result.metadata.page_end == 8
    assert result.court == "Fictional Commercial Court"
    assert result.metadata.source_url == "https://example.invalid/prec-001"
    assert result.metadata.citation == "[2025] FCT 12"
    assert any("do not establish authority" in limit for limit in response.verification_limitations)


def test_hybrid_filters_and_serialization_are_preserved() -> None:
    agent = RecordingRetrievalAgent()
    response = Coordinator(
        retrieval_adapter=CoordinatorRetrievalAdapter(agent)  # type: ignore[arg-type]
    ).execute_retrieval(_request("hybrid", retrieval_filters={"court": "Fictional Commercial Court"}))

    assert agent.calls[0][3] == {"court": "Fictional Commercial Court"}
    restored = type(response).model_validate_json(response.model_dump_json())
    assert restored.retrieval_results[0].hybrid_score == 0.79


def test_missing_service_and_missing_index_do_not_execute_search() -> None:
    request = _request()
    missing_service = Coordinator().execute_retrieval(request)
    missing_index = Coordinator(
        retrieval_adapter=CoordinatorRetrievalAdapter(RecordingRetrievalAgent(bm25=0))  # type: ignore[arg-type]
    ).execute_retrieval(request)

    assert missing_service.status == "requires_input"
    assert missing_service.executed is False
    assert missing_service.errors[0].code == "missing_retrieval_service"
    assert missing_index.status == "requires_input"
    assert missing_index.executed is False
    assert missing_index.errors[0].code == "missing_retrieval_index"


def test_empty_search_is_completed_with_an_explicit_warning() -> None:
    agent = RecordingRetrievalAgent()
    agent.search_bm25 = lambda *args, **kwargs: []  # type: ignore[method-assign]
    response = Coordinator(
        retrieval_adapter=CoordinatorRetrievalAdapter(agent)  # type: ignore[arg-type]
    ).execute_retrieval(_request())

    assert response.status == "completed"
    assert response.executed is True
    assert response.retrieval_results == []
    assert response.warnings[0].code == "no_retrieval_results"


def test_runtime_failure_is_controlled_and_other_routes_remain_unchanged() -> None:
    failed = Coordinator(
        retrieval_adapter=CoordinatorRetrievalAdapter(RecordingRetrievalAgent(failure=RuntimeError("private path")))  # type: ignore[arg-type]
    ).execute_retrieval(_request())
    planned = asyncio.run(
        Coordinator().handle(
            _request("bm25", requested_task="summarize_evidence", document_ids=["DOC-001"])
        )
    )

    assert failed.status == "failed"
    assert failed.executed is True
    assert failed.errors[0].message == "Retrieval could not complete safely."
    # Step 5 executes this route and safely rejects missing supplied evidence.
    assert planned.executed is False
    assert planned.status == "requires_input"


@pytest.mark.parametrize(
    "values",
    [
        {"top_k": 0},
        {"top_k": 21},
        {"top_k": True},
        {"retrieval_mode": "unknown"},
        {"retrieval_mode": "bm25", "retrieval_filters": {"court": "Fictional"}},
    ],
)
def test_retrieval_request_validation(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _request(**values)
