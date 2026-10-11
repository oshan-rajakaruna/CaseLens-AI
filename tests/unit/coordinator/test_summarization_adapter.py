"""Coordinator Step 4 source-grounded precedent handoff tests."""

import asyncio

import pytest

from agents.coordinator import (
    Coordinator,
    CoordinatorRequest,
    CoordinatorRetrievalAdapter,
)
from agents.coordinator.summarization_adapter import CitationCollisionError, CoordinatorSummarizationAdapter
from retrieval.preprocessing.metadata import BM25SearchResult, LegalDocumentMetadata


def _result(
    document_id: str = "JUDG-001",
    chunk_id: str = "JUDG-001-C01",
    text: str = "The issue before the court was whether notice was valid. The court reasoned that notice was timely. Accordingly, the appeal is dismissed.",
    document_type: str = "judgment",
) -> BM25SearchResult:
    metadata = LegalDocumentMetadata(
        document_id=document_id,
        case_name=f"Fictional {document_id}",
        court="Fictional Court of Appeal",
        citation="[2025] FCA 11",
        document_type=document_type,
        source="fictional archive",
        source_url=f"https://example.invalid/{document_id.lower()}",
        page_number=4,
        page_start=4,
        page_end=5,
    )
    return BM25SearchResult(
        rank=1,
        document_id=document_id,
        chunk_id=chunk_id,
        score=2.0,
        chunk_text=text,
        metadata=metadata,
        case_name=metadata.case_name,
        court=metadata.court,
        citation=metadata.citation,
        source=metadata.source,
        document_type=document_type,
    )


class _Retrieval:
    bm25_index_size = 1
    semantic_index_size = 1

    def __init__(self, results: list[BM25SearchResult]) -> None:
        self.results = results

    def search_bm25(self, query: str, top_k: int = 5, **kwargs: object) -> list[BM25SearchResult]:
        return self.results

    def search_semantic(self, query: str, top_k: int = 5, **kwargs: object) -> list[BM25SearchResult]:
        return self.results

    def search_hybrid(self, query: str, top_k: int = 5, **kwargs: object) -> list[BM25SearchResult]:
        return self.results


def _request() -> CoordinatorRequest:
    return CoordinatorRequest(
        case_id="CASE-STEP4-001",
        user_query="Was contractual notice timely?",
        requested_task="summarize_precedents",
        retrieval_mode="bm25",
        top_k=5,
    )


def _coordinator(results: list[BM25SearchResult]) -> Coordinator:
    return Coordinator(retrieval_adapter=CoordinatorRetrievalAdapter(_Retrieval(results)))  # type: ignore[arg-type]


def test_actual_deterministic_handoff_preserves_passages_and_provenance() -> None:
    response = _coordinator([_result()]).execute_summarize_precedents(_request())

    assert response.status == "completed"
    assert response.executed is True
    assert [stage.status for stage in response.stages] == ["completed", "completed"]
    assert response.retrieval_results[0].chunk_text.startswith("The issue before")
    assert response.precedent_summaries[0].holding == "Accordingly, the appeal is dismissed."
    source = response.source_references[0]
    assert source.document_id == "JUDG-001"
    assert source.excerpt == response.retrieval_results[0].chunk_text
    assert source.locator == "chunk:JUDG-001-C01; pages:4-5"
    assert source.citation_id and source.citation_id.startswith("SRC-")
    mapped = response.citation_provenance[source.citation_id]
    assert mapped.chunk_id == "JUDG-001-C01"
    assert mapped.page_start == 4 and mapped.page_end == 5
    assert mapped.court == "Fictional Court of Appeal"
    assert mapped.official_citation == "[2025] FCA 11"
    assert mapped.source_url == "https://example.invalid/judg-001"
    assert type(response).model_validate_json(response.model_dump_json()) == response


def test_multiple_chunks_stay_with_one_judgment_and_documents_stay_separate() -> None:
    second_chunk = _result(
        chunk_id="JUDG-001-C02",
        text="The claimant alleged that payment was late. The Court dismisses the appeal.",
    )
    other = _result(
        document_id="JUDG-002",
        chunk_id="JUDG-002-C01",
        text="The issue before the court was whether delivery was complete. The Court allows the appeal.",
    )
    response = _coordinator([_result(), second_chunk, other]).execute_summarize_precedents(_request())

    assert len(response.summarization_responses) == 2
    assert [
        summary.precedent_summaries[0].source_references[0].document_id
        for summary in response.summarization_responses
    ] == ["JUDG-001", "JUDG-002"]
    assert [item.case_identifier for item in response.precedent_summaries] == [
        "Fictional JUDG-001", "Fictional JUDG-002"
    ]
    assert [source.document_id for source in response.source_references] == ["JUDG-001", "JUDG-001", "JUDG-002"]


def test_non_judgment_material_is_preserved_but_never_summarized_as_precedent() -> None:
    response = _coordinator([_result(document_type="contract")]).execute_summarize_precedents(_request())

    assert response.status == "partial"
    assert response.retrieval_results[0].document_type == "contract"
    assert response.precedent_summaries == []
    assert response.source_references == []
    assert {warning.code for warning in response.warnings} == {
        "unsupported_precedent_document_type", "no_supported_judgment_material"
    }
    assert response.stages[1].status == "partial"


def test_empty_and_missing_retrieval_are_explicit_and_do_not_invoke_summarization() -> None:
    empty = _coordinator([]).execute_summarize_precedents(_request())
    missing = Coordinator().execute_summarize_precedents(_request())

    assert empty.status == "partial"
    assert empty.executed is True
    assert empty.stages[1].status == "not_run"
    assert empty.warnings[0].code == "no_retrieval_results"
    assert missing.status == "requires_input"
    assert missing.executed is False
    assert missing.errors[0].code == "missing_retrieval_service"


def test_conflicting_reused_chunk_identifier_is_rejected_without_summary() -> None:
    first = _result(text="The Court dismisses the appeal.")
    conflicting = _result(text="The Court allows the appeal.")

    with pytest.raises(CitationCollisionError):
        CoordinatorSummarizationAdapter().summarize_precedents(_request(), [first, conflicting])

    response = _coordinator([first, conflicting]).execute_summarize_precedents(_request())
    assert response.status == "partial"
    assert response.stages[1].status == "failed"
    assert response.errors[0].code == "summarization_failed"
    assert response.precedent_summaries == []


def test_existing_analysis_and_retrieval_routes_remain_executable() -> None:
    retrieval = _coordinator([_result()]).execute_retrieval(
        _request().model_copy(update={"requested_task": "retrieve_precedents"})
    )
    analysis = asyncio.run(
        Coordinator().handle(
            CoordinatorRequest(
                case_id="CASE-STEP4-AN",
                case_title="Fictional dispute",
                user_query="Analyze this evidence.",
                requested_task="analyze_evidence",
                document_ids=["DOC-001"],
                evidence_documents=[{"document_id": "DOC-001", "name": "notice.txt", "text": "Payment was made."}],
            )
        )
    )

    assert retrieval.status == "completed"
    assert analysis.status == "completed"
