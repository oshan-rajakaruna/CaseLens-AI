"""Offline real-agent Coordinator full-research integration."""

import asyncio

from agents.coordinator import (
    Coordinator,
    CoordinatorEvidenceDocument,
    CoordinatorRequest,
    CoordinatorRetrievalAdapter,
)
from agents.retrieval import RetrievalAgent
from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk
from tests.unit.retrieval._fakes import DeterministicEmbeddingService


def _judgment() -> LegalTextChunk:
    metadata = LegalDocumentMetadata(
        document_id="JUDG-FULL-001", case_name="Fictional Perera v. Atlas",
        court="Fictional Court of Appeal", citation="[2025] FCA 44", document_type="judgment",
        source="fictional archive", source_url="https://example.invalid/judg-full-001",
        page_number=6, page_start=6, page_end=6,
    )
    return LegalTextChunk(
        document_id=metadata.document_id, chunk_id="JUDG-FULL-001-C06",
        chunk_text="The issue before the court was whether notice was timely. The court reasoned that notice complied with section 12. Accordingly, the appeal is dismissed.",
        metadata=metadata,
    )


def test_full_research_invokes_real_analysis_retrieval_summarization_and_verification() -> None:
    retrieval = RetrievalAgent([_judgment()], embedding_service=DeterministicEmbeddingService())
    coordinator = Coordinator(retrieval_adapter=CoordinatorRetrievalAdapter(retrieval))
    request = CoordinatorRequest(
        case_id="CASE-FULL-001", case_title="Fictional payment notice dispute",
        user_query="Was notice timely under the contract?", requested_task="full_research",
        retrieval_mode="bm25", document_ids=["DOC-FULL-001"],
        evidence_documents=[CoordinatorEvidenceDocument(
            document_id="DOC-FULL-001", name="fictional-notice.txt",
            text="Ms. Nadeesha Silva alleged that payment may not be due unless notice was served."
        )],
    )

    response = asyncio.run(coordinator.handle(request))

    assert response.executed is True
    assert response.analysis_results and response.analysis_results[0].document_id == "DOC-FULL-001"
    assert response.precedent_response is not None
    assert response.precedent_response.retrieval_results[0].chunk_id == "JUDG-FULL-001-C06"
    assert response.evidence_summary_response is not None
    assert response.evidence_summary_response.source_references[0].document_id == "DOC-FULL-001"
    assert response.verification_response is not None
    assert response.verification_response.overall_status == "supported"
    assert any("does not establish legal truth" in limit for limit in response.verification_limitations)
    assert response.model_validate_json(response.model_dump_json()) == response
    assert any(stage.agent == "analysis" for stage in response.stages)
    assert any(stage.agent == "retrieval" for stage in response.stages)
    assert any(stage.agent == "verification" for stage in response.stages)
