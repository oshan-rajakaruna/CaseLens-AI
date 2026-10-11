"""Backend-owned, authorized bridge to the public Coordinator interface."""
from collections.abc import Sequence
from agents.coordinator import Coordinator, CoordinatorEvidenceDocument, CoordinatorRequest, CoordinatorRetrievalAdapter
from agents.retrieval import RetrievalAgent
from backend.config import get_settings
from backend.db.models import DocumentRecord
from retrieval.evaluation.run import DeterministicEvaluationEmbeddingService
from retrieval.preprocessing.metadata import LegalDocumentMetadata, LegalTextChunk


def _demo_agent() -> RetrievalAgent | None:
    """Return an explicitly enabled fictional corpus; never index case documents."""
    settings = get_settings()
    if settings.app_env != "development" or not settings.enable_demo_precedent_corpus:
        return None
    metadata = LegalDocumentMetadata(
        document_id="DEMO-PREC-001", case_name="Fernando v. Meridian Trading",
        court="Fictional Commercial Court", citation="[2025] FCC 12",
        document_type="judgment", source="fictional development corpus",
        source_url="https://example.invalid/demo-prec-001", page_number=4, page_start=4, page_end=4,
    )
    chunks = [LegalTextChunk(
        document_id=metadata.document_id, chunk_id="DEMO-PREC-001-C04", metadata=metadata,
        chunk_text="The issue before the court was whether contractual notice was timely. The court reasoned that notice complied with section 12. Accordingly, the appeal is dismissed.",
    )]
    agent = RetrievalAgent(chunks, embedding_service=DeterministicEvaluationEmbeddingService())
    agent.index_semantic_chunks(chunks)
    return agent


def execute_authorized(request: CoordinatorRequest, documents: Sequence[DocumentRecord]):
    """Replace client evidence with private extracted text already authorized by the API."""
    requested = set(request.document_ids)
    available = {document.id: document for document in documents if document.extracted_text}
    if requested and not requested.issubset(available):
        raise ValueError("Requested document identifiers are not available to this case.")
    evidence_document_ids = {
        item.source.document_id for item in request.verification_evidence
    }
    if evidence_document_ids and not evidence_document_ids.issubset(available):
        raise ValueError("Verification evidence must reference extracted documents owned by this case.")
    selected = [available[key] for key in request.document_ids] if request.document_ids else list(available.values())
    evidence = [CoordinatorEvidenceDocument(
        document_id=item.id, name=item.original_name, text=item.extracted_text or "",
        media_type=item.media_type,
    ) for item in selected]
    enriched = request.model_copy(update={
        "document_ids": [item.document_id for item in evidence],
        "evidence_documents": evidence,
    })
    agent = _demo_agent()
    coordinator = Coordinator(
        retrieval_adapter=CoordinatorRetrievalAdapter(agent) if agent is not None else None
    )
    return coordinator, enriched
