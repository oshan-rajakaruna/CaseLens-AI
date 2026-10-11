"""Real Coordinator-to-Analysis integration without external services."""

import asyncio

from agents.coordinator import (
    Coordinator,
    CoordinatorEvidenceDocument,
    CoordinatorRequest,
)
from agents.analysis.schemas import EvidenceSource
from nlp.extraction import ExtractionSegment


def test_coordinator_runs_analysis_and_preserves_fact_provenance_and_semantics() -> None:
    text = "Ayesha Perera alleges that Nimal Silva did not pay Rs. 150,000 on 12 March 2025."
    segment = ExtractionSegment(
        segment_id="DOC-AN-001:segment:1",
        document_id="DOC-AN-001",
        text=text,
        start_char=0,
        end_char=len(text),
        source=EvidenceSource(
            document_id="DOC-AN-001",
            page=1,
            locator="page:1",
            start_char=0,
            end_char=len(text),
            excerpt=text,
        ),
    )
    request = CoordinatorRequest(
        case_id="CASE-AN-001",
        case_title="Fictional payment dispute",
        user_query="Analyze this evidence.",
        requested_task="analyze_evidence",
        document_ids=["DOC-AN-001"],
        evidence_documents=[
            CoordinatorEvidenceDocument(
                document_id="DOC-AN-001",
                name="statement.txt",
                text=text,
                segments=[segment],
            )
        ],
    )

    response = asyncio.run(Coordinator().handle(request))

    assert response.executed is True
    assert response.status == "completed"
    assert response.stages[0].status == "completed"
    result = response.analysis_results[0]
    fact = next(item for item in result.facts if item.category == "non_payment_or_breach")
    assert fact.statement == text
    assert fact.epistemic_status == "alleged"
    assert fact.is_attributed is True
    assert fact.is_negated is True
    assert fact.verification_status == "not_independently_verified"
    assert fact.source.document_id == "DOC-AN-001"
    assert fact.source.page == 1
    assert fact.source.locator == "page:1"
    assert fact.source.start_char == 0
    assert fact.source.end_char == len(text)
    assert fact.source.excerpt == text
    assert text[fact.source.start_char : fact.source.end_char] == fact.source.excerpt
