"""Provider-neutral Summarization Agent contract tests."""

import pytest

from agents.coordinator.schemas import (
    ContextCitationMapEntry,
    RetrievalContextDiagnostics,
    RetrievalContextMetadata,
    SummarizationReadyPayload,
)
from agents.summarization import (
    INSUFFICIENT_CONTEXT_ANSWER,
    SummarizationAgent,
    SummarizationProviderError,
    SummarizationValidationError,
    UnknownContextCitationError,
    build_summarization_provider_request,
    render_citation_provenance,
)
from retrieval.rag import ChunkRetrievalScore, RAGCitation, RAGContextPassage
from tests.unit.summarization._fakes import FakeSummarizationProvider


def _passage(context_id: str, document_id: str, rank: int) -> RAGContextPassage:
    page = rank + 10
    chunk_id = f"{document_id}-chunk-{rank:04d}"
    return RAGContextPassage(
        context_id=context_id,
        text=f"Retrieved legal evidence for {document_id}.",
        document_id=document_id,
        title=f"Case {document_id}",
        court="International Court",
        source="Official source",
        citation=RAGCitation(
            document_id=document_id,
            citation=f"OFFICIAL-{document_id}",
            source="Official source",
            source_url=f"https://example.test/{document_id}",
            provenance="official_court_source",
            chunk_ids=[chunk_id],
            pages=[page],
        ),
        chunk_ids=[chunk_id],
        pages=[page],
        ranks=[rank],
        score=1.0 / rank,
        retrieval_scores=[
            ChunkRetrievalScore(
                chunk_id=chunk_id,
                rank=rank,
                final_score=1.0 / rank,
                hybrid_score=1.0 / rank,
            )
        ],
    )


def _payload(
    *,
    warnings: list[str] | None = None,
    empty: bool = False,
) -> SummarizationReadyPayload:
    passages = [] if empty else [
        _passage("CTX-001", "DOC-A", 1),
        _passage("CTX-002", "DOC-B", 2),
    ]
    citation_map = {
        passage.context_id: ContextCitationMapEntry(
            context_id=passage.context_id,
            document_id=passage.document_id,
            title=passage.title,
            court=passage.court,
            source=passage.source,
            chunk_ids=passage.chunk_ids,
            pages=passage.pages,
            official_citation=passage.citation.citation,
            source_url=passage.citation.source_url,
        )
        for passage in passages
    }
    active_warnings = warnings or []
    return SummarizationReadyPayload(
        query="What is the relevant rule?",
        legal_issue="State responsibility",
        context_text="\n\n".join(
            f"[{passage.context_id}] {passage.text}" for passage in passages
        ),
        context_passages=passages,
        citation_map=citation_map,
        retrieval=RetrievalContextMetadata(
            mode="hybrid",
            top_k=2,
            result_count=len(passages),
            bm25_weight=0.8,
            semantic_weight=0.2,
            diversified=True,
            max_chunks_per_document=2,
        ),
        diagnostics=RetrievalContextDiagnostics(
            retrieval_result_count=len(passages),
            rag_passage_count=len(passages),
            source_document_count=len(passages),
            estimated_context_tokens=20 if passages else 0,
            context_truncated=False,
            empty_retrieval=not passages,
            no_context=not passages,
            weak_context_ids=["CTX-002"] if active_warnings else [],
            warnings=active_warnings,
        ),
    )


def _response(
    answer: str,
    citations: list[str],
    *,
    limitations: list[str] | None = None,
) -> dict[str, object]:
    return {
        "answer": answer,
        "citations": citations,
        "confidence": 0.8,
        "limitations": limitations or [],
    }


def test_prompt_contains_all_grounding_sections_and_allowed_ids() -> None:
    request = build_summarization_provider_request(_payload())

    assert "SYSTEM / INSTRUCTIONS" in request.prompt
    assert "USER QUERY\nWhat is the relevant rule?" in request.prompt
    assert "LEGAL ISSUE\nState responsibility" in request.prompt
    assert "RETRIEVED CONTEXT\n[CTX-001]" in request.prompt
    assert "ALLOWED CITATIONS\nCTX-001:" in request.prompt
    assert "CTX-002:" in request.prompt
    assert "Do not invent legal authorities" in request.prompt
    assert "legal advice" in request.prompt
    assert request.citation_map["CTX-001"].document_id == "DOC-A"


def test_valid_answer_and_repeated_citations_pass_without_rewriting() -> None:
    answer = "Rule one [CTX-001]. Supporting detail [CTX-001] [CTX-002]."
    provider = FakeSummarizationProvider(
        _response(answer, ["CTX-001", "CTX-002"])
    )

    result = SummarizationAgent(provider).summarize(_payload())

    assert result.output.answer == answer
    assert result.used_context_ids == ["CTX-001", "CTX-002"]
    assert result.citation_validation.inline.cited_context_ids == [
        "CTX-001",
        "CTX-001",
        "CTX-002",
    ]
    assert result.citation_validation.all_citations_valid is True
    assert result.provider_metadata.provider_called is True


@pytest.mark.parametrize(
    ("answer", "citations", "expected_field", "expected_value"),
    [
        (
            "Unsupported claim [CTX-999].",
            ["CTX-999"],
            "invalid_declared_citations",
            ["CTX-999"],
        ),
        (
            "Supported claim [CTX-001].",
            ["CTX-001", "CTX-999"],
            "invalid_declared_citations",
            ["CTX-999"],
        ),
        (
            "Supported claim [CTX-001].",
            ["CTX-001", "CTX-002"],
            "declared_missing_inline",
            ["CTX-002"],
        ),
        (
            "Two claims [CTX-001] [CTX-002].",
            ["CTX-001"],
            "inline_missing_declared",
            ["CTX-002"],
        ),
    ],
)
def test_unknown_or_inconsistent_citations_fail_without_fallback(
    answer: str,
    citations: list[str],
    expected_field: str,
    expected_value: list[str],
) -> None:
    provider = FakeSummarizationProvider(_response(answer, citations))

    with pytest.raises(SummarizationValidationError) as caught:
        SummarizationAgent(provider).summarize(_payload())

    validation = caught.value.citation_validation
    assert validation is not None
    assert getattr(validation, expected_field) == expected_value
    assert provider.response["answer"] == answer


def test_malformed_inline_citation_is_reported() -> None:
    provider = FakeSummarizationProvider(_response("Claim [CTX-1].", []))

    with pytest.raises(SummarizationValidationError) as caught:
        SummarizationAgent(provider).summarize(_payload())

    assert caught.value.citation_validation is not None
    assert caught.value.citation_validation.inline.malformed_citations == [
        "[CTX-1]"
    ]


def test_answer_without_citations_is_rejected() -> None:
    provider = FakeSummarizationProvider(_response("Uncited claim.", []))

    with pytest.raises(
        SummarizationValidationError,
        match="must cite retrieved context inline",
    ):
        SummarizationAgent(provider).summarize(_payload())


def test_no_context_returns_controlled_answer_without_provider_call() -> None:
    provider = FakeSummarizationProvider(error=AssertionError("must not call"))

    result = SummarizationAgent(provider).summarize(_payload(empty=True))

    assert result.output.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert result.output.citations == []
    assert result.provider_metadata.provider_called is False
    assert provider.calls == []


def test_weak_context_is_in_prompt_and_requires_limitation() -> None:
    warning = "Selected context contains a weak lower-ranked passage"
    payload = _payload(warnings=[warning])
    provider = FakeSummarizationProvider(
        _response("Grounded claim [CTX-001].", ["CTX-001"])
    )

    with pytest.raises(
        SummarizationValidationError,
        match="acknowledge weak context",
    ):
        SummarizationAgent(provider).summarize(payload)

    assert warning in provider.calls[0].prompt

    provider = FakeSummarizationProvider(
        _response(
            "Grounded claim [CTX-001].",
            ["CTX-001"],
            limitations=["The lower-ranked evidence is weak"],
        )
    )
    result = SummarizationAgent(provider).summarize(payload)
    assert result.warnings == [warning]


def test_provider_failure_is_wrapped_without_secret_leakage() -> None:
    provider = FakeSummarizationProvider(
        error=RuntimeError("secret-token-should-not-leak")
    )

    with pytest.raises(SummarizationProviderError) as caught:
        SummarizationAgent(provider).summarize(_payload())

    assert str(caught.value) == "Summarization provider failed"
    assert "secret-token" not in str(caught.value)


def test_empty_answer_is_rejected_as_invalid_structured_output() -> None:
    provider = FakeSummarizationProvider(_response("   ", ["CTX-001"]))

    with pytest.raises(
        SummarizationProviderError,
        match="invalid structured output",
    ):
        SummarizationAgent(provider).summarize(_payload())


def test_provenance_rendering_uses_only_stored_values() -> None:
    payload = _payload()

    rendered = render_citation_provenance("CTX-001", payload.citation_map)

    assert "Document: DOC-A" in rendered
    assert "Title: Case DOC-A" in rendered
    assert "Court: International Court" in rendered
    assert "Pages: 11" in rendered
    assert "Citation: OFFICIAL-DOC-A" in rendered
    assert "URL: https://example.test/DOC-A" in rendered
    assert "not available" not in rendered
    with pytest.raises(UnknownContextCitationError):
        render_citation_provenance("CTX-999", payload.citation_map)
