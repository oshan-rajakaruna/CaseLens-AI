"""Offline tests for explicit, deterministic fact-to-authority links."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from agents.analysis import AnalysisAgent
from agents.summarization import GeminiSummarizationProvider
from agents.verification import VerificationAgent
from backend.schemas.legal_comparison import FactAuthoritySelection
from backend.services.legal_comparison import (
    LegalComparisonAssembler,
    LegalComparisonValidationError,
)
from tests.unit.summarization.test_verification_bridge import _passage, _payload
from tests.unit.test_legal_report import _analysis


def _selection(
    *,
    document_id: str = "PRIVATE-DOC-1",
    fact_index: int = 0,
    context_id: str = "CTX-001",
) -> FactAuthoritySelection:
    return FactAuthoritySelection(
        analysis_document_id=document_id,
        fact_index=fact_index,
        context_id=context_id,
    )


def test_valid_explicit_pair_preserves_fact_and_authority_provenance() -> None:
    analysis = _analysis()
    payload = _payload(_passage())

    record = LegalComparisonAssembler().build(
        [analysis], payload, [_selection()]
    )[0]

    assert record.id == "CMP-001"
    assert record.fact.statement == analysis.facts[0].statement
    assert record.fact.source == analysis.facts[0].source
    assert record.fact.epistemic_status == "alleged"
    assert record.fact.verification_status == "not_independently_verified"
    assert record.authority.context_id == "CTX-001"
    assert record.authority.document_id == "DOC-A"
    assert record.authority.chunk_ids == payload.citation_map["CTX-001"].chunk_ids
    assert record.authority.pages == [3]
    assert record.authority.provenance == "official_court_source"
    assert record.authority.official_citation == "CIT-DOC-A"
    assert record.authority.source_url == "https://example.test/DOC-A"


def test_private_evidence_and_authority_document_ids_remain_separate() -> None:
    record = LegalComparisonAssembler().build(
        [_analysis()], _payload(_passage()), [_selection()]
    )[0]

    assert record.fact.source.document_id == "PRIVATE-DOC-1"
    assert record.authority.document_id == "DOC-A"
    assert record.fact.source.document_id != record.authority.document_id


def test_empty_selections_need_no_analysis_and_return_empty() -> None:
    assert LegalComparisonAssembler().build([], _payload(_passage()), []) == []


def test_nonempty_selection_requires_analysis() -> None:
    with pytest.raises(
        LegalComparisonValidationError,
        match="Analysis results are required",
    ):
        LegalComparisonAssembler().build([], _payload(_passage()), [_selection()])


def test_unknown_or_ambiguous_analysis_document_is_rejected() -> None:
    payload = _payload(_passage())
    with pytest.raises(LegalComparisonValidationError, match="was not found"):
        LegalComparisonAssembler().build(
            [_analysis()],
            payload,
            [_selection(document_id="UNKNOWN")],
        )
    with pytest.raises(LegalComparisonValidationError, match="ambiguous"):
        LegalComparisonAssembler().build(
            [_analysis(), _analysis()],
            payload,
            [_selection()],
        )


def test_invalid_fact_indices_and_source_identity_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _selection(fact_index=-1)
    with pytest.raises(LegalComparisonValidationError, match="outside"):
        LegalComparisonAssembler().build(
            [_analysis()],
            _payload(_passage()),
            [_selection(fact_index=99)],
        )
    analysis = _analysis()
    analysis.facts[0].source.document_id = "PRIVATE-OTHER"
    with pytest.raises(LegalComparisonValidationError, match="fact source"):
        LegalComparisonAssembler().build(
            [analysis],
            _payload(_passage()),
            [_selection()],
        )


@pytest.mark.parametrize(
    "problem",
    ["unknown", "missing", "missing_map", "ambiguous"],
)
def test_unknown_missing_or_ambiguous_ctx_passage_is_rejected(problem: str) -> None:
    passage = _passage()
    payload = _payload(passage)
    selection = _selection(context_id="CTX-999") if problem == "unknown" else _selection()
    if problem == "missing":
        payload.context_passages = []
    elif problem == "missing_map":
        payload.citation_map = {}
    elif problem == "ambiguous":
        payload.context_passages.append(passage.model_copy(deep=True))

    with pytest.raises(LegalComparisonValidationError):
        LegalComparisonAssembler().build([_analysis()], payload, [selection])


@pytest.mark.parametrize("field", ["document_id", "chunk_ids", "pages"])
def test_citation_map_and_passage_provenance_mismatch_is_rejected(field: str) -> None:
    payload = _payload(_passage())
    replacement = {
        "document_id": "DOC-OTHER",
        "chunk_ids": ["other-chunk"],
        "pages": [99],
    }[field]
    payload.citation_map["CTX-001"] = payload.citation_map[
        "CTX-001"
    ].model_copy(update={field: replacement})

    with pytest.raises(LegalComparisonValidationError, match="provenance"):
        LegalComparisonAssembler().build(
            [_analysis()], payload, [_selection()]
        )


def test_duplicate_selections_are_rejected() -> None:
    with pytest.raises(LegalComparisonValidationError, match="Duplicate"):
        LegalComparisonAssembler().build(
            [_analysis()],
            _payload(_passage()),
            [_selection(), _selection()],
        )


def test_ids_and_order_are_deterministic_and_inputs_are_not_mutated() -> None:
    analysis = _analysis()
    second_analysis = _analysis().model_copy(
        deep=True,
        update={"document_id": "PRIVATE-DOC-2"},
    )
    second_analysis.facts[0].source.document_id = "PRIVATE-DOC-2"
    payload = _payload(
        _passage("CTX-001", "DOC-A", page=3),
        _passage("CTX-002", "DOC-B", page=4),
    )
    selections = [
        _selection(document_id="PRIVATE-DOC-2", context_id="CTX-002"),
        _selection(),
    ]
    before = deepcopy(
        ([analysis.model_dump(), second_analysis.model_dump()], payload.model_dump())
    )

    first = LegalComparisonAssembler().build(
        [analysis, second_analysis], payload, selections
    )
    second = LegalComparisonAssembler().build(
        [analysis, second_analysis], payload, selections
    )

    assert [item.id for item in first] == ["CMP-001", "CMP-002"]
    assert [item.authority.context_id for item in first] == ["CTX-002", "CTX-001"]
    assert first == second
    assert before == (
        [analysis.model_dump(), second_analysis.model_dump()],
        payload.model_dump(),
    )


def test_record_has_neutral_wording_and_mandatory_review_status() -> None:
    record = LegalComparisonAssembler().build(
        [_analysis()], _payload(_passage()), [_selection()]
    )[0]

    assert record.origin == "explicit_selection"
    assert record.review_status == "requires_human_review"
    assert "legal applicability was not assessed" in record.relationship
    assert any("not substantively verified" in item for item in record.limitations)
    assert not hasattr(record, "verified")
    assert not hasattr(record, "supported")
    assert not hasattr(record.authority, "text")


def test_builder_does_not_execute_agents_or_gemini(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("agent or provider execution is forbidden")

    monkeypatch.setattr(AnalysisAgent, "analyze", forbidden)
    monkeypatch.setattr(VerificationAgent, "verify", forbidden)
    monkeypatch.setattr(GeminiSummarizationProvider, "generate", forbidden)

    records = LegalComparisonAssembler().build(
        [_analysis()], _payload(_passage()), [_selection()]
    )

    assert len(records) == 1
