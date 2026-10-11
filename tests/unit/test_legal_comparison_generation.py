"""Offline tests for opt-in AI legal-comparison draft orchestration."""

from copy import deepcopy

import pytest

from backend.schemas.legal_comparison import (
    FactAuthoritySelection,
    LegalComparisonProviderOutput,
)
from backend.services.legal_comparison import LegalComparisonAssembler
from backend.services.legal_comparison_generation import (
    LegalComparisonDraftService,
    LegalComparisonGenerationError,
)
from tests.unit.summarization.test_verification_bridge import _passage, _payload
from tests.unit.test_legal_report import _analysis


class FakeComparisonProvider:
    provider_name = "fake"
    model_name = "offline-model"

    def __init__(self, output: LegalComparisonProviderOutput | None = None) -> None:
        self.output = output or LegalComparisonProviderOutput(
            comparison_id="CMP-001",
            context_id="CTX-001",
            explanation=(
                "The alleged non-payment may relate to the selected authority, "
                "but applicability cannot be determined. [CTX-001]"
            ),
            uncertainties=["The fact is not independently verified."],
            ai_generated=True,
            human_review_required=True,
        )
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return self.output


def _inputs():
    analysis = _analysis()
    payload = _payload(_passage())
    record = LegalComparisonAssembler().build(
        [analysis],
        payload,
        [
            FactAuthoritySelection(
                analysis_document_id="PRIVATE-DOC-1",
                fact_index=0,
                context_id="CTX-001",
            )
        ],
    )[0]
    return analysis, payload, record


def test_success_uses_only_explicit_pair_and_attaches_unverified_draft() -> None:
    analysis, payload, record = _inputs()
    provider = FakeComparisonProvider()

    result = LegalComparisonDraftService(provider).generate(
        [record], [analysis], payload
    )[0]

    assert len(provider.calls) == 1
    request = provider.calls[0]
    assert request.comparison_id == record.id
    assert request.fact_statement == analysis.facts[0].statement
    assert request.context_id == "CTX-001"
    assert request.authority_passage == payload.context_passages[0].text
    assert request.evidence.document_id == "PRIVATE-DOC-1"
    assert request.authority.document_id == "DOC-A"
    assert "excerpt" not in type(request.evidence).model_fields
    assert result.ai_draft.verification_status == "not_verified"
    assert result.review_status == "requires_human_review"
    assert result.ai_draft.provider == "fake"


def test_generation_does_not_mutate_deterministic_record_or_inputs() -> None:
    analysis, payload, record = _inputs()
    before = deepcopy(
        (analysis.model_dump(), payload.model_dump(), record.model_dump())
    )

    generated = LegalComparisonDraftService(FakeComparisonProvider()).generate(
        [record], [analysis], payload
    )

    assert record.ai_draft is None
    assert generated[0].fact == record.fact
    assert generated[0].authority == record.authority
    assert before == (
        analysis.model_dump(),
        payload.model_dump(),
        record.model_dump(),
    )


def test_empty_collection_makes_no_provider_request() -> None:
    provider = FakeComparisonProvider()

    assert LegalComparisonDraftService(provider).generate([], [], _payload()) == []
    assert provider.calls == []


@pytest.mark.parametrize("field", ["statement", "document_id", "page", "locator"])
def test_changed_fact_identity_is_rejected(field: str) -> None:
    analysis, payload, record = _inputs()
    if field == "statement":
        analysis.facts[0].statement = "Changed fact."
    elif field == "document_id":
        analysis.facts[0].source.document_id = "OTHER-PRIVATE-DOC"
    elif field == "page":
        analysis.facts[0].source.page = 99
    else:
        analysis.facts[0].source.locator = "page:99"
    provider = FakeComparisonProvider()

    with pytest.raises(LegalComparisonGenerationError, match="fact no longer"):
        LegalComparisonDraftService(provider).generate(
            [record], [analysis], payload
        )

    assert provider.calls == []


def test_changed_authority_provenance_is_rejected() -> None:
    analysis, payload, record = _inputs()
    payload.context_passages[0].citation.document_id = "OTHER-DOC"

    with pytest.raises(LegalComparisonGenerationError, match="provenance"):
        LegalComparisonDraftService(FakeComparisonProvider()).generate(
            [record], [analysis], payload
        )


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (
            LegalComparisonProviderOutput(
                comparison_id="CMP-999",
                context_id="CTX-001",
                explanation="Draft.",
                uncertainties=["Review."],
                ai_generated=True,
                human_review_required=True,
            ),
            "comparison identifier",
        ),
        (
            LegalComparisonProviderOutput(
                comparison_id="CMP-001",
                context_id="CTX-999",
                explanation="Draft.",
                uncertainties=["Review."],
                ai_generated=True,
                human_review_required=True,
            ),
            "CTX identifier",
        ),
        (
            LegalComparisonProviderOutput(
                comparison_id="CMP-001",
                context_id="CTX-001",
                explanation="Invented authority [CTX-999].",
                uncertainties=["Review."],
                ai_generated=True,
                human_review_required=True,
            ),
            "CTX identifier",
        ),
        (
            LegalComparisonProviderOutput(
                comparison_id="CMP-001",
                context_id="CTX-001",
                explanation="Invented citation [CASE-777].",
                uncertainties=["Review."],
                ai_generated=True,
                human_review_required=True,
            ),
            "fabricated citation token",
        ),
    ],
)
def test_inconsistent_or_fabricated_provider_references_are_rejected(
    output: LegalComparisonProviderOutput,
    message: str,
) -> None:
    analysis, payload, record = _inputs()

    with pytest.raises(LegalComparisonGenerationError, match=message):
        LegalComparisonDraftService(FakeComparisonProvider(output)).generate(
            [record], [analysis], payload
        )


def test_provider_failure_is_sanitized() -> None:
    analysis, payload, record = _inputs()

    class FailingProvider(FakeComparisonProvider):
        def generate(self, request):
            raise RuntimeError("PRIVATE FACT AND SECRET PROMPT")

    with pytest.raises(LegalComparisonGenerationError) as caught:
        LegalComparisonDraftService(FailingProvider()).generate(
            [record], [analysis], payload
        )

    assert "PRIVATE FACT" not in str(caught.value)
    assert "SECRET PROMPT" not in str(caught.value)


def test_provider_failure_category_survives_orchestration() -> None:
    analysis, payload, record = _inputs()

    class CategorizedFailure(RuntimeError):
        category = "provider_invalid_request"

    class FailingProvider(FakeComparisonProvider):
        def generate(self, request):
            raise CategorizedFailure("PRIVATE PROVIDER BODY")

    with pytest.raises(LegalComparisonGenerationError) as caught:
        LegalComparisonDraftService(FailingProvider()).generate(
            [record], [analysis], payload
        )

    assert caught.value.category == "provider_invalid_request"
    assert "provider_invalid_request" in str(caught.value)
    assert "PRIVATE PROVIDER BODY" not in str(caught.value)
