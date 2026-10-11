"""Offline tests for the summarization application service."""

import pytest

from agents.summarization import SummarizationAgent, SummarizationVerificationBridge
from agents.verification import VerificationAgent
from backend.schemas.summarization import SummarizationVerifyRequest
from backend.services.summarization_service import SummarizationService
from tests.unit.summarization._fakes import FakeSummarizationProvider
from tests.unit.summarization.test_verification_bridge import _passage, _payload


def _service(answer: str) -> SummarizationService:
    provider = FakeSummarizationProvider(
        {
            "answer": answer,
            "citations": ["CTX-001"],
            "confidence": 0.7,
            "limitations": ["Limited to supplied evidence"],
        }
    )
    return SummarizationService(
        SummarizationAgent(provider),
        SummarizationVerificationBridge(VerificationAgent()),
    )


@pytest.mark.parametrize(
    "evidence_text,expected",
    [
        ("The appeal was allowed.", "supported"),
        ("The appeal was not allowed.", "unsupported"),
        ("The court heard argument.", "uncertain"),
    ],
)
def test_service_preserves_real_verification_outcomes(
    evidence_text: str,
    expected: str,
) -> None:
    request = SummarizationVerifyRequest(
        task_id="task-1",
        payload=_payload(_passage(text=evidence_text)),
    )

    response = _service("The appeal was allowed. [CTX-001]").summarize_and_verify(
        request
    )

    assert response.summary.output.citations == ["CTX-001"]
    assert response.verification.overall_status == expected
    assert response.verification.verified is (expected == "supported")
    assert response.provenance["CTX-001"].document_id == "DOC-A"
    assert response.bridge_warnings


def test_service_response_omits_internal_evidence_bearing_request() -> None:
    evidence_text = "PRIVATE-EVIDENCE-SENTINEL"
    response = _service("A limited conclusion [CTX-001].").summarize_and_verify(
        SummarizationVerifyRequest(
            task_id="task-1",
            payload=_payload(_passage(text=evidence_text)),
        )
    )

    serialized = response.model_dump_json()
    assert "verification_request" not in serialized
    assert evidence_text not in serialized

