"""Development-only API for grounded summarization and verification."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from agents.summarization import (
    GeminiSummarizationConfigurationError,
    SummarizationProviderError,
    SummarizationVerificationBridgeError,
    SummaryNotVerifiableError,
    SummaryProvenanceError,
    VerificationExecutionError,
)
from backend.auth.dependencies import require_dev_summarization_access
from backend.schemas.summarization import (
    SummarizationVerifyRequest,
    SummarizationVerifyResponse,
)
from backend.services.summarization_service import (
    SummarizationService,
    get_summarization_service,
)

router = APIRouter(prefix="/summarization", tags=["summarization"])


def get_api_summarization_service() -> SummarizationService:
    """Resolve the configured service while sanitizing configuration failures."""

    try:
        return get_summarization_service()
    except GeminiSummarizationConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Summarization provider is unavailable",
        ) from exc


@router.post("/verify", response_model=SummarizationVerifyResponse)
def summarize_and_verify(
    request: SummarizationVerifyRequest,
    _access: Annotated[None, Depends(require_dev_summarization_access)],
    service: Annotated[
        SummarizationService,
        Depends(get_api_summarization_service),
    ],
) -> SummarizationVerifyResponse:
    """Generate and verify a summary from caller-supplied assembled context."""

    try:
        return service.summarize_and_verify(request)
    except SummarizationProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Summarization provider is unavailable",
        ) from exc
    except (SummaryNotVerifiableError, SummaryProvenanceError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Summary could not be verified from the supplied context",
        ) from exc
    except VerificationExecutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Summary verification failed",
        ) from exc
    except SummarizationVerificationBridgeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Summary could not be verified",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Summarization workflow failed",
        ) from exc
