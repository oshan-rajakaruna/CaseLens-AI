"""Development-only APIs for draft summaries and legal research reports."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from agents.summarization import (
    CaseInformation,
    EmptyModelResponseError,
    GeminiAuthenticationError,
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiTimeoutError,
    InvalidModelResponseError,
    MissingEvidenceError,
    ModelUnavailableError,
    SummarizationAgent,
    SummarizationRequest,
    SummarizationResponse,
    UnsupportedSourceReferenceError,
)
from backend.api.dependencies import (
    get_markdown_report_renderer,
    get_report_assembler,
    get_summarization_agent,
    require_unauthenticated_dev_api,
)
from backend.services import (
    InconsistentCaseInformationError,
    InvalidReportInputError,
    LegalReportAssembler,
    LegalResearchReport,
    MarkdownLegalReportRenderer,
    UnsupportedReportSourceError,
)


draft_router = APIRouter(
    dependencies=[Depends(require_unauthenticated_dev_api)],
)


class DraftReportRequest(BaseModel):
    """Validated inputs needed to assemble a report without another LLM call."""

    model_config = ConfigDict(extra="forbid")

    case: CaseInformation
    source_request: SummarizationRequest
    summarization: SummarizationResponse


class DraftReportResponse(BaseModel):
    """Structured report plus deterministic Markdown preview."""

    model_config = ConfigDict(extra="forbid")

    report: LegalResearchReport
    markdown: str


@draft_router.post(
    "/summarization/draft",
    response_model=SummarizationResponse,
    tags=["summarization"],
)
async def create_draft_summary(
    request: SummarizationRequest,
    agent: Annotated[SummarizationAgent, Depends(get_summarization_agent)],
) -> SummarizationResponse:
    """Create one evidence-grounded draft summary through the configured model."""

    try:
        return await agent.summarize(request)
    except MissingEvidenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one supported evidence source is required.",
        ) from exc
    except UnsupportedSourceReferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Generated output contained an unsupported source reference.",
        ) from exc
    except (ModelUnavailableError, GeminiAuthenticationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The summarization model is not configured or available.",
        ) from exc
    except GeminiRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The summarization provider is temporarily unavailable.",
        ) from exc
    except GeminiTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The summarization provider timed out.",
        ) from exc
    except (
        EmptyModelResponseError,
        InvalidModelResponseError,
        GeminiProviderError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The summarization provider returned an unusable response.",
        ) from exc


@draft_router.post(
    "/reports/draft",
    response_model=DraftReportResponse,
    tags=["reports"],
)
async def create_draft_report(
    request: DraftReportRequest,
    assembler: Annotated[LegalReportAssembler, Depends(get_report_assembler)],
    renderer: Annotated[
        MarkdownLegalReportRenderer, Depends(get_markdown_report_renderer)
    ],
) -> DraftReportResponse:
    """Assemble and render a draft report without invoking a model."""

    try:
        report = assembler.assemble(
            request.case,
            request.source_request,
            request.summarization,
        )
        return DraftReportResponse(report=report, markdown=renderer.render(report))
    except (
        InconsistentCaseInformationError,
        InvalidReportInputError,
        UnsupportedReportSourceError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The report inputs are inconsistent or contain unsupported sources.",
        ) from exc
