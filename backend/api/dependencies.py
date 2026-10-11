"""Lazy dependencies and development-only access control for draft APIs."""

from typing import Annotated

from fastapi import Depends, HTTPException, status

from agents.summarization import SummarizationAgent
from agents.summarization.exceptions import ModelUnavailableError
from agents.summarization.gemini import GeminiSummaryGenerator
from backend.config.settings import Settings, get_settings
from backend.services import LegalReportAssembler, MarkdownLegalReportRenderer


def require_unauthenticated_dev_api(
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Refuse unauthenticated draft APIs unless explicitly enabled locally."""

    if (
        settings.app_env.strip().lower() != "development"
        or not settings.enable_unauthenticated_dev_api
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Development-only API access is disabled.",
        )


def get_summarization_agent(
    _access: Annotated[None, Depends(require_unauthenticated_dev_api)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> SummarizationAgent:
    """Create the Gemini dependency lazily when an allowed request needs it."""

    try:
        return SummarizationAgent(
            generator=GeminiSummaryGenerator(
                api_key=settings.llm_api_key,
                model=settings.gemini_model,
                max_output_tokens=settings.gemini_max_output_tokens,
            )
        )
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The summarization model is not configured or available.",
        ) from exc


def get_report_assembler() -> LegalReportAssembler:
    """Provide the deterministic report assembly service."""

    return LegalReportAssembler()


def get_markdown_report_renderer() -> MarkdownLegalReportRenderer:
    """Provide the deterministic Markdown renderer."""

    return MarkdownLegalReportRenderer()
