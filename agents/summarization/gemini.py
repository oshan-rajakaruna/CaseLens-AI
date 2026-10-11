"""Synchronous Gemini provider for evidence-grounded legal summarization."""

from dataclasses import dataclass
import json
import logging
from os import getenv
from time import monotonic
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from agents.summarization.schemas import (
    SummarizationOutput,
    SummarizationProviderRequest,
)

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = 2048
DEFAULT_GEMINI_REQUEST_TIMEOUT_MS = 30_000

logger = logging.getLogger(__name__)


class GeminiSummarizationError(RuntimeError):
    """Base class for sanitized Gemini provider failures."""


class GeminiSummarizationConfigurationError(GeminiSummarizationError):
    """Raised when Gemini generation settings are missing or invalid."""


class GeminiSummarizationAuthenticationError(GeminiSummarizationError):
    """Raised when Gemini rejects the configured credentials."""


class GeminiSummarizationModelError(GeminiSummarizationError):
    """Raised when the configured Gemini model is unavailable."""


class GeminiSummarizationRateLimitError(GeminiSummarizationError):
    """Raised when Gemini quota or rate limits prevent generation."""


class GeminiSummarizationTimeoutError(GeminiSummarizationError):
    """Raised when Gemini generation times out."""


class GeminiSummarizationResponseError(GeminiSummarizationError):
    """Raised when Gemini returns empty or invalid structured output."""


@dataclass(frozen=True, slots=True)
class GeminiSummarizationSettings:
    """Backend-only environment configuration for Gemini generation."""

    api_key: str
    model: str = DEFAULT_GEMINI_MODEL
    max_output_tokens: int = DEFAULT_GEMINI_MAX_OUTPUT_TOKENS
    request_timeout_ms: int = DEFAULT_GEMINI_REQUEST_TIMEOUT_MS

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise GeminiSummarizationConfigurationError(
                "GEMINI_API_KEY is required for Gemini summarization"
            )
        if not isinstance(self.model, str) or not self.model.strip():
            raise GeminiSummarizationConfigurationError(
                "GEMINI_MODEL must contain a model name"
            )
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise GeminiSummarizationConfigurationError(
                "GEMINI_MAX_OUTPUT_TOKENS must be a positive integer"
            )
        if (
            isinstance(self.request_timeout_ms, bool)
            or not isinstance(self.request_timeout_ms, int)
            or self.request_timeout_ms <= 0
        ):
            raise GeminiSummarizationConfigurationError(
                "GEMINI_REQUEST_TIMEOUT_MS must be a positive integer"
            )

    @classmethod
    def from_environment(cls) -> "GeminiSummarizationSettings":
        raw_max_tokens = getenv(
            "GEMINI_MAX_OUTPUT_TOKENS",
            str(DEFAULT_GEMINI_MAX_OUTPUT_TOKENS),
        )
        try:
            max_output_tokens = int(raw_max_tokens)
        except ValueError as exc:
            raise GeminiSummarizationConfigurationError(
                "GEMINI_MAX_OUTPUT_TOKENS must be a positive integer"
            ) from exc
        request_timeout_ms = _positive_integer_from_environment(
            "GEMINI_REQUEST_TIMEOUT_MS",
            DEFAULT_GEMINI_REQUEST_TIMEOUT_MS,
        )
        return cls(
            api_key=getenv("GEMINI_API_KEY", ""),
            model=getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            max_output_tokens=max_output_tokens,
            request_timeout_ms=request_timeout_ms,
        )


class GeminiSummarizationProvider:
    """Generate one schema-constrained summary through the official SDK."""

    provider_name = "gemini"

    def __init__(
        self,
        settings: GeminiSummarizationSettings | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        if settings is None:
            settings = (
                GeminiSummarizationSettings(
                    api_key="injected-client",
                    model=getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
                    max_output_tokens=_max_output_tokens_from_environment(),
                )
                if client is not None
                else GeminiSummarizationSettings.from_environment()
            )
        self.settings = settings
        self.model_name = settings.model.strip()
        self._client = client

    def generate(self, request: SummarizationProviderRequest) -> SummarizationOutput:
        """Return validated structured output without changing agent contracts."""

        started_at = monotonic()
        _log_timing("gemini_generate_content", started_at, "started")
        try:
            response = self._get_client().models.generate_content(
                model=self.model_name,
                contents=request.prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=SummarizationOutput.model_json_schema(),
                    max_output_tokens=self.settings.max_output_tokens,
                ),
            )
        except GeminiSummarizationError:
            _log_timing("gemini_generate_content", started_at, "error")
            raise
        except Exception as exc:
            _log_timing(
                "gemini_generate_content",
                started_at,
                "error",
                type(exc).__name__,
            )
            raise _classified_provider_error(exc) from exc
        _log_timing("gemini_generate_content", started_at, "completed")
        return _parse_response(response)

    def _get_client(self) -> Any:
        if self._client is None:
            started_at = monotonic()
            try:
                self._client = genai.Client(
                    api_key=self.settings.api_key,
                    http_options=types.HttpOptions(
                        timeout=self.settings.request_timeout_ms,
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                )
            except Exception as exc:
                _log_timing(
                    "gemini_client_initialization",
                    started_at,
                    "error",
                    type(exc).__name__,
                )
                raise GeminiSummarizationConfigurationError(
                    "Gemini client could not be configured; check GEMINI_API_KEY"
                ) from exc
            _log_timing("gemini_client_initialization", started_at, "completed")
        return self._client


def _max_output_tokens_from_environment() -> int:
    raw_value = getenv(
        "GEMINI_MAX_OUTPUT_TOKENS",
        str(DEFAULT_GEMINI_MAX_OUTPUT_TOKENS),
    )
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise GeminiSummarizationConfigurationError(
            "GEMINI_MAX_OUTPUT_TOKENS must be a positive integer"
        ) from exc
    if value <= 0:
        raise GeminiSummarizationConfigurationError(
            "GEMINI_MAX_OUTPUT_TOKENS must be a positive integer"
        )
    return value


def _positive_integer_from_environment(name: str, default: int) -> int:
    raw_value = getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise GeminiSummarizationConfigurationError(
            f"{name} must be a positive integer"
        ) from exc
    if value <= 0:
        raise GeminiSummarizationConfigurationError(
            f"{name} must be a positive integer"
        )
    return value


def _log_timing(
    stage: str,
    started_at: float,
    outcome: str,
    exception_type: str | None = None,
) -> None:
    if getenv("SUMMARIZATION_TIMING_DIAGNOSTICS", "false").casefold() != "true":
        return
    logger.info(
        "summarization_stage stage=%s duration_seconds=%.3f outcome=%s exception_type=%s",
        stage,
        monotonic() - started_at,
        outcome,
        exception_type or "none",
    )


def _parse_response(response: Any) -> SummarizationOutput:
    parsed = getattr(response, "parsed", None)
    try:
        if parsed is not None:
            return SummarizationOutput.model_validate(parsed)
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise GeminiSummarizationResponseError(
                "Gemini returned an empty summarization response"
            )
        return SummarizationOutput.model_validate(json.loads(text))
    except GeminiSummarizationResponseError:
        raise
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise GeminiSummarizationResponseError(
            "Gemini returned invalid structured summarization output"
        ) from exc


def _classified_provider_error(error: Exception) -> GeminiSummarizationError:
    """Map provider failures to safe messages without echoing request data."""

    status = getattr(error, "code", None)
    if not isinstance(status, int):
        status = getattr(error, "status_code", None)
    name = type(error).__name__.casefold()
    message = str(error).casefold()
    if status in {401, 403} or "authentication" in name or "unauthorized" in message:
        return GeminiSummarizationAuthenticationError(
            "Gemini authentication failed; check GEMINI_API_KEY"
        )
    if status == 404 or "not found" in message or "model" in name:
        return GeminiSummarizationModelError(
            "Configured Gemini model is unavailable; change GEMINI_MODEL"
        )
    if status == 429 or "rate" in name or "quota" in message:
        return GeminiSummarizationRateLimitError(
            "Gemini rate limit or quota was exceeded"
        )
    if status in {408, 504} or "timeout" in name or "timed out" in message:
        return GeminiSummarizationTimeoutError("Gemini summarization timed out")
    return GeminiSummarizationError("Gemini summarization request failed")
