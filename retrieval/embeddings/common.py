"""Shared validation, hashing, and bounded retry helpers for embeddings."""

from collections.abc import Callable, Sequence
import hashlib
import math
from numbers import Real
import time
from typing import TypeVar


_T = TypeVar("_T")


def content_hash(prepared_text: str) -> str:
    """Return a stable hash of the exact provider-formatted embedding input."""

    return hashlib.sha256(prepared_text.encode("utf-8")).hexdigest()


def validate_embedding_values(
    values: Sequence[object] | None,
    dimension: int,
    *,
    provider_name: str,
    error_type: type[Exception],
) -> list[float]:
    """Validate dimension, numeric types, and finiteness for one vector."""

    if values is None:
        raise error_type(f"{provider_name} returned no embedding values")
    if len(values) != dimension:
        raise error_type(
            f"{provider_name} embedding dimension does not match configured dimension"
        )

    vector: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise error_type(
                f"{provider_name} embedding contains a non-numeric value"
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise error_type(f"{provider_name} embedding contains a non-finite value")
        vector.append(numeric_value)
    return vector


def is_transient_embedding_error(error: Exception) -> bool:
    """Recognize common provider rate-limit and transient transport failures."""

    code = getattr(error, "status_code", None)
    if code is None:
        code = getattr(error, "code", None)
    if code in {408, 409, 429} or (isinstance(code, int) and 500 <= code <= 599):
        return True
    error_name = type(error).__name__.casefold()
    return isinstance(error, (ConnectionError, TimeoutError)) or any(
        marker in error_name
        for marker in ("timeout", "connect", "connection", "rate", "servererror")
    )


def execute_with_retries(
    operation: Callable[[], _T],
    *,
    max_retries: int,
    retry_base_seconds: float,
    on_attempt: Callable[[], None],
    on_retry: Callable[[], None],
) -> _T:
    """Execute an operation with bounded exponential backoff."""

    for attempt in range(max_retries + 1):
        on_attempt()
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_retries or not is_transient_embedding_error(exc):
                raise
            on_retry()
            time.sleep(retry_base_seconds * (2**attempt))
    raise RuntimeError("unreachable")
