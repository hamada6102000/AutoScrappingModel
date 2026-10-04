"""Map SDK / HTTP exceptions onto the provider error types."""

from __future__ import annotations

from .base import LLMError, RateLimitExhausted, RequestTooLarge


def wrap_sdk_error(exc: Exception) -> LLMError:
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    message = f"{type(exc).__name__}: {exc}"[:300]
    if status == 429:
        return RateLimitExhausted(message)
    if status == 413:
        return RequestTooLarge(message)
    return LLMError(message)
