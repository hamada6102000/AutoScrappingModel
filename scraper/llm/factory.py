"""
Provider factory — resolves the configured LLM provider from Django settings.

    SCRAPER["LLM_PROVIDER"] in {"heuristic", "groq", "anthropic", "openai", "ollama"}

``get_provider`` raises on misconfiguration (used by the evaluation, which must
never silently switch providers). ``get_provider_safe`` falls back to the
heuristic provider for ordinary dashboard / CLI use.
"""

from __future__ import annotations

import logging

from django.conf import settings

from .base import LLMProvider
from .heuristic import HeuristicProvider

logger = logging.getLogger(__name__)

PROVIDERS = ("heuristic", "groq", "anthropic", "openai", "ollama")


def get_provider(name: str | None = None, model: str | None = None) -> LLMProvider:
    cfg = settings.SCRAPER
    name = (name or cfg.get("LLM_PROVIDER") or "heuristic").lower()
    max_chars = cfg.get("LLM_MAX_INPUT_CHARS", 60000)

    if name == "heuristic":
        return HeuristicProvider()

    if name == "groq":
        from .groq_provider import GroqProvider

        return GroqProvider(
            api_key=cfg.get("GROQ_API_KEY", ""),
            model=model or cfg.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            host=cfg.get("GROQ_HOST", "https://api.groq.com/openai/v1"),
            max_retries=cfg.get("LLM_MAX_RETRIES", 6),
            max_wait_seconds=cfg.get("LLM_MAX_WAIT_SECONDS", 900),
            max_input_chars=max_chars,
        )

    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=cfg.get("ANTHROPIC_API_KEY", ""),
            model=model or cfg.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
            max_input_chars=max_chars,
        )

    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=cfg.get("OPENAI_API_KEY", ""),
            model=model or cfg.get("OPENAI_MODEL", "gpt-4o"),
            max_input_chars=max_chars,
        )

    if name == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(
            host=cfg.get("OLLAMA_HOST", "http://localhost:11434"),
            model=model or cfg.get("OLLAMA_MODEL", "llama3.1"),
            max_input_chars=max_chars,
        )

    raise ValueError(f"Unknown LLM provider: {name!r} (choose from {', '.join(PROVIDERS)})")


def get_provider_safe(name: str | None = None, model: str | None = None) -> LLMProvider:
    """Like :func:`get_provider` but falls back to heuristic on any error."""
    try:
        return get_provider(name, model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Falling back to heuristic provider: %s", exc)
        return HeuristicProvider()
