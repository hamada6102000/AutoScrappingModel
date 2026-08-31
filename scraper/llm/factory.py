"""
Provider factory — resolves the configured LLM provider from Django settings.

    SCRAPER["LLM_PROVIDER"] in {"heuristic", "anthropic", "openai", "ollama"}

Falls back to the heuristic provider (never crashes the pipeline) if a real
provider is misconfigured, and records the reason.
"""

from __future__ import annotations

import logging

from django.conf import settings

from .base import LLMProvider
from .heuristic import HeuristicProvider

logger = logging.getLogger(__name__)


def get_provider(name: str | None = None) -> LLMProvider:
    cfg = settings.SCRAPER
    name = (name or cfg.get("LLM_PROVIDER") or "heuristic").lower()

    if name == "heuristic":
        return HeuristicProvider()

    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=cfg.get("ANTHROPIC_API_KEY", ""),
            model=cfg.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        )

    if name == "groq":
        from .groq_provider import GroqProvider

        return GroqProvider(
            api_key=cfg.get("GROQ_API_KEY", ""),
            model=cfg.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            host=cfg.get("GROQ_HOST", "https://api.groq.com/openai/v1"),
        )

    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=cfg.get("OPENAI_API_KEY", ""),
            model=cfg.get("OPENAI_MODEL", "gpt-4o"),
        )

    if name == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(
            host=cfg.get("OLLAMA_HOST", "http://localhost:11434"),
            model=cfg.get("OLLAMA_MODEL", "llama3.1"),
        )

    raise ValueError(f"Unknown LLM provider: {name!r}")


def get_provider_safe(name: str | None = None) -> LLMProvider:
    """Like :func:`get_provider` but falls back to heuristic on any error."""
    try:
        return get_provider(name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Falling back to heuristic provider: %s", exc)
        return HeuristicProvider()
