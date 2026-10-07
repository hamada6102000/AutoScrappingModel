"""
Provider factory — resolves the configured LLM provider from Django settings.

    SCRAPER["LLM_PROVIDER"] in {"heuristic", "groq", "anthropic", "claude_code", "openai", "ollama"}

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

PROVIDERS = ("heuristic", "rule_based", "groq", "anthropic", "claude_code", "openai", "ollama")


GROQ_OPTIONS = ("reasoning_effort", "include_reasoning", "tokens_per_minute",
                "completion_counts_in_tpm")
ANTHROPIC_OPTIONS = ("reasoning_effort", "tokens_per_minute", "completion_counts_in_tpm")


def get_provider(name: str | None = None, model: str | None = None, **options) -> LLMProvider:
    """``options`` (from the experiment's llm section): max_input_chars and,
    for Groq, reasoning_effort, include_reasoning, tokens_per_minute,
    completion_counts_in_tpm; for Anthropic, reasoning_effort (sent as
    output_config.effort), tokens_per_minute and completion_counts_in_tpm."""
    cfg = settings.SCRAPER
    name = (name or cfg.get("LLM_PROVIDER") or "heuristic").lower()
    max_chars = options.get("max_input_chars") or cfg.get("LLM_MAX_INPUT_CHARS", 60000)

    if name == "heuristic":
        return HeuristicProvider()

    if name == "rule_based":
        from .rule_based import RuleBasedProvider

        return RuleBasedProvider()

    if name == "groq":
        from .groq_provider import GroqProvider

        return GroqProvider(
            api_key=cfg.get("GROQ_API_KEY", ""),
            model=model or cfg.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            host=cfg.get("GROQ_HOST", "https://api.groq.com/openai/v1"),
            max_retries=cfg.get("LLM_MAX_RETRIES", 6),
            max_wait_seconds=cfg.get("LLM_MAX_WAIT_SECONDS", 900),
            max_input_chars=max_chars,
            **{k: options[k] for k in GROQ_OPTIONS if options.get(k) is not None},
        )

    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=cfg.get("ANTHROPIC_API_KEY", ""),
            model=model or cfg.get("ANTHROPIC_MODEL", "claude-opus-5-5"),
            max_retries=cfg.get("LLM_MAX_RETRIES", 6),
            max_wait_seconds=cfg.get("LLM_MAX_WAIT_SECONDS", 900),
            max_input_chars=max_chars,
            **{k: options[k] for k in ANTHROPIC_OPTIONS if options.get(k) is not None},
        )

    if name == "claude_code":
        # Claude subscription via the installed Claude Code CLI (its own OAuth login).
        from .claude_code_provider import ClaudeCodeProvider, resolve_cli

        return ClaudeCodeProvider(
            model=model or cfg.get("CLAUDE_CODE_MODEL", "claude-opus-5-5"),
            cli=resolve_cli(cfg.get("CLAUDE_CODE_CLI") or None),
            max_retries=cfg.get("LLM_MAX_RETRIES", 6),
            max_wait_seconds=cfg.get("LLM_MAX_WAIT_SECONDS", 900),
            max_input_chars=max_chars,
            **{k: options[k] for k in ANTHROPIC_OPTIONS if options.get(k) is not None},
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
