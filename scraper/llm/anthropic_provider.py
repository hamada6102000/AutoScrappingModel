"""
Anthropic / Claude provider (optional real LLM).

Requires an Anthropic API key (ANTHROPIC_API_KEY). Note: this is the Anthropic
*API*, billed via console.anthropic.com — it is separate from a claude.ai
Pro/Team/Enterprise chat subscription, which cannot authenticate API calls.
Set ANTHROPIC_MODEL to a model ID that is currently available on your account.
"""

from __future__ import annotations

import time
from typing import Optional

from .base import FailureHint, FieldSpec, LLMProvider, LLMResult
from .errors import wrap_sdk_error
from .prompts import SYSTEM_PROMPT, build_prompt, parse_xpath_json


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-latest",
                 max_input_chars: int = 60000):
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Set it in your environment/.env "
                "or switch SCRAPER_LLM_PROVIDER back to 'heuristic'."
            )
        try:
            import anthropic  # imported lazily so the dependency is optional
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required for the Claude provider. "
                "Install it with: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_input_chars = max_input_chars

    def generate_xpaths(
        self,
        pruned_html: str,
        fields: list[FieldSpec],
        failure_history: Optional[list[FailureHint]] = None,
    ) -> LLMResult:
        prompt, meta = build_prompt(pruned_html, fields, failure_history, self.max_input_chars)
        start = time.perf_counter()
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise wrap_sdk_error(exc) from exc
        latency_ms = int((time.perf_counter() - start) * 1000)
        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(message, "usage", None)
        return LLMResult(
            xpaths=parse_xpath_json(text, fields),
            model=self.model,
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            raw_response=text,
            latency_ms=latency_ms,
            **meta,
        )
