"""
Anthropic / Claude provider (primary real LLM — proposal GPT-4 / LLaMA class).

Requires an Anthropic API key (ANTHROPIC_API_KEY). Note: this is the Anthropic
*API*, billed via console.anthropic.com — it is separate from a claude.ai
Pro/Team/Enterprise chat subscription, which cannot authenticate API calls.
"""

from __future__ import annotations

import time
from typing import Optional

from .base import FailureHint, FieldSpec, LLMProvider, LLMResult
from .prompts import SYSTEM_PROMPT, build_user_prompt, parse_xpath_json


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-latest"):
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Set it in your environment/.env "
                "or switch SCRAPER_LLM_PROVIDER back to 'heuristic'."
            )
        try:
            import anthropic  # noqa: WPS433 (imported lazily so the dep is optional)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required for the Claude provider. "
                "Install it with: pip install anthropic"
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate_xpaths(
        self,
        pruned_html: str,
        fields: list[FieldSpec],
        failure_history: Optional[list[FailureHint]] = None,
    ) -> LLMResult:
        prompt = build_user_prompt(pruned_html, fields, failure_history)
        start = time.perf_counter()
        message = self._client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        _ = time.perf_counter() - start  # latency captured by the caller

        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        xpaths = parse_xpath_json(text, fields)
        usage = getattr(message, "usage", None)
        return LLMResult(
            xpaths=xpaths,
            model=self.model,
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            raw_response=text,
        )
