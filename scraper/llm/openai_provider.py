"""OpenAI GPT provider (optional real LLM — proposal GPT-4 class)."""

from __future__ import annotations

import time
from typing import Optional

from .base import FailureHint, FieldSpec, LLMProvider, LLMResult
from .errors import wrap_sdk_error
from .prompts import SYSTEM_PROMPT, build_prompt, parse_xpath_json


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o", max_input_chars: int = 60000):
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Set it in your environment/.env or "
                "switch SCRAPER_LLM_PROVIDER back to 'heuristic'."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=api_key)
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
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise wrap_sdk_error(exc) from exc
        latency_ms = int((time.perf_counter() - start) * 1000)
        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return LLMResult(
            xpaths=parse_xpath_json(text, fields),
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw_response=text,
            latency_ms=latency_ms,
            **meta,
        )
