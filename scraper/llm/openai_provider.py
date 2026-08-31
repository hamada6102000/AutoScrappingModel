"""OpenAI GPT provider (optional real LLM — proposal GPT-4 class)."""

from __future__ import annotations

from typing import Optional

from .base import FailureHint, FieldSpec, LLMProvider, LLMResult
from .prompts import SYSTEM_PROMPT, build_user_prompt, parse_xpath_json


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Set it in your environment/.env or "
                "switch SCRAPER_LLM_PROVIDER back to 'heuristic'."
            )
        try:
            from openai import OpenAI  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self.model = model

    def generate_xpaths(
        self,
        pruned_html: str,
        fields: list[FieldSpec],
        failure_history: Optional[list[FailureHint]] = None,
    ) -> LLMResult:
        prompt = build_user_prompt(pruned_html, fields, failure_history)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content or ""
        xpaths = parse_xpath_json(text, fields)
        usage = getattr(response, "usage", None)
        return LLMResult(
            xpaths=xpaths,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw_response=text,
        )
