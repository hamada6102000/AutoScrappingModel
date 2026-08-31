"""
Groq provider (free, fast LLaMA inference — proposal LLaMA class).

Groq's Cloud API is OpenAI-compatible, so this talks to it with plain ``requests``
(no extra SDK needed). Get a free API key at https://console.groq.com/keys and
set GROQ_API_KEY. Free-tier models include:

    * llama-3.3-70b-versatile   (default; strongest reasoning)
    * llama-3.1-8b-instant      (fastest / lightest on rate limits)

The free tier is rate-limited but has no cost.
"""

from __future__ import annotations

from typing import Optional

import requests

from .base import FailureHint, FieldSpec, LLMProvider, LLMResult
from .prompts import SYSTEM_PROMPT, build_user_prompt, parse_xpath_json


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        host: str = "https://api.groq.com/openai/v1",
        timeout: int = 120,
    ):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys, or switch "
                "SCRAPER_LLM_PROVIDER back to 'heuristic'."
            )
        self.api_key = api_key
        self.model = model
        self.base = host.rstrip("/")
        self.timeout = timeout

    def generate_xpaths(
        self,
        pruned_html: str,
        fields: list[FieldSpec],
        failure_history: Optional[list[FailureHint]] = None,
    ) -> LLMResult:
        prompt = build_user_prompt(pruned_html, fields, failure_history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        data = self._chat(messages)
        text = data["choices"][0]["message"]["content"] or ""
        xpaths = parse_xpath_json(text, fields)
        usage = data.get("usage", {}) or {}
        return LLMResult(
            xpaths=xpaths,
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
            raw_response=text,
        )

    # ------------------------------------------------------------------
    def _chat(self, messages: list[dict]) -> dict:
        url = f"{self.base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1500,
            # Ask for a JSON object; the prompt already requires JSON output.
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        # Some models reject response_format — retry once without JSON mode.
        if resp.status_code == 400 and "response_format" in resp.text:
            payload.pop("response_format", None)
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
