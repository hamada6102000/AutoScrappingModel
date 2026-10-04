"""
Ollama / local LLaMA provider (optional, free — proposal LLaMA class).

Talks to a locally running Ollama server (default http://localhost:11434).
Install Ollama and pull a model first, e.g.:  ollama pull llama3.1
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from .base import FailureHint, FieldSpec, LLMError, LLMProvider, LLMResult
from .prompts import SYSTEM_PROMPT, build_prompt, parse_xpath_json


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.1",
                 max_input_chars: int = 60000):
        self.host = host.rstrip("/")
        self.model = model
        self.max_input_chars = max_input_chars

    def generate_xpaths(
        self,
        pruned_html: str,
        fields: list[FieldSpec],
        failure_history: Optional[list[FailureHint]] = None,
    ) -> LLMResult:
        prompt, meta = build_prompt(pruned_html, fields, failure_history, self.max_input_chars)
        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        start = time.perf_counter()
        try:
            resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=300)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"ollama error: {exc}"[:300]) from exc
        latency_ms = int((time.perf_counter() - start) * 1000)
        data = resp.json()
        text = data.get("response", "")
        return LLMResult(
            xpaths=parse_xpath_json(text, fields),
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            completion_tokens=data.get("eval_count", 0) or 0,
            raw_response=text,
            latency_ms=latency_ms,
            **meta,
        )
