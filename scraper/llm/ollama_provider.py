"""
Ollama / local LLaMA provider (optional, free — proposal LLaMA class).

Talks to a locally running Ollama server (default http://localhost:11434).
Install Ollama and pull a model first, e.g.:  ollama pull llama3.1
"""

from __future__ import annotations

from typing import Optional

import requests

from .base import FailureHint, FieldSpec, LLMProvider, LLMResult
from .prompts import SYSTEM_PROMPT, build_user_prompt, parse_xpath_json


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.1"):
        self.host = host.rstrip("/")
        self.model = model

    def generate_xpaths(
        self,
        pruned_html: str,
        fields: list[FieldSpec],
        failure_history: Optional[list[FailureHint]] = None,
    ) -> LLMResult:
        prompt = build_user_prompt(pruned_html, fields, failure_history)
        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        xpaths = parse_xpath_json(text, fields)
        return LLMResult(
            xpaths=xpaths,
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            completion_tokens=data.get("eval_count", 0) or 0,
            raw_response=text,
        )
