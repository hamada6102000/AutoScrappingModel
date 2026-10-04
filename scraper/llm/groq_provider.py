"""
Groq provider (free, fast LLaMA inference — proposal LLaMA class).

Groq's Cloud API is OpenAI-compatible, so this talks to it with plain
``requests`` (no extra SDK). Get a free key at https://console.groq.com/keys.

Rate limits: on HTTP 429 the provider waits for ``retry-after`` (or the delay
quoted in the error message) and retries, up to SCRAPER["LLM_MAX_RETRIES"]
times and SCRAPER["LLM_MAX_WAIT_SECONDS"] of total waiting. Waiting time is
reported separately (``rate_limit_wait_ms``) and excluded from latency. If the
limit is still hit, ``RateLimitExhausted`` is raised so the evaluation harness
can stop cleanly and resume later instead of recording a bogus failure.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import requests

from .base import (
    FailureHint, FieldSpec, LLMError, LLMProvider, LLMResult,
    RateLimitExhausted, RequestTooLarge,
)
from .prompts import SYSTEM_PROMPT, build_prompt, parse_xpath_json

_WAIT_IN_MSG = re.compile(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s", re.I)


def _parse_wait_seconds(resp) -> Optional[float]:
    headers = getattr(resp, "headers", {}) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    m = _WAIT_IN_MSG.search(getattr(resp, "text", "") or "")
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2))
    return None


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        host: str = "https://api.groq.com/openai/v1",
        timeout: int = 120,
        max_retries: int = 6,
        max_wait_seconds: int = 900,
        max_input_chars: int = 60000,
        sleep=time.sleep,
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
        self.max_retries = max_retries
        self.max_wait_seconds = max_wait_seconds
        self.max_input_chars = max_input_chars
        self._sleep = sleep
        self.last_rate_limits: dict = {}

    def generate_xpaths(
        self,
        pruned_html: str,
        fields: list[FieldSpec],
        failure_history: Optional[list[FailureHint]] = None,
    ) -> LLMResult:
        prompt, meta = build_prompt(pruned_html, fields, failure_history, self.max_input_chars)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        # Room for one XPath per field (~70 tokens each in JSON), bounded.
        max_tokens = min(4096, 300 + 70 * len(fields))
        data, call = self._chat(messages, max_tokens=max_tokens)
        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {}) or {}
        return LLMResult(
            xpaths=parse_xpath_json(text, fields),
            model=data.get("model") or self.model,
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
            raw_response=text,
            latency_ms=call["latency_ms"],
            rate_limit_wait_ms=call["wait_ms"],
            retries=call["retries"],
            input_chars=meta["input_chars"],
            input_truncated=meta["input_truncated"],
            prompt_sha=meta["prompt_sha"],
        )

    # ------------------------------------------------------------------
    def _chat(self, messages: list[dict], max_tokens: int = 1500) -> tuple[dict, dict]:
        url = f"{self.base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        waited = 0.0
        retries = 0
        attempt = 0
        while True:
            attempt += 1
            start = time.perf_counter()
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                if retries >= self.max_retries:
                    raise LLMError(f"network error: {exc}") from exc
                delay = min(2 ** retries, 30)
                self._sleep(delay)
                waited += delay
                retries += 1
                continue
            latency_ms = int((time.perf_counter() - start) * 1000)
            status = resp.status_code
            limits = {k.lower(): v for k, v in (getattr(resp, "headers", {}) or {}).items()
                      if k.lower().startswith("x-ratelimit")}
            if limits:
                self.last_rate_limits = limits

            if status == 429:
                delay = _parse_wait_seconds(resp) or min(2 ** (retries + 1), 60)
                if retries >= self.max_retries or waited + delay > self.max_wait_seconds:
                    raise RateLimitExhausted(
                        f"Groq rate limit still exceeded after {retries} retries / "
                        f"{int(waited)}s waiting: {resp.text[:300]}",
                        retry_after=delay,
                    )
                self._sleep(delay)
                waited += delay
                retries += 1
                continue
            if status == 413:
                raise RequestTooLarge(f"prompt too large for {self.model}: {resp.text[:200]}")
            if status == 400 and "response_format" in resp.text and "response_format" in payload:
                payload.pop("response_format")  # model without JSON mode
                retries += 1
                continue
            if status >= 500:
                if retries >= self.max_retries:
                    raise LLMError(f"HTTP {status}: {resp.text[:200]}")
                delay = min(2 ** retries, 30)
                self._sleep(delay)
                waited += delay
                retries += 1
                continue
            if status >= 400:
                raise LLMError(f"HTTP {status}: {resp.text[:200]}")
            return resp.json(), {
                "latency_ms": latency_ms,
                "wait_ms": int(waited * 1000),
                "retries": retries,
            }
