"""
Anthropic / Claude provider (Claude Messages API over HTTP, no SDK).

Requires an Anthropic API key (ANTHROPIC_API_KEY), billed via the Claude
Console; a claude.ai chat subscription cannot authenticate API calls.

Request shape for the current models (Claude Opus/Sonnet/Haiku 5.5,
documentation read 2026-10-07):

* no ``temperature``/``top_p``: these models reject non-default sampling values;
* thinking is adaptive (always on for Opus 5.5); its depth is steered with
  ``output_config.effort`` (low/medium/high/...), and ``max_tokens`` caps
  thinking plus answer together;
* the answer is constrained to a JSON object with one string per field via
  ``output_config.format`` (json_schema structured output).

Rate limits: on HTTP 429 the provider waits ``retry-after`` seconds and retries
(up to LLM_MAX_RETRIES / LLM_MAX_WAIT_SECONDS), then raises
``RateLimitExhausted`` so the evaluation stops cleanly and resumes later. A
429 caused by the organization's spend cap carries no retry-after and cannot
be waited out: it is raised as a fatal HTTP 429 error. 529 (overloaded) and
5xx answers are retried with backoff. Waiting time is reported separately
(``rate_limit_wait_ms``) and excluded from latency.
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from .base import FailureHint, FieldSpec, LLMError, LLMProvider, LLMResult, RateLimitExhausted, RequestTooLarge
from .prompts import SYSTEM_PROMPT, build_prompt, parse_xpath_json, prompt_token_budget

API_VERSION = "2023-06-01"


def answer_schema(fields: list[FieldSpec]) -> dict:
    """JSON schema of the answer: one XPath string per field (empty if not found)."""
    return {
        "type": "object",
        "properties": {f.name: {"type": "string"} for f in fields},
        "required": [f.name for f in fields],
        "additionalProperties": False,
    }


def completion_tokens_for(n_fields: int) -> int:
    """max_tokens (thinking + answer) for one call. Billing follows actual use,
    so a generous cap costs nothing unless it is reached."""
    return min(32000, 12000 + 150 * max(1, n_fields))


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    # Anthropic counts input and output tokens in separate per-minute limits
    # (ITPM / OTPM), so the answer allowance never reduces the prompt budget.
    completion_counts_in_tpm_documented = False

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-5-5",
        host: str = "https://api.anthropic.com/v1",
        timeout: int = 300,
        max_retries: int = 6,
        max_wait_seconds: int = 900,
        max_input_chars: int = 60000,
        sleep=time.sleep,
        reasoning_effort: Optional[str] = None,
        tokens_per_minute: Optional[int] = None,
        completion_counts_in_tpm: bool = False,
        include_reasoning: Optional[bool] = None,   # accepted for config compatibility; unused
    ):
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Create a key in the Claude Console and put "
                "ANTHROPIC_API_KEY=... in the .env file, or switch the provider back."
            )
        self.api_key = api_key
        self.model = model
        self.base = host.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_wait_seconds = max_wait_seconds
        self.max_input_chars = max_input_chars
        self._sleep = sleep
        self.reasoning_effort = reasoning_effort
        self.tokens_per_minute = tokens_per_minute
        self.completion_counts_in_tpm = completion_counts_in_tpm
        self.last_rate_limits: dict = {}
        self.last_thinking_tokens = 0

    # ------------------------------------------------------------------
    def _headers(self) -> dict:
        return {"x-api-key": self.api_key, "anthropic-version": API_VERSION,
                "content-type": "application/json"}

    def prompt_token_budget(self, max_tokens: int) -> Optional[int]:
        return prompt_token_budget(self.tokens_per_minute, max_tokens, self.completion_counts_in_tpm)

    def generate_xpaths(
        self,
        pruned_html: str,
        fields: list[FieldSpec],
        failure_history: Optional[list[FailureHint]] = None,
    ) -> LLMResult:
        max_tokens = completion_tokens_for(len(fields))
        prompt, meta = build_prompt(pruned_html, fields, failure_history, self.max_input_chars,
                                    max_prompt_tokens=self.prompt_token_budget(max_tokens))
        output_config: dict = {"format": {"type": "json_schema", "schema": answer_schema(fields)}}
        if self.reasoning_effort:
            output_config["effort"] = self.reasoning_effort
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }
        data, call = self._post(payload)
        notes = []
        first_usage = {}
        if data.get("stop_reason") == "max_tokens" and not _text(data).strip():
            # Thinking used the whole allowance before the answer: resend once with twice the cap.
            first_usage = data.get("usage") or {}
            bigger = dict(payload, max_tokens=min(64000, max_tokens * 2))
            notes.append(f"[max_tokens {max_tokens} reached; resent with {bigger['max_tokens']}]")
            data2, call2 = self._post(bigger)
            call = {"latency_ms": call["latency_ms"] + call2["latency_ms"],
                    "wait_ms": call["wait_ms"] + call2["wait_ms"],
                    "retries": call["retries"] + call2["retries"] + 1}
            data = data2
        text = _text(data)
        usage = data.get("usage") or {}
        # No cache_control is sent, so cached input should be 0; any cached tokens
        # are still counted as input (an upper bound on cost) and noted.
        prompt_tokens = _input(usage) + _input(first_usage)
        cached = _cached(usage) + _cached(first_usage)
        completion_tokens = (usage.get("output_tokens") or 0) + (first_usage.get("output_tokens") or 0)
        self.last_thinking_tokens = _thinking(usage) + _thinking(first_usage)
        stop = data.get("stop_reason")
        if stop == "max_tokens":
            notes.append("[finish_reason=length]")
        elif stop == "refusal":
            notes.append("[stop_reason=refusal]")
        if self.last_thinking_tokens:
            notes.append(f"[thinking_tokens={self.last_thinking_tokens}]")
        if cached:
            notes.append(f"[cached_input_tokens={cached}]")
        return LLMResult(
            xpaths=parse_xpath_json(text, fields),
            model=data.get("model") or self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_response=(" ".join(notes) + " " if notes else "") + text,
            latency_ms=call["latency_ms"],
            rate_limit_wait_ms=call["wait_ms"],
            retries=call["retries"],
            input_chars=meta["input_chars"],
            input_truncated=meta["input_truncated"],
            prompt_sha=meta["prompt_sha"],
        )

    def list_models(self) -> list[str]:
        """Model ids this API key can use (GET /v1/models, all pages)."""
        ids, after = [], None
        for _ in range(20):
            params = {"limit": 100}
            if after:
                params["after_id"] = after
            resp = requests.get(f"{self.base}/models", headers=self._headers(), params=params,
                                timeout=self.timeout)
            if resp.status_code >= 400:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            body = resp.json()
            ids += [m.get("id", "") for m in body.get("data", [])]
            if not body.get("has_more"):
                break
            after = body.get("last_id")
        return sorted(i for i in ids if i)

    def reported_tokens_per_minute(self) -> Optional[int]:
        """Input-token per-minute limit reported by the last response."""
        value = self.last_rate_limits.get("anthropic-ratelimit-input-tokens-limit")
        try:
            return int(float(value)) if value is not None else None
        except ValueError:
            return None

    # ------------------------------------------------------------------
    def _post(self, payload: dict) -> tuple[dict, dict]:
        url = f"{self.base}/messages"
        waited = 0.0
        retries = 0
        while True:
            start = time.perf_counter()
            try:
                resp = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
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
            headers = getattr(resp, "headers", {}) or {}
            limits = {k.lower(): v for k, v in headers.items() if k.lower().startswith("anthropic-ratelimit")}
            if limits:
                self.last_rate_limits = limits
            if status == 429:
                body = _json(resp)
                code = ((body.get("error") or {}).get("details") or {}).get("error_code")
                retry_after = _float(headers.get("retry-after") or headers.get("Retry-After"))
                if code == "enforced_spend_limit_reached":
                    raise LLMError("HTTP 429 spend limit reached (raise the limit in the Claude Console): "
                                   f"{resp.text[:200]}")
                delay = retry_after if retry_after is not None else min(2 ** (retries + 1), 60)
                if retries >= self.max_retries or waited + delay > self.max_wait_seconds:
                    raise RateLimitExhausted(
                        f"Anthropic rate limit still exceeded after {retries} retries / {int(waited)}s "
                        f"waiting: {resp.text[:300]}", retry_after=delay)
                self._sleep(delay)
                waited += delay
                retries += 1
                continue
            if status == 413:
                raise RequestTooLarge(f"request too large for {self.model}: {resp.text[:200]}")
            if status == 529 or status >= 500:
                if retries >= self.max_retries:
                    raise LLMError(f"HTTP {status}: {resp.text[:200]}")
                delay = min(2 ** (retries + 1), 60)
                self._sleep(delay)
                waited += delay
                retries += 1
                continue
            if status >= 400:
                raise LLMError(f"HTTP {status}: {resp.text[:300]}")
            return resp.json(), {"latency_ms": latency_ms, "wait_ms": int(waited * 1000), "retries": retries}


def _text(data: dict) -> str:
    return "".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text")


def _cached(usage: dict) -> int:
    usage = usage or {}
    return (usage.get("cache_creation_input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0)


def _input(usage: dict) -> int:
    return ((usage or {}).get("input_tokens") or 0) + _cached(usage)


def _thinking(usage: dict) -> int:
    return ((usage or {}).get("output_tokens_details") or {}).get("thinking_tokens") or 0


def _json(resp) -> dict:
    try:
        return resp.json() or {}
    except ValueError:
        return {}


def _float(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
