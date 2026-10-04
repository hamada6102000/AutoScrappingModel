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

Request size: prompts are planned to fit the per-minute token limit, but
Groq's own count of a request can come out higher than the plan. When Groq
rejects one request as too large (HTTP 413, "Limit L, Requested N"), that
request is resent with the page shortened by the reported excess plus a 5%
margin (at most three times); the resend is recorded in the call's raw
response (``[request_too_large_resent: ...]``), its retries and its
truncation flag. Later requests keep the original plan.
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
from .prompts import (
    SYSTEM_PROMPT, build_prompt, completion_budget, conservative_counter, parse_xpath_json,
    prompt_token_budget,
)

_WAIT_IN_MSG = re.compile(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s", re.I)
_TOO_LARGE = re.compile(r"Limit\s+(\d+),\s*Requested\s+(\d+)", re.I)


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
        reasoning_effort: Optional[str] = None,
        include_reasoning: Optional[bool] = None,
        tokens_per_minute: Optional[int] = None,
        completion_counts_in_tpm: bool = True,
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
        # Reasoning models (openai/gpt-oss-*): reasoning depth and whether the
        # reasoning text is returned. Sent only when set.
        self.reasoning_effort = reasoning_effort
        self.include_reasoning = include_reasoning
        # Per-minute token limit of the account: every request (prompt, plus
        # max_completion_tokens if the provider counts it) is planned to fit.
        self.tokens_per_minute = tokens_per_minute
        self.completion_counts_in_tpm = completion_counts_in_tpm

    def generate_xpaths(
        self,
        pruned_html: str,
        fields: list[FieldSpec],
        failure_history: Optional[list[FailureHint]] = None,
    ) -> LLMResult:
        max_tokens = completion_budget(len(fields), self.reasoning_effort)
        budget = self.prompt_token_budget(max_tokens)
        shrunk: list[str] = []
        extra_retries = 0
        while True:
            prompt, meta = build_prompt(pruned_html, fields, failure_history, self.max_input_chars,
                                        max_prompt_tokens=budget)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            try:
                data, call = self._chat(messages, max_tokens=max_tokens)
                break
            except RequestTooLarge as exc:
                # The provider's own count of this request exceeded its limit
                # (its tokenizer / accounting differs from our plan). Shorten
                # the page by the reported excess plus a margin and resend;
                # only this request is affected, later ones keep the plan.
                new = self._shrunk_budget(exc, budget, prompt)
                if new is None or len(shrunk) >= self.MAX_SHRINKS:
                    raise
                shrunk.append(f"{exc.requested}>{exc.limit} tokens: prompt budget "
                              f"{budget if budget else 'none'}->{new}")
                budget = new
                extra_retries += 1
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage", {}) or {}
        note = "".join(f"[request_too_large_resent: {s}] " for s in shrunk)
        call["retries"] += extra_retries
        if call.get("json_failed"):
            note += "[json_validate_failed] "
        elif choice.get("finish_reason") == "length":
            note += "[finish_reason=length] "
        return LLMResult(
            xpaths=parse_xpath_json(text, fields),
            model=data.get("model") or self.model,
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
            raw_response=note + text,
            latency_ms=call["latency_ms"],
            rate_limit_wait_ms=call["wait_ms"],
            retries=call["retries"],
            input_chars=meta["input_chars"],
            input_truncated=meta["input_truncated"],
            prompt_sha=meta["prompt_sha"],
        )

    def prompt_token_budget(self, max_tokens: int) -> Optional[int]:
        """Largest prompt (tokens) that keeps one request within the TPM limit."""
        return prompt_token_budget(self.tokens_per_minute, max_tokens, self.completion_counts_in_tpm)

    MAX_SHRINKS = 3
    SHRINK_MARGIN = 0.05     # of the provider's limit, on top of the reported excess

    def _shrunk_budget(self, exc: RequestTooLarge, budget: Optional[int], prompt: str) -> Optional[int]:
        """Prompt budget for resending a request the provider rejected as too
        large, or None when it cannot be shortened (no figures, or the page
        would have to go below the minimum)."""
        if not exc.limit or not exc.requested or exc.requested <= exc.limit:
            return None
        current = conservative_counter()(SYSTEM_PROMPT + prompt)   # same counter as build_prompt
        if budget:
            current = min(current, budget)
        # Covers both a fixed overhead (subtract the excess) and a tokenizer
        # ratio (scale by limit/requested), whichever needs the shorter prompt.
        new = min(current - (exc.requested - exc.limit), int(current * exc.limit / exc.requested))
        new -= int(exc.limit * self.SHRINK_MARGIN)
        return new if new >= 500 else None

    def list_models(self) -> list[str]:
        """Model ids this API key can use (GET /models)."""
        resp = requests.get(f"{self.base}/models", timeout=self.timeout,
                            headers={"Authorization": f"Bearer {self.api_key}"})
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return sorted(m.get("id", "") for m in resp.json().get("data", []) if m.get("active", True))

    def probe_completion_accounting(self, tokens_per_minute: int) -> Optional[bool]:
        """Does the per-minute limit count max_completion_tokens up front?
        Sends a tiny prompt asking for more completion tokens than the limit:
        rejected as too large -> counted; accepted -> not counted."""
        payload = {"model": self.model, "temperature": 0,
                   "messages": [{"role": "user", "content": "Reply with the word ok."}],
                   "max_completion_tokens": tokens_per_minute + 200}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        resp = requests.post(f"{self.base}/chat/completions", json=payload, timeout=self.timeout,
                             headers={"Authorization": f"Bearer {self.api_key}"})
        if resp.status_code == 200:
            return False
        if resp.status_code in (413, 429) and re.search(r"(?i)request too large|requested", resp.text):
            return True
        return None

    # ------------------------------------------------------------------
    def _chat(self, messages: list[dict], max_tokens: int = 1500) -> tuple[dict, dict]:
        url = f"{self.base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.include_reasoning is not None:
            payload["include_reasoning"] = self.include_reasoning
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
                m = _TOO_LARGE.search(resp.text or "")
                raise RequestTooLarge(f"prompt too large for {self.model}: {resp.text[:200]}",
                                      limit=int(m.group(1)) if m else None,
                                      requested=int(m.group(2)) if m else None)
            if status == 400 and "json_validate_failed" in resp.text:
                # The model answered, but not with valid JSON: a model output
                # failure (scored as such), not an infrastructure error.
                try:
                    failed = (resp.json().get("error") or {}).get("failed_generation") or ""
                except ValueError:
                    failed = ""
                return {"choices": [{"message": {"content": failed}, "finish_reason": "error"}],
                        "model": self.model, "usage": {}}, {
                    "latency_ms": latency_ms, "wait_ms": int(waited * 1000),
                    "retries": retries, "json_failed": True}
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
