"""
LLM provider abstraction (proposal Section 1.4 — LLM Analysis Layer).

Every provider takes the LLM view of a pruned DOM plus a list of target fields
and returns one XPath expression per field. During self-healing the provider
also receives a "failure-history memory" (recent failed XPaths and why they
failed) so it avoids repeating broken selectors.

Providers report per-call metadata (pure request latency, time spent waiting
on rate limits, retries, input truncation, prompt hash) so the evaluation can
report latency without free-tier throttling mixed in.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional


class LLMError(Exception):
    """A provider call failed (network, HTTP error, bad response)."""


class RateLimitExhausted(LLMError):
    """Rate limit / quota still exceeded after the allowed waiting time.

    The evaluation harness treats this as an infrastructure stop (roll the run
    back and resume later), never as a method failure.
    """


class RequestTooLarge(LLMError):
    """The prompt exceeds the provider's per-request token limit."""


@dataclass
class FieldSpec:
    """A field the pipeline must extract."""

    name: str
    description: str = ""
    field_type: str = "string"
    cardinality: str = "single"


@dataclass
class FailureHint:
    """One entry of the failure-history memory fed back during regeneration."""

    field_name: str
    failed_xpath: str
    reason: str


@dataclass
class LLMResult:
    """Return value of a provider call."""

    xpaths: dict[str, str] = field(default_factory=dict)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: str = ""
    # Call metadata (0 / "" when a provider cannot report it).
    latency_ms: int = 0            # successful request only, excludes waits
    rate_limit_wait_ms: int = 0
    retries: int = 0
    input_chars: int = 0
    input_truncated: bool = False
    prompt_sha: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(abc.ABC):
    """Base class for all XPath-generation providers."""

    #: short identifier stored on LLMCallLog / XPathRule.source
    name: str = "base"
    model: str = ""

    @abc.abstractmethod
    def generate_xpaths(
        self,
        pruned_html: str,
        fields: list[FieldSpec],
        failure_history: Optional[list[FailureHint]] = None,
    ) -> LLMResult:
        """Return {field_name: xpath} for every field in ``fields``."""
        raise NotImplementedError

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"<LLMProvider {self.name}>"
