"""
LLM provider abstraction (proposal Section 1.4 — LLM Analysis Layer).

Every provider takes a *pruned* DOM plus a list of target fields and returns
one XPath expression per field. During self-healing the provider is also given
a "failure-history memory" (recent failed XPaths and why they failed), which it
uses to avoid repeating broken selectors (proposal Self-Healing Layer).

Keeping this behind a single interface is what makes the system model-agnostic:
the heuristic provider runs entirely offline, while Anthropic / OpenAI / Ollama
providers plug into the exact same contract.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FieldSpec:
    """A field the pipeline must extract."""

    name: str
    description: str = ""
    field_type: str = "string"


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

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(abc.ABC):
    """Base class for all XPath-generation providers."""

    #: short identifier stored on LLMCallLog / XPathRule.source
    name: str = "base"

    @abc.abstractmethod
    def generate_xpaths(
        self,
        pruned_html: str,
        fields: list[FieldSpec],
        failure_history: Optional[list[FailureHint]] = None,
    ) -> LLMResult:
        """Return {field_name: xpath} for every field in ``fields``."""
        raise NotImplementedError

    # Convenience so callers can `str(provider)`.
    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"<LLMProvider {self.name}>"
