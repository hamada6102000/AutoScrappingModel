"""
Layer 4 — Self-Healing Layer (proposal Section 1.4).

On failure detection, re-invokes the LLM with fresh HTML and a failure-history
memory (recent failed XPaths + reasons) to regenerate corrected selectors,
without human intervention.
"""

from __future__ import annotations

import time

from ..llm.base import FailureHint, FieldSpec, LLMProvider, LLMResult
from .extractor import FieldOutcome


def build_failure_hints(outcomes: list[FieldOutcome]) -> list[FailureHint]:
    """Convert failed extraction outcomes into failure-history memory entries."""
    return [
        FailureHint(
            field_name=o.field_name,
            failed_xpath=o.xpath,
            reason=o.reason or "extraction-failed",
        )
        for o in outcomes
        if not o.success
    ]


def regenerate(
    provider: LLMProvider,
    pruned_html: str,
    fields: list[FieldSpec],
    failure_history: list[FailureHint],
) -> tuple[LLMResult, int]:
    """
    Regenerate XPaths given the current DOM and the failure-history memory.
    Returns (result, latency_ms).
    """
    start = time.perf_counter()
    result = provider.generate_xpaths(
        pruned_html, fields, failure_history=failure_history
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    return result, latency_ms
