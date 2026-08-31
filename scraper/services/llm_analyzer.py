"""
Layer 2 — LLM Analysis Layer (proposal Section 1.4).

Thin, timed wrapper around whichever LLM provider is configured. Prompts the
model with the pruned DOM and the target field schema and returns one XPath per
field. Timing is captured here so the pipeline can record regeneration latency
(proposal RQ4 / Table 5.1).
"""

from __future__ import annotations

import time

from ..llm.base import FieldSpec, LLMProvider, LLMResult


def generate(
    provider: LLMProvider,
    pruned_html: str,
    fields: list[FieldSpec],
) -> tuple[LLMResult, int]:
    """Initial XPath generation. Returns (result, latency_ms)."""
    start = time.perf_counter()
    result = provider.generate_xpaths(pruned_html, fields, failure_history=None)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return result, latency_ms
