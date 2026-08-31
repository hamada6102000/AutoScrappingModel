"""
Health monitoring (proposal Section 1.4 — Validation & Change-Detection Layer;
RQ1 — structural divergence metrics).

Computes a continuous health score in [0, 1] from the extraction outcome. The
score combines:

    * XPath failure rate   — fraction of fields whose selector failed.
    * Field-count mismatch — required fields that produced no value.
    * Structural distance  — tree-edit distance vs. the previous DOM (optional
                             corroborating signal).

When the score drops below SCRAPER["HEALTH_THRESHOLD"] the pipeline triggers
self-healing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .extractor import FieldOutcome


@dataclass
class HealthReport:
    score: float
    xpath_failure_rate: float
    field_count_mismatch: int
    fields_total: int
    fields_extracted: int
    structural_distance: float = 0.0

    @property
    def failed_fields(self) -> int:
        return self.fields_total - self.fields_extracted


def compute_health(
    outcomes: list[FieldOutcome],
    required_fields: set[str],
    structural_distance: float = 0.0,
) -> HealthReport:
    total = len(outcomes)
    if total == 0:
        return HealthReport(0.0, 1.0, 0, 0, 0, structural_distance)

    extracted = sum(1 for o in outcomes if o.success)
    failure_rate = 1.0 - (extracted / total)

    # Required fields that failed weigh more heavily.
    required_failures = sum(
        1 for o in outcomes if not o.success and o.field_name in required_fields
    )
    required_total = max(1, len(required_fields))
    required_failure_rate = required_failures / required_total

    # Weighted health: required-field success dominates, overall success and
    # structural stability contribute the remainder.
    score = (
        0.6 * (1.0 - required_failure_rate)
        + 0.3 * (1.0 - failure_rate)
        + 0.1 * (1.0 - min(1.0, structural_distance))
    )
    score = max(0.0, min(1.0, score))

    return HealthReport(
        score=round(score, 4),
        xpath_failure_rate=round(failure_rate, 4),
        field_count_mismatch=required_failures,
        fields_total=total,
        fields_extracted=extracted,
        structural_distance=round(structural_distance, 4),
    )
