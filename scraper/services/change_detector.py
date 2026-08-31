"""
Change detection (proposal Section 1.4 — Change-Detection Layer; RQ1).

Quantifies how much the DOM structure diverged from the previous snapshot using
a tag-only skeleton comparison (a lightweight tree-edit-distance proxy). A large
structural distance, combined with a drop in health, is strong evidence of a
layout change rather than a transient content update.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass
class ChangeReport:
    structural_distance: float  # 0.0 (identical) .. 1.0 (completely different)
    changed: bool
    previous_available: bool


def structural_distance(prev_skeleton: str, curr_skeleton: str) -> float:
    """1 - similarity ratio between two tag-only DOM skeletons."""
    if not prev_skeleton or not curr_skeleton:
        return 0.0
    prev_tags = prev_skeleton.split(">")
    curr_tags = curr_skeleton.split(">")
    ratio = difflib.SequenceMatcher(None, prev_tags, curr_tags).ratio()
    return round(1.0 - ratio, 4)


def detect_change(
    prev_skeleton: str,
    curr_skeleton: str,
    threshold: float = 0.05,
) -> ChangeReport:
    if not prev_skeleton:
        return ChangeReport(0.0, changed=False, previous_available=False)
    dist = structural_distance(prev_skeleton, curr_skeleton)
    return ChangeReport(
        structural_distance=dist,
        changed=dist >= threshold,
        previous_available=True,
    )
