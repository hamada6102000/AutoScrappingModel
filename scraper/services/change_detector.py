"""
Change detection (proposal Section 1.4 — Change-Detection Layer; RQ1).

Two structural divergence signals are computed between the current DOM and the
DOM of the previous run, both in [0, 1]:

* structural_distance — weighted Jaccard distance between the multisets of
  root-to-node *tag paths* (e.g. ``html/body/main/div/span``). Captures added,
  removed, wrapped or re-nested elements; blind to class renames.
* markup_distance — the same over paths whose segments include class and id
  tokens (e.g. ``div.price-box#buy``). Also captures class/id renames.

Both are linear-time in the number of elements (unlike sequence alignment),
so they stay fast on large real pages. These are proxies for tree-edit
distance; the paper should describe them as such.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass


@dataclass
class ChangeReport:
    structural_distance: float
    markup_distance: float
    changed: bool
    previous_available: bool


def _h(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8", "ignore")).hexdigest()[:12]


def _markup_segment(el) -> str:
    seg = el.tag
    classes = sorted((el.get("class") or "").split())[:3]
    if classes:
        seg += "." + ".".join(classes)
    el_id = el.get("id")
    if el_id:
        seg += "#" + el_id
    return seg


def dom_profiles(root) -> tuple[dict, dict]:
    """Return (tag_profile, markup_profile): hashed path -> count."""
    tag_counts: Counter = Counter()
    markup_counts: Counter = Counter()
    if root is None:
        return {}, {}
    stack = [(root, root.tag if isinstance(root.tag, str) else "root",
              _markup_segment(root) if isinstance(root.tag, str) else "root")]
    while stack:
        el, tpath, mpath = stack.pop()
        for child in el:
            if not isinstance(child.tag, str):
                continue
            tp = f"{tpath}/{child.tag}"
            mp = f"{mpath}/{_markup_segment(child)}"
            tag_counts[_h(tp)] += 1
            markup_counts[_h(mp)] += 1
            stack.append((child, tp, mp))
    return dict(tag_counts), dict(markup_counts)


def weighted_jaccard_distance(p: dict, q: dict) -> float:
    if not p and not q:
        return 0.0
    keys = set(p) | set(q)
    inter = sum(min(p.get(k, 0), q.get(k, 0)) for k in keys)
    union = sum(max(p.get(k, 0), q.get(k, 0)) for k in keys)
    return round(1.0 - (inter / union), 4) if union else 0.0


def detect_change(prev: tuple[dict, dict] | None, curr: tuple[dict, dict],
                  threshold: float = 0.05) -> ChangeReport:
    """
    Compare (tag_profile, markup_profile) pairs. ``prev`` is None when there is
    no earlier DOM for this target.
    """
    if not prev or not (prev[0] or prev[1]):
        return ChangeReport(0.0, 0.0, changed=False, previous_available=False)
    s = weighted_jaccard_distance(prev[0], curr[0])
    m = weighted_jaccard_distance(prev[1], curr[1])
    return ChangeReport(
        structural_distance=s,
        markup_distance=m,
        changed=(s >= threshold or m >= threshold),
        previous_available=True,
    )
