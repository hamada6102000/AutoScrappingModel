"""
Value normalization (proposal Section 4.3, step 4 — Value Normalization).

Strips whitespace, decodes HTML entities, unifies numeric/price formats, and
standardises dates to ISO 8601 (best-effort, dependency-free).
"""

from __future__ import annotations

import html
import re
from datetime import datetime

_WS_RE = re.compile(r"\s+")
_PRICE_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y",
    "%d %B %Y", "%d %b %Y", "%Y/%m/%d", "%d-%m-%Y",
]


def _collapse(value: str) -> str:
    return _WS_RE.sub(" ", html.unescape(value or "")).strip()


def normalize(value: str, field_type: str = "string") -> str:
    """Normalise ``value`` according to its declared field type."""
    text = _collapse(value)
    if not text:
        return ""

    if field_type == "price":
        match = _PRICE_RE.search(text.replace(",", ""))
        return match.group(0) if match else text

    if field_type == "number":
        match = _PRICE_RE.search(text.replace(",", ""))
        return match.group(0) if match else text

    if field_type == "date":
        return _normalize_date(text)

    return text


def _normalize_date(text: str) -> str:
    candidate = text.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).date().isoformat()
        except ValueError:
            continue
    # Try to salvage an ISO-like prefix (e.g. "2026-01-15T10:00").
    iso = re.match(r"(\d{4}-\d{2}-\d{2})", candidate)
    return iso.group(1) if iso else candidate
