"""
Value normalization (proposal Section 4.3, step 4 — Value Normalization).

Strips whitespace, decodes HTML entities, maps Arabic-Indic / Persian digits
to ASCII, unifies numeric/price formats (US "1,149.99", European "1.149,99",
Arabic "١٬١٤٩٫٩٩"), and standardises dates to ISO 8601 where possible.
"""

from __future__ import annotations

import html
import re
from datetime import datetime

_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"[-+]?\d[\d.,'   ]*\d|[-+]?\d")
_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬",
    "01234567890123456789.,",
)
_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y",
    "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y", "%Y/%m/%d", "%d-%m-%Y",
    "%d.%m.%Y", "%b. %d, %Y", "%A, %B %d, %Y",
]


def clean_text(value: str) -> str:
    text = html.unescape(value or "").translate(_DIGITS)
    return _WS_RE.sub(" ", text).strip()


def parse_number(text: str) -> str | None:
    """Return a plain decimal string ('1149.99') or None if no number found."""
    match = _NUM_RE.search(clean_text(text))
    if not match:
        return None
    s = re.sub(r"[\s'  ]", "", match.group(0))
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):      # 1.149,99
            s = s.replace(".", "").replace(",", ".")
        else:                                 # 1,149.99
            s = s.replace(",", "")
    elif "," in s:
        if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", s):   # 1,149 (thousands)
            s = s.replace(",", "")
        else:                                            # 149,99 (decimal)
            s = s.replace(",", ".")
    elif s.count(".") > 1:                    # 1.149.000
        s = s.replace(".", "")
    return s


def normalize(value: str, field_type: str = "string") -> str:
    """Normalise ``value`` according to its declared field type."""
    text = clean_text(value)
    if not text:
        return ""
    if field_type in ("price", "number"):
        number = parse_number(text)
        return number if number is not None else text
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
    iso = re.match(r"(\d{4}-\d{2}-\d{2})", candidate)
    return iso.group(1) if iso else candidate
