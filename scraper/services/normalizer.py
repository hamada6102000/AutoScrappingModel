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
from urllib.parse import urljoin, urlsplit, urlunsplit

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
# How an ambiguous "10/12/2026" is read: "dmy" (default; Arabic/European
# pages) or "mdy" (US pages). Set per experiment via set_date_order().
DATE_ORDER = "dmy"


def set_date_order(order: str) -> None:
    global DATE_ORDER
    if order not in ("dmy", "mdy"):
        raise ValueError("date order must be 'dmy' or 'mdy'")
    DATE_ORDER = order


def _date_formats() -> list[str]:
    if DATE_ORDER == "mdy":
        return ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"] + _DATE_FORMATS[3:]
    return _DATE_FORMATS


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


# Spelled-out numbers ("star-rating Three").
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_WORD_RE = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.I)
TRUE_WORDS = {"true", "yes", "y", "1", "on", "\u2713", "\u2714"}
FALSE_WORDS = {"false", "no", "n", "0", "off", "none", "\u2717", "\u2718"}


def number_word(text: str) -> str | None:
    m = _WORD_RE.search(text or "")
    return str(_NUMBER_WORDS[m.group(1).lower()]) if m else None


def parse_bool(text: str) -> str | None:
    """'true' / 'false' for yes/no-like text, else None."""
    t = clean_text(text).lower().strip(" .:!")
    if t in TRUE_WORDS:
        return "true"
    if t in FALSE_WORDS:
        return "false"
    return None


def resolve_url(value: str, base_url: str | None) -> str:
    """Make a relative URL absolute against the page URL (href/src values)."""
    v = clean_text(value)
    if not v or not base_url or v.startswith(("data:", "javascript:", "mailto:", "tel:")):
        return v
    return urljoin(base_url, v)


def canonical_url(value: str) -> str:
    """Scheme-, case- and trailing-slash-insensitive form of a URL."""
    v = clean_text(value)
    parts = urlsplit(v)
    if not parts.netloc:
        return v.rstrip("/").lower()
    path = parts.path.rstrip("/")
    return urlunsplit(("", parts.netloc.lower(), path, parts.query, "")).lstrip("/")


def normalize(value: str, field_type: str = "string") -> str:
    """Normalise ``value`` according to its declared field type."""
    text = clean_text(value)
    if not text:
        return ""
    if field_type in ("price", "number"):
        number = parse_number(text)
        if number is None:
            number = number_word(text)
        return number if number is not None else text
    if field_type == "date":
        return _normalize_date(text)
    if field_type == "boolean":
        return parse_bool(text) or text
    return text


_DATE_SEARCH = [
    # (regex, strptime format applied to the match)
    (re.compile(r"\b(\d{4}-\d{2}-\d{2})"), "%Y-%m-%d"),
    (re.compile(r"\b(\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]* \d{4})\b", re.I), None),
    (re.compile(r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.? \d{1,2},? \d{4})\b", re.I), None),
    (re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"), None),
]


def _parse_date(candidate: str) -> str | None:
    candidate = re.sub(r"\b([Ss])ept\b", r"\1ep", candidate)
    for fmt in _date_formats():
        try:
            return datetime.strptime(candidate, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _normalize_date(text: str) -> str:
    candidate = text.strip()
    parsed = _parse_date(candidate)
    if parsed:
        return parsed
    # A date inside a longer text ("12 Jun 2017 (v1)", "Updated 2:48 PM, October 3, 2026").
    for rx, fmt in _DATE_SEARCH:
        m = rx.search(candidate)
        if not m:
            continue
        if fmt:
            try:
                return datetime.strptime(m.group(1), fmt).date().isoformat()
            except ValueError:
                continue
        parsed = _parse_date(re.sub(r"\s+", " ", m.group(1)))
        if parsed:
            return parsed
    return candidate
