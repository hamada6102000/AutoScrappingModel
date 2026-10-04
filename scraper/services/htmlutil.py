"""
HTML helpers shared by the fetcher, extractor, mutations and providers.

Centralising parsing avoids subtle inconsistencies (e.g. one module stripping
an XML declaration and another crashing on it).
"""

from __future__ import annotations

import re

from lxml import etree, html as lxml_html

_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>", re.I)
_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([a-zA-Z0-9_\-:]+)""", re.I)
_HEADER_CHARSET = re.compile(r"charset\s*=\s*([a-zA-Z0-9_\-:]+)", re.I)


def parse_html(text: str):
    """Parse an HTML document string; return the root element or None."""
    if not text or not text.strip():
        return None
    text = _XML_DECL.sub("", text, count=1)
    try:
        return lxml_html.document_fromstring(text)
    except (etree.ParserError, ValueError):
        try:
            return lxml_html.fromstring(text)
        except (etree.ParserError, ValueError):
            return None


def to_html(root) -> str:
    return lxml_html.tostring(root, encoding="unicode", method="html")


def decode_html(content: bytes, content_type: str | None = None) -> tuple[str, str]:
    """
    Decode raw response bytes. Order: charset from the Content-Type header,
    then <meta charset>, then UTF-8, then UTF-8 with replacement. (requests
    defaults to ISO-8859-1 for text/html without a charset, which garbles
    UTF-8 pages such as Arabic news sites — so we never rely on it.)
    """
    candidates: list[str] = []
    if content_type:
        m = _HEADER_CHARSET.search(content_type)
        if m:
            candidates.append(m.group(1))
    m = _META_CHARSET.search(content[:8192])
    if m:
        candidates.append(m.group(1).decode("ascii", "ignore"))
    candidates.append("utf-8")
    for enc in candidates:
        try:
            return content.decode(enc), enc.lower()
        except (LookupError, UnicodeDecodeError):
            continue
    return content.decode("utf-8", errors="replace"), "utf-8-replace"


def visible_text(root) -> str:
    """Whitespace-collapsed text of an element tree."""
    if root is None:
        return ""
    return re.sub(r"\s+", " ", root.text_content() or "").strip()


SEARCH_ATTRS = ("content", "href", "src", "datetime", "title", "alt", "aria-label", "class", "value")


def searchable_text(root, base_url: str | None = None) -> str:
    """Element text plus attribute values (meta content, href/src resolved,
    datetime, class ...): everything an XPath could return as a value."""
    if root is None:
        return ""
    from urllib.parse import urljoin

    parts = [visible_text(root)]
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        for a in SEARCH_ATTRS:
            v = el.get(a)
            if v:
                parts.append(urljoin(base_url, v) if base_url and a in ("href", "src") else v)
    return " \n ".join(parts)
