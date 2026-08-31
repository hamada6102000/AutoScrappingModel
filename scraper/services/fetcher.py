"""
Layer 1 — Fetcher Layer (proposal Section 1.4).

Retrieves raw HTML and produces a *pruned* DOM suitable for LLM analysis by
removing non-informational content. Implements steps 1-2 of the data-cleaning
pipeline (proposal Section 4.3):

    1. HTML Fetch & Strip — remove <script>, <style>, inline handlers,
       comments, ad containers.
    2. DOM Pruning       — remove non-semantic nodes (nav, footer, sidebars,
       forms) and keep content-bearing elements. Targets a large token
       reduction while preserving all target data fields (AXE-inspired).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import requests
from lxml import etree, html as lxml_html

# Elements that never carry target data — pruned wholesale.
_PRUNE_TAGS = [
    "script", "style", "noscript", "template", "svg", "iframe",
    "nav", "footer", "header", "aside", "form", "button", "input",
    "select", "option", "label",
]

# Attributes that only add noise / tokens.
_STRIP_ATTRS = {"style", "onclick", "onload", "onmouseover", "srcset"}


@dataclass
class FetchResult:
    url: str
    http_status: int
    raw_html: str
    pruned_html: str
    dom_skeleton: str
    raw_html_signature: str
    dom_signature: str
    raw_token_estimate: int
    pruned_token_estimate: int
    error: str = ""

    @property
    def token_reduction(self) -> float:
        if not self.raw_token_estimate:
            return 0.0
        return 1.0 - (self.pruned_token_estimate / self.raw_token_estimate)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


def _token_estimate(text: str) -> int:
    # ~4 chars per token is the usual rough approximation.
    return max(1, len(text) // 4)


def _dom_skeleton(root) -> str:
    """A tag-only fingerprint of document structure (for change detection)."""
    tags = [
        el.tag
        for el in root.iter()
        if isinstance(el.tag, str) and el.tag not in ("html", "body")
    ]
    return ">".join(tags)


def prune_html(raw_html: str) -> tuple[str, str]:
    """
    Clean and prune raw HTML.

    Returns (pruned_html_string, dom_skeleton_string).
    """
    try:
        root = lxml_html.fromstring(raw_html)
    except (etree.ParserError, ValueError):
        # Empty or unparseable document.
        return "", ""

    # Remove HTML comments (step 1: strip comments).
    for comment in root.xpath("//comment()"):
        parent = comment.getparent()
        if parent is not None:
            parent.remove(comment)

    # Drop noise / non-semantic tags entirely (steps 1-2: strip + prune).
    for tag in _PRUNE_TAGS:
        for el in root.findall(f".//{tag}"):
            parent = el.getparent()
            if parent is not None:
                parent.remove(el)

    # Strip noisy / inline-handler attributes.
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        for attr in list(el.attrib):
            if attr in _STRIP_ATTRS or attr.startswith("on"):
                del el.attrib[attr]

    skeleton = _dom_skeleton(root)
    pruned = lxml_html.tostring(root, encoding="unicode", pretty_print=False)
    return pruned, skeleton


def prune_result(url: str, raw_html: str, http_status: int = 200, error: str = "") -> FetchResult:
    """Build a FetchResult from already-obtained HTML (used by tests/demo)."""
    pruned, skeleton = prune_html(raw_html)
    return FetchResult(
        url=url,
        http_status=http_status,
        raw_html=raw_html,
        pruned_html=pruned,
        dom_skeleton=skeleton,
        raw_html_signature=_sha(raw_html),
        dom_signature=_sha(skeleton),
        raw_token_estimate=_token_estimate(raw_html),
        pruned_token_estimate=_token_estimate(pruned),
        error=error,
    )


def fetch(url: str, timeout: int = 20, user_agent: str = "SelfHealingScraper/1.0") -> FetchResult:
    """
    Fetch a URL over HTTP and return a pruned FetchResult.

    Network / HTTP errors are captured on the result rather than raised, so the
    pipeline can record them as a run failure (proposal: distinguish HTTP
    errors from genuine XPath failures — Table 4.1, HTTP Status Code).
    """
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": user_agent},
        )
        raw = resp.text
        result = prune_result(url, raw, http_status=resp.status_code)
        if resp.status_code >= 400:
            result.error = f"HTTP {resp.status_code}"
        return result
    except requests.RequestException as exc:
        return FetchResult(
            url=url,
            http_status=0,
            raw_html="",
            pruned_html="",
            dom_skeleton="",
            raw_html_signature="",
            dom_signature="",
            raw_token_estimate=0,
            pruned_token_estimate=0,
            error=f"fetch error: {exc}",
        )
