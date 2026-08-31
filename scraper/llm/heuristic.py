"""
Heuristic (offline) XPath provider.

This provider is a *stand-in for the LLM* so the whole self-healing pipeline
runs end-to-end with zero configuration and no API keys. It inspects the pruned
DOM and locates each field using semantic cues, then emits an XPath.

Why this still demonstrates real self-healing
---------------------------------------------
The provider deliberately separates two phases, mirroring how a brittle
generated selector behaves in practice:

* MATCH phase  — finds the element using *synonym-expanded* tokens
  (e.g. "price" also matches "amount", "cost"). This generalises across
  redesigns, exactly like an LLM reasoning about semantics.
* EMIT phase   — writes the XPath using the *specific* class token observed in
  the current DOM (e.g. ``contains(@class,"product-title")``). This is
  intentionally brittle: when the site is redesigned and that class is renamed,
  the stored XPath stops matching — triggering detection and regeneration.

So a rule generated against DOM v1 breaks on DOM v2, is detected as a failure,
and regeneration against the fresh DOM produces a new working rule. That is the
closed self-healing loop the proposal describes, reproduced deterministically.

When a real LLM provider (Anthropic / OpenAI / Ollama) is configured, it
replaces this class behind the same interface with far stronger generalisation.
"""

from __future__ import annotations

import re
from typing import Optional

from lxml import html as lxml_html

from .base import FailureHint, FieldSpec, LLMResult, LLMProvider


# Synonyms let the matcher survive class renames across a redesign.
FIELD_SYNONYMS: dict[str, list[str]] = {
    "title": ["title", "name", "heading", "headline", "pdp", "productname"],
    "name": ["name", "title", "heading"],
    "price": ["price", "amount", "cost", "money", "pricing"],
    "availability": ["availability", "stock", "instock", "avail", "inventory"],
    "stock": ["stock", "availability", "inventory", "instock"],
    "rating": ["rating", "stars", "score", "rank"],
    "reviews": ["reviews", "review", "ratingcount", "votes"],
    "description": ["description", "desc", "summary", "details", "overview", "about"],
    "brand": ["brand", "manufacturer", "vendor", "maker"],
    "sku": ["sku", "productid", "itemid", "model", "mpn"],
    "author": ["author", "byline", "writer", "creator"],
    "date": ["date", "published", "time", "datetime", "timestamp", "posted"],
    "headline": ["headline", "title", "heading"],
    "category": ["category", "breadcrumb", "section", "department"],
    "company": ["company", "employer", "organization", "org"],
    "location": ["location", "place", "city", "region", "address"],
    "salary": ["salary", "pay", "compensation", "wage"],
    "symbol": ["symbol", "ticker", "code"],
    "change": ["change", "delta", "movement", "percent"],
}

# Tags we prefer for certain field kinds (small nudge in scoring).
PREFERRED_TAGS: dict[str, set[str]] = {
    "title": {"h1", "h2"},
    "name": {"h1", "h2"},
    "headline": {"h1", "h2"},
    "price": {"span", "strong", "b", "p", "div"},
    "date": {"time", "span"},
}

PRICE_RE = re.compile(r"[$€£¥]\s?\d|\d[\d.,]*\s?(usd|eur|gbp|egp)", re.I)
STOPWORDS = {
    "the", "a", "of", "for", "and", "to", "field", "value", "product",
    "item", "page", "data", "text", "its", "with",
}


def _dedupe(tokens: list[str]) -> list[str]:
    seen, ordered = set(), []
    for t in tokens:
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _strong_tokens(spec: FieldSpec) -> list[str]:
    """
    High-confidence tokens: the field name plus curated synonyms. Only these
    can select an element and drive the emitted XPath.
    """
    tokens: list[str] = []
    for part in re.split(r"[^a-z0-9]+", spec.name.lower()):
        if part and part not in STOPWORDS:
            tokens.append(part)
            tokens.extend(FIELD_SYNONYMS.get(part, []))
    return _dedupe(tokens)


def _weak_tokens(spec: FieldSpec, strong: set[str]) -> list[str]:
    """
    Low-confidence hints mined from the free-text description. They add a small
    score nudge but never select an element on their own (avoids matching
    generic container classes like "product-info").
    """
    tokens = []
    for part in re.split(r"[^a-z0-9]+", spec.description.lower()):
        if len(part) > 3 and part not in STOPWORDS and part not in strong:
            tokens.append(part)
    return _dedupe(tokens)


def _class_tokens(el) -> list[str]:
    return [c for c in (el.get("class") or "").split() if c]


def _xpath_escape(value: str) -> str:
    """Return an XPath string literal safe for values containing quotes."""
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    parts = value.split('"')
    return "concat(" + ', \'"\', '.join(f'"{p}"' for p in parts) + ")"


class HeuristicProvider(LLMProvider):
    name = "heuristic"

    def generate_xpaths(
        self,
        pruned_html: str,
        fields: list[FieldSpec],
        failure_history: Optional[list[FailureHint]] = None,
    ) -> LLMResult:
        try:
            tree = lxml_html.fromstring(pruned_html)
        except Exception:
            return LLMResult(xpaths={}, model="heuristic-v1")

        elements = list(tree.iter())
        # XPaths already known to be broken (from failure memory) are avoided.
        banned = {h.failed_xpath for h in (failure_history or [])}

        result: dict[str, str] = {}
        for spec in fields:
            xpath = self._best_xpath(spec, elements, banned)
            if xpath:
                result[spec.name] = xpath

        # Rough token estimate so the LLMCallLog reflects "work done".
        approx = max(1, len(pruned_html) // 4)
        return LLMResult(
            xpaths=result,
            model="heuristic-v1",
            prompt_tokens=approx,
            completion_tokens=len(result) * 12,
            raw_response=repr(result),
        )

    # ------------------------------------------------------------------
    def _best_xpath(self, spec: FieldSpec, elements, banned: set[str]) -> Optional[str]:
        strong = _strong_tokens(spec)
        weak = _weak_tokens(spec, set(strong))
        preferred = PREFERRED_TAGS.get(spec.name.lower(), set())
        best_score, best_xpath = 0.0, None

        for el in elements:
            tag = el.tag if isinstance(el.tag, str) else None
            if tag in (None, "script", "style", "html", "body"):
                continue

            text = (el.text_content() or "").strip()
            classes = _class_tokens(el)
            el_id = el.get("id") or ""
            itemprop = el.get("itemprop") or ""
            child_count = sum(1 for c in el if isinstance(c.tag, str))

            score = 0.0
            matched_class: Optional[str] = None

            # 1) microdata itemprop — strongest semantic signal.
            if itemprop and any(t in itemprop.lower() for t in strong):
                score += 6.0

            # 2) strong class-token match — the primary, intentionally brittle
            #    signal that drives the emitted (rename-fragile) XPath.
            for cls in classes:
                low = cls.lower()
                if any(t in low for t in strong):
                    score += 5.0
                    matched_class = cls
                    break

            # 3) weak (description) hints — a small nudge only, never selecting.
            if any(any(t in c.lower() for t in weak) for c in classes):
                score += 1.0

            # 4) id match.
            if el_id and any(t in el_id.lower() for t in strong):
                score += 3.0

            # 5) tag preference.
            if tag in preferred:
                score += 1.5

            # 6) type-specific text pattern (price).
            if spec.field_type == "price" and PRICE_RE.search(text):
                score += 2.0

            # 7) content shape: reward small leaf-ish elements, penalise big
            #    wrapper/container elements (fewer false positives on parents).
            if not text:
                score -= 3.0
            elif len(text) < 200:
                score += 0.5
            elif len(text) > 400:
                score -= 1.0
            score -= 0.4 * min(child_count, 5)

            if score <= best_score:
                continue

            xpath = self._emit(tag, matched_class, itemprop, el_id, strong)
            if xpath and xpath not in banned:
                best_score, best_xpath = score, xpath

        # Require at least a class / itemprop / id level signal (>= 4.0);
        # weak text-only matches are discarded so failures stay honest.
        return best_xpath if best_score >= 4.0 else None

    # ------------------------------------------------------------------
    def _emit(
        self,
        tag: Optional[str],
        matched_class: Optional[str],
        itemprop: str,
        el_id: str,
        tokens: list[str],
    ) -> Optional[str]:
        """
        Emit a *specific* XPath for the located element.

        Preference order favours the class token (brittle, realistic) so that a
        later redesign breaks it and forces regeneration.
        """
        t = tag or "*"
        if matched_class:
            return f'//{t}[contains(concat(" ", normalize-space(@class), " "), " {matched_class} ")]'
        if itemprop and any(tok in itemprop.lower() for tok in tokens):
            return f'//{t}[@itemprop={_xpath_escape(itemprop)}]'
        if el_id and any(tok in el_id.lower() for tok in tokens):
            return f'//*[@id={_xpath_escape(el_id)}]'
        return None
