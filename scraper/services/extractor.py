"""
Extraction + validation (proposal Section 1.4 — Validation Layer;
Section 4.3 step 3 — XPath Validation).

Applies an XPath to the clean DOM and classifies the outcome as a success or a
typed failure: ``xpath-syntax-error``, ``no-match``, ``empty-value``,
``type-mismatch``, ``no-rule`` or ``no-dom``. Supports single-value fields and
list fields (e.g. every price on a category page).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

from .htmlutil import parse_html
from .normalizer import normalize

_DIGIT = re.compile(r"\d")
_WS = re.compile(r"\s+")


@dataclass
class FieldOutcome:
    field_name: str
    xpath: str
    success: bool
    raw_value: str = ""
    normalized_value: str = ""
    values: list = field(default_factory=list)  # list fields (normalised)
    raw_values: list = field(default_factory=list)
    reason: str = ""
    cardinality: str = "single"


def _node_text(node) -> str:
    if isinstance(node, str):  # attribute / text() results (lxml smart strings)
        return node
    if hasattr(node, "text_content"):
        return node.text_content()
    if isinstance(node, (int, float, bool)):
        return str(node)
    return str(node)


def type_ok(value: str, field_type: str) -> bool:
    value = (value or "").strip()
    if not value:
        return False
    if field_type in ("price", "number", "date"):
        return bool(_DIGIT.search(value))
    if field_type == "url":
        return not _WS.search(value)
    return True


def evaluate(tree, xpath: str):
    """Return (nodes, reason). ``nodes`` is a list (possibly empty)."""
    try:
        result = tree.xpath(xpath)
    except (etree.XPathEvalError, etree.XPathSyntaxError, etree.XPathError,
            ValueError, SyntaxError, TypeError):
        return None, "xpath-syntax-error"
    if isinstance(result, (str, int, float, bool)):
        result = [result]
    if not result:
        return [], "no-match"
    return list(result), ""


def extract_one(tree, name: str, field_type: str, cardinality: str, xpath: str) -> FieldOutcome:
    if tree is None:
        return FieldOutcome(name, xpath, False, reason="no-dom", cardinality=cardinality)
    if not xpath:
        return FieldOutcome(name, xpath, False, reason="no-rule", cardinality=cardinality)

    nodes, reason = evaluate(tree, xpath)
    if nodes is None or not nodes:
        return FieldOutcome(name, xpath, False, reason=reason, cardinality=cardinality)

    texts = [_WS.sub(" ", _node_text(n)).strip() for n in nodes]

    if cardinality == "many":
        raw = [t for t in texts if t]
        if not raw:
            return FieldOutcome(name, xpath, False, reason="empty-value", cardinality=cardinality)
        valid = [t for t in raw if type_ok(t, field_type)]
        if not valid:
            return FieldOutcome(name, xpath, False, raw_values=raw,
                                reason="type-mismatch", cardinality=cardinality)
        normalised = [normalize(t, field_type) for t in valid]
        return FieldOutcome(
            field_name=name, xpath=xpath, success=True,
            raw_value=" || ".join(valid)[:5000],
            normalized_value=" || ".join(normalised)[:5000],
            values=normalised, raw_values=valid, cardinality=cardinality,
        )

    raw = texts[0]
    if not raw:
        return FieldOutcome(name, xpath, False, reason="empty-value", cardinality=cardinality)
    if not type_ok(raw, field_type):
        return FieldOutcome(name, xpath, False, raw_value=raw,
                            reason="type-mismatch", cardinality=cardinality)
    return FieldOutcome(
        field_name=name, xpath=xpath, success=True, raw_value=raw,
        normalized_value=normalize(raw, field_type), cardinality=cardinality,
    )


def extract_fields(clean_html: str, field_rules: list[tuple]) -> list[FieldOutcome]:
    """
    Apply rules to the clean DOM. Each rule is (name, field_type, xpath) or
    (name, field_type, cardinality, xpath).
    """
    tree = parse_html(clean_html) if clean_html else None
    outcomes = []
    for rule in field_rules:
        if len(rule) == 3:
            name, ftype, xpath = rule
            cardinality = "single"
        else:
            name, ftype, cardinality, xpath = rule
        outcomes.append(extract_one(tree, name, ftype, cardinality or "single", xpath))
    return outcomes


# Backward-compatible helper used by older tests.
def apply_xpath(tree, xpath: str) -> tuple[bool, str, str]:
    out = extract_one(tree, "f", "string", "single", xpath)
    return out.success, out.raw_value, out.reason
