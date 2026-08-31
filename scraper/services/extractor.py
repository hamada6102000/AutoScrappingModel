"""
Extraction + validation helpers (proposal Section 1.4 — Validation Layer;
Section 4.3 step 3 — XPath Validation).

Applies an XPath against the live DOM and classifies the outcome as success or
a typed failure (empty / not-found / syntax error / type-mismatch).
"""

from __future__ import annotations

from dataclasses import dataclass

from lxml import etree, html as lxml_html

from .normalizer import normalize


@dataclass
class FieldOutcome:
    field_name: str
    xpath: str
    success: bool
    raw_value: str = ""
    normalized_value: str = ""
    reason: str = ""


def _node_to_text(node) -> str:
    if isinstance(node, str):
        return node
    # lxml smart strings (attribute/text results) are str subclasses handled above.
    if hasattr(node, "text_content"):
        return node.text_content()
    return str(node)


def apply_xpath(tree, xpath: str) -> tuple[bool, str, str]:
    """
    Evaluate an XPath against a parsed tree.

    Returns (found, raw_value, reason). ``found`` is False for syntax errors,
    no match, or an empty/whitespace-only value.
    """
    try:
        nodes = tree.xpath(xpath)
    except (etree.XPathEvalError, etree.XPathError, ValueError, SyntaxError):
        return False, "", "xpath-syntax-error"

    if not nodes:
        return False, "", "no-match"

    raw = _node_to_text(nodes[0]).strip()
    if not raw:
        return False, "", "empty-value"
    return True, raw, ""


def extract_fields(pruned_html: str, field_rules: list[tuple]) -> list[FieldOutcome]:
    """
    Apply each (field_name, field_type, xpath) rule to the pruned DOM.

    ``field_rules`` is a list of (field_name, field_type, xpath) tuples. A rule
    with an empty xpath is reported as a not-generated failure.
    """
    outcomes: list[FieldOutcome] = []
    try:
        tree = lxml_html.fromstring(pruned_html) if pruned_html else None
    except (etree.ParserError, ValueError):
        tree = None

    for field_name, field_type, xpath in field_rules:
        if tree is None:
            outcomes.append(
                FieldOutcome(field_name, xpath, False, reason="no-dom")
            )
            continue
        if not xpath:
            outcomes.append(
                FieldOutcome(field_name, xpath, False, reason="no-rule")
            )
            continue

        found, raw, reason = apply_xpath(tree, xpath)
        if found:
            outcomes.append(
                FieldOutcome(
                    field_name=field_name,
                    xpath=xpath,
                    success=True,
                    raw_value=raw,
                    normalized_value=normalize(raw, field_type),
                )
            )
        else:
            outcomes.append(
                FieldOutcome(field_name, xpath, False, reason=reason)
            )
    return outcomes
