"""
Prompt construction for real LLM providers (proposal Section 5.2, O2 —
"Develop an Effective LLM Prompting Strategy").

The same prompt shape is used for initial generation and for self-healing; the
only difference is that during healing a failure-history block is appended.
"""

from __future__ import annotations

import json
from typing import Optional

from .base import FailureHint, FieldSpec

SYSTEM_PROMPT = (
    "You are an expert web-scraping engineer. You are given the pruned HTML of a "
    "web page and a list of data fields to extract. For each field, return a single "
    "robust XPath 1.0 expression that selects the element containing that field's "
    "value on THIS page.\n\n"
    "Rules:\n"
    "- Return ONLY a JSON object mapping each field name to its XPath string.\n"
    "- Prefer stable attributes (itemprop, data-*, semantic class names) over "
    "absolute positional paths.\n"
    "- Each XPath must select exactly one element (or its text). Do not invent "
    "fields that are not requested.\n"
    "- If a field cannot be found, map it to an empty string.\n"
    "- Do not wrap the JSON in markdown fences or add commentary."
)


def build_user_prompt(
    pruned_html: str,
    fields: list[FieldSpec],
    failure_history: Optional[list[FailureHint]] = None,
    max_html_chars: int = 45000,
) -> str:
    field_lines = "\n".join(
        f'- "{f.name}" (type: {f.field_type}): {f.description or "the " + f.name}'
        for f in fields
    )

    healing_block = ""
    if failure_history:
        hints = "\n".join(
            f'- field "{h.field_name}": previous XPath `{h.failed_xpath}` failed '
            f"({h.reason}). Do NOT reuse it; the page structure has changed."
            for h in failure_history
        )
        healing_block = (
            "\n\nThe following selectors recently FAILED (failure-history memory). "
            "The layout has changed — regenerate corrected XPaths against the HTML "
            f"below:\n{hints}\n"
        )

    html_snippet = pruned_html[:max_html_chars]
    example = json.dumps({f.name: "" for f in fields}, indent=None)

    return (
        f"Fields to extract:\n{field_lines}\n"
        f"{healing_block}\n"
        f"Return JSON exactly in this shape (fill in the XPaths):\n{example}\n\n"
        f"Pruned HTML:\n```html\n{html_snippet}\n```"
    )


def parse_xpath_json(text: str, fields: list[FieldSpec]) -> dict[str, str]:
    """Best-effort extraction of the {field: xpath} JSON from a model reply."""
    text = text.strip()
    # Strip accidental markdown fences.
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    # Find the outermost JSON object.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return {}
    valid = {f.name for f in fields}
    return {
        str(k): str(v)
        for k, v in data.items()
        if k in valid and isinstance(v, str) and v.strip()
    }
