"""
Prompt construction for real LLM providers (proposal Section 5.2, O2 —
"Develop an Effective LLM Prompting Strategy").

The same prompt shape is used for initial generation, validation repair and
self-healing; repair/healing append a failure-history block. ``PROMPT_VERSION``
is a hash of the prompt text so every logged call can be tied to the exact
prompt that produced it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from .base import FailureHint, FieldSpec

SYSTEM_PROMPT = (
    "You are an expert web-scraping engineer. You are given the cleaned HTML of a "
    "web page and a list of data fields to extract. For each field, return one "
    "robust XPath 1.0 expression that selects that field's value on THIS page.\n\n"
    "Rules:\n"
    "- Return ONLY a JSON object mapping each field name to its XPath string.\n"
    "- Prefer stable anchors (itemprop, data-* test ids, meaningful class names, "
    "nearby label text) over absolute positional paths.\n"
    "- For a single-value field, the XPath must select exactly one element (or "
    "one attribute/text node). For a list field, it must select every item.\n"
    "- Long texts are shortened with '…' and long runs of similar siblings are "
    "summarised by an HTML comment; do not depend on the hidden parts.\n"
    "- If a field cannot be found, map it to an empty string.\n"
    "- Do not wrap the JSON in markdown fences or add commentary."
)

_TEMPLATE_ID = "user-v2"
PROMPT_VERSION = hashlib.sha256(
    (SYSTEM_PROMPT + _TEMPLATE_ID).encode("utf-8")
).hexdigest()[:10]


def prepare_html(pruned_html: str, max_chars: int) -> tuple[str, bool]:
    if max_chars and len(pruned_html) > max_chars:
        return pruned_html[:max_chars] + "\n<!-- [input truncated] -->", True
    return pruned_html, False


def _field_line(f: FieldSpec) -> str:
    kind = "list — select ALL items" if f.cardinality == "many" else "single value"
    hint = f.description or f"the {f.name}"
    return f'- "{f.name}" (type: {f.field_type}; {kind}): {hint}'


def build_prompt(
    pruned_html: str,
    fields: list[FieldSpec],
    failure_history: Optional[list[FailureHint]] = None,
    max_html_chars: int = 60000,
) -> tuple[str, dict]:
    """Return (user_prompt, meta) where meta has input_chars/truncated/sha."""
    field_lines = "\n".join(_field_line(f) for f in fields)

    healing_block = ""
    if failure_history:
        hints = "\n".join(
            f'- field "{h.field_name}": XPath `{h.failed_xpath}` failed ({h.reason}). '
            "Do NOT reuse it."
            for h in failure_history
        )
        healing_block = (
            "\n\nThe following selectors recently FAILED (failure-history memory). "
            "The page structure has probably changed — write corrected XPaths "
            f"against the HTML below:\n{hints}\n"
        )

    html_snippet, truncated = prepare_html(pruned_html, max_html_chars)
    example = json.dumps({f.name: "" for f in fields})
    prompt = (
        f"Fields to extract:\n{field_lines}\n"
        f"{healing_block}\n"
        f"Return JSON exactly in this shape (fill in the XPaths):\n{example}\n\n"
        f"Page HTML:\n```html\n{html_snippet}\n```"
    )
    meta = {
        "input_chars": len(html_snippet),
        "input_truncated": truncated,
        "prompt_sha": hashlib.sha256((SYSTEM_PROMPT + prompt).encode("utf-8")).hexdigest(),
    }
    return prompt, meta


def build_user_prompt(pruned_html, fields, failure_history=None, max_html_chars=60000) -> str:
    """Backward-compatible wrapper returning only the prompt text."""
    return build_prompt(pruned_html, fields, failure_history, max_html_chars)[0]


def parse_xpath_json(text: str, fields: list[FieldSpec]) -> dict[str, str]:
    """Best-effort extraction of the {field: xpath} JSON from a model reply."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    valid = {f.name for f in fields}
    return {
        str(k): str(v).strip()
        for k, v in data.items()
        if k in valid and isinstance(v, str) and v.strip()
    }
