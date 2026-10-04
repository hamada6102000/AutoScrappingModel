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
    "- A value may be in an attribute: select it with /@content, /@href, /@src, "
    "/@datetime or /@class.\n"
    "- If present, <section id=\"page-metadata\"> holds the page's <title>, <meta> tags, "
    "canonical <link> and JSON-LD structured data (nested div[@data-key] elements); "
    "use it when a field is only there.\n"
    "- For a yes/no (boolean) field, select the element whose text states yes/no, or "
    "write an XPath boolean() expression.\n"
    "- If a field cannot be found, map it to an empty string.\n"
    "- Do not wrap the JSON in markdown fences or add commentary."
)

_TEMPLATE_ID = "user-v2"
PROMPT_VERSION = hashlib.sha256(
    (SYSTEM_PROMPT + _TEMPLATE_ID).encode("utf-8")
).hexdigest()[:10]


TRUNCATION_MARK = "\n<!-- [input truncated] -->"


def prepare_html(pruned_html: str, max_chars: int) -> tuple[str, bool]:
    if max_chars and len(pruned_html) > max_chars:
        return pruned_html[:max_chars] + TRUNCATION_MARK, True
    return pruned_html, False


def _field_line(f: FieldSpec) -> str:
    kind = "list — select ALL items" if f.cardinality == "many" else "single value"
    hint = f.description or f"the {f.name}"
    return f'- "{f.name}" (type: {f.field_type}; {kind}): {hint}'


def completion_budget(n_fields: int, reasoning_effort: Optional[str] = None) -> int:
    """max_completion_tokens for one call: room for one XPath per field in the
    JSON answer, plus the hidden reasoning of reasoning models."""
    reserve = {"low": 2000, "medium": 5000, "high": 10000}.get(reasoning_effort or "", 0)
    return min(16384, 300 + 60 * max(1, n_fields) + reserve)


def prompt_token_budget(tokens_per_minute: Optional[int], max_completion_tokens: int,
                        completion_counts: bool = True) -> Optional[int]:
    """Largest prompt (tokens) that keeps one request within a per-minute
    token limit; 15% margin for tokenizer and chat-format differences."""
    if not tokens_per_minute:
        return None
    usable = int(tokens_per_minute * 0.85)
    if completion_counts:
        usable -= max_completion_tokens
    return max(500, usable)


def conservative_counter():
    """Token counter for budgeting requests. With the exact tokenizer
    unavailable, chars/3 (HTML averages ~3 characters per token) errs on the
    large side, so a request is never planned bigger than the limit."""
    from ..services import tokens

    if tokens.is_estimate():
        return lambda text: len(text) // 3 + 1
    return tokens.count_tokens


def build_prompt(
    pruned_html: str,
    fields: list[FieldSpec],
    failure_history: Optional[list[FailureHint]] = None,
    max_html_chars: int = 60000,
    max_prompt_tokens: Optional[int] = None,
    count=None,
) -> tuple[str, dict]:
    """Return (user_prompt, meta) where meta has input_chars/truncated/sha.

    ``max_html_chars`` caps the page HTML; ``max_prompt_tokens`` (system +
    user prompt) additionally shortens the HTML so the request fits a
    provider's per-request token budget. Cuts are always at the end of the
    page and are flagged (``input_truncated``).
    """
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
    example = json.dumps({f.name: "" for f in fields})

    def assemble(snippet: str) -> str:
        return (
            f"Fields to extract:\n{field_lines}\n"
            f"{healing_block}\n"
            f"Return JSON exactly in this shape (fill in the XPaths):\n{example}\n\n"
            f"Page HTML:\n```html\n{snippet}\n```"
        )

    html_snippet, truncated = prepare_html(pruned_html, max_html_chars)
    prompt = assemble(html_snippet)
    if max_prompt_tokens:
        count = count or conservative_counter()

        def fits(n: int) -> bool:
            return count(SYSTEM_PROMPT + assemble(pruned_html[:n] + TRUNCATION_MARK)) <= max_prompt_tokens

        if count(SYSTEM_PROMPT + prompt) > max_prompt_tokens:
            lo, hi = 0, min(len(pruned_html), max_html_chars or len(pruned_html))
            while lo < hi:  # largest prefix that fits
                mid = (lo + hi + 1) // 2
                if fits(mid):
                    lo = mid
                else:
                    hi = mid - 1
            html_snippet, truncated = pruned_html[:lo] + TRUNCATION_MARK, True
            prompt = assemble(html_snippet)
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
