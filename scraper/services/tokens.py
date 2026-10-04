"""
Token counting.

Pages that are never sent to an LLM (raw HTML) still need a token count to
measure pruning (proposal §4.3 step 2, target >= 90% reduction). We use a
standard BPE tokenizer (tiktoken ``cl100k_base`` by default). If it cannot be
loaded, a chars/4 estimate is used instead and the tokenizer name records
that, so estimated and exact counts are never silently mixed.

Tokens actually sent to the LLM are taken from the provider's own usage
report (e.g. Groq ``usage.prompt_tokens``), which is authoritative.
"""

from __future__ import annotations

from django.conf import settings

_ENCODER = None
_NAME: str | None = None
ESTIMATE_NAME = "estimate:chars/4"


def _load() -> None:
    global _ENCODER, _NAME
    if _NAME is not None:
        return
    wanted = (settings.SCRAPER.get("TOKENIZER") or "cl100k_base").strip()
    if wanted.lower() in {"estimate", "chars/4", ESTIMATE_NAME}:
        _ENCODER, _NAME = None, ESTIMATE_NAME
        return
    try:
        import tiktoken

        _ENCODER = tiktoken.get_encoding(wanted)
        _NAME = f"tiktoken:{wanted}"
    except Exception:  # missing package, offline, unknown encoding
        _ENCODER, _NAME = None, ESTIMATE_NAME


def count_tokens(text: str) -> int:
    _load()
    if not text:
        return 0
    if _ENCODER is not None:
        return len(_ENCODER.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)


def tokenizer_name() -> str:
    _load()
    return _NAME or ESTIMATE_NAME


def is_estimate() -> bool:
    return tokenizer_name() == ESTIMATE_NAME


def reset() -> None:
    """Forget the cached tokenizer (used by tests)."""
    global _ENCODER, _NAME
    _ENCODER, _NAME = None, None
