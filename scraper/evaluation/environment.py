"""Capture the software/hardware environment of an evaluation (reproducibility)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys

from django.conf import settings
from django.utils import timezone

from ..llm.prompts import PROMPT_VERSION
from ..services import tokens


def _version(module_name: str) -> str:
    try:
        from importlib.metadata import version

        return version(module_name)
    except Exception:
        return "unknown"


def _git(args: list[str]) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=settings.BASE_DIR, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def capture() -> dict:
    return {
        "captured_at": timezone.now().isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {
            name: _version(name)
            for name in ("Django", "lxml", "requests", "tiktoken", "PyYAML", "openpyxl", "anthropic",
                         "playwright", "scipy")
        },
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "tokenizer": tokens.tokenizer_name(),
        "prompt_version": PROMPT_VERSION,
        "llm_max_input_chars": settings.SCRAPER.get("LLM_MAX_INPUT_CHARS"),
        "structured_data": settings.SCRAPER.get("STRUCTURED_DATA", True),
        "llm_max_retries": settings.SCRAPER.get("LLM_MAX_RETRIES"),
        "llm_max_wait_seconds": settings.SCRAPER.get("LLM_MAX_WAIT_SECONDS"),
        "pricing": settings.SCRAPER.get("PRICING") or {},
    }
