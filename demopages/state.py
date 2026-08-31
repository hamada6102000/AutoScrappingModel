"""
Tiny persisted toggle for which version of the demo product page is "live".

Lets us simulate a real-world layout change (proposal Section 4.1: "controlled
structural modifications will be introduced to the locally cached HTML to
simulate realistic change scenarios"). The state is a file on disk so it is
shared between the dev server process and management commands.
"""

from __future__ import annotations

from pathlib import Path

from django.conf import settings

_STATE_FILE = Path(settings.BASE_DIR) / "demo_state.txt"
_FIXTURES = Path(__file__).resolve().parent / "fixtures"

VALID_VERSIONS = ("v1", "v2")


def get_active_version() -> str:
    try:
        value = _STATE_FILE.read_text(encoding="utf-8").strip()
        return value if value in VALID_VERSIONS else "v1"
    except FileNotFoundError:
        return "v1"


def set_active_version(version: str) -> str:
    if version not in VALID_VERSIONS:
        raise ValueError(f"Unknown demo version: {version!r}")
    _STATE_FILE.write_text(version, encoding="utf-8")
    return version


def load_fixture(version: str) -> str:
    if version not in VALID_VERSIONS:
        raise ValueError(f"Unknown demo version: {version!r}")
    return (_FIXTURES / f"sample_product_{version}.html").read_text(encoding="utf-8")
