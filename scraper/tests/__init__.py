"""
Test package. Tests must never touch the network, a real LLM account or real
data folders, whatever is configured in .env:

* the token counter is pinned to the offline estimate;
* the default LLM provider is the offline heuristic and every API key is
  blanked, so a key in .env is never used (or charged) by a test;
* any real HTTP request raises a ConnectionError (tests that exercise an HTTP
  client mock ``requests.get`` / ``requests.post`` themselves).
"""

import requests
from django.conf import settings

settings.SCRAPER["TOKENIZER"] = "estimate"
settings.SCRAPER["LLM_PROVIDER"] = "heuristic"
for _key in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
    settings.SCRAPER[_key] = ""


def _no_network(self, method, url, *args, **kwargs):
    raise requests.ConnectionError(f"network access is disabled in tests ({method} {url})")


requests.sessions.Session.request = _no_network
