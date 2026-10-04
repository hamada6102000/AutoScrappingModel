"""
Test package. Tests must never touch the network or real data folders, so the
token counter is pinned to the offline estimate for the whole test process.
"""

from django.conf import settings

settings.SCRAPER["TOKENIZER"] = "estimate"
