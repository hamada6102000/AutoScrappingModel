"""Shared test helpers."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from django.conf import settings
from django.test import override_settings

from scraper.services import snapshots as snapstore
from scraper.services import tokens

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PAGES = FIXTURES / "pages"
SELFTEST_CONFIG = FIXTURES / "selftest.yaml"

SELFTEST_GT = {
    "shop-product": {"name": "Nova Smart Watch", "price": "$199.00", "availability": "In stock",
                     "rating": "4.3", "brand": "Nova"},
    "news-article": {"headline": "City council approves new transit plan", "author": "Mona Adel",
                     "date": "September 28, 2026"},
    "job-posting": {"title": "Backend Engineer (Python)", "company": "Delta Analytics",
                    "location": "Cairo, Egypt", "salary": "EGP 45,000"},
    "finance-quote": {"symbol": "ACME", "price": "1,248.50", "change": "+2.35%"},
    "category-listing": {
        "title": "\n".join(["Alpha Lamp", "Beta Chair", "Gamma Desk", "Delta Shelf", "Epsilon Sofa",
                            "Zeta Rug", "Eta Mirror", "Theta Stool", "Iota Bench", "Kappa Clock",
                            "Lambda Vase", "Mu Table"]),
        "price": "\n".join(["$24.99", "$89.00", "$149.50", "$59.99", "$499.00", "$120.00",
                            "$45.25", "$32.00", "$75.80", "$19.99", "$28.40", "$210.00"]),
    },
}


def page(name: str) -> str:
    return (PAGES / f"{name}.html").read_text(encoding="utf-8")


class TempDataDirMixin:
    """Point SCRAPER['DATA_DIR'] at a throwaway folder for the test."""

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.mkdtemp(prefix="scraper-test-")
        self._override = override_settings(SCRAPER={**settings.SCRAPER, "DATA_DIR": self._tmp,
                                                    "TOKENIZER": "estimate"})
        self._override.enable()
        tokens.reset()
        snapstore._PAGE_CACHE.clear()

    def tearDown(self):
        self._override.disable()
        shutil.rmtree(self._tmp, ignore_errors=True)
        tokens.reset()
        snapstore._PAGE_CACHE.clear()
        super().tearDown()
