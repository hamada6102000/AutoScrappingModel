from django.test import SimpleTestCase

from scraper.services.normalizer import normalize


class NormalizerTests(SimpleTestCase):
    def test_price_strips_currency_and_commas(self):
        self.assertEqual(normalize("$1,149.99", "price"), "1149.99")

    def test_whitespace_and_entities(self):
        self.assertEqual(normalize("  Hello&amp;World \n ", "string"), "Hello&World")

    def test_date_to_iso(self):
        self.assertEqual(normalize("January 15, 2026", "date"), "2026-01-15")
        self.assertEqual(normalize("15/01/2026", "date"), "2026-01-15")

    def test_empty(self):
        self.assertEqual(normalize("", "string"), "")
