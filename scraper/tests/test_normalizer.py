from django.test import SimpleTestCase

from scraper.services.normalizer import normalize, parse_number


class NormalizerTests(SimpleTestCase):
    def test_price_strips_currency_and_commas(self):
        self.assertEqual(normalize("$1,149.99", "price"), "1149.99")

    def test_european_and_arabic_number_formats(self):
        self.assertEqual(normalize("1.149,99 €", "price"), "1149.99")
        self.assertEqual(normalize("149,99 €", "price"), "149.99")
        self.assertEqual(normalize("١٬١٤٩٫٩٩ ج.م", "price"), "1149.99")
        self.assertEqual(normalize("EGP 45,000", "price"), "45000")
        self.assertEqual(normalize("1.234.567", "number"), "1234567")

    def test_number_with_trailing_text(self):
        self.assertEqual(normalize("4.6 out of 5", "number"), "4.6")
        self.assertIsNone(parse_number("no digits here"))

    def test_whitespace_and_entities(self):
        self.assertEqual(normalize("  Hello&amp;World \n ", "string"), "Hello&World")

    def test_date_to_iso(self):
        self.assertEqual(normalize("January 15, 2026", "date"), "2026-01-15")
        self.assertEqual(normalize("15/01/2026", "date"), "2026-01-15")
        self.assertEqual(normalize("2026-09-28T10:00:00Z", "date"), "2026-09-28")

    def test_empty(self):
        self.assertEqual(normalize("", "string"), "")
