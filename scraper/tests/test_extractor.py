from django.test import SimpleTestCase

from scraper.services.extractor import extract_fields, extract_one
from scraper.services.fetcher import prune
from scraper.services.htmlutil import parse_html

from .helpers import page


class ExtractorTests(SimpleTestCase):
    def setUp(self):
        self.listing = parse_html(prune(page("category_listing")).clean_html)
        self.shop = parse_html(prune(page("shop_product")).clean_html)

    def test_list_field_returns_all_items(self):
        out = extract_one(self.listing, "price", "price", "many", "//span[@class='card-price']")
        self.assertTrue(out.success)
        self.assertEqual(len(out.values), 12)
        self.assertEqual(out.values[0], "24.99")

    def test_attribute_xpath(self):
        out = extract_one(self.shop, "price", "price", "single", "//meta[@itemprop='price']/@content")
        self.assertTrue(out.success)
        self.assertEqual(out.normalized_value, "199.00")

    def test_failure_reasons(self):
        cases = {
            "//div[": "xpath-syntax-error",
            "//nothing": "no-match",
            "": "no-rule",
        }
        for xpath, reason in cases.items():
            out = extract_one(self.shop, "f", "string", "single", xpath)
            self.assertFalse(out.success)
            self.assertEqual(out.reason, reason, xpath)

    def test_type_mismatch(self):
        # A "price" selector that lands on text without any digit is rejected.
        out = extract_one(self.shop, "price", "price", "single", "//p[@class='availability-status']")
        self.assertFalse(out.success)
        self.assertEqual(out.reason, "type-mismatch")

    def test_extract_fields_accepts_3_and_4_tuples(self):
        outs = extract_fields(prune(page("shop_product")).clean_html, [
            ("name", "string", "//h1"),
            ("price", "price", "single", "//span[@class='price-current']"),
        ])
        self.assertEqual([o.success for o in outs], [True, True])
