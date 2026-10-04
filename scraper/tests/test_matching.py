from django.test import SimpleTestCase

from scraper.services.matching import canonical, exact_match, list_match, token_prf, value_in_text


class MatchingTests(SimpleTestCase):
    def test_exact_match_is_type_aware(self):
        self.assertTrue(exact_match("$149.90", "149.9", "price"))
        self.assertTrue(exact_match("USD 1,248.50", "1.248,50", "price"))
        self.assertFalse(exact_match("$149.99", "$149.90", "price"))
        self.assertTrue(exact_match("Aurora  wireless HEADPHONES", "Aurora Wireless Headphones"))
        self.assertTrue(exact_match("September 28, 2026", "2026-09-28", "date"))
        self.assertFalse(exact_match("anything", "", "string"))  # empty label never matches

    def test_canonical_numbers(self):
        self.assertEqual(canonical("$100.00", "price"), "100")
        self.assertEqual(canonical("٤٫٦", "number"), "4.6")

    def test_token_prf(self):
        p, r, f = token_prf("Aurora Wireless Headphones - Black", "Aurora Wireless Headphones")
        self.assertAlmostEqual(p, 0.75)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f, 6 / 7)
        self.assertEqual(token_prf("x", "y"), (0.0, 0.0, 0.0))
        self.assertIsNone(token_prf("x", ""))

    def test_list_match_multiset(self):
        self.assertEqual(list_match(["$1", "$2", "$3"], ["1.00", "2", "4"], "price"), (2, 1, 1))
        self.assertEqual(list_match(["a", "a"], ["a"], "string"), (1, 1, 0))
        self.assertEqual(list_match([], ["a", "b"], "string"), (0, 0, 2))

    def test_value_in_text(self):
        text = "Price: $1,248.50 today. Category: Home furniture"
        self.assertTrue(value_in_text("1248.5", text, "price"))
        self.assertTrue(value_in_text("home FURNITURE", text))
        self.assertFalse(value_in_text("Garden", text))
