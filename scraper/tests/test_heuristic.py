from django.test import SimpleTestCase
from lxml import html as lxml_html

from demopages import state
from scraper.llm.base import FieldSpec
from scraper.llm.heuristic import HeuristicProvider
from scraper.services.fetcher import prune_html

FIELDS = [
    FieldSpec("title", "the product name / title", "string"),
    FieldSpec("price", "the product price", "price"),
    FieldSpec("availability", "stock availability status", "string"),
    FieldSpec("rating", "the average customer rating score", "number"),
    FieldSpec("sku", "the product SKU / item code", "string"),
]

EXPECTED = {
    "title": "Aurora Wireless Headphones",
    "price": "$149.99",
    "availability": "In Stock",
    "rating": "4.6",
    "sku": "AUR-1024",
}


def _extract_all(pruned, xpaths):
    tree = lxml_html.fromstring(pruned)
    out = {}
    for name, xp in xpaths.items():
        nodes = tree.xpath(xp)
        out[name] = nodes[0].text_content().strip() if nodes else None
    return out


class HeuristicTests(SimpleTestCase):
    def setUp(self):
        self.provider = HeuristicProvider()

    def test_generates_all_fields_v1(self):
        pruned, _ = prune_html(state.load_fixture("v1"))
        result = self.provider.generate_xpaths(pruned, FIELDS)
        self.assertEqual(set(result.xpaths), set(EXPECTED))
        values = _extract_all(pruned, result.xpaths)
        for name, expected in EXPECTED.items():
            self.assertEqual(values[name], expected, f"field {name}")

    def test_v1_xpaths_break_on_v2(self):
        """Selectors generated against v1 must fail on the redesigned v2 DOM."""
        pruned_v1, _ = prune_html(state.load_fixture("v1"))
        v1_xpaths = self.provider.generate_xpaths(pruned_v1, FIELDS).xpaths

        pruned_v2, _ = prune_html(state.load_fixture("v2"))
        values = _extract_all(pruned_v2, v1_xpaths)
        # Every stale selector should now return nothing.
        self.assertTrue(all(v is None for v in values.values()), values)

    def test_regenerates_working_xpaths_on_v2(self):
        pruned_v2, _ = prune_html(state.load_fixture("v2"))
        result = self.provider.generate_xpaths(pruned_v2, FIELDS)
        values = _extract_all(pruned_v2, result.xpaths)
        for name, expected in EXPECTED.items():
            self.assertEqual(values[name], expected, f"field {name} on v2")
