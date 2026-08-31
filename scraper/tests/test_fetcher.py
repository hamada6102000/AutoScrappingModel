from django.test import SimpleTestCase

from demopages import state
from scraper.services.fetcher import prune_html, prune_result


class FetcherTests(SimpleTestCase):
    def setUp(self):
        self.v1 = state.load_fixture("v1")

    def test_prunes_scripts_and_chrome(self):
        pruned, skeleton = prune_html(self.v1)
        low = pruned.lower()
        self.assertNotIn("<script", low)
        self.assertNotIn("window.analytics", low)
        self.assertNotIn("<nav", low)
        self.assertNotIn("<footer", low)
        self.assertNotIn("add to cart", low)  # button/form pruned

    def test_keeps_target_content(self):
        pruned, _ = prune_html(self.v1)
        self.assertIn("Aurora Wireless Headphones", pruned)
        self.assertIn("149.99", pruned)

    def test_token_reduction_and_signatures(self):
        fr = prune_result("http://example/", self.v1)
        self.assertTrue(fr.token_reduction > 0)
        self.assertTrue(fr.dom_signature)
        self.assertTrue(fr.dom_skeleton)

    def test_v1_and_v2_have_different_skeletons(self):
        _, sk1 = prune_html(state.load_fixture("v1"))
        _, sk2 = prune_html(state.load_fixture("v2"))
        self.assertNotEqual(sk1, sk2)
