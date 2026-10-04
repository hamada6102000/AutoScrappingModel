from django.test import SimpleTestCase

from demopages import state
from scraper.services.fetcher import prune, prune_html, prune_result
from scraper.services.htmlutil import decode_html, parse_html, visible_text

from .helpers import page


class PruningTests(SimpleTestCase):
    def test_removes_noise_keeps_content(self):
        p = prune(page("news_article"))
        low = p.clean_html.lower()
        for gone in ("<script", "<style", "_tracking", "advertisement", "daily test gazette 2026",
                     "shared 1,204 times", "<nav", "masthead"):
            self.assertNotIn(gone, low, gone)
        self.assertIn("City council approves new transit plan", p.clean_html)

    def test_keeps_article_header_with_byline_and_date(self):
        # The old pruner deleted every <header>, losing headline/author/date.
        p = prune(page("news_article"))
        self.assertIn('class="story-header"', p.clean_html)
        self.assertIn("Mona Adel", p.clean_html)
        self.assertIn('datetime="2026-09-28"', p.clean_html)

    def test_unwraps_form_but_drops_controls(self):
        p = prune(page("shop_product"))
        self.assertIn("$199.00", p.clean_html)
        self.assertIn("In stock", p.clean_html)
        self.assertNotIn("<form", p.clean_html)
        self.assertNotIn("<button", p.clean_html)
        self.assertNotIn("<select", p.clean_html)
        self.assertIn('itemprop="price"', p.clean_html)  # body microdata kept

    def test_keeps_sidebar_inside_main_and_honours_prune_keep(self):
        self.assertIn("Delta Analytics", prune(page("job_posting")).clean_html)
        html = "<html><body><aside class='x'><p>Keep me</p></aside><main><p>A</p></main></body></html>"
        self.assertNotIn("Keep me", prune(html).clean_html)
        self.assertIn("Keep me", prune(html, keep_tags=["aside"]).clean_html)

    def test_llm_view_collapses_repeated_siblings_only_in_view(self):
        p = prune(page("category_listing"))
        cards = "//li[@class='product-card']"
        self.assertEqual(len(parse_html(p.clean_html).xpath(cards)), 12)
        self.assertEqual(len(parse_html(p.llm_view).xpath(cards)), 3)
        self.assertIn("+9 more similar", p.llm_view)

    def test_llm_view_truncates_long_text_but_clean_dom_does_not(self):
        long_text = "word " * 100
        p = prune(f"<html><body><main><p class='d'>{long_text}</p></main></body></html>")
        self.assertIn("…", p.llm_view)
        self.assertIn(long_text.strip(), visible_text(parse_html(p.clean_html)))

    def test_deterministic_and_counts_tokens(self):
        a, b = prune(page("shop_product")), prune(page("shop_product"))
        self.assertEqual(a.clean_html, b.clean_html)
        self.assertEqual(a.llm_view, b.llm_view)
        self.assertGreater(a.raw_tokens, a.llm_view_tokens)
        self.assertTrue(0 < a.token_reduction < 1)
        self.assertTrue(a.tag_profile and a.markup_profile)

    def test_prune_result_and_compat_helper(self):
        fr = prune_result("http://example/", state.load_fixture("v1"))
        self.assertTrue(fr.dom_signature)
        self.assertIn("Aurora Wireless Headphones", fr.pruned_html)
        view, skeleton = prune_html(state.load_fixture("v1"))
        self.assertIn("Aurora Wireless Headphones", view)
        self.assertTrue(skeleton)

    def test_empty_page(self):
        self.assertTrue(prune("").empty)


class DecodeTests(SimpleTestCase):
    def test_header_charset_wins(self):
        text, enc = decode_html("café".encode("latin-1"), "text/html; charset=ISO-8859-1")
        self.assertEqual(text, "café")
        self.assertEqual(enc, "iso-8859-1")

    def test_meta_charset_for_arabic_windows_1256(self):
        body = '<html><head><meta charset="windows-1256"></head><body>سعر</body></html>'
        text, enc = decode_html(body.encode("windows-1256"), "text/html")
        self.assertIn("سعر", text)
        self.assertEqual(enc, "windows-1256")

    def test_defaults_to_utf8_not_latin1(self):
        text, _ = decode_html("السعر ١٤٩".encode("utf-8"), "text/html")
        self.assertEqual(text, "السعر ١٤٩")
