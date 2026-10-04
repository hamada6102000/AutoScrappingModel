from django.test import SimpleTestCase

from scraper.services.htmlutil import parse_html, visible_text
from scraper.services.matching import value_in_text
from scraper.services.mutations import ALL_OPS, STRUCTURAL_OPS, apply_mutation, field_nodes

from .helpers import SELFTEST_GT, page

PAGE_FOR = {
    "shop-product": "shop_product", "news-article": "news_article", "job-posting": "job_posting",
    "finance-quote": "finance_quote", "category-listing": "category_listing",
}
TYPES = {
    "shop-product": {"price": "price", "rating": "number"},
    "finance-quote": {"price": "price"},
    "job-posting": {"salary": "price"},
    "category-listing": {"price": "price"},
    "news-article": {"date": "date"},
}


def gt_fields(site):
    out = {}
    for name, value in SELFTEST_GT[site].items():
        values = value.split("\n")
        out[name] = (values, TYPES.get(site, {}).get(name, "string"))
    return out


class MutationTests(SimpleTestCase):
    def test_structural_ops_preserve_every_labelled_value(self):
        for site, fixture in PAGE_FOR.items():
            html = page(fixture)
            gt = gt_fields(site)
            for op in STRUCTURAL_OPS:
                for focus in (True, False):
                    res = apply_mutation(html, op, seed=3, severity=1.0, focus=focus, gt_fields=gt)
                    after = visible_text(parse_html(res.html))
                    for name, (values, ftype) in gt.items():
                        for v in values:
                            self.assertTrue(value_in_text(v, after, ftype), f"{site}/{op}/{name}")
                    self.assertTrue(all(res.spec["gt_preserved"].values()), f"{site}/{op}")

    def test_deterministic_for_same_seed(self):
        gt = gt_fields("shop-product")
        for op in ALL_OPS:
            a = apply_mutation(page("shop_product"), op, seed=11, severity=0.6, gt_fields=gt)
            b = apply_mutation(page("shop_product"), op, seed=11, severity=0.6, gt_fields=gt)
            self.assertEqual(a.html, b.html, op)

    def test_structural_ops_actually_change_the_page(self):
        gt = gt_fields("news-article")
        for op in STRUCTURAL_OPS:
            if op == "rename_ids":  # the news fixture has ids only outside the content region
                continue
            res = apply_mutation(page("news_article"), op, seed=5, severity=1.0, gt_fields=gt)
            self.assertNotEqual(res.html, page("news_article"), op)
            self.assertGreater(res.spec["affected"], 0, op)

    def test_perturb_numbers_updates_ground_truth_consistently(self):
        res = apply_mutation(page("shop_product"), "perturb_numbers", seed=2, gt_fields=gt_fields("shop-product"))
        self.assertIn("price", res.gt_updates)
        new_price = res.gt_updates["price"]
        self.assertNotEqual(new_price, "$199.00")
        self.assertIn(new_price, res.html)
        self.assertNotIn("$199.00", res.html)
        # The microdata copy of the price is changed too, so the page stays consistent.
        self.assertNotIn('content="199.00"', res.html)

    def test_rating_change_stays_plausible(self):
        res = apply_mutation(page("shop_product"), "perturb_numbers", seed=9, gt_fields=gt_fields("shop-product"))
        if "rating" in res.gt_updates:
            self.assertLessEqual(float(res.gt_updates["rating"]), 4.3 * 1.05 + 0.05)

    def test_control_is_identity(self):
        res = apply_mutation(page("finance_quote"), "control", seed=1)
        self.assertEqual(res.html, page("finance_quote"))

    def test_field_nodes_locates_values(self):
        nodes = field_nodes(parse_html(page("job_posting")), gt_fields("job-posting"))
        texts = {n.text_content().strip() for n in nodes}
        self.assertIn("Delta Analytics", texts)
        self.assertIn("EGP 45,000", texts)

    def test_unknown_operator(self):
        with self.assertRaises(ValueError):
            apply_mutation(page("shop_product"), "explode", seed=1)
