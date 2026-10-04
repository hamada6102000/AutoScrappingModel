"""Annotated dataset -> config, structured metadata, audit and pre-filled labels."""

from pathlib import Path

from django.core.management import call_command
from django.test import TestCase

from scraper.evaluation.audit import run_audit, write_outputs
from scraper.evaluation.config import resolve
from scraper.evaluation.dataset import field_config, load_dataset, probes, to_config
from scraper.evaluation.gt_io import import_ground_truth
from scraper.evaluation.protocol import baseline_snapshot, capture_snapshot, init_experiment
from scraper.services.extractor import extract_one, set_base_url
from scraper.services.fetcher import prune
from scraper.services.htmlutil import parse_html
from scraper.services.matching import exact_match
from scraper.services.mutations import apply_mutation

from .helpers import TempDataDirMixin

FIX = Path(__file__).parent / "fixtures"
PAGE = FIX / "pages" / "dataset_product.html"
DATASET = FIX / "dataset" / "mini_dataset.json"


def _config(**over):
    data = to_config(load_dataset(DATASET), name="mini", repetitions=1, planned_repetitions=3,
                     provider="heuristic", render="static")
    site = data["sites"][0]
    site["html_file"] = str(PAGE)
    data["experiment"]["fetch"]["respect_robots"] = False
    for key, value in over.items():
        data["experiment"][key] = value
    return resolve(data, base_dir=FIX)


class DatasetImportTests(TestCase):
    def test_mapping_keeps_urls_and_fields(self):
        cfg = _config()
        site = cfg["sites"][0]
        self.assertEqual(site["url"], "https://lumen.example.test/products/trail-lamp-300")
        self.assertEqual(site["category"], "ecommerce")
        self.assertEqual(len(site["fields"]), 16)
        types = {f["name"]: f["type"] for f in site["fields"]}
        self.assertEqual(types["product_title"], "string")
        self.assertEqual(types["waterproof"], "boolean")
        self.assertEqual(types["weight"], "mixed")
        box = next(f for f in site["fields"] if f["name"] == "in_the_box")
        self.assertEqual(box["cardinality"], "many")
        self.assertIn("Trail Lamp 300", site["expect_text"])
        self.assertEqual(cfg["experiment"]["planned_repetitions"], 3)

    def test_probes_for_annotated_values(self):
        f = field_config({"field_name": "a", "type": "text", "cardinality": "Single",
                          "current_value": "InStock (http://schema.org/InStock)"})
        self.assertEqual(probes(f).values, ["InStock (http://schema.org/InStock)", "InStock",
                                            "http://schema.org/InStock"])
        f = field_config({"field_name": "a", "type": "text", "cardinality": "List",
                          "current_value": "Peng Zhou; … ; Zheng-Li Shi (29 authors)"})
        p = probes(f)
        self.assertEqual(p.expected_count, 29)
        self.assertFalse(p.complete)
        self.assertEqual([i[0] for i in p.items], ["Peng Zhou", "Zheng-Li Shi"])
        f = field_config({"field_name": "a", "type": "text", "current_value": "(article body, ~816 words)"})
        self.assertEqual(probes(f).kind, "none")
        f = field_config({"field_name": "a", "type": "other", "current_value": "e.g. size 8 sku X"})
        self.assertEqual(probes(f).kind, "none")

    def test_exclusion_needs_reason(self):
        data = to_config(load_dataset(DATASET), name="mini", repetitions=1, planned_repetitions=None)
        data["sites"][0]["fields"][0]["exclude"] = True
        from scraper.evaluation.config import ConfigError
        with self.assertRaises(ConfigError):
            resolve(data)
        data["sites"][0]["fields"][0]["exclude_reason"] = "test"
        cfg = resolve(data)
        self.assertEqual(len(cfg["sites"][0]["fields"]), 15)
        self.assertEqual(cfg["sites"][0]["excluded_fields"][0]["name"], "product_title")

    def test_import_command_writes_yaml(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "sites.yaml"
            call_command("eval_import_dataset", dataset=str(DATASET), out=str(out), verbosity=0,
                         stdout=open(Path(d) / "log.txt", "w"))
            text = out.read_text(encoding="utf-8")
            self.assertIn("lumen.example.test/products/trail-lamp-300", text)
            self.assertIn("dataset_value: Trail Lamp 300", text)


class StructuredMetadataTests(TestCase):
    def setUp(self):
        self.page = prune(PAGE.read_text(encoding="utf-8"), structured=True)
        self.tree = parse_html(self.page.clean_html)
        set_base_url(self.tree, "https://lumen.example.test/products/trail-lamp-300")

    def test_jsonld_and_meta_are_addressable(self):
        out = extract_one(self.tree, "sku", "string", "single",
                          "//div[@data-jsonld]/div[@data-key='sku']")
        self.assertEqual(out.raw_value, "TL-300-BLK")
        out = extract_one(self.tree, "price", "price", "single",
                          "//div[@data-key='offers']/div[@data-key='price']")
        self.assertEqual(out.raw_value, "49.00")  # number kept as written
        out = extract_one(self.tree, "summary", "string", "single",
                          "//meta[@name='description']/@content")
        self.assertTrue(out.success)
        out = extract_one(self.tree, "url", "url", "single", "//link[@rel='canonical']/@href")
        self.assertTrue(exact_match(out.raw_value, "https://lumen.example.test/products/trail-lamp-300", "url"))

    def test_structured_data_can_be_switched_off(self):
        off = prune(PAGE.read_text(encoding="utf-8"), structured=False)
        self.assertNotIn("page-metadata", off.clean_html)
        self.assertNotIn("TL-300-BLK", off.clean_html)

    def test_relative_urls_resolve_against_page(self):
        out = extract_one(self.tree, "img", "url", "single", "//img[@class='hero']/@src")
        self.assertEqual(out.raw_value, "https://lumen.example.test/media/trail-lamp.jpg")

    def test_boolean_and_number_word_fields(self):
        out = extract_one(self.tree, "w", "boolean", "single", "//dt[.='Waterproof']/following-sibling::dd[1]")
        self.assertEqual(out.normalized_value, "true")
        self.assertTrue(exact_match(out.raw_value, "true", "boolean"))
        out = extract_one(self.tree, "w", "boolean", "single", "boolean(//dt[.='Waterproof'])")
        self.assertEqual(out.raw_value, "true")
        out = extract_one(self.tree, "r", "number", "single", "//p[contains(@class,'star-rating')]/@class")
        self.assertTrue(exact_match(out.raw_value, "4", "number"))

    def test_perturbation_keeps_jsonld_consistent_and_whole_numbers_only(self):
        html = PAGE.read_text(encoding="utf-8")
        res = apply_mutation(html, "perturb_numbers", seed=7, gt_fields={"price": (["$49.00"], "price")})
        new_price = res.gt_updates["price"]
        page = prune(res.html, structured=True)
        tree = parse_html(page.clean_html)
        visible = extract_one(tree, "p", "price", "single", "//span[@class='price']")
        jsonld = extract_one(tree, "p", "price", "single", "//div[@data-key='offers']/div[@data-key='price']")
        self.assertTrue(exact_match(visible.raw_value, new_price, "price"))
        self.assertTrue(exact_match(jsonld.raw_value, new_price, "price"))

        res = apply_mutation(html, "perturb_numbers", seed=3, gt_fields={"stock": (["22"], "number")})
        self.assertIn("2026", res.html)  # '22' inside '2026'/'2022' untouched


class AuditTests(TempDataDirMixin, TestCase):
    def test_audit_and_prefilled_sheet(self):
        cfg = _config()
        exp = init_experiment(cfg)
        capture_snapshot(exp, cfg["sites"][0], write=lambda *_: None)
        snap = baseline_snapshot(exp, "site_001")
        self.assertTrue(snap.fetch_meta["check"]["content_ok"])
        rows = run_audit(exp)
        status = {f["field"]: f for f in rows[0]["fields"]}
        self.assertEqual(status["product_title"]["status"], "found")
        self.assertEqual(status["sku"]["found_in"], "metadata")
        self.assertEqual(status["availability"]["status"], "found-variant")
        self.assertEqual(status["availability"]["value"], "https://schema.org/InStock")
        self.assertEqual(status["description"]["status"], "prefix")
        self.assertTrue(status["description"]["value"].endswith("sealed battery."))
        self.assertEqual(status["in_the_box"]["status"], "list-completed")
        self.assertEqual(len(status["in_the_box"]["value"]), 5)
        self.assertEqual(status["waterproof"]["status"], "boolean")
        self.assertEqual(status["ship_date"]["value"], "Sep 8, 2026")
        self.assertIn("attribute", status["image_url"]["found_in"])
        self.assertEqual(status["product_url"]["found_in"], "metadata")
        self.assertEqual(status["star_rating"]["status"], "found-partial")
        self.assertEqual(status["body_text"]["status"], "not-literal")
        self.assertEqual(status["gift_wrap"]["status"], "found-partial")  # 'available' inside a text

        paths = write_outputs(exp, rows, Path(self._tmp) / "audit")
        for p in paths.values():
            self.assertTrue(Path(p).exists())
        report = import_ground_truth(exp, paths["gt_sheet"], default_verifier="test")
        self.assertEqual(report["errors"], [])
        self.assertFalse([w for w in report["warnings"] if "pruning" in w], report["warnings"])
        self.assertEqual(report["imported"], 15)  # all but body_text
        gt = {g.field_name: g for g in snap.ground_truth.all()}
        self.assertEqual(len(gt["in_the_box"].values), 5)
        self.assertIn("prefilled", gt["sku"].notes)
