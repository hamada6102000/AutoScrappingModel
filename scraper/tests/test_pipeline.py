from django.test import TestCase

from demopages import state
from scraper.models import (
    Category,
    ChangeEvent,
    FieldType,
    Resolution,
    RunStatus,
    ScrapeTarget,
    TargetField,
    XPathRule,
)
from scraper.services.pipeline import ScrapePipeline

FIELDS = [
    ("title", "the product name / title", FieldType.STRING, True, "Aurora Wireless Headphones"),
    ("price", "the product price", FieldType.PRICE, True, "149.99"),
    ("availability", "stock availability status", FieldType.STRING, True, "In Stock"),
    ("rating", "the average customer rating score", FieldType.NUMBER, False, "4.6"),
    ("sku", "the product SKU / item code", FieldType.STRING, False, "AUR-1024"),
]


def make_target():
    target = ScrapeTarget.objects.create(
        name="Demo Product", url="http://testserver/demo/product/",
        category=Category.ECOMMERCE,
    )
    for i, (name, desc, ftype, req, gt) in enumerate(FIELDS, start=1):
        TargetField.objects.create(
            target=target, name=name, description=desc, field_type=ftype,
            is_required=req, ground_truth=gt, order=i,
        )
    return target


class PipelineTests(TestCase):
    def setUp(self):
        self.target = make_target()
        self.pipeline = ScrapePipeline()  # heuristic provider by default
        self.v1 = state.load_fixture("v1")
        self.v2 = state.load_fixture("v2")

    def test_initial_run_succeeds(self):
        run = self.pipeline.run(self.target, html=self.v1)
        self.assertEqual(run.status, RunStatus.SUCCESS)
        self.assertGreaterEqual(run.health_score, 0.8)
        self.assertEqual(run.fields_extracted, 5)
        # All values match ground truth.
        for rec in run.records.all():
            if rec.matches_ground_truth is not None:
                self.assertTrue(rec.matches_ground_truth, f"{rec.field.name}={rec.normalized_value}")
        # Five active rules were created.
        self.assertEqual(
            XPathRule.objects.filter(field__target=self.target, is_active=True).count(), 5
        )

    def test_layout_change_triggers_self_heal(self):
        # First run establishes baseline rules against v1.
        self.pipeline.run(self.target, html=self.v1)

        # Redesign the site: run again with v2 HTML.
        run2 = self.pipeline.run(self.target, html=self.v2)

        self.assertEqual(run2.status, RunStatus.HEALED)
        self.assertTrue(run2.healed)
        self.assertGreaterEqual(run2.heal_attempts, 1)
        self.assertGreaterEqual(run2.health_score, 0.8)
        self.assertEqual(run2.fields_extracted, 5)
        self.assertGreater(run2.structural_distance, 0.0)

        # A change event was logged and auto-resolved.
        event = ChangeEvent.objects.filter(run=run2).first()
        self.assertIsNotNone(event)
        self.assertEqual(event.resolution, Resolution.AUTO_HEALED)
        self.assertGreater(event.time_to_recovery_ms, -1)

        # Rules were versioned up (v2 selectors replace v1).
        title_rule = (
            XPathRule.objects.filter(field__target=self.target, field__name="title", is_active=True)
            .first()
        )
        self.assertIsNotNone(title_rule)
        self.assertGreaterEqual(title_rule.version, 2)

        # Post-heal values still match ground truth (sustained accuracy).
        for rec in run2.records.all():
            if rec.matches_ground_truth is not None:
                self.assertTrue(rec.matches_ground_truth, f"{rec.field.name}={rec.normalized_value}")

    def test_stable_rerun_no_change(self):
        self.pipeline.run(self.target, html=self.v1)
        run2 = self.pipeline.run(self.target, html=self.v1)
        self.assertEqual(run2.status, RunStatus.SUCCESS)
        self.assertFalse(run2.healed)
        # No new change event on a stable re-run.
        self.assertEqual(ChangeEvent.objects.count(), 0)
