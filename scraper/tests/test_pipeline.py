from django.conf import settings
from django.test import TestCase, override_settings

from demopages import state
from scraper.llm.base import LLMError, LLMProvider, LLMResult, RateLimitExhausted
from scraper.models import (
    Category, ChangeEvent, FieldType, GroundTruth, LLMPurpose, RecordPhase, Resolution,
    RulePurpose, RunStatus, ScrapeTarget, SnapshotKind, TargetField, XPathRule,
)
from scraper.services import snapshots as snapstore
from scraper.services.pipeline import ScrapePipeline

from .helpers import PAGES, TempDataDirMixin

FIELDS = [
    ("title", "the product name / title", FieldType.STRING, True, "Aurora Wireless Headphones"),
    ("price", "the product price", FieldType.PRICE, True, "149.99"),
    ("availability", "stock availability status", FieldType.STRING, True, "In Stock"),
    ("rating", "the average customer rating score", FieldType.NUMBER, False, "4.6"),
    ("sku", "the product SKU / item code", FieldType.STRING, False, "AUR-1024"),
]
GOOD = {
    "title": "//h1",
    "price": "//span[@class='price']",
    "availability": "//div[contains(@class,'availability')]",
    "rating": "//span[@class='rating-value']",
    "sku": "//span[@class='product-sku']",
}


def make_target(**extra):
    target = ScrapeTarget.objects.create(
        name="Demo Product", url="http://testserver/demo/product/", category=Category.ECOMMERCE, **extra)
    for i, (name, desc, ftype, req, gt) in enumerate(FIELDS, start=1):
        TargetField.objects.create(target=target, name=name, description=desc, field_type=ftype,
                                   is_required=req, ground_truth=gt, order=i)
    return target


class ScriptedProvider(LLMProvider):
    """Fake LLM returning scripted answers (dicts) or raising scripted errors."""

    name = "groq"
    model = "test-model"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_xpaths(self, pruned_html, fields, failure_history=None):
        self.calls.append(([f.name for f in fields], list(failure_history or [])))
        item = self.script.pop(0) if self.script else {}
        if isinstance(item, Exception):
            raise item
        return LLMResult(xpaths={f.name: item[f.name] for f in fields if item.get(f.name)},
                         model=self.model, prompt_tokens=100, completion_tokens=10, latency_ms=50)


class HeuristicLoopTests(TestCase):
    """The original v1 -> v2 demo, now asserting the richer bookkeeping."""

    def setUp(self):
        self.target = make_target()
        self.v1, self.v2 = state.load_fixture("v1"), state.load_fixture("v2")

    def test_initial_run_succeeds(self):
        run = ScrapePipeline().run(self.target, html=self.v1)
        self.assertEqual(run.status, RunStatus.SUCCESS)
        self.assertEqual(run.fields_extracted, 5)
        self.assertTrue(run.completed)
        for rec in run.records.filter(phase=RecordPhase.FINAL):
            if rec.matches_ground_truth is not None:
                self.assertTrue(rec.matches_ground_truth, f"{rec.field.name}={rec.normalized_value}")
        call = run.llm_calls.get()
        self.assertEqual((call.provider, call.purpose), ("heuristic", LLMPurpose.INITIAL))
        self.assertEqual(call.prompt_tokens + call.completion_tokens, 0)  # not an LLM

    def test_layout_change_triggers_self_heal(self):
        pipeline = ScrapePipeline()
        pipeline.run(self.target, html=self.v1)
        run2 = pipeline.run(self.target, html=self.v2)
        self.assertEqual(run2.status, RunStatus.HEALED)
        self.assertTrue(run2.detected)
        self.assertLess(run2.pre_heal_health, 0.8)
        self.assertEqual(len(run2.pre_heal_failed), 5)
        self.assertGreater(run2.structural_distance, 0.0)
        self.assertGreater(run2.markup_distance, 0.0)
        pre = run2.records.filter(phase=RecordPhase.PRE_HEAL)
        self.assertEqual(pre.count(), 5)
        self.assertFalse(any(r.success for r in pre))
        for rec in run2.records.filter(phase=RecordPhase.FINAL):
            if rec.matches_ground_truth is not None:
                self.assertTrue(rec.matches_ground_truth, f"{rec.field.name}={rec.normalized_value}")
        event = ChangeEvent.objects.get(run=run2)
        self.assertEqual(event.resolution, Resolution.AUTO_HEALED)
        self.assertEqual(sorted(event.fields_recovered), sorted(f[0] for f in FIELDS))
        self.assertGreater(event.failure_history_len, 0)
        self.assertEqual(run2.llm_calls.get().purpose, LLMPurpose.HEALING)
        title = XPathRule.objects.get(field__target=self.target, field__name="title", is_active=True)
        self.assertGreaterEqual(title.version, 2)
        self.assertEqual(title.purpose, RulePurpose.HEALING)

    def test_stable_rerun_no_change(self):
        pipeline = ScrapePipeline()
        pipeline.run(self.target, html=self.v1)
        run2 = pipeline.run(self.target, html=self.v1)
        self.assertEqual(run2.status, RunStatus.SUCCESS)
        self.assertEqual(run2.structural_distance, 0.0)
        self.assertEqual(ChangeEvent.objects.count(), 0)
        self.assertEqual(run2.llm_calls.count(), 0)

    def test_no_heal_mode_detects_but_does_not_repair(self):
        ScrapePipeline().run(self.target, html=self.v1)
        run2 = ScrapePipeline(heal_enabled=False).run(self.target, html=self.v2)
        self.assertEqual(run2.status, RunStatus.DEGRADED)
        self.assertTrue(run2.detected)
        self.assertFalse(run2.heal_enabled)
        self.assertEqual(run2.llm_calls.count(), 0)
        self.assertEqual(ChangeEvent.objects.get(run=run2).resolution, Resolution.NOT_ATTEMPTED)


class ScriptedLLMTests(TestCase):
    def setUp(self):
        self.target = make_target()
        self.v1 = state.load_fixture("v1")

    def test_generation_validates_and_repairs_before_storing(self):
        bad_price = {**GOOD, "price": "//nothing"}
        provider = ScriptedProvider([bad_price, {"price": GOOD["price"]}])
        run = ScrapePipeline(provider).run(self.target, html=self.v1)
        self.assertEqual(run.status, RunStatus.SUCCESS)
        calls = list(run.llm_calls.order_by("id"))
        self.assertEqual([c.purpose for c in calls], [LLMPurpose.INITIAL, LLMPurpose.REPAIR])
        self.assertEqual((calls[0].xpaths_returned, calls[0].xpaths_valid), (5, 4))
        self.assertEqual(calls[1].fields_requested, 1)
        self.assertEqual(provider.calls[1][0], ["price"])
        self.assertEqual(provider.calls[1][1][0].failed_xpath, "//nothing")  # failure memory
        # The invalid XPath was never stored.
        self.assertFalse(XPathRule.objects.filter(xpath="//nothing").exists())
        price = XPathRule.objects.get(field__target=self.target, field__name="price")
        self.assertEqual(price.purpose, RulePurpose.REPAIR)

    def test_tokens_and_cost_are_recorded(self):
        pricing = {"test-model": {"input": 1.0, "output": 2.0}}
        with override_settings(SCRAPER={**settings.SCRAPER, "PRICING": pricing}):
            run = ScrapePipeline(ScriptedProvider([GOOD])).run(self.target, html=self.v1)
        call = run.llm_calls.get()
        self.assertEqual((call.prompt_tokens, call.completion_tokens, call.latency_ms), (100, 10, 50))
        self.assertAlmostEqual(call.cost_usd, 100 / 1e6 + 20 / 1e6)
        self.assertEqual((run.prompt_tokens, run.completion_tokens, run.token_usage), (100, 10, 110))
        self.assertAlmostEqual(run.cost_usd, call.cost_usd)
        self.assertEqual(call.requested_provider, "groq")
        self.assertFalse(call.fallback_used)

    def test_fallback_is_flagged_when_allowed(self):
        provider = ScriptedProvider([LLMError("HTTP 500")])
        run = ScrapePipeline(provider, allow_fallback=True).run(self.target, html=self.v1)
        self.assertTrue(run.fallback_used)
        self.assertEqual(run.llm_errors, 1)
        call = run.llm_calls.order_by("id").first()
        self.assertEqual((call.provider, call.requested_provider), ("heuristic", "groq"))
        self.assertTrue(call.fallback_used)
        self.assertIn("HTTP 500", call.error)

    def test_no_fallback_records_errors_and_never_uses_heuristic(self):
        provider = ScriptedProvider([LLMError("bad key")] * 20)
        run = ScrapePipeline(provider, allow_fallback=False, max_attempts=2).run(self.target, html=self.v1)
        self.assertFalse(run.fallback_used)
        self.assertEqual(run.llm_errors, run.llm_calls_count)
        self.assertFalse(run.llm_calls.filter(provider="heuristic").exists())
        self.assertEqual(run.status, RunStatus.DEGRADED)

    def test_quota_exhaustion_propagates_only_when_requested(self):
        with self.assertRaises(RateLimitExhausted):
            ScrapePipeline(ScriptedProvider([RateLimitExhausted("TPD")]), raise_on_quota=True,
                           allow_fallback=False).run(self.target, html=self.v1)
        target2 = make_target()
        run = ScrapePipeline(ScriptedProvider([RateLimitExhausted("TPD")] * 10), allow_fallback=False,
                             max_attempts=1).run(target2, html=self.v1)
        self.assertGreaterEqual(run.llm_errors, 1)

    def test_generate_missing_false_never_calls_llm(self):
        provider = ScriptedProvider([GOOD])
        run = ScrapePipeline(provider, generate_missing=False, heal_enabled=False).run(self.target, html=self.v1)
        self.assertEqual(provider.calls, [])
        self.assertEqual(run.llm_calls_count, 0)
        self.assertTrue(any(s["step"] == "llm-generate:skipped" for s in run.log))


class GroqThroughPipelineTests(TestCase):
    """The real Groq adapter inside the pipeline, with only HTTP mocked."""

    def test_call_log_captures_tokens_waits_and_prompt(self):
        import json
        from unittest import mock

        from scraper.llm.groq_provider import GroqProvider
        from scraper.llm.prompts import PROMPT_VERSION

        from .test_groq_provider import Resp

        answer = Resp({"choices": [{"message": {"content": json.dumps(GOOD)}}],
                       "model": "llama-3.3-70b-versatile",
                       "usage": {"prompt_tokens": 812, "completion_tokens": 96}})
        rate_limited = Resp(status_code=429, headers={"retry-after": "3"}, text="rate limit")
        provider = GroqProvider(api_key="gsk_test", sleep=lambda s: None)
        with mock.patch("scraper.llm.groq_provider.requests.post", side_effect=[rate_limited, answer]):
            run = ScrapePipeline(provider, allow_fallback=False).run(make_target(), html=state.load_fixture("v1"))
        self.assertEqual(run.status, RunStatus.SUCCESS)
        call = run.llm_calls.get()
        self.assertEqual((call.provider, call.model), ("groq", "llama-3.3-70b-versatile"))
        self.assertEqual((call.prompt_tokens, call.completion_tokens), (812, 96))
        self.assertEqual((call.retries, call.rate_limit_wait_ms), (1, 3000))
        self.assertEqual(call.prompt_version, PROMPT_VERSION)
        self.assertEqual(len(call.prompt_sha), 64)
        self.assertIn("//h1", call.raw_response)
        self.assertEqual(call.xpaths_valid, 5)
        self.assertEqual(run.rate_limit_wait_ms, 3000)
        self.assertFalse(run.fallback_used)


class SnapshotScoringTests(TempDataDirMixin, TestCase):
    def test_scores_against_snapshot_ground_truth(self):
        target = make_target()
        snap = snapstore.create_snapshot(html=state.load_fixture("v1"), site_key="demo",
                                         kind=SnapshotKind.BASELINE, label="baseline")
        GroundTruth.objects.create(snapshot=snap, field_name="title", value="Aurora Wireless Headphones")
        GroundTruth.objects.create(snapshot=snap, field_name="price", value="$149.99")
        GroundTruth.objects.create(snapshot=snap, field_name="sku", value="AUR-9999")  # deliberately wrong
        run = ScrapePipeline(ScriptedProvider([GOOD])).run(target, snapshot=snap)
        recs = {r.field.name: r for r in run.records.filter(phase=RecordPhase.FINAL)}
        self.assertTrue(recs["title"].exact_match)
        self.assertTrue(recs["price"].exact_match)
        self.assertFalse(recs["sku"].exact_match)
        self.assertLess(recs["sku"].token_f1, 1.0)
        self.assertIsNone(recs["rating"].exact_match)  # no label for this snapshot
        self.assertEqual(run.raw_tokens, snap.raw_tokens)

    def test_live_run_from_html_file_saves_a_snapshot(self):
        target = make_target(html_file=str(PAGES / "shop_product.html"))
        run = ScrapePipeline().run(target)
        self.assertIsNotNone(run.snapshot)
        self.assertEqual(run.snapshot.kind, SnapshotKind.LIVE)
        self.assertIn("Nova Smart Watch", snapstore.read_html(run.snapshot))
        self.assertEqual(run.snapshot.sha256, run.raw_html_signature)
