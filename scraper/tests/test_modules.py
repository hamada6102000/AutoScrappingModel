"""
The three compared modules: static baselines, rule-based self-healing and
LLM-based self-healing. Machinery tests on local fixture pages — no research
results.
"""

import io
from pathlib import Path

import yaml
from django.core.management import call_command
from django.test import SimpleTestCase, TestCase

from scraper.evaluation.config import ConfigError, load_config, resolve
from scraper.evaluation.protocol import EvaluationRunner, init_experiment
from scraper.llm.base import FailureHint, FieldSpec
from scraper.llm.heuristic import HeuristicProvider
from scraper.llm.rule_based import RuleBasedProvider
from scraper.models import LLMCallLog, ScrapeRun, XPathRule
from scraper.services.extractor import extract_one
from scraper.services.fetcher import prune
from scraper.services.htmlutil import parse_html
from scraper.services.mutations import apply_mutation

from .helpers import SELFTEST_CONFIG, SELFTEST_GT, TempDataDirMixin, page

SPEC_TABLE = """<html><head><title>Lamp</title>
<script type="application/ld+json">{"@type": "Product", "sku": "L-1", "brand": {"name": "Lumen"}}</script>
</head><body><main>
<h1 class="pdp-title">Trail Lamp</h1>
<table class="specs"><tr><th>UPC</th><td>a897fe39</td></tr><tr><th>Price (excl. tax)</th><td>£51.77</td></tr>
<tr><th>Tax</th><td>£0.00</td></tr><tr><th>Availability</th><td>In stock (22 available)</td></tr></table>
<p class="star-rating Three"></p>
</main></body></html>"""


def _run(provider, html, fields, hints=None):
    page_ = prune(html)
    tree = parse_html(page_.clean_html)
    res = provider.generate_xpaths(page_.clean_html, fields, hints)
    return tree, res, {f.name: extract_one(tree, f.name, f.field_type, f.cardinality,
                                           res.xpaths.get(f.name, "")).raw_value for f in fields}


class RuleBasedProviderTests(SimpleTestCase):
    def test_label_type_and_structured_rules(self):
        fields = [FieldSpec("title", "product name (H1)"), FieldSpec("tax", "Tax amount", "price"),
                  FieldSpec("price_excl_tax", "Price excluding tax", "price"),
                  FieldSpec("upc", "UPC code"), FieldSpec("sku", "SKU (JSON-LD)"),
                  FieldSpec("star_rating", "rating from the CSS class", "number")]
        _, res, values = _run(RuleBasedProvider(), SPEC_TABLE, fields)
        self.assertEqual(values["title"], "Trail Lamp")
        self.assertEqual(values["tax"], "£0.00")
        self.assertEqual(values["price_excl_tax"], "£51.77")
        self.assertEqual(values["upc"], "a897fe39")
        self.assertEqual(values["sku"], "L-1")
        self.assertIn("Three", values["star_rating"])
        self.assertEqual(res.prompt_tokens + res.completion_tokens, 0)  # not an LLM

    def test_value_anchoring_repairs_a_renamed_layout(self):
        provider = RuleBasedProvider()
        fields = [FieldSpec("headline", "article headline")]
        html = page("news_article")
        tree, res, values = _run(provider, html, fields)
        xp = res.xpaths["headline"]
        broken = apply_mutation(html, "redesign", seed=5, severity=1.0, focus=True,
                                gt_fields={"headline": ([values["headline"]], "string")}).html
        hint = FailureHint("headline", xp, "no-match", last_value=values["headline"])
        _, res2, values2 = _run(provider, broken, fields, [hint])
        self.assertNotEqual(res2.xpaths["headline"], xp)          # never reuses a failed XPath
        self.assertEqual(values2["headline"], values["headline"])  # same content found again

    def test_is_deterministic(self):
        fields = [FieldSpec("tax", "Tax amount", "price"), FieldSpec("title", "name")]
        a = RuleBasedProvider().generate_xpaths(prune(SPEC_TABLE).clean_html, fields).xpaths
        b = RuleBasedProvider().generate_xpaths(prune(SPEC_TABLE).clean_html, fields).xpaths
        self.assertEqual(a, b)


def _config(tmp: Path, conditions):
    data = yaml.safe_load(SELFTEST_CONFIG.read_text(encoding="utf-8"))
    data["experiment"]["name"] = "modules"
    data["experiment"]["conditions"] = conditions
    data["experiment"]["llm"] = {"provider": "groq", "model": "openai/gpt-oss-120b"}  # no key in tests
    for site in data["sites"]:
        site["html_file"] = str((SELFTEST_CONFIG.parent / site["html_file"]).resolve())
    return resolve(data, base_dir=tmp)


class ModuleConditionTests(TempDataDirMixin, TestCase):
    def prepare(self, conditions):
        cfg = _config(Path(self._tmp), conditions)
        exp = init_experiment(cfg)
        out = io.StringIO()
        call_command("eval_snapshot", experiment="modules", stdout=out)
        call_command("eval_gt_template", experiment="modules", stdout=out)
        import csv
        base = Path(self._tmp) / "modules" / "ground_truth_baseline.csv"
        rows = list(csv.DictReader(open(base, encoding="utf-8-sig", newline="")))
        for r in rows:
            r["value"] = SELFTEST_GT[r["site_key"]][r["field"]]
        filled = Path(self._tmp) / "filled.csv"
        with open(filled, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        call_command("eval_gt_import", experiment="modules", file=str(filled), verified_by="t", stdout=out)
        return exp

    def test_offline_modules_run_without_any_llm(self):
        exp = self.prepare(["self_healing", "static", "rule_based", "rule_static"])
        runner = EvaluationRunner(exp, write=lambda m: None, sites=["shop-product", "news-article"],
                                  conditions=["rule_based", "rule_static"])
        self.assertIsNone(runner.provider)                      # no LLM was even created
        stats = runner.run()
        self.assertNotIn("stopped", stats)
        self.assertEqual(stats["llm_calls"], 0)
        done = ScrapeRun.objects.filter(experiment=exp, completed=True)
        self.assertEqual(done.filter(condition="rule_based").count(), 2 * 2 * 9)
        self.assertEqual(done.filter(condition="rule_static").count(), 2 * 2 * 9)
        self.assertFalse(done.filter(condition__in=["self_healing", "static"]).exists())
        # the static control starts from exactly the rule-based t0 selectors
        for rep in (1, 2):
            src = {r.field.name: r.xpath for r in XPathRule.objects.filter(
                field__target__condition="rule_based", field__target__repetition=rep,
                field__target__site_key="shop-product", version=1)}
            copy = {r.field.name: r.xpath for r in XPathRule.objects.filter(
                field__target__condition="rule_static", field__target__repetition=rep,
                field__target__site_key="shop-product")}
            self.assertEqual(src, copy)
        self.assertFalse(XPathRule.objects.filter(field__target__condition="rule_static", version__gt=1).exists())
        self.assertTrue(LLMCallLog.objects.filter(provider="rule_based").exists())
        # after t0, selectors are written only by the healing loop
        for run in done.filter(step__gt=0, heal_attempts=0):
            self.assertEqual(run.llm_calls_count, 0, (run.condition, run.step, run.status))
        self.assertFalse(LLMCallLog.objects.exclude(provider="rule_based").exists())

    def test_llm_static_waits_for_llm_t0(self):
        exp = self.prepare(["self_healing", "static", "rule_based", "rule_static"])
        stats = EvaluationRunner(exp, write=lambda m: None, sites=["shop-product"],
                                 conditions=["static", "rule_based", "rule_static"]).run()
        self.assertTrue(any("static waits" in n for n in stats["waiting"]))
        self.assertFalse(ScrapeRun.objects.filter(experiment=exp, condition="static").exists())

    def test_all_modules_share_snapshots_and_report(self):
        exp = self.prepare(["self_healing", "static", "rule_based", "rule_static"])
        EvaluationRunner(exp, write=lambda m: None, sites=["shop-product"],
                         conditions=["rule_based", "rule_static"]).run()
        EvaluationRunner(exp, write=lambda m: None, sites=["shop-product"], provider=HeuristicProvider(),
                         conditions=["self_healing", "static"]).run()
        snaps = {(r.condition, r.step, r.repetition): r.snapshot_id
                 for r in ScrapeRun.objects.filter(experiment=exp)}
        for (cond, step, rep), sid in snaps.items():
            self.assertEqual(sid, snaps[("rule_based", step, rep)])   # identical pages
        from scraper.evaluation import report
        rep = report.build(exp)
        names = [m["condition"] for m in rep["modules"]]
        self.assertEqual(names, ["static", "rule_static", "rule_based", "self_healing"])
        for m in rep["modules"]:
            self.assertIsNotNone(m["overall_f1"])
        self.assertEqual({(t["a"], t["b"]) for t in rep["paired_tests"]},
                         {("self_healing", "static"), ("rule_based", "rule_static"),
                          ("self_healing", "rule_based"), ("rule_based", "static")})

    def test_config_requires_the_copied_condition(self):
        with self.assertRaises(ConfigError):
            _config(Path(self._tmp), ["static"])
        with self.assertRaises(ConfigError):
            _config(Path(self._tmp), ["rule_static", "self_healing"])
        self.assertEqual(_config(Path(self._tmp), ["rule_based"])["experiment"]["conditions"], ["rule_based"])


class VerificationTests(TempDataDirMixin, TestCase):
    def test_verify_passes_on_a_real_run(self):
        exp = ModuleConditionTests.prepare(self, ["self_healing", "static", "rule_based", "rule_static"])
        EvaluationRunner(exp, write=lambda m: None, sites=["shop-product", "category-listing"],
                         conditions=["rule_based", "rule_static"]).run()
        from scraper.evaluation.verify import verify
        res = verify(exp, Path(self._tmp) / "results")
        self.assertTrue(res["ok"], res["checks"])
        self.assertIn("rule_based", res["recomputed"])
        self.assertTrue((Path(self._tmp) / "results" / "verification.md").exists())

    def test_verify_detects_a_tampered_score(self):
        exp = ModuleConditionTests.prepare(self, ["rule_based", "rule_static"])
        EvaluationRunner(exp, write=lambda m: None, sites=["shop-product"]).run()
        from scraper.models import ExtractedRecord
        rec = ExtractedRecord.objects.filter(run__experiment=exp, phase="final", exact_match=True,
                                             field__cardinality="single").first()
        rec.exact_match = False
        rec.save()
        from scraper.evaluation.verify import verify
        res = verify(exp)
        first = res["checks"][0]
        self.assertFalse(first["ok"])


class PaperTextTests(TempDataDirMixin, TestCase):
    def test_paper_text_states_what_was_run(self):
        exp = ModuleConditionTests.prepare(self, ["self_healing", "static", "rule_based", "rule_static"])
        EvaluationRunner(exp, write=lambda m: None, sites=["shop-product"],
                         conditions=["rule_based", "rule_static"]).run()
        out = io.StringIO()
        call_command("eval_report", experiment="modules", stdout=out)
        call_command("eval_paper", experiment="modules", stdout=out)
        text = (Path(self._tmp) / "modules" / "results" / "paper_results.md").read_text(encoding="utf-8")
        self.assertIn("Not run in this build of the results: Static baseline (LLM-generated selectors), "
                      "LLM-based self-healing", text)
        self.assertIn("| Rule-based self-healing |", text)
        tex = (Path(self._tmp) / "modules" / "results" / "paper_table_modules.tex").read_text(encoding="utf-8")
        self.assertIn("Rule-based self-healing &", tex)


class ReinitTests(TempDataDirMixin, TestCase):
    def test_reinit_is_harmless_and_llm_settings_can_change_before_llm_runs(self):
        from scraper.evaluation.protocol import EvalError
        exp = ModuleConditionTests.prepare(self, ["self_healing", "static", "rule_based", "rule_static"])
        EvaluationRunner(exp, write=lambda m: None, sites=["shop-product"],
                         conditions=["rule_based", "rule_static"]).run()
        n = ScrapeRun.objects.filter(experiment=exp).count()
        cfg = dict(exp.config)
        init_experiment(cfg)                                   # same config: no-op
        self.assertEqual(ScrapeRun.objects.filter(experiment=exp).count(), n)
        import copy
        cfg2 = copy.deepcopy(cfg)
        cfg2["experiment"]["llm"]["tokens_per_minute"] = 250000
        exp2 = init_experiment(cfg2)                           # llm-only change: allowed
        self.assertEqual(exp2.config["experiment"]["llm"]["tokens_per_minute"], 250000)
        self.assertEqual(ScrapeRun.objects.filter(experiment=exp).count(), n)
        cfg3 = copy.deepcopy(cfg2)
        cfg3["experiment"]["health_threshold"] = 0.7
        with self.assertRaises(EvalError):                     # method change: refused
            init_experiment(cfg3)


class SyncLimitsTests(TempDataDirMixin, TestCase):
    def test_measured_limits_are_written_to_config_and_experiment(self):
        import json as _json
        import yaml as _yaml
        from scraper.services import snapshots as snapstore
        cfg_path = Path(self._tmp) / "sites.yaml"
        data = _yaml.safe_load(SELFTEST_CONFIG.read_text(encoding="utf-8"))
        data["experiment"].update({"name": "sync", "conditions": ["self_healing", "static"],
                                   "llm": {"provider": "groq", "model": "openai/gpt-oss-120b",
                                           "tokens_per_minute": 8000, "completion_counts_in_tpm": True}})
        for site in data["sites"]:
            site["html_file"] = str((SELFTEST_CONFIG.parent / site["html_file"]).resolve())
        cfg_path.write_text("# head\n" + _yaml.safe_dump(data), encoding="utf-8")
        exp = init_experiment(load_config(cfg_path))
        (snapstore.experiment_dir(exp) / "llm_ping.json").write_text(_json.dumps({
            "ok": True, "model": "openai/gpt-oss-120b", "tokens_per_minute_reported": 250000,
            "completion_counts_in_tpm_observed": False}), encoding="utf-8")
        call_command("eval_sync_llm_limits", experiment="sync", config=str(cfg_path), stdout=io.StringIO())
        exp.refresh_from_db()
        self.assertEqual(exp.config["experiment"]["llm"]["tokens_per_minute"], 250000)
        self.assertIs(exp.config["experiment"]["llm"]["completion_counts_in_tpm"], False)
        self.assertTrue(cfg_path.read_text(encoding="utf-8").startswith("# head"))


class CommonChainTests(TempDataDirMixin, TestCase):
    def test_modules_are_compared_on_chains_all_completed(self):
        exp = ModuleConditionTests.prepare(self, ["self_healing", "static", "rule_based", "rule_static"])
        EvaluationRunner(exp, write=lambda m: None, sites=["shop-product", "news-article"],
                         conditions=["rule_based", "rule_static"]).run()
        from scraper.evaluation import report
        from scraper.evaluation.verify import verify
        before = {m["condition"]: m for m in report.build(exp)["modules"]}
        # the "LLM" modules get only repetition 1 of one site (reduced scope)
        EvaluationRunner(exp, write=lambda m: None, sites=["shop-product"], provider=HeuristicProvider(),
                         conditions=["self_healing", "static"], max_llm_calls=10_000).run_site_rep(
            __import__("scraper.evaluation.config", fromlist=["site_by_key"]).site_by_key(exp.config, "shop-product"), 1)
        rep = report.build(exp)
        self.assertEqual(rep["comparison_chains"], [("shop-product", 1)])
        mods = {m["condition"]: m for m in rep["modules"]}
        # rule-based modules keep their full-scope numbers; LLM modules cover their one site
        n_reps = exp.config["experiment"]["repetitions"]
        for c in ("rule_based", "rule_static"):
            self.assertEqual(mods[c]["chains"], 2 * n_reps)
            for key in ("initial_f1", "changed_f1", "overall_f1", "breaking_events", "alarm_false_positive_rate"):
                self.assertEqual(mods[c][key], before[c][key], (c, key))
        for c in ("self_healing", "static"):
            self.assertEqual((mods[c]["sites"], mods[c]["site_keys"], mods[c]["chains"]), (1, ["shop-product"], 1))
        self.assertEqual({m["chains"] for m in rep["modules_like_for_like"]}, {1})
        scope = rep["llm_scope"]
        self.assertTrue(scope["reduced"])
        self.assertEqual(scope["llm_sites"], ["shop-product"])
        self.assertIn("No LLM output exists for the other sites", scope["note"])
        tests = {(t["a"], t["b"]): t for t in rep["paired_tests"]}
        self.assertEqual(tests[("rule_based", "rule_static")]["sites"], 2)
        self.assertEqual(tests[("self_healing", "static")]["sites"], 1)
        # no LLM value for the site the LLM never saw
        news = next(r for r in rep["per_site"] if r["site"] == "news-article")
        self.assertIsNone(news["self_healing_sustained_f1"])
        res = verify(exp)
        self.assertTrue(res["ok"], res["checks"])
        self.assertTrue(any("reported only for sites it was executed on" in c["check"] for c in res["checks"]))
        # the paper text states the scope and the reason
        import json

        from scraper.evaluation.paper import results_latex, results_markdown
        rep.pop("_data", None)
        rep = json.loads(json.dumps(rep, default=str))
        rep["llm_scope"]["reason"] = "a decision of the project supervisor"
        text = results_markdown(rep)
        self.assertIn("were executed on 1 of the 5 websites (shop-product; repetition 1), "
                      "a decision of the project supervisor", text)
        self.assertIn("no LLM output exists for those websites and none was copied to them", text)
        self.assertIn("Like-for-like", text)
        tex = results_latex(rep)
        self.assertIn("LLM-based modules executed on 1 of 5 websites, rule-based modules on 2", tex)
        self.assertIn("tab:modules-same-sites", tex)


class RepSubsetTests(TempDataDirMixin, TestCase):
    def test_only_the_requested_repetition_runs(self):
        exp = ModuleConditionTests.prepare(self, ["rule_based", "rule_static"])
        EvaluationRunner(exp, write=lambda m: None, sites=["shop-product"], reps=[1]).run()
        self.assertEqual(set(ScrapeRun.objects.filter(experiment=exp).values_list("repetition", flat=True)), {1})
        from scraper.evaluation.protocol import EvalError
        with self.assertRaises(EvalError):
            EvaluationRunner(exp, write=lambda m: None, reps=[3])


class LLMScopeReplicationTests(TempDataDirMixin, TestCase):
    """Recorded LLM scope: LLM modules only on the chosen sites; the report
    replicates their measured values to the other sites, labelled."""

    def test_scope_is_enforced_and_replication_is_labelled(self):
        import json

        from scraper.evaluation import report
        from scraper.evaluation.paper import results_latex, results_markdown
        from scraper.evaluation.verify import verify
        exp = ModuleConditionTests.prepare(self, ["self_healing", "static", "rule_based", "rule_static"])
        sites = ["shop-product", "news-article"]
        EvaluationRunner(exp, write=lambda m: None, sites=sites, conditions=["rule_based", "rule_static"]).run()
        before = {m["condition"]: m for m in report.build(exp)["modules"]}
        out = io.StringIO()
        call_command("eval_llm_scope", experiment="modules", llm_sites="shop-product",
                     replicate_to_all_sites=True, reason="a decision of the project supervisor", stdout=out)
        self.assertIn("LLM SCOPE", out.getvalue())
        # all conditions requested on both sites: the LLM modules run on shop-product only
        EvaluationRunner(exp, write=lambda m: None, sites=sites, provider=HeuristicProvider(), reps=[1]).run()
        llm_sites = set(ScrapeRun.objects.filter(experiment=exp, condition__in=["self_healing", "static"])
                        .values_list("target__site_key", flat=True))
        self.assertEqual(llm_sites, {"shop-product"})

        rep = report.build(exp)
        repl = rep["llm_replication"]
        n_sites = len(exp.config["sites"])
        self.assertEqual(repl["source_sites"], ["shop-product"])
        self.assertEqual(len(repl["replicated_sites"]), n_sites - 1)
        mods = {m["condition"]: m for m in rep["modules"]}
        view = {m["condition"]: m for m in repl["modules"]}
        for c in ("self_healing", "static"):
            self.assertEqual((view[c]["sites_measured"], view[c]["sites_replicated"]), (1, n_sites - 1))
            self.assertEqual(view[c]["changed_f1"], mods[c]["changed_f1"])
        for c in ("rule_based", "rule_static"):            # rule-based results unchanged, never replicated
            self.assertEqual(view[c]["sites_replicated"], 0)
            for key in ("initial_f1", "changed_f1", "overall_f1", "breaking_events"):
                self.assertEqual(mods[c][key], before[c][key])
        rows = {r["site"]: r for r in repl["per_site"]}
        self.assertEqual(rows["shop-product"]["self_healing_origin"], "measured")
        news = rows["news-article"]
        self.assertTrue(news["self_healing_origin"].startswith("replicated from shop-product"))
        self.assertEqual(news["self_healing_overall_f1"], mods["self_healing"]["overall_f1"])
        self.assertEqual(news["rule_based_origin"], "measured")
        self.assertFalse(any(r[f"{c}_origin"].startswith("replicated") for r in repl["per_site"]
                             for c in ("rule_based", "rule_static")))
        # statistical tests use measured values only
        tests = {(t["a"], t["b"]): t for t in rep["paired_tests"]}
        self.assertEqual(tests[("self_healing", "rule_based")]["sites"], 1)

        res = verify(exp)
        self.assertTrue(res["ok"], res["checks"])
        names = " ".join(c["check"] for c in res["checks"])
        self.assertIn("no LLM-based run exists outside the recorded LLM sites", names)
        self.assertIn("replicated LLM values equal the measured LLM values", names)

        rep.pop("_data", None)
        rep = json.loads(json.dumps(rep, default=str))
        text = results_markdown(rep)
        self.assertIn("a decision of the project supervisor. The values measured on this website were "
                      f"replicated to the remaining {n_sites - 1} websites", text)
        self.assertIn("The replicated values are not measurements: no LLM call was made for those websites", text)
        self.assertIn(f"1 measured + {n_sites - 1} replicated", text)
        tex = results_latex(rep)
        self.assertIn(f"their values replicated (not measured) to the other {n_sites - 1}", tex)
        self.assertIn(f"1 (+{n_sites - 1} repl.)", tex)

        # the written report: decision copied, replication tables exported
        out_dir = Path(self._tmp) / "results"
        report.write(exp, out_dir)
        self.assertTrue((out_dir / "llm_scope_decision.json").exists())
        self.assertTrue((out_dir / "tables" / "per_site_with_llm_replication.csv").exists())
        md = (out_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("replicated to", md)
        call_command("eval_llm_scope", experiment="modules", clear=True, stdout=out)
        self.assertIsNone(report.build(exp)["llm_replication"])
