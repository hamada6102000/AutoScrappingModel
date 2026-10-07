"""
Claude (Anthropic Messages API) provider tests with a mocked HTTP layer: the
request shape for the current models (no temperature, effort and JSON-schema
output in output_config), parsing around thinking blocks, usage and thinking
tokens, rate-limit waits, the spend cap, overload retries, the max_tokens
resend and the model list — without a real API key or network.
"""

import json
from unittest import mock

from django.test import SimpleTestCase

from scraper.llm.anthropic_provider import AnthropicProvider, answer_schema
from scraper.llm.base import FieldSpec, LLMError, RateLimitExhausted, RequestTooLarge

FIELDS = [FieldSpec("title", "the product title"), FieldSpec("price", "the price", "price")]
POST = "scraper.llm.anthropic_provider.requests.post"
GET = "scraper.llm.anthropic_provider.requests.get"


class Resp:
    def __init__(self, payload=None, status_code=200, headers=None, text=None):
        self._payload = payload or {}
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


def ok(answer=None, stop="end_turn", usage=None, headers=None):
    answer = answer or {"title": "//h1", "price": "//span[@class='p']"}
    return Resp({"model": "claude-opus-5-5", "stop_reason": stop,
                 "content": [{"type": "thinking", "thinking": "..."},
                             {"type": "text", "text": json.dumps(answer)}],
                 "usage": usage or {"input_tokens": 900, "output_tokens": 300,
                                    "output_tokens_details": {"thinking_tokens": 250}}},
                headers=headers or {"anthropic-ratelimit-input-tokens-limit": "2000000",
                                    "anthropic-ratelimit-output-tokens-limit": "400000"})


class AnthropicProviderTests(SimpleTestCase):
    def make(self, **kw):
        self.sleeps = []
        kw.setdefault("sleep", self.sleeps.append)
        return AnthropicProvider(api_key="sk-ant-test", model="claude-opus-5-5", max_retries=3,
                                 max_wait_seconds=100, **kw)

    def test_requires_key(self):
        with self.assertRaises(ValueError):
            AnthropicProvider(api_key="")

    def test_request_shape_for_current_models(self):
        p = self.make(reasoning_effort="medium")
        with mock.patch(POST, return_value=ok()) as post:
            res = p.generate_xpaths("<html><body><h1>x</h1></body></html>", FIELDS)
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.anthropic.com/v1/messages")
        self.assertEqual(kwargs["headers"]["x-api-key"], "sk-ant-test")
        self.assertEqual(kwargs["headers"]["anthropic-version"], "2023-06-01")
        body = kwargs["json"]
        self.assertEqual(body["model"], "claude-opus-5-5")
        for banned in ("temperature", "top_p", "top_k", "thinking"):
            self.assertNotIn(banned, body)                  # rejected / not needed on these models
        self.assertEqual(body["output_config"]["effort"], "medium")
        self.assertEqual(body["output_config"]["format"],
                         {"type": "json_schema", "schema": answer_schema(FIELDS)})
        self.assertEqual(body["output_config"]["format"]["schema"]["required"], ["title", "price"])
        self.assertGreaterEqual(body["max_tokens"], 12000)
        self.assertEqual(body["messages"][0]["role"], "user")
        # parsed from the text block, the thinking block ignored
        self.assertEqual(res.xpaths, {"title": "//h1", "price": "//span[@class='p']"})
        self.assertEqual((res.prompt_tokens, res.completion_tokens), (900, 300))
        self.assertIn("[thinking_tokens=250]", res.raw_response)
        self.assertEqual(p.reported_tokens_per_minute(), 2000000)

    def test_no_effort_unless_configured(self):
        with mock.patch(POST, return_value=ok()) as post:
            self.make().generate_xpaths("<html></html>", FIELDS)
        self.assertNotIn("effort", post.call_args.kwargs["json"]["output_config"])

    def test_rate_limit_waits_retry_after_and_records_wait(self):
        limited = Resp({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
                       status_code=429, headers={"retry-after": "7"})
        p = self.make()
        with mock.patch(POST, side_effect=[limited, ok()]):
            res = p.generate_xpaths("<html></html>", FIELDS)
        self.assertEqual(self.sleeps, [7.0])
        self.assertEqual(res.retries, 1)
        self.assertEqual(res.rate_limit_wait_ms, 7000)

    def test_rate_limit_beyond_max_wait_stops_cleanly(self):
        limited = Resp({"error": {"type": "rate_limit_error"}}, status_code=429, headers={"retry-after": "500"})
        p = self.make()
        with mock.patch(POST, return_value=limited):
            with self.assertRaises(RateLimitExhausted):
                p.generate_xpaths("<html></html>", FIELDS)
        self.assertEqual(self.sleeps, [])

    def test_spend_cap_is_fatal_not_waited(self):
        cap = Resp({"type": "error", "error": {"type": "rate_limit_error", "message": "usage limits",
                                                "details": {"error_code": "enforced_spend_limit_reached"}}},
                   status_code=429)
        p = self.make()
        with mock.patch(POST, return_value=cap):
            with self.assertRaises(LLMError) as ctx:
                p.generate_xpaths("<html></html>", FIELDS)
        self.assertNotIsInstance(ctx.exception, RateLimitExhausted)
        self.assertIn("HTTP 429 spend limit", str(ctx.exception))   # matched as fatal by the pipeline
        self.assertEqual(self.sleeps, [])

    def test_overloaded_is_retried(self):
        with mock.patch(POST, side_effect=[Resp(status_code=529, text="overloaded"), ok()]):
            res = self.make().generate_xpaths("<html></html>", FIELDS)
        self.assertEqual(res.retries, 1)

    def test_too_large_and_bad_request(self):
        with mock.patch(POST, return_value=Resp(status_code=413, text="too large")):
            with self.assertRaises(RequestTooLarge):
                self.make().generate_xpaths("<html></html>", FIELDS)
        with mock.patch(POST, return_value=Resp(status_code=400, text="invalid temperature")):
            with self.assertRaisesRegex(LLMError, "HTTP 400"):
                self.make().generate_xpaths("<html></html>", FIELDS)

    def test_thinking_used_all_tokens_is_resent_with_a_larger_cap(self):
        cut = Resp({"model": "claude-opus-5-5", "stop_reason": "max_tokens",
                    "content": [{"type": "thinking", "thinking": "..."}],
                    "usage": {"input_tokens": 900, "output_tokens": 12300,
                              "output_tokens_details": {"thinking_tokens": 12300}}})
        p = self.make()
        with mock.patch(POST, side_effect=[cut, ok()]) as post:
            res = p.generate_xpaths("<html></html>", FIELDS)
        first, second = (c.kwargs["json"]["max_tokens"] for c in post.call_args_list)
        self.assertEqual(second, 2 * first)
        self.assertEqual(res.xpaths["title"], "//h1")
        self.assertEqual((res.prompt_tokens, res.completion_tokens), (1800, 12600))   # both calls billed
        self.assertTrue(res.raw_response.startswith("[max_tokens"))
        self.assertEqual(res.retries, 1)

    def test_cached_input_is_counted_and_noted(self):
        usage = {"input_tokens": 100, "cache_read_input_tokens": 800, "cache_creation_input_tokens": 0,
                 "output_tokens": 50}
        with mock.patch(POST, return_value=ok(usage=usage)):
            res = self.make().generate_xpaths("<html></html>", FIELDS)
        self.assertEqual(res.prompt_tokens, 900)
        self.assertIn("[cached_input_tokens=800]", res.raw_response)

    def test_list_models_pages(self):
        pages = [Resp({"data": [{"id": "claude-opus-5-5"}], "has_more": True, "last_id": "claude-opus-5-5"}),
                 Resp({"data": [{"id": "claude-haiku-5-5"}], "has_more": False})]
        with mock.patch(GET, side_effect=pages) as get:
            self.assertEqual(self.make().list_models(), ["claude-haiku-5-5", "claude-opus-5-5"])
        self.assertEqual(get.call_args_list[1].kwargs["params"]["after_id"], "claude-opus-5-5")

    def test_full_page_is_sent_when_the_limits_allow_it(self):
        page = "<html><body>" + "".join(f"<p class='c{i}'>row {i}</p>" for i in range(3000)) + "</body></html>"
        p = self.make(tokens_per_minute=2000000)
        p.max_input_chars = 400000
        with mock.patch(POST, return_value=ok()):
            res = p.generate_xpaths(page, FIELDS)
        self.assertFalse(res.input_truncated)


class CostAndRunnerScriptTests(SimpleTestCase):
    def test_dated_model_id_uses_base_price(self):
        from types import SimpleNamespace

        from scraper.services.pipeline import ScrapePipeline
        holder = SimpleNamespace(pricing={"claude-opus-5-5": {"input": 4, "output": 20}})
        cost = ScrapePipeline._cost
        self.assertAlmostEqual(cost(holder, "claude-opus-5-5", 1_000_000, 100_000), 6.0)
        self.assertAlmostEqual(cost(holder, "claude-opus-5-5-20261001", 1_000_000, 0), 4.0)
        self.assertIsNone(cost(holder, "claude-opus-5-50", 1000, 0))      # not a dated id of the model
        self.assertIsNone(cost(holder, "other-model", 1000, 1000))

    def test_evaluate_script_runs_claude_on_all_sites(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "eval" / "evaluate.py"
        spec = importlib.util.spec_from_file_location("evaluate_script", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        commands = [cmd for _, cmd, _ in mod.SETUP + mod.LLM] + [mod.PING[1]]
        flat = [" ".join(c) for c in commands]
        self.assertFalse(any("eval_llm_scope" in c for c in flat))            # no two-site scope
        self.assertFalse(any("--sites" in c or "--reps" in c for c in flat))   # all sites, all reps
        switch = next(c for c in commands if "eval_switch_llm" in c)
        args = dict(zip(switch[3::2], switch[4::2]))
        self.assertEqual(args["--provider"], "anthropic")
        self.assertEqual(args["--model"], "claude-opus-5-5")
        self.assertEqual(args["--repetitions"], "3")
        steps = [c[2] for c in commands if c[1] == "manage.py"]
        self.assertLess(steps.index("eval_init"), steps.index("eval_switch_llm"))
        with mock.patch.object(mod, "dotenv", return_value={"SCRAPER_PRICING": '{"x": {"input": 1}}'}), \
                mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("SCRAPER_PRICING", None)
            pricing = json.loads(mod.child_env()["SCRAPER_PRICING"])
        self.assertEqual(pricing["claude-opus-5-5"], {"input": 4.0, "output": 20.0})
        self.assertIn("x", pricing)
