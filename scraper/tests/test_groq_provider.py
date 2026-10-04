"""
Groq provider tests with a mocked HTTP layer: request shape, usage parsing,
rate-limit waiting (accounted separately from latency), quota exhaustion,
oversized prompts and JSON-mode fallback — all without a real API key.
"""

import json
from unittest import mock

from django.test import SimpleTestCase

from scraper.llm.base import FieldSpec, LLMError, RateLimitExhausted, RequestTooLarge
from scraper.llm.groq_provider import GroqProvider, _parse_wait_seconds

FIELDS = [FieldSpec("title", "the product title"), FieldSpec("price", "the price", "price")]
POST = "scraper.llm.groq_provider.requests.post"


class Resp:
    def __init__(self, payload=None, status_code=200, headers=None, text=None):
        self._payload = payload or {}
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


def ok(content=None, usage=None):
    content = content or json.dumps({"title": '//h1[@class="t"]', "price": '//span[@class="p"]'})
    return Resp({"choices": [{"message": {"content": content}}], "model": "llama-3.3-70b-versatile",
                 "usage": usage or {"prompt_tokens": 321, "completion_tokens": 42}})


class GroqProviderTests(SimpleTestCase):
    def setUp(self):
        self.sleeps = []
        self.provider = GroqProvider(api_key="gsk_test", sleep=self.sleeps.append,
                                     max_retries=3, max_wait_seconds=100)

    def test_requires_key(self):
        with self.assertRaises(ValueError):
            GroqProvider(api_key="")

    def test_request_and_parsing(self):
        with mock.patch(POST, return_value=ok()) as post:
            res = self.provider.generate_xpaths("<html><body><h1 class='t'>x</h1></body></html>", FIELDS)
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer gsk_test")
        self.assertEqual(kwargs["json"]["temperature"], 0)
        self.assertEqual(res.xpaths["price"], '//span[@class="p"]')
        self.assertEqual((res.prompt_tokens, res.completion_tokens), (321, 42))
        self.assertEqual(res.retries, 0)
        self.assertEqual(res.rate_limit_wait_ms, 0)
        self.assertEqual(len(res.prompt_sha), 64)
        self.assertGreater(res.input_chars, 0)

    def test_429_waits_and_records_wait_separately(self):
        responses = [Resp(status_code=429, headers={"retry-after": "2"}, text="rate limit"), ok()]
        with mock.patch(POST, side_effect=responses):
            res = self.provider.generate_xpaths("<html></html>", FIELDS)
        self.assertEqual(self.sleeps, [2.0])
        self.assertEqual(res.retries, 1)
        self.assertEqual(res.rate_limit_wait_ms, 2000)
        self.assertLess(res.latency_ms, 2000)  # latency excludes the wait

    def test_wait_parsed_from_message(self):
        resp = Resp(status_code=429, text="Rate limit reached ... Please try again in 1m2.5s.")
        self.assertAlmostEqual(_parse_wait_seconds(resp), 62.5)

    def test_quota_exhausted_raises(self):
        responses = [Resp(status_code=429, headers={"retry-after": "500"}, text="TPD limit")]
        with mock.patch(POST, side_effect=responses):
            with self.assertRaises(RateLimitExhausted):
                self.provider.generate_xpaths("<html></html>", FIELDS)
        self.assertEqual(self.sleeps, [])  # never sleeps past the configured maximum

    def test_request_too_large(self):
        with mock.patch(POST, return_value=Resp(status_code=413, text="too large")):
            with self.assertRaises(RequestTooLarge):
                self.provider.generate_xpaths("<html></html>", FIELDS)

    def test_server_error_retries_then_succeeds(self):
        with mock.patch(POST, side_effect=[Resp(status_code=503, text="busy"), ok()]):
            res = self.provider.generate_xpaths("<html></html>", FIELDS)
        self.assertEqual(res.retries, 1)

    def test_other_client_error(self):
        with mock.patch(POST, return_value=Resp(status_code=401, text="invalid api key")):
            with self.assertRaises(LLMError):
                self.provider.generate_xpaths("<html></html>", FIELDS)

    def test_json_mode_fallback(self):
        responses = [Resp(status_code=400, text="response_format not supported"), ok()]
        with mock.patch(POST, side_effect=responses) as post:
            res = self.provider.generate_xpaths("<html></html>", FIELDS)
        self.assertNotIn("response_format", post.call_args_list[1].kwargs["json"])
        self.assertEqual(res.xpaths["title"], '//h1[@class="t"]')

    def test_input_truncation_flagged(self):
        provider = GroqProvider(api_key="k", sleep=lambda s: None, max_input_chars=50)
        with mock.patch(POST, return_value=ok()):
            res = provider.generate_xpaths("<html>" + "x" * 500 + "</html>", FIELDS)
        self.assertTrue(res.input_truncated)


class ReasoningModelTests(SimpleTestCase):
    """openai/gpt-oss-* options, per-minute token budget, JSON-mode failures."""

    def make(self, **kw):
        return GroqProvider(api_key="gsk_test", model="openai/gpt-oss-120b", sleep=lambda s: None, **kw)

    def test_reasoning_options_and_completion_budget_are_sent(self):
        p = self.make(reasoning_effort="low", include_reasoning=False)
        with mock.patch(POST, return_value=ok()) as post:
            p.generate_xpaths("<html><body><h1>x</h1></body></html>", FIELDS)
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "openai/gpt-oss-120b")
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertIs(body["include_reasoning"], False)
        self.assertNotIn("max_tokens", body)
        self.assertGreaterEqual(body["max_completion_tokens"], 2000)  # reasoning reserve

    def test_no_reasoning_options_unless_configured(self):
        with mock.patch(POST, return_value=ok()) as post:
            self.make().generate_xpaths("<html><body><h1>x</h1></body></html>", FIELDS)
        body = post.call_args.kwargs["json"]
        self.assertNotIn("reasoning_effort", body)
        self.assertNotIn("include_reasoning", body)

    def test_prompt_is_shortened_to_fit_the_per_minute_limit(self):
        from scraper.llm.prompts import SYSTEM_PROMPT, conservative_counter
        page = "<html><body>" + "".join(f"<p class='c{i}'>row {i} text</p>" for i in range(3000)) + "</body></html>"
        p = self.make(reasoning_effort="low", tokens_per_minute=8000)
        with mock.patch(POST, return_value=ok()) as post:
            res = p.generate_xpaths(page, FIELDS)
        body = post.call_args.kwargs["json"]
        prompt = body["messages"][1]["content"]
        count = conservative_counter()
        self.assertTrue(res.input_truncated)
        self.assertLessEqual(count(SYSTEM_PROMPT + prompt) + body["max_completion_tokens"], int(8000 * 0.85))
        self.assertIn("[input truncated]", prompt)

    def test_json_validate_failed_is_a_model_answer_not_an_outage(self):
        err = Resp({"error": {"code": "json_validate_failed", "message": "Failed to generate JSON",
                              "failed_generation": '{"title": "//h1"'}}, status_code=400)
        with mock.patch(POST, return_value=err):
            res = self.make().generate_xpaths("<html><body><h1>x</h1></body></html>", FIELDS)
        self.assertTrue(res.raw_response.startswith("[json_validate_failed]"))
        self.assertEqual(res.xpaths, {})

    def test_truncated_answer_is_flagged(self):
        resp = ok()
        resp._payload["choices"][0]["finish_reason"] = "length"
        with mock.patch(POST, return_value=resp):
            res = self.make().generate_xpaths("<html><body><h1>x</h1></body></html>", FIELDS)
        self.assertTrue(res.raw_response.startswith("[finish_reason=length]"))

    def _page(self):
        return "<html><body>" + "".join(f"<p class='c{i}'>row {i} text</p>" for i in range(3000)) + "</body></html>"

    def test_request_too_large_is_resent_shorter(self):
        from scraper.llm.prompts import SYSTEM_PROMPT, conservative_counter
        count = conservative_counter()
        too_large = Resp(status_code=413, text='{"error":{"message":"Request too large for model ... on tokens '
                                                'per minute (TPM): Limit 8000, Requested 8259, please reduce"}}')
        p = self.make(reasoning_effort="low", tokens_per_minute=8000, completion_counts_in_tpm=False)
        with mock.patch(POST, side_effect=[too_large, ok()]) as post:
            res = p.generate_xpaths(self._page(), FIELDS)
        first = post.call_args_list[0].kwargs["json"]["messages"][1]["content"]
        second = post.call_args_list[1].kwargs["json"]["messages"][1]["content"]
        n1, n2 = count(SYSTEM_PROMPT + first), count(SYSTEM_PROMPT + second)
        self.assertLessEqual(n1, 6800)
        expected = min(n1 - 259, int(n1 * 8000 / 8259)) - 400
        self.assertLessEqual(n2, expected)
        self.assertGreater(n2, expected - 50)       # shortened by what was needed, not more
        self.assertTrue(res.input_truncated)
        self.assertEqual(res.retries, 1)
        self.assertTrue(res.raw_response.startswith(
            f"[request_too_large_resent: 8259>8000 tokens: prompt budget 6800->{expected}] "))
        self.assertEqual(res.xpaths["title"], '//h1[@class="t"]')
        # the next request is planned as before (no state carried over)
        with mock.patch(POST, return_value=ok()) as post:
            p.generate_xpaths(self._page(), FIELDS)
        self.assertEqual(post.call_args.kwargs["json"]["messages"][1]["content"], first)

    def test_request_too_large_gives_up_after_three_resends(self):
        too_large = Resp(status_code=413, text="Limit 8000, Requested 9000")
        p = self.make(tokens_per_minute=8000, completion_counts_in_tpm=False)
        with mock.patch(POST, return_value=too_large) as post:
            with self.assertRaises(RequestTooLarge):
                p.generate_xpaths(self._page(), FIELDS)
        self.assertEqual(post.call_count, 4)

    def test_request_too_large_without_figures_is_not_resent(self):
        p = self.make(tokens_per_minute=8000)
        with mock.patch(POST, return_value=Resp(status_code=413, text="too large")) as post:
            with self.assertRaises(RequestTooLarge):
                p.generate_xpaths(self._page(), FIELDS)
        self.assertEqual(post.call_count, 1)

    def test_list_models_and_tpm_probe(self):
        models = Resp({"data": [{"id": "openai/gpt-oss-120b", "active": True},
                                {"id": "old-model", "active": False}]})
        with mock.patch("scraper.llm.groq_provider.requests.get", return_value=models):
            self.assertEqual(self.make().list_models(), ["openai/gpt-oss-120b"])
        too_large = Resp(status_code=413, text="Request too large ... Limit 8000, Requested 8212")
        with mock.patch(POST, return_value=too_large):
            self.assertTrue(self.make().probe_completion_accounting(8000))
        with mock.patch(POST, return_value=ok()):
            self.assertFalse(self.make().probe_completion_accounting(8000))
