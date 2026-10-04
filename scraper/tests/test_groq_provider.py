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
