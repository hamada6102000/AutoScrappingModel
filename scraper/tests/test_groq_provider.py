"""
Unit tests for the Groq provider using a mocked HTTP layer, so the OpenAI-
compatible request/parse path is verified without a real API key.
"""

import json
from unittest import mock

from django.test import SimpleTestCase

from scraper.llm.base import FieldSpec
from scraper.llm.groq_provider import GroqProvider

FIELDS = [
    FieldSpec("title", "the product title", "string"),
    FieldSpec("price", "the product price", "price"),
]


class _FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")


class GroqProviderTests(SimpleTestCase):
    def test_requires_key(self):
        with self.assertRaises(ValueError):
            GroqProvider(api_key="")

    def test_parses_response_and_usage(self):
        content = json.dumps({
            "title": '//h1[@class="t"]',
            "price": '//span[@class="p"]',
        })
        payload = {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 321, "completion_tokens": 42},
        }
        provider = GroqProvider(api_key="gsk_test", model="llama-3.3-70b-versatile")

        with mock.patch("scraper.llm.groq_provider.requests.post",
                        return_value=_FakeResp(payload)) as post:
            result = provider.generate_xpaths("<html></html>", FIELDS)

        # Correct endpoint + auth header were used.
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer gsk_test")
        self.assertEqual(kwargs["json"]["model"], "llama-3.3-70b-versatile")

        # Parsed XPaths + token usage.
        self.assertEqual(result.xpaths["title"], '//h1[@class="t"]')
        self.assertEqual(result.xpaths["price"], '//span[@class="p"]')
        self.assertEqual(result.prompt_tokens, 321)
        self.assertEqual(result.completion_tokens, 42)

    def test_retries_without_json_mode_on_400(self):
        good = {
            "choices": [{"message": {"content": '{"title": "//h1", "price": "//span"}'}}],
            "usage": {},
        }
        responses = [
            _FakeResp({"error": "response_format not supported"}, status_code=400),
            _FakeResp(good),
        ]
        provider = GroqProvider(api_key="gsk_test")

        with mock.patch("scraper.llm.groq_provider.requests.post",
                        side_effect=responses) as post:
            result = provider.generate_xpaths("<html></html>", FIELDS)

        self.assertEqual(post.call_count, 2)
        # Second call dropped the response_format field.
        self.assertNotIn("response_format", post.call_args_list[1].kwargs["json"])
        self.assertEqual(result.xpaths["title"], "//h1")
