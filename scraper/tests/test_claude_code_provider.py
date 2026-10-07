"""
Claude Code provider tests (no network, no real CLI, no credentials): the
command and environment it uses, parsing of Claude Code's stream-json output
(shapes taken from Claude Code 2.1.293), subscription-only authentication,
error mapping (login, model, usage limits, overload), the single-turn rule,
token redaction, CLI discovery and a real subprocess round trip with a fake CLI.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase, override_settings

from scraper.llm.base import FieldSpec, LLMError, RateLimitExhausted
from scraper.llm.claude_code_provider import (
    ClaudeCodeProvider, parse_stream, redact, resolve_cli,
)
from scraper.llm.prompts import SYSTEM_PROMPT, build_prompt

FIELDS = [FieldSpec("title", "the product title"), FieldSpec("price", "the price", "price")]
ANSWER = {"title": "//h1", "price": "//span[@class='p']"}


def stream(answer=ANSWER, *, model="claude-opus-5-5", source="none", error=None, text=None,
           is_error=False, num_turns=1, usage=None, rate=None, retries=(), continued=0, subtype="success"):
    lines = [{"type": "system", "subtype": "init", "model": model, "apiKeySource": source, "tools": [],
              "claude_code_version": "2.1.293"}]
    for delay in retries:
        lines.append({"type": "system", "subtype": "api_retry", "attempt": 1, "retry_delay_ms": delay,
                      "error_status": 529, "error": "overloaded"})
    body = text if text is not None else json.dumps(answer)
    if error:
        lines.append({"type": "assistant", "error": error,
                      "message": {"model": "<synthetic>", "content": [{"type": "text", "text": body}]}})
    else:
        lines.append({"type": "assistant", "message": {"model": model, "content": [{"type": "thinking"}]}})
        for _ in range(continued):
            lines.append({"type": "user", "message": {"role": "user", "content": [
                {"type": "text", "text": "Output token limit hit. Resume directly"}]}})
        lines.append({"type": "assistant", "message": {"model": model, "content": [{"type": "text", "text": body}]}})
    if rate is not None:
        lines.append({"type": "rate_limit_event", "rate_limit_info": rate})
    lines.append({"type": "result", "subtype": subtype, "is_error": is_error, "result": body,
                  "num_turns": num_turns, "stop_reason": "end_turn", "total_cost_usd": 0.0123,
                  "duration_ms": 4100, "duration_api_ms": 3900,
                  "usage": usage or {"input_tokens": 50, "cache_creation_input_tokens": 900,
                                     "cache_read_input_tokens": 0, "output_tokens": 300,
                                     "output_tokens_details": {"thinking_tokens": 250}},
                  "modelUsage": {model: {"inputTokens": 50}}})
    return "\n".join(json.dumps(x) for x in lines) + "\n"


def proc(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout.encode("utf-8"), stderr=stderr.encode("utf-8"))


class FakeRun:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        out = self.outputs.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


class ClaudeCodeProviderTests(SimpleTestCase):
    def make(self, *outputs, **kw):
        self.sleeps = []
        self.tmp = tempfile.mkdtemp()
        self.fake = FakeRun(*outputs)
        return ClaudeCodeProvider(model="claude-opus-5-5", cli=["/opt/claude"], run=self.fake,
                                  sleep=self.sleeps.append, workdir=self.tmp, max_retries=2,
                                  max_wait_seconds=100, **kw)

    def test_command_environment_and_prompt(self):
        p = self.make(proc(stream()), reasoning_effort="medium")
        html = "<html><body><h1>x</h1></body></html>"
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-api03-SHOULDNOTPASS",
                                          "ANTHROPIC_BASE_URL": "http://elsewhere",
                                          "CLAUDE_CODE_OAUTH_TOKEN": "kept-by-claude-code"}):
            res = p.generate_xpaths(html, FIELDS)
        cmd, kw = self.fake.calls[0]
        self.assertEqual(cmd[:2], ["/opt/claude", "-p"])
        for flag in ("--no-session-persistence", "--safe-mode", "--strict-mcp-config", "--disable-slash-commands"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], "claude-opus-5-5")
        self.assertEqual(cmd[cmd.index("--effort") + 1], "medium")
        self.assertEqual(cmd[cmd.index("--tools") + 1], "")               # no tools at all
        self.assertEqual(cmd[cmd.index("--max-turns") + 1], "1")
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "stream-json")
        sys_file = Path(cmd[cmd.index("--system-prompt-file") + 1])
        self.assertEqual(sys_file.read_text(encoding="utf-8"), SYSTEM_PROMPT)   # project prompt, unchanged
        prompt, _ = build_prompt(html, FIELDS, None, p.max_input_chars,
                                 max_prompt_tokens=p.prompt_token_budget(12300))
        self.assertEqual(kw["input"], prompt.encode("utf-8"))                 # user prompt via stdin, unchanged
        env = kw["env"]
        for blocked in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"):
            self.assertNotIn(blocked, env)
        self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "kept-by-claude-code")  # passed through, never read
        self.assertEqual(env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "12300")
        self.assertEqual(env["DISABLE_AUTOUPDATER"], "1")
        self.assertEqual(kw["cwd"], str(p.workdir))
        self.assertEqual(list(p.workdir.iterdir()), [])                       # empty working folder
        # parsed reply
        self.assertEqual(res.xpaths, ANSWER)
        self.assertEqual(res.model, "claude-opus-5-5")
        self.assertEqual((res.prompt_tokens, res.completion_tokens), (950, 300))   # cache counted as input
        self.assertEqual(res.latency_ms, 3900)                                # model time, no CLI start-up
        self.assertIn("[thinking_tokens=250]", res.raw_response)
        self.assertIn("auth none", res.raw_response)
        self.assertNotIn("kept-by-claude-code", res.raw_response)

    def test_api_key_source_is_refused(self):
        p = self.make(proc(stream(source="ANTHROPIC_API_KEY")))
        with self.assertRaisesRegex(LLMError, "HTTP 403 .*API key"):
            p.generate_xpaths("<html></html>", FIELDS)

    def test_login_problems_are_fatal(self):
        out = stream(error="authentication_failed", text="Not logged in · Please run /login", is_error=True)
        p = self.make(proc(out, returncode=1))
        with self.assertRaisesRegex(LLMError, r"HTTP 401 Claude Code authentication_failed"):
            p.generate_xpaths("<html></html>", FIELDS)
        self.assertEqual(self.sleeps, [])

    def test_unknown_model_is_fatal(self):
        out = stream(error="model_not_found", text="There's an issue with the selected model", is_error=True)
        with self.assertRaisesRegex(LLMError, "HTTP 404"):
            self.make(proc(out, returncode=1)).generate_xpaths("<html></html>", FIELDS)

    def test_usage_limit_pauses_until_reset(self):
        resets = int(time.time()) + 7200
        out = stream(error="rate_limit", text="You've hit your session limit", is_error=True,
                     rate={"status": "rejected", "resetsAt": resets, "rateLimitType": "five_hour"})
        p = self.make(proc(out, returncode=1))
        with self.assertRaises(RateLimitExhausted) as ctx:
            p.generate_xpaths("<html></html>", FIELDS)
        self.assertAlmostEqual(ctx.exception.retry_after, 7200, delta=30)
        self.assertEqual(self.sleeps, [])                          # waited out by eval_run --wait
        self.assertEqual(p.last_rate_limits["rateLimitType"], "five_hour")

    def test_overload_is_retried_and_waits_reported(self):
        bad = stream(error="server_error", text="API Error: 529 Overloaded", is_error=True, retries=[600])
        p = self.make(proc(bad, returncode=1), proc(stream(retries=[400])))
        res = p.generate_xpaths("<html></html>", FIELDS)
        self.assertEqual(res.xpaths, ANSWER)
        self.assertEqual(self.sleeps, [2])
        self.assertEqual(res.rate_limit_wait_ms, 2000 + 600 + 400)   # our backoff + Claude Code's retries
        self.assertEqual(res.retries, 1 + 1 + 1)

    def test_transient_failures_end_in_a_retryable_error(self):
        p = self.make(subprocess.TimeoutExpired("claude", 900), proc("", returncode=1, stderr="boom"),
                      proc("", returncode=1, stderr="boom"))
        with self.assertRaises(LLMError) as ctx:
            p.generate_xpaths("<html></html>", FIELDS)
        self.assertNotRegex(str(ctx.exception), r"HTTP 4\d\d")           # pipeline: retryable stop
        self.assertEqual(len(self.fake.calls), 3)

    def test_continued_reply_is_discarded_and_resent_with_a_larger_cap(self):
        cut = stream(continued=3, num_turns=4, text="partial", error=None,
                     usage={"input_tokens": 400, "output_tokens": 12000})
        p = self.make(proc(cut), proc(stream()))
        res = p.generate_xpaths("<html></html>", FIELDS)
        first, second = (kw["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] for _, kw in self.fake.calls)
        self.assertEqual(int(second), 2 * int(first))
        self.assertEqual(res.xpaths, ANSWER)
        self.assertEqual((res.prompt_tokens, res.completion_tokens), (400 + 950, 12000 + 300))
        self.assertTrue(res.raw_response.startswith("[claude_code_continued:3"))

    def test_twice_continued_gives_no_answer(self):
        cut = stream(continued=1, num_turns=2, text=json.dumps(ANSWER))
        p = self.make(proc(cut), proc(cut))
        res = p.generate_xpaths("<html></html>", FIELDS)
        self.assertEqual(res.xpaths, {})                       # never an answer from a continued chat
        self.assertIn("[finish_reason=length", res.raw_response)

    def test_other_model_is_refused(self):
        out = stream(model="claude-sonnet-5-5")
        with self.assertRaisesRegex(LLMError, "HTTP 409 .*claude-sonnet-5-5"):
            self.make(proc(out)).generate_xpaths("<html></html>", FIELDS)

    def test_auth_status_keeps_only_the_method(self):
        status = {"loggedIn": True, "authMethod": "oauth_token", "apiProvider": "firstParty",
                  "email": "someone@example.com", "configDirectory": "C:/Users/x/.claude"}
        p = self.make(proc(json.dumps(status)))
        self.assertEqual(p.auth_status(), {"loggedIn": True, "authMethod": "oauth_token",
                                           "apiProvider": "firstParty"})
        self.assertEqual(self.fake.calls[0][0][1:], ["auth", "status", "--json"])

    def test_redaction(self):
        self.assertEqual(redact("token sk-ant-oat01-AbC_123-xyz end"), "token sk-ant-[redacted] end")
        p = self.make(proc("", returncode=1, stderr="bad sk-ant-oat01-SECRETSECRET"),
                      proc("", returncode=1), proc("", returncode=1))
        with self.assertRaises(LLMError) as ctx:
            p.generate_xpaths("<html></html>", FIELDS)
        self.assertNotIn("SECRETSECRET", str(ctx.exception))

    def test_parse_stream_of_a_cut_reply(self):
        out = parse_stream(stream(error="max_output_tokens", text="API Error: exceeded", is_error=True))
        self.assertEqual(out["continued"], 1)


class ResolveCliTests(SimpleTestCase):
    def test_npm_shim_uses_the_native_binary(self):
        tmp = Path(tempfile.mkdtemp())
        shim = tmp / "claude.cmd"
        shim.write_text("@echo off")
        exe = tmp / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        exe.parent.mkdir(parents=True)
        exe.write_text("")
        self.assertEqual(resolve_cli(str(shim)), [str(exe)])

    def test_missing_cli(self):
        with mock.patch("scraper.llm.claude_code_provider.shutil.which", return_value=None), \
                mock.patch("scraper.llm.claude_code_provider.Path.home", return_value=Path(tempfile.mkdtemp())):
            with self.assertRaisesRegex(ValueError, "Claude Code"):
                resolve_cli("/nonexistent/claude")

    def test_factory_builds_the_provider(self):
        from scraper.llm.factory import get_provider

        tmp = Path(tempfile.mkdtemp()) / ("claude.exe" if os.name == "nt" else "claude")
        tmp.write_text("")
        with override_settings(SCRAPER={"CLAUDE_CODE_CLI": str(tmp), "LLM_MAX_RETRIES": 6,
                                        "LLM_MAX_WAIT_SECONDS": 900}):
            p = get_provider("claude_code", "claude-opus-5-5", reasoning_effort="medium", max_input_chars=200000)
        self.assertEqual((p.name, p.model, p.cli, p.reasoning_effort, p.max_input_chars),
                         ("claude_code", "claude-opus-5-5", [str(tmp)], "medium", 200000))


FAKE_CLI = r'''
import json, os, sys
args = sys.argv[1:]
prompt = sys.stdin.buffer.read().decode("utf-8")
sysprompt = open(args[args.index("--system-prompt-file") + 1], encoding="utf-8").read()
info = {"api_key_seen": "ANTHROPIC_API_KEY" in os.environ, "prompt_len": len(prompt),
        "has_field": "title" in prompt, "sys_ok": sysprompt.startswith("You are an expert"),
        "tools": args[args.index("--tools") + 1], "max": os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS")}
lines = [{"type": "system", "subtype": "init", "model": args[args.index("--model") + 1],
          "apiKeySource": "none", "claude_code_version": "fake"},
         {"type": "assistant", "message": {"model": "claude-opus-5-5", "content": [{"type": "text", "text": "x"}]}},
         {"type": "result", "subtype": "success", "is_error": False, "num_turns": 1,
          "result": json.dumps({"title": "//h1", "_info": json.dumps(info)}),
          "duration_api_ms": 5, "usage": {"input_tokens": 3, "output_tokens": 2}}]
for line in lines:
    print(json.dumps(line))
'''


class FakeCliRoundTripTests(SimpleTestCase):
    def test_real_subprocess_with_a_fake_cli(self):
        tmp = Path(tempfile.mkdtemp())
        script = tmp / "fake_claude.py"
        script.write_text(FAKE_CLI, encoding="utf-8")
        p = ClaudeCodeProvider(model="claude-opus-5-5", cli=[sys.executable, str(script)], workdir=str(tmp / "w"))
        fields = [FieldSpec("title", "the title")]
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-api03-NOPE"}):
            res = p.generate_xpaths("<html><body><h1>é ünïcode</h1></body></html>", fields)
        self.assertEqual(res.xpaths, {"title": "//h1"})
        info = json.loads(json.loads(res.raw_response[res.raw_response.index("{"):])["_info"])
        self.assertEqual(info["api_key_seen"], False)
        self.assertTrue(info["has_field"] and info["sys_ok"])
        self.assertEqual(info["tools"], "")
        self.assertEqual(info["max"], "12150")


class PingRecordTests(SimpleTestCase):
    def test_eval_check_ping_records_claude_code_without_secrets(self):
        from io import StringIO
        from types import SimpleNamespace

        from django.conf import settings

        from scraper.management.commands.eval_check import Command

        tmp = tempfile.mkdtemp()
        status = json.dumps({"loggedIn": True, "authMethod": "oauth_token", "email": "x@example.com"})
        fake = FakeRun(proc("2.1.293 (Claude Code)"), proc(status),
                       proc(stream({"title": "//h1[@class='title']"},
                                   rate={"status": "allowed", "rateLimitType": "five_hour"})))
        p = ClaudeCodeProvider(model="claude-opus-5-5", cli=["/opt/claude"], run=fake, workdir=tmp)
        cmd = Command(stdout=StringIO())
        problems, notes = [], []
        scraper_cfg = {**settings.SCRAPER, "DATA_DIR": tmp}
        with override_settings(SCRAPER=scraper_cfg):
            cmd._ping(SimpleNamespace(name="t-exp"), p, {"provider": "claude_code", "model": "claude-opus-5-5"},
                      problems, notes)
            record = json.loads((Path(tmp) / "t-exp" / "llm_ping.json").read_text())
        self.assertEqual(problems, [])
        self.assertTrue(record["ok"])
        self.assertEqual(record["claude_code"]["auth"], {"loggedIn": True, "authMethod": "oauth_token"})
        self.assertEqual(record["claude_code_call"]["api_key_source"], "none")
        self.assertIsNone(record["tokens_per_minute_reported"])
        self.assertFalse(record["completion_counts_in_tpm_observed"])
        self.assertNotIn("example.com", json.dumps(record))
