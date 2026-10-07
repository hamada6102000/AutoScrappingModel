"""
Smoke test of the Claude Code route: find the CLI, show how it will be
called, check its login (method only, never the credential) and send ONE
tiny request through it. Uses no database and no experiment data.

    python manage.py eval_claude_smoke                # one request
    python manage.py eval_claude_smoke --no-request   # checks only, no request

Writes eval_data/claude_code_smoke.json and eval_data/claude_code_smoke_log.txt
(no token is ever read, so none can appear there).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from lxml import html as lxml_html

from scraper.llm.base import FieldSpec
from scraper.llm.claude_code_provider import (
    SUBSCRIPTION_KEY_SOURCES, ClaudeCodeProvider, describe_invocation, redact, resolve_cli,
)

PING_HTML = "<html><body><main><h1 class='title'>Ping</h1></main></body></html>"


class Command(BaseCommand):
    help = "Check Claude Code (subscription login) and send one tiny request through it."

    def add_arguments(self, parser):
        parser.add_argument("--model", default=None)
        parser.add_argument("--effort", default="medium")
        parser.add_argument("--no-request", action="store_true", help="Check the CLI and login only.")

    def handle(self, *args, **opts):
        out_dir = Path(settings.SCRAPER.get("DATA_DIR") or "eval_data")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._lines: list[str] = []
        record: dict = {"at": dt.datetime.now().isoformat(timespec="seconds"), "ok": False}
        try:
            self._run(opts, record)
        finally:
            (out_dir / "claude_code_smoke.json").write_text(redact(json.dumps(record, indent=2)), encoding="utf-8")
            (out_dir / "claude_code_smoke_log.txt").write_text(redact("\n".join(self._lines)) + "\n",
                                                               encoding="utf-8")
        if not record["ok"]:
            raise CommandError(record.get("problem") or "Claude Code smoke test failed")

    def say(self, text: str, style=None) -> None:
        text = redact(text)
        self._lines.append(text)
        self.stdout.write(style(text) if style else text)

    def _fail(self, record: dict, problem: str) -> None:
        record["problem"] = problem
        self.say(f"\nFAILED: {problem}", self.style.ERROR)

    def _run(self, opts, record) -> None:
        model = opts["model"] or settings.SCRAPER.get("CLAUDE_CODE_MODEL") or "claude-opus-5-5"
        record["model_requested"] = model
        try:
            cli = resolve_cli(settings.SCRAPER.get("CLAUDE_CODE_CLI") or None)
        except ValueError as exc:
            return self._fail(record, str(exc))
        provider = ClaudeCodeProvider(model=model, cli=cli, reasoning_effort=opts["effort"],
                                      max_input_chars=200000, max_retries=2, max_wait_seconds=120)
        how = describe_invocation(provider)
        record["invocation"] = how
        record["cli_version"] = provider.cli_version()
        self.say(f"Claude Code CLI : {cli[-1]}")
        self.say(f"Version         : {record['cli_version']}")
        self.say("Command         : " + " ".join(f'"{a}"' if (" " in a or a == "") else a for a in how["command"]))
        self.say(f"stdin           : {how['stdin']}")
        self.say(f"Working folder  : {how['cwd']} (empty)")
        self.say("Environment     : sets " + ", ".join(f"{k}={v}" for k, v in how["set_env"].items()))
        self.say("                  removes " + (", ".join(how["removed_env"]) or "nothing (none of "
                                                 "ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL is set)"))
        self.say("CLAUDE_CODE_OAUTH_TOKEN in this environment: "
                 + ("set (value not read or shown)" if how["oauth_token_variable_set"] else "not set"))

        auth = provider.auth_status()
        record["auth"] = auth
        self.say(f"Claude Code login: {json.dumps(auth)}")
        if not auth.get("loggedIn"):
            return self._fail(record, (
                "Claude Code is not signed in for programs started from here. The token from "
                "'claude setup-token' is only printed, not stored: add it as a Windows user environment "
                "variable named CLAUDE_CODE_OAUTH_TOKEN (Start > 'Edit environment variables for your "
                "account' > New), or run 'claude' once and sign in with /login. Then run this test again."))
        method = str(auth.get("authMethod") or "")
        if "api" in method.lower() and "key" in method.lower():
            return self._fail(record, f"Claude Code is set to use an API key ({method}), not the subscription.")
        if opts["no_request"]:
            record["ok"] = True
            self.say("\nOK (no request sent).", self.style.SUCCESS)
            return

        self.say(f"\nSending one request to {model} (effort {opts['effort']}) through Claude Code ...")
        start = time.perf_counter()
        try:
            res = provider.generate_xpaths(PING_HTML, [FieldSpec("title", "the page title")])
        except Exception as exc:  # report every failure plainly
            return self._fail(record, f"{type(exc).__name__}: {exc}")
        wall = int((time.perf_counter() - start) * 1000)
        call = provider.last_call
        xpath = res.xpaths.get("title", "")
        try:
            values = lxml_html.fromstring(PING_HTML).xpath(xpath) if xpath else []
        except Exception as exc:
            values = [f"<invalid XPath: {exc}>"]
        found = [v.text_content().strip() if hasattr(v, "text_content") else str(v).strip() for v in values]
        record.update({
            "model_reported": res.model, "models_in_reply": call.get("models"),
            "api_key_source": call.get("api_key_source"), "num_turns": call.get("num_turns"),
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "latency_ms": res.latency_ms, "wall_ms": wall, "rate_limit_wait_ms": res.rate_limit_wait_ms,
            "api_equivalent_cost_usd": call.get("cost_usd"), "rate_limit": provider.last_rate_limits,
            "xpath": xpath, "xpath_values": found, "raw_response": res.raw_response[:500],
        })
        self.say(f"Model that answered : {res.model}")
        self.say(f"Credential source   : apiKeySource={call.get('api_key_source')!r} "
                 "('none' = Claude subscription login, no API key)")
        self.say(f"Turns               : {call.get('num_turns')}")
        self.say(f"Tokens              : {res.prompt_tokens} in (incl. cache) + {res.completion_tokens} out")
        self.say(f"Model time          : {res.latency_ms} ms (whole CLI call {wall} ms)")
        self.say(f"Subscription window : {json.dumps(provider.last_rate_limits)}")
        self.say(f"Returned XPath      : {xpath!r} -> {found}")
        if call.get("api_key_source") not in SUBSCRIPTION_KEY_SOURCES:
            return self._fail(record, f"the request did not use the subscription login "
                                      f"(apiKeySource={call.get('api_key_source')!r})")
        if res.model != model:
            return self._fail(record, f"answered by {res.model}, not {model}")
        if found != ["Ping"]:
            return self._fail(record, f"expected the XPath to select 'Ping', got {found}")
        record["ok"] = True
        self.say("\nOK: the project sent one request through Claude Code with your subscription login "
                 f"and {model} returned a correct XPath.", self.style.SUCCESS)
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.say("(ANTHROPIC_API_KEY is set on this computer; it was removed from Claude Code's "
                     "environment and not used.)")
