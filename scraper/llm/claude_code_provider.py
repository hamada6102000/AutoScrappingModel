"""
Claude through the installed Claude Code CLI, authenticated with the user's
Claude subscription (Claude Code's own OAuth login), with no Anthropic API key.

How a request is made (one ``generate_xpaths`` call = one CLI process)::

    claude -p --output-format stream-json --verbose
           --model <model> [--effort <effort>]
           --system-prompt-file <file holding prompts.SYSTEM_PROMPT>
           --tools "" --max-turns 1 --no-session-persistence
           --safe-mode --strict-mcp-config --disable-slash-commands
    stdin  = the user prompt built by prompts.build_prompt (unchanged)
    cwd    = an empty, fixed working directory under the system temp folder

Authentication is entirely Claude Code's: it reads its own credentials (the
CLAUDE_CODE_OAUTH_TOKEN variable created with ``claude setup-token``, or the
login stored by ``claude /login``). This module never reads, stores, logs or
passes a token. It removes ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
ANTHROPIC_BASE_URL and the Bedrock/Vertex switches from the child process so
Claude Code cannot fall back to API-key billing or another endpoint, and every
reply is checked: the CLI must report ``apiKeySource`` "none" (subscription
OAuth) and the configured model, otherwise the call fails.

What Claude Code adds to the request (it cannot be switched off without
losing OAuth): a billing-header block and the one-line identity "You are a
Claude agent, built on Anthropic's Claude Agent SDK." before the project's
system prompt, and an environment note (working directory, platform, model id,
date) after the user prompt. Sampling settings are the model defaults
(adaptive thinking, ``effort``); ``max_tokens`` is set through
CLAUDE_CODE_MAX_OUTPUT_TOKENS with the same allowance as the API provider.

Single-turn guarantee: if Claude Code had to continue a reply (output cut by
the token cap, or a thinking-only reply nudged), that answer came from a
different conversation and is discarded; the request is sent again once with
twice the cap, as the API provider does, and both calls are billed.

Limits: transient 429 / 529 / 5xx answers are retried inside Claude Code
(its waits are reported as rate-limit wait). A subscription usage limit
(5-hour or weekly window, ``rate_limit_event`` "rejected") raises
``RateLimitExhausted`` with the reset time, so ``eval_run --wait`` sleeps and
resumes. Authentication, model and request errors are fatal ("HTTP 4xx").
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from .anthropic_provider import completion_tokens_for
from .base import FailureHint, FieldSpec, LLMError, LLMProvider, LLMResult, RateLimitExhausted
from .prompts import SYSTEM_PROMPT, build_prompt, parse_xpath_json, prompt_token_budget

# Removed from the Claude Code process: API-key auth, other endpoints / clouds.
BLOCKED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL",
               "ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
               "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR", "CLAUDE_CODE_SIMPLE")
FIXED_ENV = {
    "DISABLE_AUTOUPDATER": "1",                         # same CLI version for the whole run
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",    # no telemetry / update checks
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",              # no CLAUDE.md memory files in the prompt
}
SUBSCRIPTION_KEY_SOURCES = {"none", "oauth"}
FATAL_ERRORS = {
    "authentication_failed": 401, "oauth_org_not_allowed": 403, "account_on_hold": 403,
    "verification_required": 403, "cloud_credential_error": 401, "model_not_found": 404,
    "invalid_request": 400,
}
AUTH_STATUS_KEYS = ("loggedIn", "authMethod", "apiProvider", "subscriptionType", "tokenSource", "apiKeySource")
_SECRET = re.compile(r"sk-ant-[A-Za-z0-9_\-]{6,}")


def redact(text: str) -> str:
    """Remove anything shaped like an Anthropic credential before text is stored or shown."""
    return _SECRET.sub("sk-ant-[redacted]", text or "")


def resolve_cli(configured: Optional[str] = None) -> list[str]:
    """The command prefix that starts Claude Code, e.g. [".../claude.exe"].

    npm installs a ``claude.cmd`` shim on Windows; running it would pass every
    argument through cmd.exe, so the native binary (or cli.js with node)
    behind the shim is used instead."""
    candidates = []
    if configured:
        candidates.append(configured)
    found = shutil.which("claude")
    if found:
        candidates.append(found)
    home = Path.home()
    candidates += [str(home / ".local" / "bin" / ("claude.exe" if os.name == "nt" else "claude")),
                   str(home / ".claude" / "local" / ("claude.exe" if os.name == "nt" else "claude"))]
    for cand in candidates:
        path = Path(cand)
        if not path.is_file():
            continue
        if path.suffix.lower() in (".cmd", ".bat", ".ps1"):
            pkg = path.parent / "node_modules" / "@anthropic-ai" / "claude-code"
            for exe in (pkg / "bin" / "claude.exe", pkg / "bin" / "claude"):
                if exe.is_file():
                    return [str(exe)]
            node = shutil.which("node")
            if node and (pkg / "cli.js").is_file():
                return [node, str(pkg / "cli.js")]
            continue
        return [str(path)]
    raise ValueError(
        "Claude Code (the 'claude' command) was not found. Install Claude Code and sign in with your "
        "Claude subscription (claude setup-token), or set CLAUDE_CODE_CLI to the full path of claude.exe.")


class ClaudeCodeProvider(LLMProvider):
    name = "claude_code"
    # Subscription usage is metered in 5-hour / weekly windows, not tokens per minute.
    completion_counts_in_tpm_documented = False
    limits_note = ("not applicable: Claude subscription usage windows (5-hour / weekly), "
                   "no per-minute token limit")

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        cli: Optional[list[str]] = None,
        timeout: int = 900,
        max_retries: int = 6,
        max_wait_seconds: int = 900,
        max_input_chars: int = 60000,
        sleep=time.sleep,
        reasoning_effort: Optional[str] = None,
        tokens_per_minute: Optional[int] = None,
        completion_counts_in_tpm: bool = False,
        include_reasoning: Optional[bool] = None,   # accepted for config compatibility; unused
        run=subprocess.run,
        workdir: Optional[str] = None,
        extra_env: Optional[dict] = None,           # tests only (e.g. a local mock endpoint)
    ):
        self.model = model
        self.cli = list(cli) if cli else resolve_cli(os.environ.get("CLAUDE_CODE_CLI"))
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_wait_seconds = max_wait_seconds
        self.max_input_chars = max_input_chars
        self._sleep = sleep
        self.reasoning_effort = reasoning_effort
        self.tokens_per_minute = tokens_per_minute
        self.completion_counts_in_tpm = completion_counts_in_tpm
        self._run = run
        self._extra_env = dict(extra_env or {})
        base = Path(workdir) if workdir else Path(tempfile.gettempdir()) / "scraper_claude_code"
        self.workdir = base / "work"                # empty: the working directory Claude Code reports
        self.workdir.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]
        self.system_prompt_file = base / f"system_prompt_{sha}.txt"
        if not self.system_prompt_file.exists():
            self.system_prompt_file.write_text(SYSTEM_PROMPT, encoding="utf-8")
        self.last_rate_limits: dict = {}
        self.last_call: dict = {}

    # ------------------------------------------------------------------
    def command(self) -> list[str]:
        cmd = [*self.cli, "-p", "--output-format", "stream-json", "--verbose", "--model", self.model]
        if self.reasoning_effort:
            cmd += ["--effort", self.reasoning_effort]
        cmd += ["--system-prompt-file", str(self.system_prompt_file), "--tools", "", "--max-turns", "1",
                "--no-session-persistence", "--safe-mode", "--strict-mcp-config", "--disable-slash-commands"]
        return cmd

    def child_env(self, max_tokens: int) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in BLOCKED_ENV}
        env.update(FIXED_ENV)
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(int(max_tokens))
        env.update(self._extra_env)
        return env

    def prompt_token_budget(self, max_tokens: int) -> Optional[int]:
        return prompt_token_budget(self.tokens_per_minute, max_tokens, self.completion_counts_in_tpm)

    def generate_xpaths(
        self,
        pruned_html: str,
        fields: list[FieldSpec],
        failure_history: Optional[list[FailureHint]] = None,
    ) -> LLMResult:
        max_tokens = completion_tokens_for(len(fields))
        prompt, meta = build_prompt(pruned_html, fields, failure_history, self.max_input_chars,
                                    max_prompt_tokens=self.prompt_token_budget(max_tokens))
        out = self._call(prompt, max_tokens)
        calls = [out]
        notes = []
        if out["continued"]:
            # Claude Code continued the reply with extra turns: not a single-turn answer.
            bigger = min(64000, max_tokens * 2)
            notes.append(f"[claude_code_continued:{out['continued']}; max_tokens {max_tokens}; "
                         f"resent with {bigger}]")
            out = self._call(prompt, bigger)
            calls.append(out)
        text = "" if out["continued"] else out["text"]
        if out["continued"]:
            notes.append(f"[finish_reason=length; claude_code_continued:{out['continued']}]")
        prompt_tokens = sum(c["input_tokens"] for c in calls)
        completion_tokens = sum(c["output_tokens"] for c in calls)
        thinking = sum(c["thinking_tokens"] for c in calls)
        cached = sum(c["cached_tokens"] for c in calls)
        if thinking:
            notes.append(f"[thinking_tokens={thinking}]")
        if cached:
            notes.append(f"[cached_input_tokens={cached}]")
        notes.append(f"[claude_code {out['version']}; auth {out['api_key_source']}; "
                     f"api-equivalent cost {sum(c['cost_usd'] for c in calls):.6f} USD]")
        self.last_call = {k: v for k, v in out.items() if k != "text"}
        return LLMResult(
            xpaths=parse_xpath_json(text, fields),
            model=(out["models"] or [self.model])[0],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw_response=redact(" ".join(notes) + " " + text),
            latency_ms=sum(c["latency_ms"] for c in calls),
            rate_limit_wait_ms=sum(c["wait_ms"] for c in calls),
            retries=sum(c["retries"] for c in calls) + len(calls) - 1,
            input_chars=meta["input_chars"],
            input_truncated=meta["input_truncated"],
            prompt_sha=meta["prompt_sha"],
        )

    # -- CLI information (no secrets) -----------------------------------
    def cli_version(self) -> str:
        proc = self._run([*self.cli, "--version"], capture_output=True, timeout=60,
                         env=self.child_env(1000), cwd=str(self.workdir))
        return redact(_decode(proc.stdout).strip() or _decode(proc.stderr).strip())

    def auth_status(self) -> dict:
        """Claude Code's own login status (which method, never the credential)."""
        proc = self._run([*self.cli, "auth", "status", "--json"], capture_output=True, timeout=60,
                         env=self.child_env(1000), cwd=str(self.workdir))
        try:
            data = json.loads(_decode(proc.stdout))
        except ValueError:
            return {"loggedIn": False, "error": redact(_decode(proc.stderr).strip())[:300]}
        return {k: data[k] for k in AUTH_STATUS_KEYS if k in data}

    # ------------------------------------------------------------------
    def _call(self, prompt: str, max_tokens: int) -> dict:
        waited = 0.0
        retries = 0
        cli_wait_ms = cli_retries = 0       # retries Claude Code made inside failed attempts
        while True:
            start = time.perf_counter()
            try:
                proc = self._run(self.command(), input=prompt.encode("utf-8"), capture_output=True,
                                 timeout=self.timeout, env=self.child_env(max_tokens), cwd=str(self.workdir))
                failure = None
            except subprocess.TimeoutExpired:
                proc, failure = None, f"Claude Code did not answer within {self.timeout} s"
            except OSError as exc:
                raise LLMError(f"HTTP 404 cannot start Claude Code ({self.cli[0]}): {exc}") from exc
            wall_ms = int((time.perf_counter() - start) * 1000)
            if proc is not None:
                out = parse_stream(_decode(proc.stdout))
                out["wall_ms"] = wall_ms
                if out["rate_limit"]:
                    self.last_rate_limits = out["rate_limit"]
                failure = self._check(out, proc)
                if failure is None:
                    out["wait_ms"] = int(waited * 1000) + cli_wait_ms + out["retry_delay_ms"]
                    out["retries"] = retries + cli_retries + out["api_retries"]
                    return out
                cli_wait_ms += out["retry_delay_ms"]
                cli_retries += out["api_retries"]
            # transient: retry here with backoff, within the waiting allowance
            delay = min(2 ** (retries + 1), 60)
            if retries >= self.max_retries or waited + delay > self.max_wait_seconds:
                if failure.startswith("rate limited"):
                    raise RateLimitExhausted(f"Claude Code: {failure}", retry_after=60)
                raise LLMError(f"Claude Code failed after {retries} retries: {failure}")
            self._sleep(delay)
            waited += delay
            retries += 1

    def _check(self, out: dict, proc) -> Optional[str]:
        """None if the call succeeded; a message for a transient failure; raises otherwise."""
        result = out["result"]
        text = redact(out["result_text"] or out["error_text"])[:300]
        source = out["api_key_source"]
        if source and source not in SUBSCRIPTION_KEY_SOURCES:
            raise LLMError(f"HTTP 403 Claude Code is using an API key ({source}) instead of the Claude "
                           "subscription; remove it from Claude Code's settings or environment")
        if result is None:
            if out["error"] in FATAL_ERRORS:
                raise LLMError(f"HTTP {FATAL_ERRORS[out['error']]} Claude Code {out['error']}: {text}")
            stderr = redact(_decode(proc.stderr).strip())[:300]
            return f"no result from Claude Code (exit {proc.returncode}): {stderr or text}"
        if out["error"] in FATAL_ERRORS:
            hint = (" - sign in again (claude setup-token, then set CLAUDE_CODE_OAUTH_TOKEN, or claude /login)"
                    if FATAL_ERRORS[out["error"]] in (401, 403) else "")
            raise LLMError(f"HTTP {FATAL_ERRORS[out['error']]} Claude Code {out['error']}: {text}{hint}")
        limit = out["rate_limit"] or {}
        if out["error"] in ("rate_limit", "billing_error") or (out["is_error"] and limit.get("status") == "rejected"):
            resets = limit.get("resetsAt")
            if limit.get("status") == "rejected" and resets:
                wait = max(60.0, float(resets) - time.time())
                raise RateLimitExhausted(
                    f"Claude subscription usage limit reached ({limit.get('rateLimitType') or 'limit'}); "
                    f"resets in {int(wait)} s: {text}", retry_after=wait)
            if out["error"] == "billing_error":
                raise LLMError(f"HTTP 402 Claude Code billing_error: {text}")
            return f"rate limited: {text}"
        if out["is_error"] and not out["continued"]:
            return f"{out['error'] or 'error'}: {text}"
        bad = [m for m in out["models"] if m != self.model]
        if bad:
            raise LLMError(f"HTTP 409 Claude Code answered with {', '.join(bad)} instead of {self.model}")
        if not out["models"] and not out["continued"]:
            return f"Claude Code returned no model reply: {text}"
        return None


def parse_stream(stdout: str) -> dict:
    """Summarise Claude Code's stream-json output (one JSON object per line)."""
    out = {"version": "", "api_key_source": "", "init_model": "", "tools": None, "models": [],
           "text": "", "result": None, "result_text": "", "error": "", "error_text": "", "is_error": False,
           "continued": 0, "num_turns": 0, "rate_limit": {}, "api_retries": 0, "retry_delay_ms": 0,
           "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "cached_tokens": 0,
           "cost_usd": 0.0, "latency_ms": 0, "stop_reason": ""}
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        kind, sub = msg.get("type"), msg.get("subtype")
        if kind == "system" and sub == "init":
            out["version"] = msg.get("claude_code_version") or ""
            out["api_key_source"] = msg.get("apiKeySource") or ""
            out["init_model"] = msg.get("model") or ""
            out["tools"] = msg.get("tools")
        elif kind == "system" and sub == "api_retry":
            out["api_retries"] += 1
            out["retry_delay_ms"] += int(msg.get("retry_delay_ms") or 0)
        elif kind == "rate_limit_event":
            out["rate_limit"] = msg.get("rate_limit_info") or {}
        elif kind == "user":
            out["continued"] += 1          # a message Claude Code added to continue the reply
        elif kind == "assistant":
            m = msg.get("message") or {}
            model = m.get("model") or ""
            if msg.get("error"):
                out["error"] = msg["error"]
                out["error_text"] = "".join(b.get("text", "") for b in m.get("content") or []
                                            if b.get("type") == "text")
            if model and model != "<synthetic>" and model not in out["models"]:
                out["models"].append(model)
        elif kind == "result":
            out["result"] = msg
            out["is_error"] = bool(msg.get("is_error"))
            out["result_text"] = msg.get("result") if isinstance(msg.get("result"), str) else ""
            out["num_turns"] = int(msg.get("num_turns") or 0)
            out["stop_reason"] = msg.get("stop_reason") or ""
            out["cost_usd"] = float(msg.get("total_cost_usd") or 0)
            out["latency_ms"] = int(msg.get("duration_api_ms") or msg.get("duration_ms") or 0)
    res = out["result"]
    if res is not None:
        if out["num_turns"] > 1:
            out["continued"] = max(out["continued"], out["num_turns"] - 1)
        if out["error"] == "max_output_tokens" or str(res.get("subtype") or "").startswith("error_max_turns"):
            out["continued"] = max(out["continued"], 1)     # the reply was cut, not answered
        usage = res.get("usage") or {}
        cached = int(usage.get("cache_creation_input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0)
        out["cached_tokens"] = cached
        out["input_tokens"] = int(usage.get("input_tokens") or 0) + cached
        out["output_tokens"] = int(usage.get("output_tokens") or 0)
        out["thinking_tokens"] = int((usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
        if not out["is_error"]:
            out["text"] = out["result_text"]
    return out


def _decode(data) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def describe_invocation(provider: ClaudeCodeProvider) -> dict:
    """How the project calls Claude Code (for logs and the smoke test; no secrets)."""
    return {
        "command": provider.command(),
        "stdin": "the user prompt (prompts.build_prompt)",
        "cwd": str(provider.workdir),
        "removed_env": [k for k in BLOCKED_ENV if k in os.environ],
        "set_env": {**FIXED_ENV, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "12000 + 150 x fields (max 32000)"},
        "oauth_token_variable_set": bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")),
        "python": sys.executable,
    }
