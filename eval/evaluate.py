"""
Run the whole 20-site evaluation on Windows, logging to eval_data\\evaluate_log.txt.

    eval\\evaluate.bat            all modules (the LLM-based ones call the Claude API)
    eval\\evaluate.bat offline    only the modules that need no LLM

LLM (decision 2026-10-07): the LLM-based modules use Claude Opus 5.5
(claude-opus-5-5, Anthropic API, effort "medium", whole pages up to 200,000
characters) on all 20 sites with 3 repetitions, the same design as the
rule-based modules (which gain a 3rd repetition; their 2 completed repetitions
are kept). On the first run eval_switch_llm backs up the database and removes
the earlier gpt-oss-120b (Groq) runs of the LLM-based modules and the recorded
two-site scope; nothing is replicated any more, every LLM value is measured.

Needs ANTHROPIC_API_KEY=... in the .env file (a Claude Console API key, with a
spend limit set in the Console). Running it again resumes where it stopped.

Steps: packages, migrate, tests, eval_init, switch the LLM modules to Claude,
save any missing pages, audit, import the labels that are your recorded dataset
values found verbatim on the saved pages (labels already imported are kept),
run the rule-based modules, check the Claude API, run the LLM-based modules
(sleeping through rate limits), then report, verify, paper text and the
results bundle. Fields without a label are scraped but not scored.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "eval_data" / "evaluate_log.txt"
PY = sys.executable
EXP = "ieee-2026"
SITES = str(Path("eval") / "sites.yaml")
SHEET = str(Path("eval_data") / EXP / "audit" / "ground_truth_prefilled.xlsx")

LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-opus-5-5"
LLM_EFFORT = "medium"
LLM_MAX_INPUT_CHARS = "200000"
REPETITIONS = "3"
# USD per 1M tokens (Claude API pricing page, read 2026-10-07); thinking tokens are billed as output.
PRICING = {LLM_MODEL: {"input": 4.0, "output": 20.0}}


def M(*args):
    return [PY, "manage.py", *args]


SETUP = [
    ("Install / update Python packages", [PY, "-m", "pip", "install", "-q", "-r", "requirements.txt"], True),
    ("Create / update the database", M("migrate"), True),
    ("Run the test suite (offline)", M("test", "scraper"), True),
    ("Create / update the experiment", M("eval_init", "--config", SITES), True),
    (f"Use {LLM_MODEL} for the LLM-based modules on all sites, {REPETITIONS} repetitions "
     "(first time: back up, then remove the old model's LLM runs)",
     M("eval_switch_llm", "--config", SITES, "--provider", LLM_PROVIDER, "--model", LLM_MODEL,
       "--reasoning-effort", LLM_EFFORT, "--max-input-chars", LLM_MAX_INPUT_CHARS,
       "--repetitions", REPETITIONS), True),
    ("Save any missing pages", M("eval_snapshot"), False),
    ("Check the saved pages against the dataset (and exclude URL fields not in the HTML)",
     M("eval_audit", "--apply-url-exclusions", SITES), True),
    ("Update the experiment with the exclusions", M("eval_init", "--config", SITES), True),
    ("Re-check the saved pages", M("eval_audit"), True),
    ("Import labels: recorded values found verbatim on the saved pages",
     M("eval_gt_import", "--file", SHEET, "--only-status", "found,list-complete", "--source", "dataset",
       "--keep-existing", "--verified-by", "dataset 2026-10-04"), True),
    ("Run the rule-based modules (no LLM)",
     M("eval_run", "--conditions", "rule_based,rule_static", "--allow-unlabelled",
       "--skip-unavailable"), True),
]
PING = ("Check the Claude API (model list, one tiny call, limits)", M("eval_check", "--ping", "--allow-unlabelled"))
LLM = [
    ("Use the account's measured token limits", M("eval_sync_llm_limits", "--config", SITES), True),
    ("Re-check the saved pages with those limits", M("eval_audit"), True),
    (f"Run the LLM-based modules with {LLM_MODEL} on all sites and repetitions (sleeps through rate limits)",
     M("eval_run", "--allow-unlabelled", "--skip-unavailable", "--wait"), False),
]
FINISH = [
    ("Build the report, tables and figures", M("eval_report"), True),
    ("Verify every reported number", M("eval_verify"), False),
    ("Write the results section text", M("eval_paper"), False),
    ("Export the results bundle", M("export_results"), False),
]


def log(text: str) -> None:
    print(text, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def dotenv() -> dict:
    path = ROOT / ".env"
    if not path.exists():
        return {}
    try:
        from dotenv import dotenv_values
        return {k: v or "" for k, v in dotenv_values(path).items()}
    except ImportError:
        out = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                out[key.strip()] = value.strip().strip("'\"")
        return out


def child_env() -> dict:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    values = dotenv()
    try:
        pricing = json.loads(env.get("SCRAPER_PRICING") or values.get("SCRAPER_PRICING") or "{}") or {}
    except ValueError:
        pricing = {}
    env["SCRAPER_PRICING"] = json.dumps({**pricing, **PRICING})
    return env


def run(title: str, cmd: list[str]) -> int:
    log(f"\n=== {title} ===\n$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=child_env(), text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        log(line.rstrip("\n"))
    code = proc.wait()
    log(f"--- exit code {code} ---")
    return code


def api_key_problem() -> str | None:
    key = (os.environ.get("ANTHROPIC_API_KEY") or dotenv().get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        return ("ANTHROPIC_API_KEY is missing. Create an API key in the Claude Console "
                "(platform.claude.com, formerly console.anthropic.com > API keys), add credit and set a "
                "spend limit there, then add the line\n"
                "    ANTHROPIC_API_KEY=sk-ant-...\n"
                f"to {ROOT / '.env'} and double-click evaluate.bat again. "
                "(A claude.ai chat subscription cannot be used for API calls.)")
    if not key.startswith("sk-ant-"):
        log("NOTE: ANTHROPIC_API_KEY does not start with 'sk-ant-'; check that it is a Claude API key.")
    return None


def ping_problem() -> str | None:
    data_dir = Path(os.environ.get("SCRAPER_DATA_DIR") or dotenv().get("SCRAPER_DATA_DIR") or "eval_data")
    ping = (data_dir if data_dir.is_absolute() else ROOT / data_dir) / EXP / "llm_ping.json"
    try:
        record = json.loads(ping.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"the API check wrote no result ({ping})"
    if record.get("model") != LLM_MODEL:
        return f"the API check used {record.get('model')!r}, not {LLM_MODEL}"
    if not record.get("ok"):
        return record.get("error") or "the test call returned no XPath (see the messages above)"
    return None


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    LOG.parent.mkdir(parents=True, exist_ok=True)
    offline = "offline" in sys.argv[1:]
    log(f"\n##### evaluate.py started {dt.datetime.now().isoformat(timespec='seconds')} "
        f"({'offline modules only' if offline else 'all modules'}) #####")
    log(f"LLM: {LLM_MODEL} (Anthropic API, effort {LLM_EFFORT}, input up to {int(LLM_MAX_INPUT_CHARS):,} "
        f"characters) on all 20 sites, {REPETITIONS} repetitions; rule-based modules on all 20 sites, "
        f"{REPETITIONS} repetitions. Every reported value is measured (no replication).")
    if not offline:
        problem = api_key_problem()
        if problem:
            log("\nSTOPPED: " + problem)
            return 1
    failed = []
    for title, cmd, stop in SETUP:
        code = run(title, cmd)
        if code != 0:
            failed.append(title)
            if stop:
                log(f"\nSTOPPED: '{title}' failed. See the messages above.")
                return code
    if not offline:
        run(*PING)
        problem = ping_problem()
        if problem:
            log(f"\nSTOPPED: the Claude API check failed: {problem}\nCheck ANTHROPIC_API_KEY in .env, "
                "the credit balance and the spend limit in the Claude Console, then double-click "
                "evaluate.bat again (completed work is kept).")
            return 1
        for title, cmd, stop in LLM:
            code = run(title, cmd)
            if code != 0:
                failed.append(title)
                if stop:
                    log(f"\nSTOPPED: '{title}' failed. See the messages above.")
                    return code
    for title, cmd, stop in FINISH:
        code = run(title, cmd)
        if code != 0:
            failed.append(title)
            if stop:
                log(f"\nSTOPPED: '{title}' failed. See the messages above.")
                return code
    log("\n##### done" + (f" with problems in: {', '.join(failed)}" if failed else "") + " #####")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
