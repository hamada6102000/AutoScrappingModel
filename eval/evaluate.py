"""
Run the whole 20-site evaluation on Windows, logging to eval_data\\evaluate_log.txt.

    eval\\evaluate.bat            all modules (the LLM-based ones need Groq quota)
    eval\\evaluate.bat offline    only the modules that need no LLM

Steps: migrate, tests, eval_init, save any missing pages, audit, import the
labels that are your recorded dataset values found verbatim on the saved
pages (source "dataset"; labels you already imported are kept), run the
rule-based modules, check the LLM, run the LLM-based modules (sleeping through
rate limits), then report, verify, paper text and the results bundle.
Fields without a label are scraped but not scored, and the report says how many.
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "eval_data" / "evaluate_log.txt"
PY = sys.executable
EXP = "ieee-2026"
SHEET = str(Path("eval_data") / EXP / "audit" / "ground_truth_prefilled.xlsx")


def M(*args):
    return [PY, "manage.py", *args]


SETUP = [
    ("Install / update Python packages", [PY, "-m", "pip", "install", "-q", "-r", "requirements.txt"], True),
    ("Create / update the database", M("migrate"), True),
    ("Run the test suite (offline)", M("test", "scraper"), True),
    ("Create / update the experiment", M("eval_init", "--config", str(Path("eval") / "sites.yaml")), True),
    ("Save any missing pages", M("eval_snapshot"), False),
    ("Check the saved pages against the dataset (and exclude URL fields not in the HTML)",
     M("eval_audit", "--apply-url-exclusions", str(Path("eval") / "sites.yaml")), True),
    ("Update the experiment with the exclusions", M("eval_init", "--config", str(Path("eval") / "sites.yaml")), True),
    ("Re-check the saved pages", M("eval_audit"), True),
    ("Import labels: recorded values found verbatim on the saved pages",
     M("eval_gt_import", "--file", SHEET, "--only-status", "found,list-complete", "--source", "dataset",
       "--keep-existing", "--verified-by", "dataset 2026-10-04"), True),
    ("Run the rule-based modules (no LLM)",
     M("eval_run", "--conditions", "rule_based,rule_static", "--allow-unlabelled",
       "--skip-unavailable"), True),
]
LLM = [
    ("Check the LLM (model list, one tiny call, limits)", M("eval_check", "--ping", "--allow-unlabelled"), False),
    ("Use the account's measured token limits", M("eval_sync_llm_limits", "--config", str(Path("eval") / "sites.yaml")), True),
    ("Re-check the saved pages with those limits", M("eval_audit"), True),
    ("Run the LLM-based modules (sleeps through rate limits)",
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


def run(title: str, cmd: list[str]) -> int:
    log(f"\n=== {title} ===\n$ {' '.join(cmd)}")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=env, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        log(line.rstrip("\n"))
    code = proc.wait()
    log(f"--- exit code {code} ---")
    return code


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
    steps = SETUP + ([] if offline else LLM) + FINISH
    failed = []
    for title, cmd, stop in steps:
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
