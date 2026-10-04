"""
One-shot preparation of the 20-site evaluation on Windows (no LLM runs).

Run from the project folder:   eval\\prepare.bat
(or: venv\\Scripts\\python.exe eval\\prepare.py)

Steps: install requirements + the headless browser, migrate, run the test
suite, create the experiment from eval\\sites.yaml, save every page
(eval_snapshot), check the pages against the dataset (eval_audit) and make
one tiny LLM call (eval_check --ping). Everything printed is also written to
eval_data\\prepare_log.txt.
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "eval_data" / "prepare_log.txt"
PY = sys.executable

STEPS = [
    # (title, command, stop_on_failure)
    ("Install Python packages", [PY, "-m", "pip", "install", "-r", "requirements.txt"], True),
    ("Install headless Chromium (Playwright)", [PY, "-m", "playwright", "install", "chromium"], True),
    ("Create / update the database", [PY, "manage.py", "migrate"], True),
    ("Run the test suite (offline)", [PY, "manage.py", "test", "scraper"], True),
    ("Create the experiment from eval/sites.yaml",
     [PY, "manage.py", "eval_init", "--config", str(Path("eval") / "sites.yaml")], True),
    ("Save every page (static + browser-rendered)", [PY, "manage.py", "eval_snapshot"], False),
    ("Check the saved pages against the dataset", [PY, "manage.py", "eval_audit"], False),
    ("One tiny LLM call to verify the Groq key", [PY, "manage.py", "eval_check", "--ping"], False),
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
    log(f"\n##### prepare.py started {dt.datetime.now().isoformat(timespec='seconds')} "
        f"(Python {sys.version.split()[0]}, {sys.platform}) #####")
    env_file = ROOT / ".env"
    key_set = env_file.exists() and any(
        line.strip().startswith("GROQ_API_KEY=") and len(line.strip()) > len("GROQ_API_KEY=") + 10
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines())
    if not key_set:
        log("NOTE: GROQ_API_KEY is empty in .env - the last step (LLM ping) will fail until you "
            "paste your key after GROQ_API_KEY= in .env. Everything else still runs.")
    failed = []
    for title, cmd, stop in STEPS:
        code = run(title, cmd)
        if code != 0:
            failed.append(title)
            if stop:
                log(f"\nSTOPPED: '{title}' failed. See the messages above.")
                return code
    log("\n##### done" + (f" with problems in: {', '.join(failed)}" if failed else "") + " #####")
    log(f"Log: {LOG}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
