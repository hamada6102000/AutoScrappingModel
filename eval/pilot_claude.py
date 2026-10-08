"""
Separate two-site pilot with Claude Opus 5.5 (a replication of the original
gpt-oss-120b pilot of 2026-10-04 with another model).

    eval\\pilot_claude.bat

Everything is kept apart from the main experiment: its own database and data
folder (eval_data\\pilot_claude\\), its own experiment name
(pilot-claude-2sites). The main experiment, its results, its backups and the
original pilot are only read, never written.

Conditions copied from the original pilot (backup of 2026-10-08, made before
the switch to Claude):
  * the experiment settings of that run (seed 20261004, the eight change steps,
    threshold, attempts, memory, scoring), one repetition, conditions
    self_healing and static (LLM-based modules only, as in the original pilot);
  * site_001 and site_002 with their fields, loaded from the very same saved
    pages (html_file copies, SHA-256 checked against the original snapshots),
    so the seeded simulated pages are byte-identical;
  * labels imported by the same rule (recorded values found verbatim);
  * the same prompt budget as the original pilot (8000 tokens per minute ->
    6800-token prompt budget, 60,000-character input cap), so the model reads
    the same prompts; the LLM is claude-opus-5-5 through Claude Code with the
    subscription login (effort medium).

The report is built only when all 36 runs (2 sites x 2 modules x 9 steps) are
complete; otherwise the script says so and stops. Running it again resumes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PILOT = Path(os.environ.get("PILOT_CLAUDE_DIR") or ROOT / "eval_data" / "pilot_claude")   # override: tests only
DB = PILOT / "db.sqlite3"
CONFIG = PILOT / "pilot_claude.yaml"
LOG = PILOT / "pilot_log.txt"
EXP = "pilot-claude-2sites"
SITES = ("site_001", "site_002")
EXPECTED_RUNS = len(SITES) * 2 * 9
MAIN = ROOT / "eval_data" / "ieee-2026"
ORIGINAL = MAIN / "backups" / "20261008-020953_before_claude-opus-5-5"   # original pilot (gpt-oss-120b)
LLM = {"provider": "claude_code", "model": "claude-opus-5-5", "reasoning_effort": "medium",
       "max_input_chars": 60000, "tokens_per_minute": 8000, "completion_counts_in_tpm": False}
PRICING = {"claude-opus-5-5": {"input": 4.0, "output": 20.0}}
PY = sys.executable


def original_snapshots() -> dict:
    """site -> (relative path, sha256) of the original pilot's baseline snapshots."""
    con = sqlite3.connect(f"file:{ORIGINAL / 'db.sqlite3'}?mode=ro", uri=True)
    rows = con.execute("SELECT s.site_key, s.path, s.sha256 FROM scraper_snapshot s "
                       "JOIN scraper_experiment e ON e.id = s.experiment_id "
                       "WHERE e.name = 'ieee-2026' AND s.kind = 'baseline'").fetchall()
    con.close()
    return {k: (p, h) for k, p, h in rows if k in SITES}


def build_config(source_yaml: Path, inputs: dict[str, Path]) -> dict:
    """Pilot config from the original pilot's sites file: same experiment
    settings and site entries, new name/LLM, one repetition, LLM modules only."""
    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    e = dict(data["experiment"])
    e.update({"name": EXP, "llm": dict(LLM), "repetitions": 1, "planned_repetitions": 1,
              "conditions": ["self_healing", "static"],
              "description": "Two-site pilot with Claude Opus 5.5 (replication of the gpt-oss-120b pilot of "
                             "2026-10-04 on the same saved pages, fields, labels and seeded changes)"})
    sites = []
    for s in data["sites"]:
        if s["key"] in SITES:
            s = dict(s)
            s["html_file"] = str(inputs[s["key"]])
            sites.append(s)
    if [s["key"] for s in sites] != list(SITES):
        raise SystemExit(f"original sites file does not contain {SITES}")
    return {**data, "experiment": e, "sites": sites}


def log(text: str) -> None:
    print(text, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(text + "\n")


def env() -> dict:
    out = {**os.environ, "DJANGO_DB_PATH": str(DB), "SCRAPER_DATA_DIR": str(PILOT),
           "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "SCRAPER_PRICING": json.dumps(PRICING)}
    return out


def run(title: str, *args: str) -> int:
    cmd = [PY, "manage.py", *args]
    log(f"\n=== {title} ===\n$ {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env(),
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        log(line.rstrip("\n"))
    code = proc.wait()
    log(f"--- exit code {code} ---")
    return code


def query(sql: str, *params):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def prepare() -> None:
    PILOT.mkdir(parents=True, exist_ok=True)
    if CONFIG.exists():
        return
    snaps = original_snapshots()
    inputs = {}
    for site in SITES:
        rel, sha = snaps[site]
        src = MAIN.parent / rel                      # eval_data/<rel>
        dst = PILOT / "inputs" / f"{site}.html"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        inputs[site] = dst
        log(f"{site}: copied {src} (original snapshot sha256 {sha[:16]}...)")
    cfg = build_config(ORIGINAL / "sites.yaml", inputs)
    CONFIG.write_text("# Two-site Claude pilot (generated by eval/pilot_claude.py)\n"
                      + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, width=110), encoding="utf-8")


def snapshots_identical() -> bool:
    orig = original_snapshots()
    new = dict(query("SELECT site_key, sha256 FROM scraper_snapshot WHERE kind = 'baseline'"))
    ok = all(new.get(s) == orig[s][1] for s in SITES)
    for s in SITES:
        log(f"{s}: baseline sha256 {str(new.get(s))[:16]}... "
            f"{'identical to' if new.get(s) == orig[s][1] else 'DIFFERENT FROM'} the original pilot")
    return ok


def complete() -> bool:
    n = query("SELECT COUNT(*) FROM scraper_scraperun r JOIN scraper_experiment e ON e.id = r.experiment_id "
              "WHERE e.name = ? AND r.completed = 1", EXP)[0][0]
    log(f"Completed runs: {n} of {EXPECTED_RUNS}")
    return n == EXPECTED_RUNS


def main() -> int:
    PILOT.mkdir(parents=True, exist_ok=True)
    log(f"\n##### pilot_claude.py started {dt.datetime.now().isoformat(timespec='seconds')} #####")
    log(f"Separate experiment '{EXP}' in {PILOT} (the main experiment is not touched).")
    prepare()
    steps = [
        ("Create / update the pilot database", ("migrate",)),
        ("Create the pilot experiment", ("eval_init", "--config", str(CONFIG))),
        ("Load the two saved pages (no download)", ("eval_snapshot",)),
    ]
    for title, args in steps:
        if run(title, *args) != 0:
            log(f"STOPPED: '{title}' failed.")
            return 1
    if not snapshots_identical():
        log("STOPPED: the pilot pages differ from the original pilot's pages.")
        return 1
    for title, args in [
        ("Check the saved pages against the dataset", ("eval_audit",)),
        ("Import labels: recorded values found verbatim on the saved pages",
         ("eval_gt_import", "--file", str(PILOT / EXP / "audit" / "ground_truth_prefilled.xlsx"),
          "--only-status", "found,list-complete", "--source", "dataset", "--keep-existing",
          "--verified-by", "dataset 2026-10-04")),
        ("Check Claude Code and its subscription login", ("eval_claude_smoke", "--no-request")),
        ("One test request through Claude Code", ("eval_check", "--ping", "--allow-unlabelled")),
    ]:
        if run(title, *args) != 0:
            log(f"STOPPED: '{title}' failed.")
            return 1
    run("Run the pilot: self_healing and static on site_001 and site_002, repetition 1 "
        "(waits through usage limits)", "eval_run", "--allow-unlabelled", "--wait", "--max-hours", "168")
    if not complete():
        log("\nNOT FINISHED: not every pilot run is complete, so no result is reported. "
            "Double-click eval\\pilot_claude.bat again to resume.")
        return 1
    for title, args in [("Build the pilot report", ("eval_report",)),
                        ("Verify every reported number", ("eval_verify",)),
                        ("Export the pilot results bundle", ("export_results",))]:
        run(title, *args)
    log("\n##### pilot finished: all runs complete #####")
    return 0


if __name__ == "__main__":
    sys.exit(main())
