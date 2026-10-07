"""
Switch the experiment's LLM-based modules to another model (and optionally
add repetitions), keeping everything else: the saved pages, the ground truth
and the completed rule-based runs.

    python manage.py eval_switch_llm --config eval/sites.yaml --provider anthropic ^
        --model claude-opus-5-5 --reasoning-effort medium --max-input-chars 200000 --repetitions 3

When the requested model differs from the experiment's current LLM settings:

1. the database and the current LLM files are backed up to
   eval_data/<exp>/backups/<time>_before_<model>/ (nothing is lost);
2. the runs of the LLM-based modules (self_healing, static) are deleted,
   because one experiment must not mix models;
3. a recorded LLM-scope decision (llm_scope.json: LLM on some sites only,
   values replicated) is removed, so the new model runs on every site;
4. experiment.llm and experiment.repetitions are written to the sites file
   and the experiment is updated (new repetitions get their targets).

Running it again with the same arguments changes nothing, so it is safe to
call at the start of every evaluation run (resuming keeps the new model's runs).
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sqlite3
from pathlib import Path

import yaml
from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from scraper.evaluation import llm_scope
from scraper.evaluation.config import COND_INFO, ConfigError, load_config
from scraper.evaluation.protocol import EvalError, init_experiment
from scraper.models import Experiment, ScrapeRun, ScrapeTarget
from scraper.services import snapshots as snapstore

LLM_CONDITIONS = [c for c, info in COND_INFO.items() if info["generator"] == "llm"]
KEYS = ("provider", "model", "reasoning_effort", "max_input_chars")


class Command(BaseCommand):
    help = "Switch the LLM-based modules to another model (backing up and removing the old model's runs)."

    def add_arguments(self, parser):
        parser.add_argument("--config", default="eval/sites.yaml")
        parser.add_argument("--provider", required=True, choices=["groq", "anthropic", "openai", "ollama"])
        parser.add_argument("--model", required=True)
        parser.add_argument("--reasoning-effort", default=None)
        parser.add_argument("--max-input-chars", type=int, default=None)
        parser.add_argument("--repetitions", type=int, default=None,
                            help="Total repetitions (may only grow; existing runs are kept).")

    def handle(self, *args, **opts):
        path = Path(opts["config"])
        text = path.read_text(encoding="utf-8")
        header = ""
        for line in text.splitlines():          # the leading comment block is kept
            if not line.startswith("#"):
                break
            header += line + "\n"
        data = yaml.safe_load(text)
        name = data["experiment"]["name"]
        exp = Experiment.objects.filter(name=name).first()
        if exp is None:
            raise CommandError(f"experiment '{name}' does not exist; run eval_init first")

        want = {"provider": opts["provider"], "model": opts["model"],
                "reasoning_effort": opts["reasoning_effort"], "max_input_chars": opts["max_input_chars"]}
        current = exp.config["experiment"].get("llm") or {}
        switching = any(current.get(k) != want[k] for k in KEYS)
        reps_now = int(exp.config["experiment"]["repetitions"])
        reps = opts["repetitions"] or reps_now
        if reps < reps_now:
            raise CommandError(f"repetitions can only grow ({reps_now} -> {reps} refused)")
        if not switching and reps == reps_now:
            self.stdout.write(f"LLM modules already use {want['provider']} / {want['model']}"
                              f" ({reps} repetitions); nothing to change.")
            return

        if switching:
            old = f"{current.get('provider')} / {current.get('model')}"
            backup = self._backup(exp, path, want["model"])
            n_runs = ScrapeRun.objects.filter(experiment=exp, condition__in=LLM_CONDITIONS).count()
            ScrapeTarget.objects.filter(experiment=exp, condition__in=LLM_CONDITIONS).delete()
            scope = llm_scope.load(exp)
            llm_scope.clear(exp)
            ping = snapstore.experiment_dir(exp) / "llm_ping.json"
            if ping.exists():
                ping.unlink()
            self.stdout.write(self.style.WARNING(
                f"LLM model switched: {old} -> {want['provider']} / {want['model']}. Backed up to {backup}. "
                f"Deleted {n_runs} runs of the LLM-based modules ({', '.join(LLM_CONDITIONS)}) made with the "
                "old model; rule-based runs, saved pages and labels are unchanged."
                + (f" Removed the LLM-scope decision (LLM on {', '.join(scope['llm_sites'])} only"
                   + (", values replicated" if scope["replicate_to_all_sites"] else "")
                   + "): the new model runs on every site." if scope else "")))

        llm = {k: v for k, v in want.items() if v is not None}
        if want["provider"] == "anthropic":
            llm["completion_counts_in_tpm"] = False     # separate input / output per-minute limits
        if not switching:
            llm = {**current, **llm}                    # keep measured limits (tokens_per_minute)
        data["experiment"]["llm"] = llm
        data["experiment"]["repetitions"] = reps
        path.write_text(header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=110),
                        encoding="utf-8")
        try:
            init_experiment(load_config(path))
        except (ConfigError, EvalError) as exc:
            path.write_text(text, encoding="utf-8")
            raise CommandError(f"{exc} (sites file restored)") from exc
        self.stdout.write(self.style.SUCCESS(
            f"Experiment '{name}': LLM {llm['provider']} / {llm['model']}"
            + (f", effort {llm['reasoning_effort']}" if llm.get("reasoning_effort") else "")
            + (f", input cap {llm['max_input_chars']} characters" if llm.get("max_input_chars") else "")
            + f"; {reps} repetitions" + (f" (was {reps_now})" if reps != reps_now else "") + "."))

    # ------------------------------------------------------------------
    def _backup(self, exp, sites_file: Path, model: str) -> Path:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)
        out = snapstore.experiment_dir(exp) / "backups" / f"{stamp}_before_{safe}"
        out.mkdir(parents=True, exist_ok=True)
        db = Path(str(connection.settings_dict["NAME"]))
        if db.exists():
            src = sqlite3.connect(str(db))
            dst = sqlite3.connect(str(out / "db.sqlite3"))
            with dst:
                src.backup(dst)
            src.close()
            dst.close()
        shutil.copy2(sites_file, out / sites_file.name)
        base = snapstore.experiment_dir(exp)
        for name in ("llm_scope.json", "llm_ping.json"):
            if (base / name).exists():
                shutil.copy2(base / name, out / name)
        results = base / "results"
        for name in ("summary.json", "report.md", "paper_results.md", "verification.md"):
            if (results / name).exists():
                (out / "results").mkdir(exist_ok=True)
                shutil.copy2(results / name, out / "results" / name)
        (out / "README.txt").write_text(
            "Backup made by eval_switch_llm before switching the LLM-based modules to "
            f"{model}. db.sqlite3 holds every run made until then (including the old model's).\n"
            + json.dumps(exp.config["experiment"].get("llm") or {}, indent=2) + "\n", encoding="utf-8")
        return out
