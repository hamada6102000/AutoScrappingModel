"""
Copy the LLM limits measured by `eval_check --ping` into the sites file and
the experiment (before any LLM-module run).

    python manage.py eval_sync_llm_limits --config eval/sites.yaml

Reads eval_data/<exp>/llm_ping.json: the per-minute token limit Groq reported
for this key/model and whether max_completion_tokens counts toward it. Only
experiment.llm changes; completed rule-based runs are unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from django.core.management.base import BaseCommand, CommandError

from scraper.evaluation.cli import add_experiment_arg, get_experiment
from scraper.evaluation.config import ConfigError, load_config
from scraper.evaluation.protocol import EvalError, init_experiment
from scraper.services import snapshots as snapstore


class Command(BaseCommand):
    help = "Set llm.tokens_per_minute / completion_counts_in_tpm from the measured account limits."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--config", default="eval/sites.yaml")

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        ping_file = snapstore.experiment_dir(exp) / "llm_ping.json"
        if not ping_file.exists():
            raise CommandError("no llm_ping.json; run eval_check --ping first")
        ping = json.loads(ping_file.read_text(encoding="utf-8"))
        if not ping.get("ok") or ping.get("model") != exp.config["experiment"]["llm"].get("model"):
            raise CommandError("the last ping did not succeed for the configured model; nothing changed")
        tpm = ping.get("tokens_per_minute_reported")
        counted = ping.get("completion_counts_in_tpm_observed")
        path = Path(opts["config"])
        text = path.read_text(encoding="utf-8")
        header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
        data = yaml.safe_load(text)
        llm = data["experiment"].setdefault("llm", {})
        before = dict(llm)
        if tpm:
            llm["tokens_per_minute"] = int(tpm)
        if counted is not None:
            llm["completion_counts_in_tpm"] = bool(counted)
        if llm == before:
            self.stdout.write("LLM limits already match the account; nothing to change.")
            return
        path.write_text(header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=110),
                        encoding="utf-8")
        try:
            init_experiment(load_config(path))
        except (ConfigError, EvalError) as exc:
            path.write_text(text, encoding="utf-8")   # keep the file and the experiment in step
            if "LLM-module runs already exist" in str(exc):
                # Runs of the LLM modules were made with the stored settings; keep
                # them for the rest of the experiment (the limit only caps prompt size).
                self.stdout.write(self.style.WARNING(
                    f"The account now reports tokens_per_minute {tpm}; the experiment keeps "
                    f"{before.get('tokens_per_minute')} because LLM-module runs already exist."))
                return
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(
            f"llm settings updated from the account: tokens_per_minute {before.get('tokens_per_minute')} -> "
            f"{llm.get('tokens_per_minute')}, completion_counts_in_tpm {before.get('completion_counts_in_tpm')} "
            f"-> {llm.get('completion_counts_in_tpm')}"))
