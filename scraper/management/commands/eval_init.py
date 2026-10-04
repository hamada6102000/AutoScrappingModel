"""
Step 1 — create (or update) an evaluation experiment from a sites file.

    python manage.py eval_init --config eval/sites.yaml
"""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from scraper.evaluation.config import ConfigError, load_config
from scraper.evaluation.protocol import EvalError, init_experiment
from scraper.llm.factory import get_provider
from scraper.services import tokens


class Command(BaseCommand):
    help = "Create or update an evaluation experiment from eval/sites.yaml."

    def add_arguments(self, parser):
        parser.add_argument("--config", default="eval/sites.yaml")
        parser.add_argument("--notes", default="")
        parser.add_argument("--force", action="store_true",
                            help="Delete this experiment's runs and start over.")

    def handle(self, *args, **opts):
        try:
            config = load_config(opts["config"])
        except ConfigError as exc:
            raise CommandError(str(exc)) from exc
        try:
            exp = init_experiment(config, notes=opts["notes"], force=opts["force"])
        except EvalError as exc:
            raise CommandError(str(exc)) from exc

        e = config["experiment"]
        self.stdout.write(self.style.SUCCESS(f"Experiment '{exp.name}' ready."))
        self.stdout.write(f"  sites        : {len(config['sites'])}")
        self.stdout.write(f"  conditions   : {', '.join(e['conditions'])}")
        self.stdout.write(f"  repetitions  : {e['repetitions']}  (targets: {exp.targets.count()})")
        self.stdout.write(f"  LLM          : {e['llm']['provider']} / {e['llm'].get('model') or 'default'}")
        self.stdout.write(f"  tokenizer    : {tokens.tokenizer_name()}")
        if tokens.is_estimate():
            self.stdout.write(self.style.WARNING(
                "  WARNING: exact tokenizer unavailable (tiktoken could not load its vocabulary). "
                "Token counts would be chars/4 estimates. Check your internet connection and "
                "'pip install tiktoken', then re-run eval_init."))
        try:
            get_provider(e["llm"]["provider"], e["llm"].get("model"))
        except Exception as exc:
            self.stdout.write(self.style.WARNING(f"  WARNING: LLM provider not usable yet: {exc}"))
        self.stdout.write("\nNext: python manage.py eval_snapshot")
