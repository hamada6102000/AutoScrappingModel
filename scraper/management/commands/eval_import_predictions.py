"""
Optional — score an external baseline (AutoScraper, AXE, ...) on the same snapshots.

    python manage.py eval_import_predictions --name autoscraper --file autoscraper_preds.csv

CSV columns: snapshot_id, field, value  (lists: items separated by || or newlines)
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from scraper.evaluation.baselines import import_predictions
from scraper.evaluation.cli import add_experiment_arg, get_experiment


class Command(BaseCommand):
    help = "Import an external baseline's predictions for scoring."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--name", required=True, help="Baseline name, e.g. autoscraper")
        parser.add_argument("--file", required=True)

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        path = Path(opts["file"])
        if not path.exists():
            raise CommandError(f"File not found: {path}")
        name = opts["name"].strip().lower().replace(" ", "-")
        rep = import_predictions(exp, name, path)
        self.stdout.write(self.style.SUCCESS(f"Imported {rep['runs']} snapshot predictions for '{name}'."))
        for err in rep["errors"]:
            self.stdout.write(self.style.ERROR(f"  {err}"))
        self.stdout.write("Re-run eval_report to include it in the comparison table.")
