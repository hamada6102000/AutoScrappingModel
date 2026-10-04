"""
Step 4 — import the filled ground-truth sheet and check it against the pages.

    python manage.py eval_gt_import --file eval_data/<exp>/ground_truth_baseline.xlsx --verified-by HM
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from scraper.evaluation.cli import add_experiment_arg, get_experiment
from scraper.evaluation.gt_io import import_ground_truth
from scraper.evaluation.protocol import baseline_snapshot, gt_status


class Command(BaseCommand):
    help = "Import ground-truth labels (xlsx or csv)."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--file", required=True)
        parser.add_argument("--verified-by", default="", help="Default verifier name/initials.")

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        path = Path(opts["file"])
        if not path.exists():
            raise CommandError(f"File not found: {path}")
        try:
            rep = import_ground_truth(exp, path, opts["verified_by"])
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(
            f"Imported {rep['imported']} labels ({rep['absent']} marked {'<ABSENT>'}); "
            f"{rep['empty']} rows still empty."))
        for err in rep["errors"]:
            self.stdout.write(self.style.ERROR(f"  ERROR {err}"))
        for warn in rep["warnings"]:
            self.stdout.write(self.style.WARNING(f"  CHECK {warn}"))

        incomplete = []
        for site in exp.config["sites"]:
            snap = baseline_snapshot(exp, site["key"])
            if snap is None:
                incomplete.append(f"{site['key']} (no snapshot)")
                continue
            missing, _ = gt_status(snap, site)
            if missing:
                incomplete.append(f"{site['key']} ({', '.join(missing)})")
        if incomplete:
            self.stdout.write(self.style.WARNING("Baseline labels still missing for: " + "; ".join(incomplete)))
        else:
            self.stdout.write(self.style.SUCCESS("All baseline labels present."))
            self.stdout.write("\nNext: python manage.py eval_check --ping")
