"""
Step 8 — export everything (tables, report, snapshots, database) as one zip.

    python manage.py export_results
    python manage.py export_results --snapshots all --out results.zip
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from scraper.evaluation.cli import add_experiment_arg, get_experiment
from scraper.evaluation.export import export
from scraper.services import snapshots as snapstore


class Command(BaseCommand):
    help = "Export an experiment's raw data, report and snapshots as a zip."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--out", help="Zip path (default: eval_data/<exp>/<exp>_results.zip).")
        parser.add_argument("--snapshots", choices=["baseline", "all", "none"], default="baseline",
                            help="Which snapshot files to include (simulated ones are reproducible "
                                 "from the baseline + seeds).")

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        out = Path(opts["out"]) if opts["out"] else snapstore.experiment_dir(exp) / f"{exp.name}_results.zip"
        path = export(exp, out, snapshots=opts["snapshots"])
        size_mb = path.stat().st_size / 1e6
        self.stdout.write(self.style.SUCCESS(f"Exported {path} ({size_mb:.1f} MB)"))
        self.stdout.write("Send this zip back for the paper. It contains no API keys.")
