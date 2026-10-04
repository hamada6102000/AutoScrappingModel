"""
Step 7 — compute all metrics (Table 5.1, RQ1-RQ5) from the stored runs.

    python manage.py eval_report
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from scraper.evaluation import report as report_mod
from scraper.evaluation.cli import add_experiment_arg, get_experiment
from scraper.services import snapshots as snapstore


class Command(BaseCommand):
    help = "Build the evaluation report (summary.json, report.md, tables/*.csv)."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--out", help="Output directory (default: eval_data/<exp>/results).")

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        out = Path(opts["out"]) if opts["out"] else snapstore.experiment_dir(exp) / "results"
        rep = report_mod.write(exp, out)
        self.stdout.write(f"Report for {exp.name}: {rep['counts']['runs']} runs\n")
        for row in rep["table_5_1"]:
            v = row["value"]
            shown = "n/a" if v is None else (f"{v:.2f} s" if row["unit"] == "s" else f"{v * 100:.1f}%")
            met = "n/a" if row["met"] is None else ("met" if row["met"] else "NOT met")
            self.stdout.write(f"  {row['metric']:<72} {shown:>10}  target {row['target']:<8} {met}")
        self.stdout.write(self.style.SUCCESS(f"\nWritten to {out}"))
