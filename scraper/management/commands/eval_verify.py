"""
Check that every reported number is correct and reproducible.

    python manage.py eval_verify

Re-scores all field records, re-aggregates P/R/F1/accuracy per module with an
independent implementation, rebuilds the report twice, checks snapshot
hashes and re-creates every simulated page from its seed. Writes
results/verification.md and verification.json.
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from scraper.evaluation.cli import add_experiment_arg, get_experiment
from scraper.evaluation.verify import verify
from scraper.services import snapshots as snapstore


class Command(BaseCommand):
    help = "Independently verify the evaluation's metrics and reproducibility."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--out", help="Results directory (default: eval_data/<exp>/results).")

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        out = Path(opts["out"]) if opts["out"] else snapstore.experiment_dir(exp) / "results"
        res = verify(exp, out)
        for c in res["checks"]:
            mark = self.style.SUCCESS("PASS") if c["ok"] else self.style.ERROR("FAIL")
            self.stdout.write(f"  {mark}  {c['check']}: {c['detail']}")
        self.stdout.write(f"\nWritten to {out / 'verification.md'}")
        if not res["ok"]:
            raise CommandError("verification failed")
