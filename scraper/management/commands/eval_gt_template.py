"""
Step 3 — write the ground-truth sheet (Excel + CSV) for you to fill in.

    python manage.py eval_gt_template
    python manage.py eval_gt_template --kind natural --label day14
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from scraper.evaluation.cli import add_experiment_arg, get_experiment
from scraper.evaluation.gt_io import template_rows, write_template
from scraper.models import SnapshotKind
from scraper.services import snapshots as snapstore


class Command(BaseCommand):
    help = "Export the ground-truth labelling sheet."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--kind", choices=["baseline", "natural"], default="baseline")
        parser.add_argument("--label", help="Snapshot label (default: baseline).")
        parser.add_argument("--out", help="Output path without extension.")

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        kind = SnapshotKind.NATURAL if opts["kind"] == "natural" else SnapshotKind.BASELINE
        label = opts["label"] or ("baseline" if kind == SnapshotKind.BASELINE else None)
        out = Path(opts["out"]) if opts["out"] else snapstore.experiment_dir(exp) / f"ground_truth_{label or 'natural'}"
        rows = template_rows(exp, kind, label)
        if not rows:
            self.stdout.write(self.style.WARNING("No snapshots found. Run eval_snapshot first."))
            return
        paths = write_template(exp, out, kind, label)
        filled = sum(1 for r in rows if r["value"])
        self.stdout.write(self.style.SUCCESS(f"{len(rows)} rows ({filled} already filled) written to:"))
        for p in paths:
            self.stdout.write(f"  {p}")
        self.stdout.write(
            "\nOpen the .xlsx, read the 'README' sheet, fill the yellow 'value' column using each "
            "row's labeling view, save, then run:\n"
            f"  python manage.py eval_gt_import --file \"{paths[-1]}\" --verified-by YOUR_NAME"
        )
