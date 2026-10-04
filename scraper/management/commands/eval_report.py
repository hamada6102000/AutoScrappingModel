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
        parser.add_argument("--llm-scope-reason", default="",
                            help="Why the LLM modules were executed on fewer sites than configured "
                                 "(recorded in summary.json, report.md and the paper text).")

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        out = Path(opts["out"]) if opts["out"] else snapstore.experiment_dir(exp) / "results"
        rep = report_mod.write(exp, out, llm_scope_reason=opts["llm_scope_reason"])
        self.stdout.write(f"Report for {exp.name}: {rep['counts']['runs']} runs\n")
        scope = rep.get("llm_scope") or {}
        if scope.get("note"):
            self.stdout.write(self.style.WARNING(
                "LLM SCOPE: " + scope["note"] + (f" Reason: {scope['reason']}." if scope.get("reason") else "")))
        for m in rep.get("modules") or []:
            self.stdout.write(f"  {m['module']:<44} {m['sites']:>2} site(s), {m.get('chains')} chain(s)")
        for row in rep["table_5_1"]:
            v = row["value"]
            shown = "n/a" if v is None else (f"{v:.2f} s" if row["unit"] == "s" else f"{v * 100:.1f}%")
            met = "n/a" if row["met"] is None else ("met" if row["met"] else "NOT met")
            self.stdout.write(f"  {row['metric']:<72} {shown:>10}  target {row['target']:<8} {met}")
        self.stdout.write(self.style.SUCCESS(f"\nWritten to {out}"))
