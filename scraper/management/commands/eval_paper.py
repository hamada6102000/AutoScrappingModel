"""
Write the paper's experimental-results text and LaTeX table from the report.

    python manage.py eval_paper

Reads eval_data/<exp>/results/summary.json (run eval_report first) and writes
paper_results.md and paper_table_modules.tex next to it. Every number comes
from summary.json; modules that were not run are stated as not run.
"""

from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from scraper.evaluation.cli import add_experiment_arg, get_experiment
from scraper.evaluation.paper import results_latex, results_markdown
from scraper.services import snapshots as snapstore


class Command(BaseCommand):
    help = "Generate the experimental-results section (Markdown + LaTeX) from summary.json."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--results", help="Results directory (default: eval_data/<exp>/results).")
        parser.add_argument("--llm-scope-reason", default="",
                            help="Why the LLM modules were executed on fewer sites than configured "
                                 "(overrides the reason recorded by eval_report).")

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        res = Path(opts["results"]) if opts["results"] else snapstore.experiment_dir(exp) / "results"
        summary = res / "summary.json"
        if not summary.exists():
            raise CommandError(f"{summary} not found; run eval_report first")
        rep = json.loads(summary.read_text(encoding="utf-8"))
        if opts["llm_scope_reason"] and (rep.get("llm_scope") or {}).get("reduced"):
            rep["llm_scope"]["reason"] = opts["llm_scope_reason"].strip()
        (res / "paper_results.md").write_text(results_markdown(rep), encoding="utf-8")
        (res / "paper_table_modules.tex").write_text(results_latex(rep), encoding="utf-8")
        self.stdout.write(f"Written {res / 'paper_results.md'} and {res / 'paper_table_modules.tex'}")
