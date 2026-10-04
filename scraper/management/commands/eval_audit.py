"""
Step 2b — check the saved snapshots against the annotated dataset.

    python manage.py eval_audit

Writes eval_data/<experiment>/audit/: audit.md (readable summary),
audit_sites.csv, audit_fields.csv and ground_truth_prefilled.xlsx (labels
pre-filled only where the recorded dataset value is literally on the saved
page; everything else left for you to label). No LLM calls.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from scraper.evaluation.audit import NEEDS_LABEL, run_audit, write_outputs
from scraper.evaluation.cli import add_experiment_arg, get_experiment
from scraper.services import snapshots as snapstore


class Command(BaseCommand):
    help = "Audit snapshots against the annotated dataset and pre-fill the ground-truth sheet."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--out", help="Output folder (default: eval_data/<experiment>/audit).")
        parser.add_argument("--max-input-chars", type=int,
                            help="Try a different prompt HTML cap (default: the experiment's).")

    def handle(self, *args, **opts):
        from pathlib import Path

        exp = get_experiment(opts["experiment"])
        cap = opts.get("max_input_chars")
        rows = run_audit(exp, cap)
        out = Path(opts["out"]) if opts["out"] else snapstore.experiment_dir(exp) / "audit"
        paths = write_outputs(exp, rows, out, cap)
        total = needs = meta_only = missing = 0
        for r in rows:
            fs = [f for f in r["fields"] if not f.get("excluded")]
            total += len(fs)
            needs += sum(1 for f in fs if f["status"] in NEEDS_LABEL or f["status"] in ("no-snapshot", "audit-error"))
            meta_only += sum(1 for f in fs if f.get("found_in") == "metadata")
            if not r.get("snapshot"):
                missing += 1
            flag = "" if r.get("snapshot") else (f"  (AUDIT ERROR: {r['error']})" if r.get("error")
                                                 else "  (NO SNAPSHOT)")
            if r.get("snapshot") and not r.get("content_ok"):
                flag = "  (CONTENT CHECK FAILED)"
            self.stdout.write(
                f"  {r['site']:<10} render {str(r.get('render', '-')):<6} prompt "
                f"{str(r.get('prompt_tokens', '-')):>6} tok  fields {len(fs):>3}  to label "
                f"{sum(1 for f in fs if f['status'] in NEEDS_LABEL or f['status'] in ('no-snapshot', 'audit-error')):>3}{flag}")
        self.stdout.write(f"\n{total} fields; {total - needs} pre-filled from the dataset, "
                          f"{needs} to label; {meta_only} found only in meta tags/JSON-LD; "
                          f"{missing} site(s) without a snapshot.")
        for key, p in paths.items():
            self.stdout.write(f"  {key}: {p}")
