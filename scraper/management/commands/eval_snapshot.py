"""
Step 2 — save a snapshot of every site's page.

    python manage.py eval_snapshot                          # baseline pages
    python manage.py eval_snapshot --sites shop-a,news-b --refresh
    python manage.py eval_snapshot --kind natural --label day14   # later re-fetch
"""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from scraper.evaluation.cli import add_experiment_arg, get_experiment, pct, site_list
from scraper.evaluation.protocol import BASELINE_LABEL, EvalError, capture_snapshot
from scraper.models import SnapshotKind
from scraper.services import snapshots as snapstore


class Command(BaseCommand):
    help = "Fetch and save page snapshots for the experiment's sites."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--sites", help="Comma-separated site keys (default: all).")
        parser.add_argument("--kind", choices=["baseline", "natural"], default="baseline")
        parser.add_argument("--label", help="Snapshot label (required for --kind natural, e.g. day14).")
        parser.add_argument("--refresh", action="store_true", help="Re-fetch even if already saved.")

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        kind = SnapshotKind.NATURAL if opts["kind"] == "natural" else SnapshotKind.BASELINE
        label = opts["label"] or (BASELINE_LABEL if kind == SnapshotKind.BASELINE else None)
        if not label:
            raise CommandError("--label is required for natural snapshots (e.g. --label day14)")
        keys = site_list(opts["sites"]) or [s["key"] for s in exp.config["sites"]]
        failures = 0
        for site in exp.config["sites"]:
            if site["key"] not in keys:
                continue
            try:
                snap = capture_snapshot(exp, site, kind=kind, label=label, refresh=opts["refresh"],
                                        write=self.stdout.write)
            except EvalError as exc:
                failures += 1
                self.stdout.write(self.style.ERROR(f"  FAIL {site['key']}: {exc}"))
                continue
            robots = {True: "allowed", False: "DISALLOWED", None: "n/a"}[snap.robots_allowed]
            self.stdout.write(
                f"  OK   {site['key']:<20} HTTP {snap.http_status}  raw {snap.raw_tokens} -> "
                f"LLM view {snap.llm_view_tokens} tokens ({pct(snap.token_reduction)} reduction)  "
                f"robots {robots}  [{snap.tokenizer}]"
            )
            self.stdout.write(f"       labeling view: {snapstore.label_view_path(snap)}")
        if failures:
            self.stdout.write(self.style.WARNING(
                f"\n{failures} site(s) failed. For blocked or JavaScript-heavy sites either set "
                "render: js (needs Playwright) or save the page from your browser and set html_file."))
        self.stdout.write(f"\nNext: python manage.py eval_gt_template"
                          + (f" --kind natural --label {label}" if kind == SnapshotKind.NATURAL else ""))
