"""
Post hoc sensitivity analysis (reported next to the primary results on all
sites, never instead of them).

    python manage.py eval_sensitivity                  # threshold 0.25
    python manage.py eval_sensitivity --min-not-found-share 0.25

Sets aside the sites on which at least the given share of the evaluated fields
has a recorded value that does not occur in the saved HTML (configuration
audit, fixed before any LLM call) and recomputes the module table, the paired
tests and the target metrics without them. Writes
eval_data/<exp>/results/sensitivity.md and sensitivity.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from scraper.evaluation import sensitivity as S
from scraper.evaluation.cli import add_experiment_arg, get_experiment
from scraper.services import snapshots as snapstore


class Command(BaseCommand):
    help = "Recompute the main metrics without sites whose saved page lacks many recorded values."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--min-not-found-share", type=float, default=S.DEFAULT_THRESHOLD)
        parser.add_argument("--out", help="Output directory (default: eval_data/<exp>/results).")

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        thr = opts["min_not_found_share"]
        if not 0 < thr <= 1:
            raise CommandError("--min-not-found-share must be in (0, 1]")
        rep = S.build(exp, thr)
        out = Path(opts["out"]) if opts["out"] else snapstore.experiment_dir(exp) / "results"
        out.mkdir(parents=True, exist_ok=True)
        (out / "sensitivity.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        (out / "sensitivity.md").write_text(S.markdown(rep), encoding="utf-8")
        aside = rep["sites_set_aside"]
        self.stdout.write(f"Post hoc sensitivity analysis (threshold {thr:.2f}): set aside "
                          f"{', '.join(aside) or 'no site'}.")
        full, sub = rep["all_sites"]["targets"], rep["without_set_aside"]["targets"]
        if full and sub:
            for k in full:
                v, w = full[k], sub[k]
                fmt = (lambda x: "n/a" if x is None else f"{x:.2f} s") if k.endswith("_s") else \
                      (lambda x: "n/a" if x is None else f"{100 * x:.1f}%")
                self.stdout.write(f"  {k:<36} all sites {fmt(v):>9}   without set-aside {fmt(w):>9}")
        self.stdout.write(f"Written to {out / 'sensitivity.md'}")
