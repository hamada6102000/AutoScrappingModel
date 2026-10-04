"""
Step 6 — run the evaluation protocol (resumable).

    python manage.py eval_run --dry-run              # plan only, no LLM calls
    python manage.py eval_run                        # full run (re-run to resume)
    python manage.py eval_run --max-llm-calls 200    # stop after a budget
    python manage.py eval_run --sites shop-a         # a subset
    python manage.py eval_run --natural              # natural-change chain (optional)
    python manage.py eval_run --wait                 # keep going: sleep through rate limits
"""

from __future__ import annotations

import time

from django.core.management.base import BaseCommand, CommandError

from scraper.evaluation.cli import add_experiment_arg, get_experiment, site_list
from scraper.evaluation.protocol import EvalError, EvaluationRunner, build_schedule, estimate


class Command(BaseCommand):
    help = "Run the paired self-healing vs. static evaluation."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--sites", help="Comma-separated site keys (default: all).")
        parser.add_argument("--max-llm-calls", type=int, help="Stop after this many LLM calls.")
        parser.add_argument("--dry-run", action="store_true", help="Show the plan and estimate only.")
        parser.add_argument("--natural", action="store_true", help="Run the natural-change chain.")
        parser.add_argument("--wait", action="store_true",
                            help="When the LLM quota is exhausted, sleep until the provider's "
                                 "suggested time and resume automatically.")
        parser.add_argument("--conditions",
                            help="Comma-separated subset to run now, e.g. rule_based,rule_static "
                                 "(no LLM needed). Default: all conditions of the experiment.")
        parser.add_argument("--allow-unlabelled", action="store_true",
                            help="Run even if some fields have no ground-truth label; those "
                                 "fields are scraped but not scored (reported as unscored).")
        parser.add_argument("--skip-unavailable", action="store_true",
                            help="Skip (and list) sites whose page could not be saved.")
        parser.add_argument("--max-hours", type=float, default=72.0,
                            help="With --wait: give up after this many hours (default 72).")

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        e = exp.config["experiment"]
        sites = site_list(opts["sites"])

        if opts["dry_run"]:
            est = estimate(exp)
            self.stdout.write(f"Plan for {exp.name}: conditions {', '.join(e['conditions'])}, "
                              f"{e['repetitions']} repetitions\n")
            for row in est["sites"]:
                if sites and row["site"] not in sites:
                    continue
                plan = build_schedule(e, row["site"], 1)
                no_severity = {"control", "perturb_numbers", "insert_banner"}
                ops = " > ".join(s["op"] if s["op"] in no_severity else f"{s['op']}({s['severity']})"
                                 for s in plan)
                self.stdout.write(f"  {row['site']:<20} prompt {row['llm_view_tokens']} tok/call, "
                                  f"{row['calls_min']}-{row['calls_max']} calls; r1: {ops}")
            t = est["totals"]
            self.stdout.write(f"\nTotal: {t['calls_min']}-{t['calls_max']} LLM calls, "
                              f"~{t['tokens_min']:,}-{t['tokens_max']:,} tokens (upper bound).")
            return

        deadline = time.time() + opts["max_hours"] * 3600
        while True:
            try:
                runner = EvaluationRunner(exp, write=self.stdout.write,
                                          max_llm_calls=opts["max_llm_calls"], sites=sites,
                                          conditions=site_list(opts.get("conditions")),
                                          allow_unlabelled=opts["allow_unlabelled"],
                                          skip_unavailable=opts["skip_unavailable"])
            except Exception as exc:
                raise CommandError(f"Cannot start: {exc}") from exc
            try:
                stats = runner.run(natural=opts["natural"])
            except EvalError as exc:
                raise CommandError(str(exc)) from exc
            stopped = str(stats.get("stopped", ""))
            quota = stopped.startswith("LLM quota")
            transient = stopped.startswith("LLM provider error") and stats.get("retryable")
            if not (opts["wait"] and (quota or transient)):
                break
            if quota:
                delay = max(60.0, min(float(stats.get("retry_after") or 300) + 15, 3 * 3600))
            else:
                delay = 300.0
            if time.time() + delay > deadline:
                self.stdout.write(self.style.WARNING("--max-hours reached; stopping."))
                break
            self.stdout.write(self.style.WARNING(
                f"{stats['stopped'][:200]}\nSleeping {int(delay)} s, then resuming... "
                "(Ctrl+C is safe: completed runs are kept.)"))
            time.sleep(delay)

        self.stdout.write(self.style.SUCCESS(
            f"\nDone this session: {stats['runs']} runs, {stats['llm_calls']} LLM calls, "
            f"{stats.get('rule_calls', 0)} rule-based generator calls "
            f"({stats['skipped']} already completed runs skipped)."))
        if stats.get("sites_without_snapshot"):
            self.stdout.write(self.style.WARNING(
                "  skipped (no saved page): " + ", ".join(stats["sites_without_snapshot"])))
        for note in stats.get("waiting") or []:
            self.stdout.write(self.style.WARNING(f"  waiting: {note}"))
        if stats.get("stopped"):
            self.stdout.write(self.style.WARNING(
                f"Stopped early: {stats['stopped']}\nNothing is lost - re-run the same command later "
                "to resume exactly where it stopped."))
        else:
            self.stdout.write("\nNext: python manage.py eval_report   then   python manage.py export_results")
