"""
Step 5 — check everything is ready before spending LLM quota.

    python manage.py eval_check            # offline checks + call/token estimate
    python manage.py eval_check --ping     # also one tiny LLM call to verify the key
"""

from __future__ import annotations

import time

from django.core.management.base import BaseCommand

from scraper.evaluation.cli import add_experiment_arg, get_experiment, pct
from scraper.evaluation.protocol import baseline_snapshot, estimate, gt_status
from scraper.llm.base import FieldSpec
from scraper.llm.factory import get_provider
from scraper.services import tokens


class Command(BaseCommand):
    help = "Readiness check for an evaluation experiment."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--ping", action="store_true", help="Make one tiny LLM call.")

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        e = exp.config["experiment"]
        problems, notes = [], []

        if tokens.is_estimate():
            problems.append("exact tokenizer unavailable (tiktoken) - token counts would be estimates")
        try:
            provider = get_provider(e["llm"]["provider"], e["llm"].get("model"))
        except Exception as exc:
            provider = None
            problems.append(f"LLM provider not usable: {exc}")

        self.stdout.write(f"Experiment {exp.name}: {len(exp.config['sites'])} sites, "
                          f"{e['repetitions']} repetitions, LLM {e['llm']['provider']}\n")
        for site in exp.config["sites"]:
            snap = baseline_snapshot(exp, site["key"])
            if snap is None:
                problems.append(f"{site['key']}: no baseline snapshot")
                self.stdout.write(f"  {site['key']:<20} NO SNAPSHOT")
                continue
            missing, labelled = gt_status(snap, site)
            flags = []
            if snap.robots_allowed is False:
                flags.append("robots DISALLOW")
            if (snap.fetch_meta or {}).get("suspected_block"):
                flags.append("possible bot-block page")
                notes.append(f"{site['key']}: open the labeling view to confirm it is the real page")
            if snap.llm_view_truncated:
                flags.append("LLM view truncated")
            if missing:
                problems.append(f"{site['key']}: unlabelled fields {', '.join(missing)}")
            for name, p in (snap.field_presence or {}).items():
                if p.get("labelled_absent"):
                    continue
                if p.get("in_raw") and not p.get("in_clean_dom"):
                    problems.append(f"{site['key']}.{name}: removed by pruning (add to prune_keep)")
                elif not p.get("in_raw"):
                    notes.append(f"{site['key']}.{name}: labelled value not found in page text (typo?)")
            self.stdout.write(
                f"  {site['key']:<20} HTTP {snap.http_status}  view {snap.llm_view_tokens} tok "
                f"({pct(snap.token_reduction)} reduction)  labels {len(labelled)}/{len(site['fields'])}"
                + (f"  [{'; '.join(flags)}]" if flags else "")
            )

        est = estimate(exp)["totals"]
        self.stdout.write(
            f"\nEstimated LLM usage for the full run: {est['calls_min']}-{est['calls_max']} calls, "
            f"~{est['tokens_min']:,}-{est['tokens_max']:,} tokens (upper bound assumes every step breaks "
            "and every heal needs all attempts). Compare with your provider's daily limits.")

        if opts["ping"] and provider is not None:
            html = "<html><body><main><h1 class='title'>Ping</h1></main></body></html>"
            start = time.perf_counter()
            try:
                res = provider.generate_xpaths(html, [FieldSpec("title", "the page title")])
                ms = int((time.perf_counter() - start) * 1000)
                if res.xpaths.get("title"):
                    self.stdout.write(self.style.SUCCESS(
                        f"\nLLM ping OK: {res.model}, {ms} ms, {res.prompt_tokens}+{res.completion_tokens} "
                        f"tokens, returned {res.xpaths['title']!r}"))
                    limits = getattr(provider, "last_rate_limits", {}) or {}
                    if limits:
                        self.stdout.write("Provider rate limits reported for this key/model:")
                        for k, v in sorted(limits.items()):
                            self.stdout.write(f"  {k}: {v}")
                        self.stdout.write("  (requests = per day, tokens = per minute on Groq; the daily "
                                          "token limit is shown at console.groq.com/settings/limits)")
                else:
                    problems.append(f"LLM ping returned no XPath (raw: {res.raw_response[:120]!r})")
            except Exception as exc:
                problems.append(f"LLM ping failed: {type(exc).__name__}: {exc}")

        for n in notes:
            self.stdout.write(self.style.WARNING(f"  NOTE {n}"))
        if problems:
            self.stdout.write(self.style.ERROR(f"\nNOT READY - {len(problems)} problem(s):"))
            for p in problems:
                self.stdout.write(self.style.ERROR(f"  - {p}"))
        else:
            self.stdout.write(self.style.SUCCESS("\nREADY. Next: python manage.py eval_run"))
