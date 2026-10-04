"""
Run the self-healing pipeline for one or all targets and print a summary.

    python manage.py run_scrape --target 1
    python manage.py run_scrape --target 1 --provider groq
    python manage.py run_scrape --target 1 --no-heal        # static baseline
    python manage.py run_scrape --all --no-fallback          # never use heuristic fallback
"""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from scraper.llm.factory import get_provider
from scraper.models import RecordPhase, ScrapeTarget
from scraper.services.pipeline import ScrapePipeline


class Command(BaseCommand):
    help = "Execute the scraping pipeline for a target (or all active targets)."

    def add_arguments(self, parser):
        parser.add_argument("--target", type=int, help="Target id to scrape.")
        parser.add_argument("--all", action="store_true", help="Scrape all active, non-evaluation targets.")
        parser.add_argument("--provider", help="LLM provider for this run (heuristic, groq, ...).")
        parser.add_argument("--model", help="Model id override for the provider.")
        parser.add_argument("--no-heal", action="store_true",
                            help="Disable self-healing (static LLM scraper baseline).")
        parser.add_argument("--no-fallback", action="store_true",
                            help="Never fall back to the heuristic if the LLM fails.")

    def handle(self, *args, **options):
        if options["target"]:
            targets = ScrapeTarget.objects.filter(pk=options["target"])
            if not targets:
                raise CommandError(f"No target with id {options['target']}")
        elif options["all"]:
            targets = ScrapeTarget.objects.filter(is_active=True, experiment__isnull=True)
        else:
            raise CommandError("Provide --target <id> or --all")

        provider = None
        if options.get("provider") or options.get("model"):
            try:
                provider = get_provider(options.get("provider"), options.get("model"))
            except Exception as exc:
                raise CommandError(f"Cannot create provider: {exc}") from exc

        pipeline = ScrapePipeline(
            provider=provider,
            heal_enabled=not options["no_heal"],
            allow_fallback=False if options["no_fallback"] else None,
        )
        for target in targets:
            self._print_run(pipeline.run(target))

    def _print_run(self, run):
        ok = run.status in ("success", "healed")
        style = self.style.SUCCESS if ok else self.style.WARNING
        self.stdout.write(style(
            f"\nRun #{run.pk} - {run.target.name}\n"
            f"  status         : {run.status} (healing {'on' if run.heal_enabled else 'OFF'})\n"
            f"  provider       : {run.provider} {run.model}"
            f"{'  [FALLBACK USED]' if run.fallback_used else ''}\n"
            f"  health         : {run.health_score:.2f} "
            f"({run.fields_extracted}/{run.fields_total} fields; before healing "
            f"{(run.pre_heal_health if run.pre_heal_health is not None else run.health_score):.2f})\n"
            f"  struct./markup : {run.structural_distance:.3f} / {run.markup_distance:.3f}\n"
            f"  heal attempts  : {run.heal_attempts} (healed={run.healed})\n"
            f"  LLM            : {run.llm_calls_count} call(s), {run.prompt_tokens}+"
            f"{run.completion_tokens} tokens, {run.llm_errors} error(s)\n"
            f"  duration       : {run.duration_ms} ms (heal {run.heal_latency_ms} ms, "
            f"rate-limit wait {run.rate_limit_wait_ms} ms)\n"
        ))
        for rec in run.records.filter(phase=RecordPhase.FINAL):
            flag = "OK" if rec.success else "XX"
            gt = "" if rec.matches_ground_truth is None else (
                "  [GT match]" if rec.matches_ground_truth else "  [GT MISMATCH]"
            )
            value = rec.normalized_value if rec.success else f"<{rec.reason}>"
            self.stdout.write(f"    [{flag}] {rec.field.name:14s} = {value!r}{gt}")
