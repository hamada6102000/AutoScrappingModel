"""
Run the self-healing pipeline for one or all targets and print a summary.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from scraper.models import ScrapeTarget
from scraper.services.pipeline import ScrapePipeline


class Command(BaseCommand):
    help = "Execute the scraping pipeline for a target (or all active targets)."

    def add_arguments(self, parser):
        parser.add_argument("--target", type=int, help="Target id to scrape.")
        parser.add_argument("--all", action="store_true", help="Scrape all active targets.")
        parser.add_argument("--provider", help="Override LLM provider for this run.")

    def handle(self, *args, **options):
        if options["target"]:
            targets = ScrapeTarget.objects.filter(pk=options["target"])
            if not targets:
                raise CommandError(f"No target with id {options['target']}")
        elif options["all"]:
            targets = ScrapeTarget.objects.filter(is_active=True)
        else:
            raise CommandError("Provide --target <id> or --all")

        provider = None
        if options.get("provider"):
            from scraper.llm.factory import get_provider_safe

            provider = get_provider_safe(options["provider"])

        pipeline = ScrapePipeline(provider=provider)
        for target in targets:
            run = pipeline.run(target)
            self._print_run(run)

    def _print_run(self, run):
        style = self.style.SUCCESS if run.status in ("success", "healed") else self.style.WARNING
        self.stdout.write(style(
            f"\nRun #{run.pk} — {run.target.name}\n"
            f"  status         : {run.status}\n"
            f"  health         : {run.health_score:.2f} "
            f"({run.fields_extracted}/{run.fields_total} fields)\n"
            f"  failure rate   : {run.xpath_failure_rate:.2f}\n"
            f"  struct. dist.  : {run.structural_distance:.3f}\n"
            f"  heal attempts  : {run.heal_attempts} (healed={run.healed})\n"
            f"  duration       : {run.duration_ms} ms "
            f"(heal {run.heal_latency_ms} ms)\n"
        ))
        for rec in run.records.all():
            flag = "OK " if rec.success else "XX "
            gt = "" if rec.matches_ground_truth is None else (
                " ✓GT" if rec.matches_ground_truth else " ✗GT"
            )
            self.stdout.write(f"    [{flag}] {rec.field.name:14s} = {rec.normalized_value!r}{gt}")
