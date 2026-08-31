"""
Simulate (or revert) a website layout change by switching which version of the
demo product page is served. Mirrors the proposal's controlled-change scenario
(Section 4.1).
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from demopages import state


class Command(BaseCommand):
    help = "Switch the demo product page between layout v1 and v2."

    def add_arguments(self, parser):
        parser.add_argument(
            "version",
            nargs="?",
            default="v2",
            choices=state.VALID_VERSIONS,
            help="Which layout to serve (default: v2 = redesigned).",
        )

    def handle(self, *args, **options):
        version = options["version"]
        state.set_active_version(version)
        label = "redesigned (v2)" if version == "v2" else "original (v1)"
        self.stdout.write(self.style.SUCCESS(
            f"Demo product page now serving {label}. "
            f"Run the pipeline again to observe self-healing."
        ))
