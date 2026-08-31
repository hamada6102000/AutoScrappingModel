"""
Seed a self-contained demo target (an e-commerce product page) with its field
schema and ground-truth labels, so the pipeline can be run end-to-end offline.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from demopages import state
from scraper.models import Category, FieldType, ScrapeTarget, TargetField

DEMO_FIELDS = [
    # (name, description, field_type, is_required, ground_truth, order)
    ("title", "the product name / title", FieldType.STRING, True, "Aurora Wireless Headphones", 1),
    ("price", "the product price", FieldType.PRICE, True, "149.99", 2),
    ("availability", "stock availability status", FieldType.STRING, True, "In Stock", 3),
    ("rating", "the average customer rating score", FieldType.NUMBER, False, "4.6", 4),
    ("sku", "the product SKU / item code", FieldType.STRING, False, "AUR-1024", 5),
]


class Command(BaseCommand):
    help = "Create the demo scrape target and reset the demo page to v1."

    def add_arguments(self, parser):
        parser.add_argument(
            "--base-url",
            default="http://127.0.0.1:8000",
            help="Base URL of the running dev server (default: %(default)s).",
        )

    def handle(self, *args, **options):
        base = options["base_url"].rstrip("/")
        url = f"{base}/demo/product/"

        target, created = ScrapeTarget.objects.get_or_create(
            name="Demo Product (ShopDemo)",
            defaults={"url": url, "category": Category.ECOMMERCE},
        )
        if not created:
            target.url = url
            target.category = Category.ECOMMERCE
            target.save()

        for name, desc, ftype, required, gt, order in DEMO_FIELDS:
            TargetField.objects.update_or_create(
                target=target,
                name=name,
                defaults={
                    "description": desc,
                    "field_type": ftype,
                    "is_required": required,
                    "ground_truth": gt,
                    "order": order,
                },
            )

        # Reset the demo site to its original (v1) layout.
        state.set_active_version("v1")

        verb = "Created" if created else "Updated"
        self.stdout.write(self.style.SUCCESS(
            f"{verb} demo target #{target.pk} '{target.name}' -> {url} "
            f"with {len(DEMO_FIELDS)} fields. Demo page reset to v1."
        ))
        self.stdout.write(
            "Next: start the server (python manage.py runserver), then run:\n"
            f"  python manage.py run_scrape --target {target.pk}"
        )
