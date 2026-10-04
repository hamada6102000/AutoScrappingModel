"""
Step 0 — turn the annotated site dataset into the evaluation config.

    python manage.py eval_import_dataset --dataset eval/dataset/dataset_20_sites.json
    python manage.py eval_import_dataset --dataset eval/dataset/sites.csv --fields eval/dataset/fields.csv

URLs, categories and fields are copied unchanged. Field types are mapped to the
harness types (text->string, other->mixed; number, price, date, url, boolean
unchanged), Single/List -> single/many, Yes/No -> required. The recorded value,
source, dynamic flag and original type of every field are kept in the config
(they are used to check snapshots and pre-fill labels, never shown to the LLM).
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from scraper.evaluation.config import ConfigError, resolve
from scraper.evaluation.dataset import load_dataset, to_config

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36")

HEADER = """\
# =============================================================================
# Evaluation configuration generated from the annotated dataset
#   {source}
# by: python manage.py eval_import_dataset
#
# URLs, categories and fields are exactly those of the dataset. Do not edit the
# dataset_* / source / dynamic keys: they document the annotation.
# To leave a field out of the study, add   exclude: true
#                                          exclude_reason: <why>
# (it stays listed in the paper's field table but is not scraped or scored).
# =============================================================================
"""


class Command(BaseCommand):
    help = "Create eval/sites.yaml from the annotated 20-site dataset (JSON, XLSX or two CSVs)."

    def add_arguments(self, parser):
        parser.add_argument("--dataset", required=True, help="dataset .json / .xlsx, or the sites .csv")
        parser.add_argument("--fields", help="fields .csv (only with a sites .csv)")
        parser.add_argument("--out", default="eval/sites.yaml")
        parser.add_argument("--name", default="ieee-2026", help="experiment name")
        parser.add_argument("--repetitions", type=int, default=2)
        parser.add_argument("--planned-repetitions", type=int, default=3,
                            help="repetitions in the study design (reported as planned)")
        parser.add_argument("--provider", default="groq")
        # Groq retired llama-3.3-70b-versatile for non-enterprise keys (deprecated
        # 2026-08-16); openai/gpt-oss-120b is its listed production replacement.
        parser.add_argument("--model", default="openai/gpt-oss-120b")
        parser.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high", "none"],
                            help="for reasoning models (gpt-oss); 'none' sends nothing")
        parser.add_argument("--tokens-per-minute", type=int, default=8000,
                            help="the account's per-minute token limit (0 = do not plan for it)")
        parser.add_argument("--render", default="auto", choices=["static", "js", "auto"])
        parser.add_argument("--user-agent", default=BROWSER_UA)
        parser.add_argument("--date-order", default="mdy", choices=["mdy", "dmy"])
        parser.add_argument("--max-input-chars", type=int, default=None)
        parser.add_argument("--force", action="store_true", help="overwrite an existing --out file")

    def handle(self, *args, **opts):
        import yaml

        out = Path(opts["out"])
        if out.exists() and not opts["force"]:
            raise CommandError(f"{out} exists; pass --force to overwrite it")
        try:
            dataset = load_dataset(opts["dataset"], opts.get("fields"))
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            raise CommandError(f"cannot read the dataset: {exc}") from exc
        config = to_config(
            dataset, name=opts["name"], repetitions=opts["repetitions"],
            planned_repetitions=opts["planned_repetitions"], provider=opts["provider"],
            model=opts["model"], render=opts["render"], user_agent=opts["user_agent"],
            date_order=opts["date_order"], max_input_chars=opts["max_input_chars"],
            reasoning_effort=None if opts["reasoning_effort"] == "none" else opts["reasoning_effort"],
            include_reasoning=False if opts["reasoning_effort"] != "none" else None,
            tokens_per_minute=opts["tokens_per_minute"] or None,
        )
        try:
            resolved = resolve(config, base_dir=out.parent)
        except ConfigError as exc:
            raise CommandError(str(exc)) from exc

        out.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False, width=110)
        out.write_text(HEADER.format(source=Path(opts["dataset"]).name) + text, encoding="utf-8")

        n_fields = sum(len(s["fields"]) for s in resolved["sites"])
        self.stdout.write(self.style.SUCCESS(f"Wrote {out}"))
        self.stdout.write(f"  sites  : {len(resolved['sites'])}")
        self.stdout.write(f"  fields : {n_fields}")
        by_type: dict = {}
        for s in resolved["sites"]:
            for f in s["fields"]:
                key = f"{f['dataset_type']}->{f['type']}"
                by_type[key] = by_type.get(key, 0) + 1
        self.stdout.write("  types  : " + ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())))
        self.stdout.write(f"\nNext: python manage.py eval_init --config {out}")
