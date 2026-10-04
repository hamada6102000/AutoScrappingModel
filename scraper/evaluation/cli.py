"""Shared helpers for the eval_* management commands."""

from __future__ import annotations

from django.core.management.base import CommandError

from ..models import Experiment


def get_experiment(name: str | None) -> Experiment:
    if name:
        try:
            return Experiment.objects.get(name=name)
        except Experiment.DoesNotExist as exc:
            raise CommandError(f"No experiment named {name!r}. Run eval_init first.") from exc
    qs = Experiment.objects.all()
    if qs.count() == 1:
        return qs.first()
    if not qs.exists():
        raise CommandError("No experiment yet. Run: python manage.py eval_init --config eval/sites.yaml")
    raise CommandError("Several experiments exist; pass --experiment NAME")


def site_list(value: str | None):
    if not value:
        return None
    return [s.strip().lower() for s in value.split(",") if s.strip()]


def pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def add_experiment_arg(parser) -> None:
    parser.add_argument("--experiment", help="Experiment name (default: the only one).")
