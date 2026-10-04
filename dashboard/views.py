"""
Dashboard controllers (the "C" in MVC).

Read-only pages plus two POST actions: run the pipeline for a target, and
simulate a layout change on the demo site.
"""

from __future__ import annotations

from django.contrib import messages
from django.db.models import Avg, Count, Q
from django.shortcuts import get_object_or_404, redirect, render

from demopages import state
from scraper.models import (
    ChangeEvent,
    Experiment,
    RecordPhase,
    RunStatus,
    ScrapeRun,
    ScrapeTarget,
)
from scraper.services.pipeline import ScrapePipeline


def target_list(request):
    # Evaluation targets (one per site x condition x repetition) are summarised
    # per experiment below instead of being listed individually.
    show_eval = request.GET.get("eval") == "1"
    targets = ScrapeTarget.objects.prefetch_related("fields", "runs")
    if not show_eval:
        targets = targets.filter(experiment__isnull=True)
    runs = ScrapeRun.objects.all()
    stats = {
        "targets": targets.count(),
        "runs": runs.count(),
        "healed": runs.filter(status=RunStatus.HEALED).count(),
        "change_events": ChangeEvent.objects.count(),
        "avg_health": runs.aggregate(v=Avg("health_score"))["v"] or 0.0,
    }
    experiments = Experiment.objects.annotate(
        n_targets=Count("targets", distinct=True),
        n_runs=Count("runs", distinct=True),
        n_snapshots=Count("snapshots", distinct=True),
    )
    context = {
        "targets": targets,
        "stats": stats,
        "experiments": experiments,
        "show_eval": show_eval,
        "active_version": state.get_active_version(),
    }
    return render(request, "dashboard/target_list.html", context)


def target_detail(request, pk: int):
    target = get_object_or_404(
        ScrapeTarget.objects.prefetch_related("fields__rules", "runs"), pk=pk
    )
    fields = target.fields.all()
    runs = target.runs.all()[:20]
    # Health series for the mini chart (oldest -> newest).
    series = list(reversed([
        {"id": r.id, "health": round(r.health_score * 100), "status": r.status}
        for r in target.runs.all()[:30]
    ]))
    context = {
        "target": target,
        "fields": fields,
        "runs": runs,
        "series": series,
        "change_events": target.change_events.all()[:10],
        "active_version": state.get_active_version(),
    }
    return render(request, "dashboard/target_detail.html", context)


def run_detail(request, pk: int):
    run = get_object_or_404(
        ScrapeRun.objects.select_related("target"), pk=pk
    )
    records = run.records.select_related("field", "xpath_used")
    context = {
        "run": run,
        "records": records.filter(phase=RecordPhase.FINAL),
        "pre_heal_records": records.filter(phase=RecordPhase.PRE_HEAL),
        "llm_calls": run.llm_calls.order_by("created_at", "id"),
        "failures": run.failures.select_related("field").all(),
        "change_events": run.change_events.all(),
    }
    return render(request, "dashboard/run_detail.html", context)


def change_event_log(request):
    events = ChangeEvent.objects.select_related("target", "run").all()[:100]
    return render(request, "dashboard/change_events.html", {"events": events})


# --- POST actions ----------------------------------------------------------
def run_target(request, pk: int):
    target = get_object_or_404(ScrapeTarget, pk=pk)
    if request.method != "POST":
        return redirect("dashboard:target_detail", pk=pk)
    try:
        run = ScrapePipeline().run(target)
    except Exception as exc:  # pragma: no cover - surfaced to the user
        messages.error(request, f"Pipeline error: {exc}")
        return redirect("dashboard:target_detail", pk=pk)

    if run.status == RunStatus.HEALED:
        messages.success(
            request,
            f"Layout change detected and auto-healed in "
            f"{run.heal_latency_ms} ms — health back to {run.health_percent}%.",
        )
    elif run.status == RunStatus.SUCCESS:
        messages.success(request, f"Scrape succeeded — health {run.health_percent}%.")
    elif run.status == RunStatus.DEGRADED:
        messages.warning(request, "Scrape degraded — self-healing exhausted its attempts.")
    else:
        messages.error(request, f"Scrape {run.status}.")
    return redirect("dashboard:run_detail", pk=run.pk)


def simulate_change(request):
    if request.method != "POST":
        return redirect("dashboard:target_list")
    current = state.get_active_version()
    new_version = "v2" if current == "v1" else "v1"
    state.set_active_version(new_version)
    label = "redesigned layout (v2)" if new_version == "v2" else "original layout (v1)"
    messages.info(
        request,
        f"Demo site now serving the {label}. Run a scrape to see the effect.",
    )
    next_url = request.POST.get("next") or "dashboard:target_list"
    if next_url.startswith("/"):
        return redirect(next_url)
    return redirect(next_url)
