"""
Evaluation report: proposal Table 5.1 metrics, RQ1-RQ5, token reduction,
cost and LLM reliability — all computed from the database, nothing assumed.

Key definitions (also written into report.md):

* correct field  — see metrics.is_correct (lists: item F1 >= list_correct_f1).
* breaking change at step k (per condition) — at least one field that was
  correct at step k-1 is not correct at step k *before* any healing.
* alarm — the system's own detector fired (pre-heal health < threshold).
* self-healing success (GT-verified) — every field broken at step k is
  correct again after healing at step k.
* false alarm — an alarm on a step that was not breaking.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from ..models import (
    ChangeEvent, ExtractedRecord, GroundTruth, LLMCallLog, LLMPurpose,
    RecordPhase, ScrapeRun, Snapshot, SnapshotKind,
)
from . import metrics as M
from .config import COND_INFO, CONDITION_ORDER

TARGETS = {
    "initial_f1": 0.85,
    "healing_success_rate": 0.80,
    "time_to_recovery_s": 60.0,
    "regeneration_latency_s": 30.0,
    "sustained_f1": 0.80,
    "false_positive_rate": 0.05,
}
THRESHOLDS = (0.01, 0.05, 0.10, 0.20)
METADATA_MARKERS = ("page-metadata", "data-jsonld", "data-key", "data-meta", "//meta", "/meta[",
                    "link[@rel")


def xpath_target(xpath: str | None) -> str:
    """Where a rule reads from: the page-metadata block (meta tags, JSON-LD,
    canonical link) or the page content."""
    if not xpath:
        return "none"
    return "metadata" if any(m in xpath for m in METADATA_MARKERS) else "content"


def source_group(source: str) -> str:
    """Annotator's location of a value: on the rendered page (visible text,
    a class name, a <time> element) or only in meta tags / JSON-LD / the URL."""
    s = (source or "visible").lower()
    return "visible" if any(k in s for k in ("visible", "html-class", "time")) else "metadata-only"


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
def collect(exp) -> dict:
    cfg = exp.config
    site_cat = {s["key"]: s["category"] for s in cfg["sites"]}
    site_fields = {s["key"]: [f["name"] for f in s["fields"]] for s in cfg["sites"]}
    field_source = {(s["key"], f["name"]): source_group(f.get("source", "")) for s in cfg["sites"]
                    for f in s["fields"]}
    has_gt = {
        (g.snapshot_id, g.field_name)
        for g in GroundTruth.objects.filter(snapshot__experiment=exp)
        if not (g.notes or "").startswith("exclude")
    }
    runs = {}
    for r in ScrapeRun.objects.filter(experiment=exp, completed=True).select_related("target", "snapshot"):
        mut = (r.snapshot.mutation if r.snapshot_id else {}) or {}
        runs[r.id] = {
            "id": r.id, "site": r.target.site_key, "category": site_cat.get(r.target.site_key, "other"),
            "condition": r.condition, "rep": r.repetition, "step": r.step, "kind": r.step_kind,
            "true_change": r.true_change, "status": r.status, "detected": r.detected,
            "healed": r.healed, "health": r.health_score,
            "pre_health": r.pre_heal_health if r.pre_heal_health is not None else r.health_score,
            "pre_failure_rate": r.pre_heal_failure_rate if r.pre_heal_failure_rate is not None else r.xpath_failure_rate,
            "pre_field_mismatch": r.pre_heal_field_mismatch if r.pre_heal_field_mismatch is not None else r.field_count_mismatch,
            "structural": r.structural_distance, "markup": r.markup_distance,
            "heal_attempts": r.heal_attempts, "llm_calls": r.llm_calls_count,
            "llm_errors": r.llm_errors, "fallback": r.fallback_used,
            "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
            "cost": r.cost_usd, "wait_ms": r.rate_limit_wait_ms, "duration_ms": r.duration_ms,
            "heal_latency_ms": r.heal_latency_ms, "truncated": r.llm_input_truncated,
            "snapshot_id": r.snapshot_id, "op": mut.get("op", "baseline" if r.step == 0 else ""),
            "family": mut.get("family", ""), "severity": mut.get("severity"),
            "focus": mut.get("focus"), "affected": mut.get("affected"),
        }
    records = []
    qs = ExtractedRecord.objects.filter(run__experiment=exp, run__completed=True).select_related(
        "field", "xpath_used")
    for rec in qs:
        run = runs.get(rec.run_id)
        if run is None:
            continue
        records.append({
            "run_id": rec.run_id, "phase": rec.phase, "field": rec.field.name,
            "cardinality": rec.field.cardinality, "type": rec.field.field_type,
            "required": rec.field.is_required, "success": rec.success, "reason": rec.reason,
            "gt_available": rec.gt_available, "exact_match": rec.exact_match,
            "token_f1": rec.token_f1, "list_tp": rec.list_tp, "list_fp": rec.list_fp,
            "list_fn": rec.list_fn, "n_values": len(rec.values or []),
            "has_gt": (run["snapshot_id"], rec.field.name) in has_gt,
            "value": (rec.normalized_value or "")[:300], "gt_value": (rec.gt_value or "")[:300],
            "site": run["site"], "category": run["category"], "condition": run["condition"],
            "rep": run["rep"], "step": run["step"],
            "source_group": field_source.get((run["site"], rec.field.name), "visible"),
            "xpath_target": xpath_target(rec.xpath_used.xpath if rec.xpath_used_id else None),
        })
    events = [{
        "run_id": e.run_id, "resolution": e.resolution, "attempts": e.attempts,
        "history_len": e.failure_history_len, "ttr_ms": e.time_to_recovery_ms,
        "ttr_excl_ms": e.time_to_recovery_excl_wait_ms, "affected": e.affected_fields,
        "recovered": e.fields_recovered, "unrecovered": e.fields_unrecovered,
        "structural": e.structural_distance, "markup": e.markup_distance,
    } for e in ChangeEvent.objects.filter(run__experiment=exp, run__completed=True)]
    calls = [{
        "run_id": c.run_id, "provider": c.provider, "requested": c.requested_provider,
        "fallback": c.fallback_used, "model": c.model, "purpose": c.purpose,
        "attempt": c.attempt, "history_len": c.failure_history_len,
        "fields_requested": c.fields_requested, "returned": c.xpaths_returned,
        "valid": c.xpaths_valid, "prompt_tokens": c.prompt_tokens,
        "completion_tokens": c.completion_tokens, "cost": c.cost_usd,
        "latency_ms": c.latency_ms, "validation_ms": c.validation_ms,
        "wait_ms": c.rate_limit_wait_ms, "retries": c.retries,
        "truncated": c.input_truncated, "success": c.success, "error": c.error,
    } for c in LLMCallLog.objects.filter(run__experiment=exp, run__completed=True)]
    baselines = [{
        "site": s.site_key, "category": site_cat.get(s.site_key, "other"),
        "http_status": s.http_status, "robots_allowed": s.robots_allowed,
        "tokenizer": s.tokenizer, "raw_tokens": s.raw_tokens, "clean_tokens": s.clean_tokens,
        "llm_view_tokens": s.llm_view_tokens, "view_truncated": s.llm_view_truncated,
        "reduction": s.token_reduction, "presence": s.field_presence,
        "suspected_block": (s.fetch_meta or {}).get("suspected_block"),
        "render": (s.fetch_meta or {}).get("render_chosen") or (s.fetch_meta or {}).get("render")
                  or (s.fetch_meta or {}).get("source"),
    } for s in Snapshot.objects.filter(experiment=exp, kind=SnapshotKind.BASELINE)]
    gt_sources = defaultdict(int)
    for g in GroundTruth.objects.filter(snapshot__experiment=exp, snapshot__kind=SnapshotKind.BASELINE):
        gt_sources["absent" if g.notes == "absent" else g.source] += 1
    return {"runs": runs, "records": records, "events": events, "calls": calls,
            "baselines": baselines, "site_fields": site_fields, "gt_sources": dict(gt_sources),
            "list_threshold": cfg["experiment"].get("list_correct_f1", 0.9)}


# ---------------------------------------------------------------------------
# Derived labels
# ---------------------------------------------------------------------------
def label_breaking(data: dict) -> None:
    thr = data["list_threshold"]
    final, pre = {}, {}
    for rec in data["records"]:
        key = (rec["run_id"], rec["field"])
        value = M.is_correct(rec, thr)
        (final if rec["phase"] == RecordPhase.FINAL else pre)[key] = value
    data["final_correct"] = final
    index = {(r["condition"], r["site"], r["rep"], r["step"]): r for r in data["runs"].values()}
    for run in data["runs"].values():
        run["breaking"] = None
        run["broken_fields"] = []
        run["gt_recovered"] = None
        run["prev_clean"] = None
        fields = data["site_fields"].get(run["site"], [])
        now = {f: pre.get((run["id"], f), final.get((run["id"], f))) for f in fields}
        if not run["step"]:
            run["degraded"] = None
            continue
        # degraded: a field that this module extracted correctly from the
        # original page (t0) is wrong before healing at this step — the ground
        # truth for failure alarms. Fields already wrong at t0 are extraction
        # errors, not failures a monitor could detect, so they are left out.
        t0 = index.get((run["condition"], run["site"], run["rep"], 0))
        good_t0 = [f for f in fields if t0 and final.get((t0["id"], f)) is True]
        checked = [now.get(f) for f in good_t0 if now.get(f) is not None]
        run["degraded"] = (False in checked) if checked else None
        prev = index.get((run["condition"], run["site"], run["rep"], run["step"] - 1))
        if prev is None:
            continue
        before = {f: final.get((prev["id"], f)) for f in fields}
        labelled_before = [v for v in before.values() if v is not None]
        # prev_clean: every labelled field was correct when this step began, so
        # an alarm here can only be caused by this step's change (RQ1).
        run["prev_clean"] = bool(labelled_before) and all(labelled_before)
        broken = [f for f in fields if before.get(f) is True and now.get(f) is False]
        run["breaking"] = bool(broken)
        run["broken_fields"] = broken
        if broken:
            run["gt_recovered"] = all(final.get((run["id"], f)) is True for f in broken)
            run["fields_recovered"] = sum(1 for f in broken if final.get((run["id"], f)) is True)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _recs(data, *, phase=RecordPhase.FINAL, condition=None, step=None, rep_min=1,
          rep_eq=None, sites=None):
    out = []
    for r in data["records"]:
        if r["phase"] != phase:
            continue
        if condition is not None and r["condition"] != condition:
            continue
        if step is not None and r["step"] != step:
            continue
        if rep_eq is not None:
            if r["rep"] != rep_eq:
                continue
        elif rep_min is not None and (r["rep"] is None or r["rep"] < rep_min):
            continue
        if sites is not None and r["site"] not in sites:
            continue
        out.append(r)
    return out


def _runs(data, *, condition=None, step_min=None, rep_min=1, rep_eq=None):
    out = []
    for r in data["runs"].values():
        if condition is not None and r["condition"] != condition:
            continue
        if step_min is not None and (r["step"] or 0) < step_min:
            continue
        if rep_eq is not None:
            if r["rep"] != rep_eq:
                continue
        elif rep_min is not None and (r["rep"] is None or r["rep"] < rep_min):
            continue
        out.append(r)
    return out


def healing_section(data, rep_min=1, rep_eq=None, condition="self_healing") -> dict:
    runs = [r for r in _runs(data, condition=condition, step_min=1, rep_min=rep_min, rep_eq=rep_eq)
            if r["breaking"]]
    alarmed = [r for r in runs if r["detected"]]
    gt_ok = [r for r in alarmed if r["gt_recovered"]]
    sys_ok = [r for r in alarmed if r["healed"]]
    broken_fields = sum(len(r["broken_fields"]) for r in runs)
    recovered_fields = sum(r.get("fields_recovered", 0) for r in runs)
    by_cat = defaultdict(lambda: [0, 0])
    for r in alarmed:
        by_cat[r["category"]][0] += 1
        by_cat[r["category"]][1] += 1 if r["gt_recovered"] else 0
    return {
        "breaking_events": len(runs),
        "detected_breaking_events": len(alarmed),
        "gt_recovered_events": len(gt_ok),
        "system_reported_healed": len(sys_ok),
        "success_rate_gt": M.safe_div(len(gt_ok), len(alarmed)),
        "success_rate_gt_all_breaking": M.safe_div(len(gt_ok), len(runs)),
        "success_rate_system": M.safe_div(len(sys_ok), len(alarmed)),
        "broken_fields": broken_fields,
        "recovered_fields": recovered_fields,
        "field_recovery_rate": M.safe_div(recovered_fields, broken_fields),
        "by_category": {c: {"events": n, "gt_recovered": k, "rate": M.safe_div(k, n)}
                        for c, (n, k) in sorted(by_cat.items())},
    }


def _signals(runs) -> dict:
    signals = {
        "alarm (health < threshold)": [bool(r["detected"]) for r in runs],
        "xpath failure rate > 0": [r["pre_failure_rate"] > 0 for r in runs],
        "required-field mismatch > 0": [r["pre_field_mismatch"] > 0 for r in runs],
    }
    for t in THRESHOLDS:
        signals[f"structural distance >= {t:.2f}"] = [r["structural"] >= t for r in runs]
        signals[f"markup distance >= {t:.2f}"] = [r["markup"] >= t for r in runs]
    return signals


def detection_section(data, condition="self_healing", rep_min=1, rep_eq=None) -> list[dict]:
    """
    Two evaluations:
    * failure detection — all monitoring runs (step >= 1); positive = some
      field actually wrong before healing ("degraded"). Gives the proposal's
      false-positive failure-detection rate.
    * change detection (RQ1) — only runs that started from a fully correct
      state; positive = this step newly broke a field, or (design label) a
      structural change was applied.
    """
    monitored = [r for r in _runs(data, condition=condition, step_min=1, rep_min=rep_min, rep_eq=rep_eq)
                 if r["degraded"] is not None]
    clean = [r for r in monitored if r["prev_clean"]]
    rows = []
    for scope, runs, labels in (
        ("all monitoring runs", monitored,
         {"degraded (field wrong before healing)": [bool(r["degraded"]) for r in monitored]}),
        ("runs starting healthy", clean,
         {"breaking change (newly broken)": [bool(r["breaking"]) for r in clean],
          "structural change (design)": [bool(r["true_change"]) for r in clean]}),
    ):
        if not runs:
            continue
        signals = _signals(runs)
        for label_name, label in labels.items():
            for name, pred in signals.items():
                ev = M.binary_eval(pred, label)
                rows.append({"condition": condition, "scope": scope, "label": label_name,
                             "signal": name, **ev})
    return rows


def rq4_rows(data, rep_min=1, condition="self_healing") -> list[dict]:
    calls_by_run = defaultdict(list)
    for c in data["calls"]:
        if c["purpose"] == LLMPurpose.HEALING:
            calls_by_run[c["run_id"]].append(c)
    rows = []
    for e in data["events"]:
        run = data["runs"].get(e["run_id"])
        if not run or run["condition"] != condition or e["resolution"] == "not_attempted":
            continue
        if run["rep"] is None or run["rep"] < rep_min:
            continue
        calls = calls_by_run.get(run["id"], [])
        rows.append({
            "site": run["site"], "category": run["category"], "rep": run["rep"], "step": run["step"],
            "op": run["op"], "severity": run["severity"], "breaking": run["breaking"],
            "resolution": e["resolution"], "gt_recovered": run["gt_recovered"],
            "attempts": e["attempts"], "affected_fields": len(e["affected"]),
            "failure_history_len": max([c["history_len"] for c in calls] or [e["history_len"]]),
            "structural_distance": e["structural"], "markup_distance": e["markup"],
            "healing_latency_ms": sum(c["latency_ms"] + c["validation_ms"] for c in calls),
            "rate_limit_wait_ms": sum(c["wait_ms"] for c in calls),
            "ttr_excl_wait_ms": e["ttr_excl_ms"] if e["resolution"] == "auto_healed" else None,
            "ttr_ms": e["ttr_ms"] if e["resolution"] == "auto_healed" else None,
            "healing_tokens": sum(c["prompt_tokens"] + c["completion_tokens"] for c in calls),
        })
    return rows


def per_site(data, conds=("self_healing", "static")) -> list[dict]:
    rows = []
    for site, fields in data["site_fields"].items():
        row = {"site": site,
               "category": next((b["category"] for b in data["baselines"] if b["site"] == site), "")}
        for cond in conds:
            row[f"{cond}_initial_f1"] = M.aggregate(_recs(data, condition=cond, step=0, sites={site}))["f1"]
            row[f"{cond}_sustained_f1"] = M.aggregate(_recs(data, condition=cond, sites={site}))["f1"]
        row["initial_f1"] = row.get("self_healing_initial_f1")
        row.setdefault("self_healing_sustained_f1", None)
        row.setdefault("static_sustained_f1", None)
        rows.append(row)
    return rows


def present_conditions(data, ecfg) -> list[str]:
    ran = {r["condition"] for r in data["runs"].values() if r["rep"]}
    return [c for c in CONDITION_ORDER if c in ecfg["conditions"] and c in ran]


def complete_chains(data, cond, n_steps) -> set:
    """(site, rep) chains this condition completed from t0 to the last step."""
    steps = defaultdict(set)
    for r in data["runs"].values():
        if r["condition"] == cond and r["rep"]:
            steps[(r["site"], r["rep"])].add(r["step"])
    return {k for k, v in steps.items() if set(range(n_steps + 1)) <= v}


def common_chains(data, conds, n_steps) -> set | None:
    """Chains completed by every module, so modules are compared on the same
    pages; None when no module has a complete chain."""
    sets = [complete_chains(data, c, n_steps) for c in conds]
    sets = [s for s in sets if s]
    if not sets:
        return None
    out = sets[0]
    for s in sets[1:]:
        out &= s
    return out


def restrict(data, chains) -> dict:
    """A view of the data limited to the given (site, rep) chains."""
    if chains is None:
        return data
    keep = {rid for rid, r in data["runs"].items() if (r["site"], r["rep"]) in chains}
    out = dict(data)
    out["runs"] = {rid: r for rid, r in data["runs"].items() if rid in keep}
    out["records"] = [r for r in data["records"] if r["run_id"] in keep]
    out["events"] = [e for e in data["events"] if e["run_id"] in keep]
    out["calls"] = [c for c in data["calls"] if c["run_id"] in keep]
    return out


def module_rows(data, conds) -> list[dict]:
    """One row per module: accuracy, P/R/F1 at t0 and under change, failure
    detection, healing success, time-to-recovery and latency."""
    rows = []
    for c in conds:
        info = COND_INFO[c]
        recs = _recs(data, condition=c)
        init = M.aggregate([r for r in recs if r["step"] == 0])
        changed = M.aggregate([r for r in recs if r["step"] and r["step"] >= 1])
        overall = M.aggregate(recs)
        det = next((d for d in detection_section(data, condition=c)
                    if d["label"].startswith("degraded") and d["signal"].startswith("alarm")), None)
        heal = healing_section(data, condition=c) if info["heal"] else None
        events = [e for e in rq4_rows(data, condition=c) if e["resolution"] == "auto_healed"]
        heal_calls = [x for x in data["calls"]
                      if x["purpose"] == LLMPurpose.HEALING and data["runs"].get(x["run_id"], {}).get("condition") == c]
        monitoring = _runs(data, condition=c, step_min=1)
        degraded = [r for r in monitoring if r["degraded"] is not None]
        scored = sum(1 for r in recs if r["has_gt"])
        sites_run = len({r["site"] for r in _runs(data, condition=c)})
        rows.append({
            "condition": c, "module": info["label"], "sites": sites_run,
            "field_decisions": len(recs), "scored_decisions": scored,
            "initial_accuracy": init["field_accuracy"], "initial_precision": init["precision"],
            "initial_recall": init["recall"], "initial_f1": init["f1"],
            "changed_accuracy": changed["field_accuracy"], "changed_precision": changed["precision"],
            "changed_recall": changed["recall"], "changed_f1": changed["f1"],
            "overall_accuracy": overall["field_accuracy"], "overall_precision": overall["precision"],
            "overall_recall": overall["recall"], "overall_f1": overall["f1"],
            "token_f1_mean": overall["token_f1_mean"],
            "degraded_run_share": M.safe_div(sum(1 for r in degraded if r["degraded"]), len(degraded)),
            "alarm_false_positive_rate": det["fpr"] if det else None,
            "false_alarms_per_monitoring_run": det["false_alarm_share"] if det else None,
            "silent_failures": det["fn"] if det else None,
            "breaking_events": sum(1 for r in monitoring if r["breaking"]),
            "detected_breaking_events": sum(1 for r in monitoring if r["breaking"] and r["detected"]),
            "change_detection_recall": M.safe_div(sum(1 for r in monitoring if r["breaking"] and r["detected"]),
                                                  sum(1 for r in monitoring if r["breaking"])),
            "healed_events_gt": heal["gt_recovered_events"] if heal else None,
            "healing_success_rate": heal["success_rate_gt"] if heal else None,
            "healing_success_all_breaking": heal["success_rate_gt_all_breaking"] if heal else None,
            "field_recovery_rate": heal["field_recovery_rate"] if heal else None,
            "ttr_median_s": M.describe([e["ttr_excl_wait_ms"] / 1000 for e in events])["median"],
            "regeneration_latency_median_s": M.describe(
                [(x["latency_ms"] + x["validation_ms"]) / 1000 for x in heal_calls])["median"],
            "run_latency_median_ms": M.describe([r["duration_ms"] for r in monitoring
                                                 if r["duration_ms"] is not None])["median"],
            "generator_calls": sum(r["llm_calls"] for r in _runs(data, condition=c)),
            "llm_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in _runs(data, condition=c)),
        })
    return rows


def paired_tests(data, conds) -> list[dict]:
    """Per-site paired comparisons of sustained F1 (sites with both values)."""
    pairs = [("self_healing", "static"), ("rule_based", "rule_static"), ("self_healing", "rule_based"),
             ("rule_based", "static")]
    out = []
    sites = per_site(data, conds)
    for a, b in pairs:
        if a not in conds or b not in conds:
            continue
        diffs = [r[f"{a}_sustained_f1"] - r[f"{b}_sustained_f1"] for r in sites
                 if r.get(f"{a}_sustained_f1") is not None and r.get(f"{b}_sustained_f1") is not None]
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        out.append({"a": a, "b": b, "sites": len(diffs), "mean_diff": M.describe(diffs)["mean"],
                    "wins": wins, "losses": losses, "ties": len(diffs) - wins - losses,
                    "sign_test_p": M.sign_test(wins, losses), "wilcoxon_p": M.wilcoxon(diffs)})
    return out


def build(exp) -> dict:
    data = collect(exp)
    label_breaking(data)
    ecfg = exp.config["experiment"]
    reps = list(range(1, ecfg["repetitions"] + 1))

    module_conds = present_conditions(data, ecfg)
    chains = common_chains(data, module_conds, len(ecfg["protocol"]["steps"]))
    compared = restrict(data, chains)
    initial = M.aggregate(_recs(data, condition="self_healing", step=0))
    sustained = {c: M.aggregate(_recs(data, condition=c)) for c in ecfg["conditions"]}
    healing = healing_section(data)
    detection = [row for c in module_conds for row in detection_section(data, condition=c)]
    alarm_vs_degraded = next((d for d in detection if d["condition"] == "self_healing"
                              and d["label"].startswith("degraded")
                              and d["signal"].startswith("alarm")), None)
    ttr = [e for e in rq4_rows(data) if e["resolution"] == "auto_healed"]
    regen = [c for c in data["calls"]
             if c["purpose"] == LLMPurpose.HEALING and not c["fallback"]
             and data["runs"].get(c["run_id"], {}).get("rep", 0)
             and data["runs"].get(c["run_id"], {}).get("condition") == "self_healing"]
    ttr_s = M.describe([e["ttr_excl_wait_ms"] / 1000 for e in ttr])
    ttr_incl_s = M.describe([e["ttr_ms"] / 1000 for e in ttr])
    regen_s = M.describe([(c["latency_ms"] + c["validation_ms"]) / 1000 for c in regen])
    regen_wait_s = M.describe([c["wait_ms"] / 1000 for c in regen])

    # F1 by step (F1 over time) per condition
    steps = sorted({r["step"] for r in data["runs"].values() if r["rep"] and r["rep"] >= 1})
    by_step = []
    for s in steps:
        row = {"step": s}
        for c in ecfg["conditions"]:
            row[f"{c}_f1"] = M.aggregate(_recs(data, condition=c, step=s))["f1"]
        ops = sorted({r["op"] for r in data["runs"].values() if r["step"] == s and r["rep"]})
        row["operators"] = ", ".join(o for o in ops if o)
        by_step.append(row)

    sites = per_site(data, module_conds)
    paired = None
    if "static" in module_conds and "self_healing" in module_conds:
        diffs = [r["self_healing_sustained_f1"] - r["static_sustained_f1"] for r in sites
                 if r["self_healing_sustained_f1"] is not None and r["static_sustained_f1"] is not None]
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        paired = {"sites": len(diffs), "mean_diff": M.describe(diffs)["mean"],
                  "wins": wins, "losses": losses, "ties": len(diffs) - wins - losses,
                  "sign_test_p": M.sign_test(wins, losses), "wilcoxon_p": M.wilcoxon(diffs)}

    per_rep = []
    for rep in reps:
        row = {"rep": rep,
               "initial_f1": M.aggregate(_recs(data, condition="self_healing", step=0, rep_eq=rep))["f1"]}
        for c in ecfg["conditions"]:
            row[f"{c}_sustained_f1"] = M.aggregate(_recs(data, condition=c, rep_eq=rep))["f1"]
        row["healing_success_rate_gt"] = healing_section(data, rep_eq=rep)["success_rate_gt"]
        per_rep.append(row)
    rep_summary = {k: M.describe([r[k] for r in per_rep]) for k in (per_rep[0] if per_rep else {}) if k != "rep"}

    reductions = [b["reduction"] for b in data["baselines"] if b["reduction"] is not None]
    presence_total = presence_view = 0
    for b in data["baselines"]:
        for p in (b["presence"] or {}).values():
            if p.get("labelled_absent"):
                continue
            presence_total += 1
            presence_view += 1 if p.get("in_llm_view") else 0
    tokens = {
        "tokenizers": sorted({b["tokenizer"] for b in data["baselines"] if b["tokenizer"]}),
        "raw_tokens": M.describe([b["raw_tokens"] for b in data["baselines"]]),
        "llm_view_tokens": M.describe([b["llm_view_tokens"] for b in data["baselines"]]),
        "reduction": M.describe(reductions),
        "sites_at_or_above_90pct": sum(1 for r in reductions if r >= 0.9),
        "sites": len(reductions),
        "field_preservation_in_llm_view": M.safe_div(presence_view, presence_total),
        "views_truncated": sum(1 for b in data["baselines"] if b["view_truncated"]),
    }

    init_tokens = [r["prompt_tokens"] + r["completion_tokens"] for r in
                   _runs(data, condition="self_healing") if r["step"] == 0]
    heal_tokens = [e["healing_tokens"] for e in rq4_rows(data)]
    priced = [c["cost"] for c in data["calls"] if c["cost"] is not None]
    cost = {
        "tokens_per_initial_generation": M.describe(init_tokens),
        "tokens_per_healing_event": M.describe(heal_tokens),
        "total_prompt_tokens": sum(c["prompt_tokens"] for c in data["calls"]),
        "total_completion_tokens": sum(c["completion_tokens"] for c in data["calls"]),
        "usd_total": sum(priced) if priced else None,
        "usd_priced_calls": len(priced),
    }

    by_purpose = defaultdict(int)
    for c in data["calls"]:
        by_purpose[c["purpose"]] += 1
    reliability = {
        "llm_calls": len(data["calls"]),
        "calls_by_purpose": dict(by_purpose),
        "errors": sum(1 for c in data["calls"] if not c["success"]),
        "fallback_calls": sum(1 for c in data["calls"] if c["fallback"]),
        "retries": sum(c["retries"] for c in data["calls"]),
        "rate_limit_wait_s": sum(c["wait_ms"] for c in data["calls"]) / 1000,
        "truncated_inputs": sum(1 for c in data["calls"] if c["truncated"]),
        "models": sorted({c["model"] for c in data["calls"] if c["model"]}),
        "providers": sorted({c["provider"] for c in data["calls"]}),
    }

    rq4 = rq4_rows(data)
    rq4_corr = {
        var: M.spearman([e[var] for e in rq4], [e["healing_latency_ms"] for e in rq4])
        for var in ("structural_distance", "markup_distance", "affected_fields",
                    "failure_history_len", "attempts")
    }

    baselines_cmp = []
    conds = sorted({r["condition"] for r in data["runs"].values() if r["condition"].startswith("baseline:")})
    for cond in conds:
        snaps = {r["snapshot_id"] for r in data["runs"].values() if r["condition"] == cond}
        theirs = [r for r in data["records"] if r["condition"] == cond and r["phase"] == RecordPhase.FINAL]
        row = {"system": cond.split(":", 1)[1], "snapshots": len(snaps), **_f(M.aggregate(theirs))}
        baselines_cmp.append(row)
        for own in ecfg["conditions"]:
            mine = [r for r in data["records"] if r["condition"] == own and r["phase"] == RecordPhase.FINAL
                    and data["runs"][r["run_id"]]["snapshot_id"] in snaps]
            baselines_cmp.append({"system": f"{own} (same snapshots)", "snapshots": len(snaps),
                                  **_f(M.aggregate(mine))})

    natural = None
    if any(r["rep"] == 0 for r in data["runs"].values()):
        natural = {
            "sustained_f1": {c: M.aggregate(_recs(data, condition=c, rep_eq=0))["f1"]
                             for c in ecfg["conditions"]},
            "healing": healing_section(data, rep_min=None, rep_eq=0),
            "detection": detection_section(data, rep_min=None, rep_eq=0),
            "human_layout_changed": sum(1 for r in data["runs"].values()
                                        if r["rep"] == 0 and r["condition"] == "self_healing"
                                        and r["true_change"]),
        }

    # Where values were read from (structured metadata is untouched by the
    # simulated layout changes, so results are also reported separately).
    location = []
    for c in ecfg["conditions"]:
        for key, label in (("source_group", "annotated location"), ("xpath_target", "rule reads from")):
            groups = sorted({r[key] for r in _recs(data, condition=c)})
            for g in groups:
                recs = [r for r in _recs(data, condition=c) if r[key] == g]
                init = [r for r in recs if r["step"] == 0]
                location.append({"condition": c, "split": label, "group": g,
                                 "initial_f1": M.aggregate(init)["f1"] if init else None,
                                 **{f"sustained_{k}": v for k, v in _f(M.aggregate(recs)).items()
                                    if k in ("f1", "fields")}})
    excluded = [{"site": s["key"], "field": f["name"], "reason": f.get("exclude_reason", "")}
                for s in exp.config["sites"] for f in s.get("excluded_fields") or []]
    n_fields = sum(len(s["fields"]) for s in exp.config["sites"])
    render_modes = defaultdict(int)
    for b in data["baselines"]:
        render_modes[b.get("render") or "static"] += 1

    rq5_sites = [s for s in sites if s["self_healing_sustained_f1"] is not None]
    table51 = [
        _target_row("Initial extraction accuracy (F1)", initial["f1"], TARGETS["initial_f1"], ">="),
        _target_row("Self-healing success rate (GT-verified)", healing["success_rate_gt"],
                    TARGETS["healing_success_rate"], ">="),
        _target_row("Time-to-recovery, median s (excl. rate-limit waits)", ttr_s["median"],
                    TARGETS["time_to_recovery_s"], "<="),
        _target_row("XPath regeneration latency, median s", regen_s["median"],
                    TARGETS["regeneration_latency_s"], "<="),
        _target_row("Sustained F1 (self-healing, all steps)",
                    sustained.get("self_healing", {}).get("f1"), TARGETS["sustained_f1"], ">="),
        _target_row("False-positive failure detections (false alarms / monitoring events)",
                    alarm_vs_degraded["false_alarm_share"] if alarm_vs_degraded else None,
                    TARGETS["false_positive_rate"], "<="),
    ]
    silent = alarm_vs_degraded["fn"] if alarm_vs_degraded else None

    return {
        "experiment": exp.name,
        "config": exp.config["experiment"],
        "environment": exp.environment,
        "counts": {
            "sites": len(exp.config["sites"]),
            "fields": n_fields,
            "excluded_fields": len(excluded),
            "repetitions": ecfg["repetitions"],
            "repetitions_completed": len({r["rep"] for r in data["runs"].values() if r["rep"]}),
            "planned_repetitions": ecfg.get("planned_repetitions"),
            "render_modes": dict(render_modes),
            "baseline_labels_by_source": data.get("gt_sources", {}),
            "unscored_decisions": sum(1 for r in data["records"] if not r["has_gt"]
                                      and r["phase"] == RecordPhase.FINAL and (r["rep"] or 0) >= 1),
            "sites_evaluated": len({r["site"] for r in data["runs"].values() if r["rep"]}),
            "runs": len(data["runs"]),
            "records": len(data["records"]),
            "change_events": len(data["events"]),
        },
        "table_5_1": table51,
        "initial": _f(initial),
        "sustained": {c: _f(v) for c, v in sustained.items()},
        "healing": healing,
        # degraded runs with no alarm: wrong-but-non-empty values the health
        # monitor cannot see (reported, never hidden).
        "silent_failures": silent,
        "time_to_recovery_s": ttr_s,
        "time_to_recovery_incl_wait_s": ttr_incl_s,
        "regeneration_latency_s": regen_s,
        "regeneration_rate_limit_wait_s": regen_wait_s,
        "detection": detection,
        "f1_by_step": by_step,
        "per_site": sites,
        "rq5": {"sites": len(rq5_sites),
                "sites_sustained_f1_ge_0_80": sum(1 for s in rq5_sites if s["self_healing_sustained_f1"] >= 0.8)},
        "paired_comparison": paired,
        "conditions_run": module_conds,
        "comparison_chains": sorted(chains) if chains is not None else None,
        "module_repetitions": {c: sorted({r for _, r in complete_chains(data, c, len(ecfg["protocol"]["steps"]))})
                               for c in module_conds},
        "modules": module_rows(compared, module_conds),
        "paired_tests": paired_tests(compared, module_conds),
        "per_repetition": per_rep,
        "repetition_summary": rep_summary,
        "tokens": tokens,
        "cost": cost,
        "reliability": reliability,
        "rq4_correlations_spearman": rq4_corr,
        "rq4_events": rq4,
        "baselines": baselines_cmp,
        "natural": natural,
        "by_location": location,
        "excluded_fields": excluded,
        "_data": data,
    }


def _f(agg: dict) -> dict:
    return {k: agg.get(k) for k in ("precision", "recall", "f1", "tp", "fp", "fn",
                                    "fields", "field_accuracy", "token_f1_mean")}


def _target_row(name, value, target, op):
    met = None
    if value is not None:
        met = value >= target if op == ">=" else value <= target
    unit = "s" if "median s" in name else "ratio"
    return {"metric": name, "value": value, "target": f"{op} {target}", "met": met, "unit": unit}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _num(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                             for k, v in row.items()})


def write(exp, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rep = build(exp)
    data = rep.pop("_data")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False, default=str)

    _write_csv(tables / "table_5_1.csv", rep["table_5_1"])
    _write_csv(tables / "f1_by_step.csv", rep["f1_by_step"])
    _write_csv(tables / "per_site.csv", rep["per_site"])
    _write_csv(tables / "per_repetition.csv", rep["per_repetition"])
    _write_csv(tables / "detection_rq1.csv", rep["detection"])
    _write_csv(tables / "healing_events_rq4.csv", rep["rq4_events"])
    _write_csv(tables / "healing_by_category_rq2.csv",
               [{"category": c, **v} for c, v in rep["healing"]["by_category"].items()])
    _write_csv(tables / "token_reduction.csv", data["baselines"])
    _write_csv(tables / "baselines_o4.csv", rep["baselines"])
    _write_csv(tables / "runs_labelled.csv", list(data["runs"].values()))
    _write_csv(tables / "accuracy_by_location.csv", rep["by_location"])
    _write_csv(tables / "module_comparison.csv", rep["modules"])
    _write_csv(tables / "paired_tests.csv", rep["paired_tests"])
    _write_csv(tables / "excluded_fields.csv", rep["excluded_fields"])
    _write_csv(tables / "field_records.csv", data["records"])

    from .figures import write_figures

    rep["figures"] = write_figures(rep, data, out_dir / "figures")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False, default=str)
    (out_dir / "report.md").write_text(render_markdown(rep), encoding="utf-8")
    return rep


def render_markdown(rep: dict) -> str:
    L = []
    c = rep["counts"]
    cfg = rep["config"]
    env = rep["environment"]
    L.append(f"# Evaluation report — {rep['experiment']}\n")
    planned = c.get("planned_repetitions")
    L.append(f"{c['sites']} sites, {c.get('fields', 'n/a')} fields ({c.get('excluded_fields', 0)} more "
             f"excluded, listed below); repetitions configured {c['repetitions']}, with runs "
             f"{c.get('repetitions_completed', 'n/a')}"
             + (f" (study design: {planned} planned)" if planned else "")
             + f"; {c['runs']} runs, {c['records']} field records, {c['change_events']} change events.  ")
    if c.get("baseline_labels_by_source"):
        L.append("Baseline labels by source: " + ", ".join(
            f"{k} {v}" for k, v in sorted(c["baseline_labels_by_source"].items()))
            + f"; unlabelled field decisions (not scored): {c.get('unscored_decisions', 0)}.  ")
    if c.get("render_modes"):
        L.append("Snapshot render modes: " + ", ".join(f"{k} {v}" for k, v in sorted(c["render_modes"].items()))
                 + ".  ")
    L.append(f"LLM: {cfg['llm']['provider']} / {cfg['llm'].get('model') or 'default model'}; "
             f"threshold {cfg['health_threshold']}, max attempts {cfg['max_heal_attempts']}, "
             f"memory {cfg['failure_memory_size']}, seed {cfg['seed']}.  ")
    L.append(f"Environment: Python {env.get('python')}, {env.get('platform')}, git "
             f"{(env.get('git_commit') or 'n/a')[:10]}{' (dirty)' if env.get('git_dirty') else ''}, "
             f"tokenizer {env.get('tokenizer')}, prompt {env.get('prompt_version')}.\n")

    if rep.get("modules"):
        L.append("## Module comparison\n")
        ch = rep.get("comparison_chains")
        if ch is not None:
            L.append(f"Compared on the {len(ch)} site-repetition chains that every module completed "
                     f"({len({c[0] for c in ch})} sites, repetitions "
                     f"{', '.join(str(r) for r in sorted({c[1] for c in ch}))}).  ")
        L.append("All modules run on the same snapshots and the same seeded change sequences. "
                 "Accuracy = correct field decisions / labelled field decisions; P/R/F1 are "
                 "micro-averaged field decisions (lists: items). 't0' = original page, "
                 "'changed' = steps 1..K. FPR = false alarms / non-degraded monitoring runs.\n")
        L.append("| Module | Acc t0 | F1 t0 | Acc changed | P changed | R changed | F1 changed | "
                 "F1 overall | Alarm FPR | Healing success | TTR median s | Run latency median ms |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for m in rep["modules"]:
            L.append(f"| {m['module']} | {_pct(m['initial_accuracy'])} | {_pct(m['initial_f1'])} | "
                     f"{_pct(m['changed_accuracy'])} | {_pct(m['changed_precision'])} | "
                     f"{_pct(m['changed_recall'])} | {_pct(m['changed_f1'])} | {_pct(m['overall_f1'])} | "
                     f"{_pct(m['alarm_false_positive_rate'])} | {_pct(m['healing_success_rate'])} | "
                     f"{_num(m['ttr_median_s'])} | {_num(m['run_latency_median_ms'], 0)} |")
        if rep.get("paired_tests"):
            L.append("\nPaired per-site comparison of sustained F1:\n")
            L.append("| A vs B | Sites | Mean diff | Wins/Ties/Losses | Sign test p | Wilcoxon p |")
            L.append("|---|---|---|---|---|---|")
            for t in rep["paired_tests"]:
                L.append(f"| {t['a']} vs {t['b']} | {t['sites']} | {_pct(t['mean_diff'])} | "
                         f"{t['wins']}/{t['ties']}/{t['losses']} | {_num(t['sign_test_p'], 4)} | "
                         f"{_num(t['wilcoxon_p'], 4)} |")
        L.append("")

    L.append("## Table 5.1 — target metrics (LLM-based self-healing)\n")
    L.append("| Metric | Value | Target | Met |\n|---|---|---|---|")
    for row in rep["table_5_1"]:
        v = row["value"]
        shown = _num(v) if row["unit"] == "s" else _pct(v)
        met = "n/a" if row["met"] is None else ("yes" if row["met"] else "no")
        L.append(f"| {row['metric']} | {shown} | {row['target']} | {met} |")

    i = rep["initial"]
    L.append("\n## Extraction accuracy\n")
    L.append(f"Initial (t0): P {_pct(i['precision'])}, R {_pct(i['recall'])}, F1 {_pct(i['f1'])} "
             f"over {i['fields']} field decisions; token-F1 {_pct(i['token_f1_mean'])}.  ")
    for cond, s in rep["sustained"].items():
        L.append(f"Sustained, {cond}: P {_pct(s['precision'])}, R {_pct(s['recall'])}, "
                 f"F1 {_pct(s['f1'])} ({s['fields']} decisions).  ")
    L.append("\n| Step | " + " | ".join(f"{k}" for k in rep["f1_by_step"][0] if k != "step")
             + " |" if rep["f1_by_step"] else "")
    if rep["f1_by_step"]:
        keys = [k for k in rep["f1_by_step"][0] if k != "step"]
        L.append("|---|" + "---|" * len(keys))
        for row in rep["f1_by_step"]:
            L.append(f"| {row['step']} | " + " | ".join(
                row[k] if k == "operators" else _pct(row[k]) for k in keys) + " |")

    h = rep["healing"]
    L.append("\n## Self-healing (RQ2)\n")
    L.append(f"Breaking change events: {h['breaking_events']} ({h['detected_breaking_events']} detected).  ")
    L.append(f"GT-verified success: {_pct(h['success_rate_gt'])} of detected; "
             f"{_pct(h['success_rate_gt_all_breaking'])} of all breaking.  ")
    L.append(f"System-reported success: {_pct(h['success_rate_system'])}.  ")
    L.append(f"Field-level recovery: {h['recovered_fields']}/{h['broken_fields']} "
             f"({_pct(h['field_recovery_rate'])}).\n")
    if h["by_category"]:
        L.append("| Category | Events | GT-recovered | Rate |\n|---|---|---|---|")
        for cat, v in h["by_category"].items():
            L.append(f"| {cat} | {v['events']} | {v['gt_recovered']} | {_pct(v['rate'])} |")

    t, ti, g = rep["time_to_recovery_s"], rep["time_to_recovery_incl_wait_s"], rep["regeneration_latency_s"]
    L.append("\n## Latency (RQ4)\n")
    L.append(f"Time-to-recovery (excl. rate-limit waits): median {_num(t['median'])} s, "
             f"p95 {_num(t['p95'])} s, n={t['n']}. Including waits: median {_num(ti['median'])} s.  ")
    L.append(f"Regeneration latency per healing call: median {_num(g['median'])} s, "
             f"p95 {_num(g['p95'])} s, n={g['n']}.  ")
    L.append("Spearman correlation of healing latency with: " + ", ".join(
        f"{k} {_num(v)}" for k, v in rep["rq4_correlations_spearman"].items()) + ".\n")

    L.append("## Failure and change detection (RQ1)\n")
    L.append(f"Silent failures (a field wrong before healing but no alarm): {rep['silent_failures']}.\n")
    L.append("| Condition | Scope | Label | Signal | n | Precision | Recall | F1 | FPR | False alarms / events |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for d in rep["detection"]:
        L.append(f"| {d['condition']} | {d['scope']} | {d['label']} | {d['signal']} | {d['n']} | "
                 f"{_pct(d['precision'])} | {_pct(d['recall'])} | {_pct(d['f1'])} | {_pct(d['fpr'])} | "
                 f"{_pct(d['false_alarm_share'])} |")

    p = rep["paired_comparison"]
    L.append("\n## Self-healing vs. static (RQ3)\n")
    if p:
        L.append(f"Per-site sustained F1 difference: mean {_pct(p['mean_diff'])}; "
                 f"wins/ties/losses {p['wins']}/{p['ties']}/{p['losses']}; "
                 f"sign test p={_num(p['sign_test_p'], 4)}; Wilcoxon p={_num(p['wilcoxon_p'], 4)}.\n")
    conds = rep.get("conditions_run") or ["self_healing", "static"]
    L.append("| Site | Category | " + " | ".join(f"{c} t0 F1 | {c} sustained F1" for c in conds) + " |")
    L.append("|---|---|" + "---|---|" * len(conds))
    for s in rep["per_site"]:
        L.append(f"| {s['site']} | {s['category']} | " + " | ".join(
            f"{_pct(s.get(f'{c}_initial_f1'))} | {_pct(s.get(f'{c}_sustained_f1'))}" for c in conds) + " |")
    r5 = rep["rq5"]
    L.append(f"\nRQ5: {r5['sites_sustained_f1_ge_0_80']} of {r5['sites']} sites keep sustained F1 >= 80%.\n")

    if rep.get("by_location"):
        L.append("## Accuracy by value location\n")
        L.append("Structured metadata (meta tags, JSON-LD, canonical link) is not modified by the "
                 "simulated layout changes, so results are split by where the value lives.\n")
        L.append("| Condition | Split | Group | Initial F1 | Sustained F1 | Field decisions |\n|---|---|---|---|---|---|")
        for row in rep["by_location"]:
            L.append(f"| {row['condition']} | {row['split']} | {row['group']} | {_pct(row['initial_f1'])} | "
                     f"{_pct(row['sustained_f1'])} | {row['sustained_fields']} |")
        L.append("")

    L.append("## Repetitions\n")
    for k, v in rep["repetition_summary"].items():
        L.append(f"- {k}: mean {_pct(v['mean'])} +/- {_pct(v['std'])} (n={v['n']})")

    tk = rep["tokens"]
    L.append("\n## Pruning and tokens\n")
    L.append(f"Tokenizer(s): {', '.join(tk['tokenizers']) or 'n/a'}.  ")
    L.append(f"Raw page tokens median {_num(tk['raw_tokens']['median'], 0)}; LLM view median "
             f"{_num(tk['llm_view_tokens']['median'], 0)}; reduction mean {_pct(tk['reduction']['mean'])}, "
             f"median {_pct(tk['reduction']['median'])}, min {_pct(tk['reduction']['min'])}.  ")
    L.append(f"Sites with >= 90% reduction: {tk['sites_at_or_above_90pct']}/{tk['sites']}. "
             f"Labelled values visible in the LLM view: {_pct(tk['field_preservation_in_llm_view'])}. "
             f"Truncated views: {tk['views_truncated']}.\n")

    co, rl = rep["cost"], rep["reliability"]
    L.append("## Cost and reliability\n")
    L.append(f"Tokens per initial generation: median {_num(co['tokens_per_initial_generation']['median'], 0)}; "
             f"per healing event: median {_num(co['tokens_per_healing_event']['median'], 0)}. "
             f"Total {co['total_prompt_tokens']} prompt + {co['total_completion_tokens']} completion tokens"
             + (f"; USD {co['usd_total']:.4f}" if co["usd_total"] is not None else "; USD not priced") + ".  ")
    L.append(f"LLM calls {rl['llm_calls']} {rl['calls_by_purpose']}; errors {rl['errors']}; "
             f"fallback calls {rl['fallback_calls']}; retries {rl['retries']}; rate-limit wait "
             f"{_num(rl['rate_limit_wait_s'], 0)} s; truncated inputs {rl['truncated_inputs']}; "
             f"models {', '.join(rl['models']) or 'n/a'}.\n")

    if rep["baselines"]:
        L.append("## External baselines (O4)\n")
        L.append("| System | Snapshots | Precision | Recall | F1 |\n|---|---|---|---|---|")
        for b in rep["baselines"]:
            L.append(f"| {b['system']} | {b['snapshots']} | {_pct(b['precision'])} | "
                     f"{_pct(b['recall'])} | {_pct(b['f1'])} |")

    if rep["natural"]:
        n = rep["natural"]
        L.append("\n## Natural changes\n")
        L.append(f"Sustained F1: {', '.join(f'{k} {_pct(v)}' for k, v in n['sustained_f1'].items())}.  ")
        L.append(f"Breaking events {n['healing']['breaking_events']}, GT-verified healing "
                 f"{_pct(n['healing']['success_rate_gt'])}; human-labelled layout changes "
                 f"{n['human_layout_changed']}.\n")

    if rep.get("excluded_fields"):
        L.append("\n## Excluded fields\n")
        for e in rep["excluded_fields"]:
            L.append(f"- {e['site']}.{e['field']}: {e['reason']}")

    L.append("\n## Definitions\n")
    L.append("- Field decision: one field in one run. TP = extracted and equal to the label after "
             "normalisation; a wrong value counts as FP and FN; a missing value as FN; any value for "
             "a field labelled absent as FP. List fields use item-level multiset matching.")
    L.append("- Degraded run: a field the module extracted correctly from the original page (t0) "
             "is wrong before healing at this step (whenever it broke). Ground truth for failure "
             "alarms; fields already wrong at t0 are extraction errors and are not counted.")
    L.append("- Breaking change: a field correct at the previous step is wrong at this step before "
             "healing (per condition). Change detection (RQ1) is scored only on runs that started "
             "with every field correct, so persistent earlier failures cannot inflate it.")
    L.append("- Alarm: pre-heal health below the threshold. False alarm: alarm on a run that is not "
             "degraded. False-positive rate (Table 5.1) = false alarms / all monitoring runs (step >= 1).")
    L.append("- Silent failure: degraded run without an alarm (e.g. a wrong but non-empty value).")
    L.append("- GT-verified healing success: every field broken at a step is correct after healing.")
    L.append("- Latencies exclude time spent waiting on provider rate limits (reported separately).")
    return "\n".join(L) + "\n"
