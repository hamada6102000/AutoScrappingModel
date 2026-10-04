"""
Independent verification of the reported numbers (``eval_verify``).

1. Re-scores every stored field record from its raw extracted value and the
   ground truth (fresh matching, not the stored flags) and counts mismatches.
2. Re-aggregates precision / recall / F1 / accuracy per module straight from
   the database with a separate implementation of the counting rules and
   compares them with the report (each module on the site-repetition chains
   it completed), and checks that no module has a result for a site it was
   not executed on, that no LLM-based run exists outside the recorded LLM
   sites, and that every replicated LLM value equals the measured value and
   is labelled as replicated.
3. Builds the report twice and checks the numbers are identical (determinism).
4. Checks every snapshot file against its stored SHA-256.
5. Re-applies every simulated change from its parent page with the stored
   operator, seed, severity and focus, and checks the result is byte-identical
   (the change sequence is reproducible).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..models import ExtractedRecord, GroundTruth, RecordPhase, ScrapeRun, Snapshot, SnapshotKind
from ..services import snapshots as snapstore
from ..services.matching import canonical, exact_match
from ..services.mutations import apply_mutation
from .config import COND_INFO, CONDITION_ORDER

TOL = 1e-9


def _gt_index(exp) -> dict:
    out = {}
    for g in GroundTruth.objects.filter(snapshot__experiment=exp):
        out[(g.snapshot_id, g.field_name)] = g
    return out


def _decision(rec, gt, list_threshold):
    """(tp, fp, fn, correct) for one record, or None when not scored."""
    if gt is None or (gt.notes or "").startswith("exclude"):
        return None
    absent = gt.notes == "absent"
    ftype = rec.field.field_type
    if rec.field.cardinality == "many":
        preds = [str(v) for v in (rec.values or [])] if rec.success else []
        if absent:
            fp = len(preds)
            return 0, fp, 0, fp == 0
        p = Counter(canonical(v, ftype) for v in preds if v.strip())
        g = Counter(canonical(v, ftype) for v in gt.values if v.strip())
        tp = sum((p & g).values())
        fp, fn = sum(p.values()) - tp, sum(g.values()) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0
        return tp, fp, fn, f1 >= list_threshold
    if absent:
        return (0, 1, 0, False) if rec.success else (0, 0, 0, True)
    if not rec.success:
        return 0, 0, 1, False
    ok = any(exact_match(rec.raw_value, v, ftype) for v in gt.values)
    return (1, 0, 0, True) if ok else (0, 1, 1, False)


def verify(exp, results_dir: Path | None = None) -> dict:
    from . import report as report_mod

    out = {"checks": []}

    def check(name, ok, detail=""):
        out["checks"].append({"check": name, "ok": bool(ok), "detail": detail})

    thr = exp.config["experiment"].get("list_correct_f1", 0.9)
    gts = _gt_index(exp)
    recs = (ExtractedRecord.objects.filter(run__experiment=exp, run__completed=True,
                                           phase=RecordPhase.FINAL, run__repetition__gte=1)
            .select_related("field", "run", "run__target"))
    rep = report_mod.build(exp)
    rep.pop("_data", None)
    # Each module is reported on the chains it completed (its own scope).
    own = {cond: {(c[0], c[1]) for c in chains}
           for cond, chains in (rep.get("module_chains") or {}).items()}

    def compared(run) -> bool:
        return (run.target.site_key, run.repetition) in own.get(run.condition, set())

    # 1 + 2: re-score and re-aggregate
    stored_mismatch = 0
    totals = defaultdict(lambda: [0, 0, 0, 0, 0])   # tp fp fn correct scored
    n = 0
    for rec in recs:
        gt = gts.get((rec.run.snapshot_id, rec.field.name))
        d = _decision(rec, gt, thr)
        if d is None:
            continue
        n += 1
        tp, fp, fn, correct = d
        if compared(rec.run):
            t = totals[rec.run.condition]
            t[0] += tp; t[1] += fp; t[2] += fn; t[3] += int(correct); t[4] += 1  # noqa: E702
        if rec.field.cardinality == "many":
            if not (gt.notes == "absent") and (rec.list_tp, rec.list_fp, rec.list_fn) != (tp, fp, fn):
                stored_mismatch += 1
        elif gt.notes != "absent" and bool(rec.exact_match) != bool(tp):
            stored_mismatch += 1
    check("stored field scores equal a fresh re-scoring", stored_mismatch == 0,
          f"{n} scored field records, {stored_mismatch} mismatches")

    mods = {m["condition"]: m for m in rep.get("modules") or []}
    agg = {}
    for cond in CONDITION_ORDER:
        if cond not in totals:
            continue
        tp, fp, fn, correct, scored = totals[cond]
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f1 = (2 * p * r / (p + r) if p + r else 0.0) if p is not None and r is not None else None
        acc = correct / scored if scored else None
        agg[cond] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1,
                     "accuracy": acc, "scored": scored}
        m = mods.get(cond) or {}
        same = all(_close(a, b) for a, b in ((p, m.get("overall_precision")), (r, m.get("overall_recall")),
                                             (f1, m.get("overall_f1")), (acc, m.get("overall_accuracy"))))
        check(f"independent P/R/F1/accuracy for {cond} equal the report", same,
              f"recomputed P={_f(p)} R={_f(r)} F1={_f(f1)} Acc={_f(acc)}; report "
              f"P={_f(m.get('overall_precision'))} R={_f(m.get('overall_recall'))} "
              f"F1={_f(m.get('overall_f1'))} Acc={_f(m.get('overall_accuracy'))}")
    out["recomputed"] = agg

    # 2a: a module's reported scope is exactly where it has completed runs
    for m in rep.get("modules") or []:
        cond = m["condition"]
        ran = set(ScrapeRun.objects.filter(experiment=exp, condition=cond, completed=True,
                                           repetition__gte=1).values_list("target__site_key", flat=True))
        claimed = set(m.get("site_keys") or [])
        check(f"{cond} is reported only for sites it was executed on", claimed <= ran,
              f"reported for {len(claimed)} site(s); executed on {len(ran)}"
              + (f"; not executed: {sorted(claimed - ran)}" if claimed - ran else ""))

    # 2c: the recorded LLM scope and the labelled replication of LLM values
    scope = (rep.get("llm_scope") or {}).get("decision")
    if scope:
        llm_conds = [c for c in CONDITION_ORDER if COND_INFO[c]["generator"] == "llm"]
        outside = (ScrapeRun.objects.filter(experiment=exp, condition__in=llm_conds)
                   .exclude(target__site_key__in=scope["llm_sites"]).count())
        check("no LLM-based run exists outside the recorded LLM sites", outside == 0,
              f"LLM sites {', '.join(scope['llm_sites'])}; runs elsewhere: {outside}")
    repl = rep.get("llm_replication")
    if repl:
        measured = {m["condition"]: m for m in rep.get("modules") or []}
        bad = []
        for row in repl["per_site"]:
            for cond in measured:
                origin = row.get(f"{cond}_origin", "")
                if not origin.startswith("replicated"):
                    continue
                if COND_INFO[cond]["generator"] != "llm" or row["site"] in scope["llm_sites"]:
                    bad.append(f"{row['site']}/{cond}: replicated where it must not be")
                    continue
                for k in ("initial_f1", "changed_f1", "overall_f1", "changed_precision", "changed_recall"):
                    if not _close(row.get(f"{cond}_{k}"), measured[cond].get(k)):
                        bad.append(f"{row['site']}/{cond}/{k}")
        for m in repl["modules"]:
            src = measured.get(m["condition"]) or {}
            for k in ("initial_f1", "changed_f1", "overall_f1", "changed_precision", "changed_recall"):
                if not _close(m.get(k), src.get(k)):
                    bad.append(f"module {m['condition']}/{k}")
        n_rows = sum(1 for r in repl["per_site"] for c in measured
                     if r.get(f"{c}_origin", "").startswith("replicated"))
        check("replicated LLM values equal the measured LLM values, only for LLM modules on non-LLM sites, "
              "every one labelled 'replicated'", not bad,
              f"{n_rows} replicated module-site values on {len(repl['replicated_sites'])} sites from "
              f"{', '.join(repl['source_sites'])}" + (f"; problems: {bad[:5]}" if bad else ""))

    # 2b: breaking changes, detection and GT-verified healing, recounted
    correct = {}   # (run_id, field, phase) -> bool
    runs = {}
    for rec in (ExtractedRecord.objects.filter(run__experiment=exp, run__completed=True,
                                               run__repetition__gte=1)
                .select_related("field", "run", "run__target")):
        gt = gts.get((rec.run.snapshot_id, rec.field.name))
        d = _decision(rec, gt, thr)
        if d is None:
            continue
        if not compared(rec.run):
            continue
        correct[(rec.run_id, rec.field.name, rec.phase)] = d[3]
        runs[rec.run_id] = rec.run
    chains = defaultdict(dict)
    for run in runs.values():
        chains[(run.condition, run.target.site_key, run.repetition)][run.step] = run
    counts = defaultdict(lambda: [0, 0, 0])   # breaking, detected, healed (GT)
    for (cond, _site, _rep), steps in chains.items():
        for k, run in steps.items():
            prev = steps.get(k - 1)
            if not k or prev is None:
                continue
            fields = {f for (rid, f, _ph) in correct if rid in (run.id, prev.id)}
            broken = [f for f in fields
                      if correct.get((prev.id, f, RecordPhase.FINAL)) is True
                      and correct.get((run.id, f, RecordPhase.PRE_HEAL),
                                      correct.get((run.id, f, RecordPhase.FINAL))) is False]
            if not broken:
                continue
            c = counts[cond]
            c[0] += 1
            if run.detected:
                c[1] += 1
                if all(correct.get((run.id, f, RecordPhase.FINAL)) is True for f in broken):
                    c[2] += 1
    for cond, (brk, det, healed) in counts.items():
        m = mods.get(cond) or {}
        same = (brk == m.get("breaking_events") and det == m.get("detected_breaking_events")
                and (m.get("healed_events_gt") is None or healed == m.get("healed_events_gt")))
        check(f"breaking / detected / healed counts for {cond} equal the report", same,
              f"recounted {brk}/{det}/{healed}; report {m.get('breaking_events')}/"
              f"{m.get('detected_breaking_events')}/{m.get('healed_events_gt')}")

    # 3: determinism of the report
    rep2 = report_mod.build(exp)
    rep2.pop("_data", None)
    a = json.dumps(rep, sort_keys=True, default=str)
    b = json.dumps(rep2, sort_keys=True, default=str)
    check("report is deterministic (built twice, identical)", a == b)

    # 4: snapshot integrity
    bad = []
    snaps = list(Snapshot.objects.filter(experiment=exp))
    for snap in snaps:
        path = snapstore.data_dir() / snap.path
        if not path.exists():
            bad.append(f"{snap.path} missing")
            continue
        digest = hashlib.sha256(snapstore.read_html(snap).encode("utf-8", "ignore")).hexdigest()
        if digest != snap.sha256:
            bad.append(f"{snap.path} changed")
    check("every snapshot file matches its stored SHA-256", not bad,
          f"{len(snaps)} snapshots" + (f"; problems: {bad[:5]}" if bad else ""))

    # 5: simulated changes are reproducible
    sims = Snapshot.objects.filter(experiment=exp, kind=SnapshotKind.SIMULATED).select_related("parent")
    types = {(s["key"], f["name"]): f["type"] for s in exp.config["sites"] for f in s["fields"]}
    diffs, total = [], 0
    for snap in sims:
        spec = snap.mutation or {}
        parent = snap.parent
        if parent is None or "seed" not in spec:
            continue
        total += 1
        gt_fields = {g.field_name: (g.values, types.get((snap.site_key, g.field_name), "string"))
                     for g in parent.ground_truth.all()}
        res = apply_mutation(snapstore.read_html(parent), spec["op"], spec["seed"],
                             float(spec.get("severity", 0.5)), bool(spec.get("focus")), gt_fields)
        if hashlib.sha256(res.html.encode("utf-8", "ignore")).hexdigest() != snap.sha256:
            diffs.append(snap.path)
    check("every simulated page is reproduced byte-for-byte from its parent and seed", not diffs,
          f"{total} simulated pages" + (f"; differing: {diffs[:5]}" if diffs else ""))

    out["ok"] = all(c["ok"] for c in out["checks"])
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "verification.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        lines = ["# Verification of the reported numbers", "",
                 f"Overall: {'ALL CHECKS PASSED' if out['ok'] else 'SOME CHECKS FAILED'}", ""]
        for c in out["checks"]:
            lines.append(f"- [{'x' if c['ok'] else ' '}] {c['check']}: {c['detail']}")
        (results_dir / "verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= TOL


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.6f}"
