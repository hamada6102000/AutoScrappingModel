"""
Import predictions from external baselines (proposal O4: AutoScraper, AXE,
HtmlRAG, WebAgents, ...) so they are scored with exactly the same ground truth
and matching rules as the proposed system, on byte-identical snapshots.

Input CSV columns: snapshot_id, field, value
  (list fields: items separated by || or newlines; empty value = no output)
Optional columns: latency_ms, tokens
"""

from __future__ import annotations

import csv
from pathlib import Path

from django.db import transaction
from django.utils import timezone

from ..models import (
    ExtractedRecord, RecordPhase, RunStatus, ScrapeRun, ScrapeTarget, Snapshot,
    SnapshotKind, StepKind, TargetField,
)
from ..services import snapshots as snapstore
from ..services.extractor import FieldOutcome, type_ok
from ..services.normalizer import normalize
from ..services.pipeline import ScrapePipeline
from .config import site_by_key


def _target(exp, site: dict, name: str) -> ScrapeTarget:
    target, _ = ScrapeTarget.objects.get_or_create(
        experiment=exp, site_key=site["key"], condition=f"baseline:{name}", repetition=None,
        defaults={"name": f"{site['name']} [baseline:{name}]", "url": site["url"] or "http://local.invalid/",
                  "category": site["category"], "is_active": False},
    )
    for order, f in enumerate(site["fields"], start=1):
        TargetField.objects.get_or_create(
            target=target, name=f["name"],
            defaults={"field_type": f["type"], "cardinality": f["cardinality"],
                      "is_required": f["required"], "description": f["description"], "order": order},
        )
    return target


def _step_kind(snap) -> str:
    if snap.kind == SnapshotKind.BASELINE:
        return StepKind.BASELINE
    if snap.kind == SnapshotKind.NATURAL:
        return StepKind.NATURAL
    fam = (snap.mutation or {}).get("family")
    return {"content": StepKind.CONTENT, "control": StepKind.CONTROL}.get(fam, StepKind.STRUCTURAL)


@transaction.atomic
def import_predictions(exp, name: str, path: Path) -> dict:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_snapshot: dict[int, list[dict]] = {}
    errors = []
    for i, row in enumerate(rows, start=2):
        try:
            by_snapshot.setdefault(int(float(row["snapshot_id"])), []).append(row)
        except (KeyError, ValueError):
            errors.append(f"row {i}: missing/invalid snapshot_id")
    created = 0
    for sid, items in by_snapshot.items():
        snap = Snapshot.objects.filter(pk=sid, experiment=exp).first()
        if snap is None:
            errors.append(f"snapshot {sid} not in experiment {exp.name}")
            continue
        site = site_by_key(exp.config, snap.site_key)
        target = _target(exp, site, name)
        step = snap.step or 0
        ScrapeRun.objects.filter(target=target, snapshot=snap).delete()
        run = ScrapeRun.objects.create(
            target=target, experiment=exp, snapshot=snap, condition=f"baseline:{name}",
            repetition=snap.repetition, step=step, step_kind=_step_kind(snap),
            status=RunStatus.SUCCESS, completed=True, heal_enabled=False, provider=name,
            finished_at=timezone.now(),
            duration_ms=sum(int(float(r.get("latency_ms") or 0)) for r in items),
            prompt_tokens=sum(int(float(r.get("tokens") or 0)) for r in items),
        )
        gt = snapstore.gt_map(snap)
        values = {r.get("field", "").strip(): (r.get("value") or "") for r in items}
        for f in target.fields.all():
            raw = values.get(f.name, "").strip()
            if f.cardinality == "many":
                items_ = [p.strip() for p in raw.replace("||", "\n").splitlines() if p.strip()]
                ok = bool(items_)
                out = FieldOutcome(f.name, "", ok, raw_value=" || ".join(items_),
                                   values=[normalize(v, f.field_type) for v in items_],
                                   cardinality="many")
            else:
                ok = type_ok(raw, f.field_type)
                out = FieldOutcome(f.name, "", ok, raw_value=raw)
            rec = ExtractedRecord(run=run, field=f, phase=RecordPhase.FINAL, raw_value=raw,
                                  normalized_value=normalize(raw, f.field_type), values=out.values,
                                  success=ok, reason="" if ok else "no-output")
            g = gt.get(f.name)
            if g is not None:
                ScrapePipeline._score(rec, out, f, g.values, g.value)
            rec.save()
        created += 1
    return {"runs": created, "errors": errors}
