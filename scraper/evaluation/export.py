"""
Export an experiment as a single zip for analysis / the paper:

  experiment.json         config + environment (versions, git commit, tokenizer)
  tables/*.csv            every raw table (runs, records, events, LLM calls, ...)
  results/                eval_report output (summary.json, report.md, tables)
  snapshots/              baseline pages + labeling views (or all, or none)
  database/db.sqlite3     a consistent copy of the whole database

Secrets are never exported (.env is not touched; API keys are not stored).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from django.conf import settings
from django.db.models import F

from ..models import (
    ChangeEvent, ExtractedRecord, FailureRecord, GroundTruth, LLMCallLog,
    ScrapeRun, ScrapeTarget, Snapshot, SnapshotKind, TargetField, XPathRule,
)
from ..services import snapshots as snapstore
from . import report as report_mod


def _values(qs, extra=None):
    names = [f.attname for f in qs.model._meta.concrete_fields]
    return list(qs.values(*names, **(extra or {})))


def _dump(path: Path, rows: list[dict]) -> None:
    report_mod._write_csv(path, rows)


def export(exp, out_zip: Path, snapshots: str = "baseline") -> Path:
    work = snapstore.experiment_dir(exp) / "_export"
    if work.exists():
        shutil.rmtree(work)
    tables = work / "tables"
    tables.mkdir(parents=True)

    with open(work / "experiment.json", "w", encoding="utf-8") as fh:
        json.dump({"name": exp.name, "created_at": exp.created_at.isoformat(), "notes": exp.notes,
                   "config": exp.config, "environment": exp.environment},
                  fh, indent=2, ensure_ascii=False)

    targets = ScrapeTarget.objects.filter(experiment=exp)
    _dump(tables / "targets.csv", _values(targets))
    _dump(tables / "fields.csv", _values(TargetField.objects.filter(target__experiment=exp),
                                         {"site_key": F("target__site_key"),
                                          "condition": F("target__condition")}))
    _dump(tables / "rules.csv", _values(XPathRule.objects.filter(field__target__experiment=exp),
                                        {"field_name": F("field__name"),
                                         "site_key": F("field__target__site_key"),
                                         "condition": F("field__target__condition"),
                                         "repetition": F("field__target__repetition")}))
    _dump(tables / "snapshots.csv", _values(Snapshot.objects.filter(experiment=exp)))
    _dump(tables / "ground_truth.csv", _values(GroundTruth.objects.filter(snapshot__experiment=exp),
                                               {"site_key": F("snapshot__site_key")}))
    run_extra = {"site_key": F("target__site_key")}
    _dump(tables / "runs.csv", [
        {k: v for k, v in row.items() if k not in ("dom_profile", "dom_skeleton")}
        for row in _values(ScrapeRun.objects.filter(experiment=exp), run_extra)
    ])
    _dump(tables / "records.csv", _values(ExtractedRecord.objects.filter(run__experiment=exp),
                                          {"field_name": F("field__name"),
                                           "site_key": F("run__target__site_key"),
                                           "condition": F("run__condition"),
                                           "repetition": F("run__repetition"),
                                           "step": F("run__step")}))
    _dump(tables / "change_events.csv", _values(ChangeEvent.objects.filter(run__experiment=exp),
                                                {"site_key": F("target__site_key")}))
    _dump(tables / "failures.csv", _values(FailureRecord.objects.filter(run__experiment=exp),
                                           {"field_name": F("field__name")}))
    _dump(tables / "llm_calls.csv", _values(LLMCallLog.objects.filter(run__experiment=exp),
                                            {"site_key": F("run__target__site_key"),
                                             "condition": F("run__condition"),
                                             "step": F("run__step")}))

    rep = report_mod.write(exp, work / "results")
    from .paper import results_latex, results_markdown
    from .verify import verify

    verify(exp, work / "results")
    rep_json = json.loads((work / "results" / "summary.json").read_text(encoding="utf-8"))
    (work / "results" / "paper_results.md").write_text(results_markdown(rep_json), encoding="utf-8")
    (work / "results" / "paper_table_modules.tex").write_text(results_latex(rep_json), encoding="utf-8")
    from . import sensitivity as sens

    srep = sens.build(exp)   # post hoc sensitivity analysis, next to the primary results
    (work / "results" / "sensitivity.json").write_text(json.dumps(srep, indent=2, default=str), encoding="utf-8")
    (work / "results" / "sensitivity.md").write_text(sens.markdown(srep), encoding="utf-8")
    del rep
    # audit of the saved pages against the annotated dataset, if one was made
    audit_dir = snapstore.experiment_dir(exp) / "audit"
    if audit_dir.exists():
        shutil.copytree(audit_dir, work / "audit")

    if snapshots in ("baseline", "all"):
        qs = Snapshot.objects.filter(experiment=exp)
        if snapshots == "baseline":
            qs = qs.filter(kind__in=[SnapshotKind.BASELINE, SnapshotKind.NATURAL])
        for snap in qs:
            src = snapstore.data_dir() / snap.path
            if src.exists():
                # snap.path = "<experiment>/snapshots/<site>/<file>" -> "snapshots/<site>/<file>"
                dst = work / Path(*Path(snap.path).parts[1:])
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                view = snapstore.label_view_path(snap)
                if view.exists():
                    shutil.copy2(view, dst.with_name(view.name))

    db_path = Path(settings.DATABASES["default"]["NAME"])
    if db_path.exists():
        (work / "database").mkdir()
        src = sqlite3.connect(str(db_path))
        dst = sqlite3.connect(str(work / "database" / "db.sqlite3"))
        with dst:
            src.backup(dst)
        src.close()
        dst.close()

    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(str(out_zip.with_suffix("")), "zip", root_dir=work)
    shutil.rmtree(work)
    return Path(archive)
