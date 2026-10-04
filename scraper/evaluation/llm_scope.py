"""
The recorded LLM-scope decision of an experiment (``eval_llm_scope``).

Stored as eval_data/<exp>/llm_scope.json:

    {"llm_sites": ["site_001", "site_002"],     # the only sites the LLM modules run on
     "replicate_to_all_sites": true,             # report: copy their values to the other sites
     "reason": "...", "recorded_at": "..."}

The runner never executes an LLM-based module on a site outside
``llm_sites`` (no LLM call is made for those sites). When
``replicate_to_all_sites`` is set, the report adds, for every other site, the
LLM-module values measured on ``llm_sites`` (pooled over those sites), each
such value labelled "replicated"; replicated values are never written to the
database and never enter a statistical test.
"""

from __future__ import annotations

import datetime as dt
import json

from ..services import snapshots as snapstore

FILE = "llm_scope.json"


def path(exp):
    return snapstore.experiment_dir(exp) / FILE


def load(exp) -> dict | None:
    p = path(exp)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    data["llm_sites"] = sorted(data.get("llm_sites") or [])
    data["replicate_to_all_sites"] = bool(data.get("replicate_to_all_sites"))
    return data


def save(exp, llm_sites: list[str], replicate: bool, reason: str) -> dict:
    known = {s["key"] for s in exp.config["sites"]}
    unknown = [s for s in llm_sites if s not in known]
    if unknown:
        raise ValueError(f"unknown site(s): {', '.join(unknown)}")
    if not llm_sites:
        raise ValueError("at least one LLM site is needed")
    data = {"llm_sites": sorted(llm_sites), "replicate_to_all_sites": bool(replicate),
            "reason": reason.strip(), "recorded_at": dt.datetime.now().isoformat(timespec="seconds")}
    p = path(exp)
    p.parent.mkdir(parents=True, exist_ok=True)
    old = load(exp)
    if old and {k: old.get(k) for k in ("llm_sites", "replicate_to_all_sites", "reason")} == \
            {k: data[k] for k in ("llm_sites", "replicate_to_all_sites", "reason")}:
        return old                                  # unchanged: keep the original timestamp
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def clear(exp) -> bool:
    p = path(exp)
    if p.exists():
        p.unlink()
        return True
    return False
