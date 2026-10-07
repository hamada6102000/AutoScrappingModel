"""
Evaluation configuration: load, validate and resolve the sites file
(``eval/sites.yaml``). Every problem is reported at once, before anything runs.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from ..services.mutations import ALL_OPS

CATEGORIES = {"ecommerce", "news", "finance", "jobs", "academic", "other"}
FIELD_TYPES = {"string", "number", "price", "date", "url", "mixed", "boolean"}
RENDER_MODES = {"static", "js", "auto"}
CARDINALITIES = {"single", "many"}
# The modules compared in the evaluation. "copy_of": the condition starts from
# an exact copy of another condition's t0 selectors (paired design).
COND_INFO = {
    "static": {"label": "Static baseline (LLM-generated selectors)", "short": "Static (LLM sel.)",
               "generator": "llm", "heal": False, "copy_of": "self_healing"},
    "rule_static": {"label": "Static baseline (rule-generated selectors)", "short": "Static (rule sel.)",
                    "generator": "rule_based", "heal": False, "copy_of": "rule_based"},
    "rule_based": {"label": "Rule-based self-healing", "short": "Rule-based SH",
                   "generator": "rule_based", "heal": True, "copy_of": None},
    "self_healing": {"label": "LLM-based self-healing", "short": "LLM-based SH",
                     "generator": "llm", "heal": True, "copy_of": None},
}
CONDITION_ORDER = ["static", "rule_static", "rule_based", "self_healing"]
CONDITIONS = set(COND_INFO)
OFFLINE_CONDITIONS = {"rule_based", "rule_static"}   # never call an LLM

DEFAULT_STEPS = [
    {"op": "rename_classes", "focus": True},
    {"op": "insert_banner", "focus": False},
    {"op": "wrap_elements", "focus": True},
    {"op": "perturb_numbers"},
    {"op": "retag_elements", "focus": True},
    {"op": "control"},
    {"op": "redesign", "focus": True},
    {"op": "insert_sibling", "focus": True},
]

DEFAULTS = {
    "name": None,
    "description": "",
    "llm": {"provider": "groq", "model": None},
    "health_threshold": 0.80,
    "max_heal_attempts": 3,
    "failure_memory_size": 10,
    "repetitions": 3,
    # Repetitions in the study design, if more than were run (reported as planned).
    "planned_repetitions": None,
    "seed": 20261004,
    # How an ambiguous date like 10/12/2026 is read: dmy or mdy.
    "date_order": "dmy",
    "conditions": ["self_healing", "static"],
    "protocol": {
        "steps": DEFAULT_STEPS,
        "severities": [0.3, 0.6, 1.0],
        "shuffle": True,
    },
    # render: static (HTTP), js (Playwright) or auto (both; keep the version
    # that contains more of the annotated values — see protocol.capture_snapshot).
    "fetch": {"user_agent": None, "timeout": 30, "respect_robots": True, "render": "static"},
    # A list field counts as "correct" when its item-level F1 reaches this.
    "list_correct_f1": 0.9,
}

_KEY = re.compile(r"^[a-z0-9][a-z0-9_-]{0,59}$")


class ConfigError(Exception):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Invalid evaluation config:\n  - " + "\n  - ".join(problems))


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError([f"config file not found: {path}"])
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    except Exception as exc:
        raise ConfigError([f"cannot parse {path.name}: {exc}"]) from exc
    return resolve(data or {}, base_dir=path.parent)


def resolve(data: dict, base_dir: Path | None = None) -> dict:
    problems: list[str] = []
    base_dir = Path(base_dir or ".").resolve()
    exp = _merge(DEFAULTS, data.get("experiment") or {})

    if not exp.get("name") or not isinstance(exp["name"], str):
        problems.append("experiment.name is required (e.g. ieee-2026)")
    llm = exp.get("llm") or {}
    if llm.get("provider") not in {"heuristic", "groq", "anthropic", "openai", "ollama"}:
        problems.append(f"experiment.llm.provider {llm.get('provider')!r} is not supported")
    if llm.get("max_input_chars") is not None and (
            not isinstance(llm["max_input_chars"], int) or llm["max_input_chars"] < 2000):
        problems.append("experiment.llm.max_input_chars must be an integer >= 2000")
    if llm.get("reasoning_effort") not in (None, "low", "medium", "high", "xhigh", "max"):
        problems.append("experiment.llm.reasoning_effort must be low, medium, high, xhigh or max")
    if llm.get("tokens_per_minute") is not None and (
            not isinstance(llm["tokens_per_minute"], int) or llm["tokens_per_minute"] < 1000):
        problems.append("experiment.llm.tokens_per_minute must be an integer >= 1000")
    for key in ("include_reasoning", "completion_counts_in_tpm"):
        if llm.get(key) is not None and not isinstance(llm[key], bool):
            problems.append(f"experiment.llm.{key} must be true or false")
    if exp.get("date_order") not in ("dmy", "mdy"):
        problems.append("experiment.date_order must be dmy or mdy")
    planned = exp.get("planned_repetitions")
    if planned is not None and (not isinstance(planned, int) or planned < 1):
        problems.append("experiment.planned_repetitions must be a positive integer")
    default_render = str((exp.get("fetch") or {}).get("render") or "static").lower()
    if default_render not in RENDER_MODES:
        problems.append("experiment.fetch.render must be static, js or auto")
    for key, lo, hi in (("health_threshold", 0, 1), ("max_heal_attempts", 1, 10),
                        ("failure_memory_size", 0, 100), ("repetitions", 1, 20)):
        value = exp.get(key)
        if not isinstance(value, (int, float)) or not lo <= value <= hi:
            problems.append(f"experiment.{key} must be between {lo} and {hi}")
    conditions = exp.get("conditions") or []
    if not conditions or not set(conditions) <= CONDITIONS:
        problems.append(f"experiment.conditions must be a non-empty subset of {', '.join(CONDITION_ORDER)}")
    for c in conditions:
        src = (COND_INFO.get(c) or {}).get("copy_of")
        if src and src not in conditions:
            problems.append(f"experiment.conditions: {c} copies the t0 selectors of {src}, so {src} "
                            "must be listed too")
    protocol = exp.get("protocol") or {}
    for i, step in enumerate(protocol.get("steps") or []):
        if not isinstance(step, dict) or step.get("op") not in ALL_OPS:
            problems.append(f"protocol.steps[{i}]: op must be one of {', '.join(ALL_OPS)}")
        elif "severity" in step and not 0 < float(step["severity"]) <= 1:
            problems.append(f"protocol.steps[{i}]: severity must be in (0, 1]")
    if not protocol.get("steps"):
        problems.append("protocol.steps must list at least one step")
    for sev in protocol.get("severities") or []:
        if not 0 < float(sev) <= 1:
            problems.append("protocol.severities values must be in (0, 1]")

    sites = data.get("sites") or []
    if not sites:
        problems.append("sites: list at least one site")
    seen_keys = set()
    resolved_sites = []
    for i, raw in enumerate(sites):
        where = f"sites[{i}]"
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be a mapping")
            continue
        site = {
            "key": str(raw.get("key", "")).strip().lower(),
            "name": str(raw.get("name") or raw.get("key") or "").strip(),
            "url": str(raw.get("url") or "").strip(),
            "category": str(raw.get("category") or "other").strip().lower(),
            "render": str(raw.get("render") or default_render).strip().lower(),
            "html_file": str(raw.get("html_file") or "").strip(),
            "prune_keep": [str(t).lower() for t in (raw.get("prune_keep") or [])],
            "notes": str(raw.get("notes") or ""),
            "website": str(raw.get("website") or "").strip(),
            "page_type": str(raw.get("page_type") or "").strip(),
            "expect_text": [str(t) for t in (raw.get("expect_text") or []) if str(t).strip()],
            "fields": [],
            "excluded_fields": [],
        }
        where = f"site '{site['key'] or i}'"
        if not _KEY.match(site["key"]):
            problems.append(f"{where}: key must be lowercase letters/digits/-/_ (max 60)")
        elif site["key"] in seen_keys:
            problems.append(f"{where}: duplicate key")
        seen_keys.add(site["key"])
        if not site["url"] and not site["html_file"]:
            problems.append(f"{where}: give a url (or an html_file)")
        if site["url"] and not site["url"].startswith(("http://", "https://")):
            problems.append(f"{where}: url must start with http:// or https://")
        if site["category"] not in CATEGORIES:
            problems.append(f"{where}: category must be one of {', '.join(sorted(CATEGORIES))}")
        if site["render"] not in RENDER_MODES:
            problems.append(f"{where}: render must be 'static', 'js' or 'auto'")
        if site["html_file"]:
            p = Path(site["html_file"])
            p = p if p.is_absolute() else (base_dir / p)
            if not p.exists():
                problems.append(f"{where}: html_file not found: {p}")
            site["html_file"] = str(p.resolve())

        names = set()
        for j, f in enumerate(raw.get("fields") or []):
            if not isinstance(f, dict):
                problems.append(f"{where} field[{j}]: must be a mapping")
                continue
            field = {
                "name": str(f.get("name") or "").strip().lower(),
                "type": str(f.get("type") or "string").strip().lower(),
                "cardinality": str(f.get("cardinality") or "single").strip().lower(),
                "required": bool(f.get("required", True)),
                "description": str(f.get("description") or "").strip(),
                # Provenance from the annotated dataset (not shown to the LLM).
                "source": str(f.get("source") or "visible").strip().lower(),
                "dynamic": bool(f.get("dynamic", False)),
                "dataset_value": "" if f.get("dataset_value") is None else str(f.get("dataset_value")),
                "dataset_type": str(f.get("dataset_type") or "").strip(),
            }
            if not re.match(r"^[a-z0-9][a-z0-9_-]{0,99}$", field["name"]):
                problems.append(f"{where} field[{j}]: name must be lowercase letters/digits/-/_")
            elif field["name"] in names:
                problems.append(f"{where}: duplicate field {field['name']}")
            names.add(field["name"])
            if field["type"] not in FIELD_TYPES:
                problems.append(f"{where} field {field['name']}: type must be one of {', '.join(sorted(FIELD_TYPES))}")
            if field["cardinality"] not in CARDINALITIES:
                problems.append(f"{where} field {field['name']}: cardinality must be single or many")
            if f.get("exclude"):
                # Kept in the record (and the paper's field table) but never
                # sent to the scraper or scored.
                reason = str(f.get("exclude_reason") or "").strip()
                if not reason:
                    problems.append(f"{where} field {field['name']}: exclude needs an exclude_reason")
                site["excluded_fields"].append({**field, "exclude_reason": reason})
                continue
            site["fields"].append(field)
        if not site["fields"]:
            problems.append(f"{where}: list at least one field")
        resolved_sites.append(site)

    if problems:
        raise ConfigError(problems)
    return {"experiment": exp, "sites": resolved_sites}


def site_by_key(config: dict, key: str) -> dict:
    for site in config["sites"]:
        if site["key"] == key:
            return site
    raise KeyError(key)
