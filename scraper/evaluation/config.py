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
FIELD_TYPES = {"string", "number", "price", "date", "url", "mixed"}
CARDINALITIES = {"single", "many"}
CONDITIONS = {"self_healing", "static"}

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
    "seed": 20261004,
    "conditions": ["self_healing", "static"],
    "protocol": {
        "steps": DEFAULT_STEPS,
        "severities": [0.3, 0.6, 1.0],
        "shuffle": True,
    },
    "fetch": {"user_agent": None, "timeout": 30, "respect_robots": True},
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
    for key, lo, hi in (("health_threshold", 0, 1), ("max_heal_attempts", 1, 10),
                        ("failure_memory_size", 0, 100), ("repetitions", 1, 20)):
        value = exp.get(key)
        if not isinstance(value, (int, float)) or not lo <= value <= hi:
            problems.append(f"experiment.{key} must be between {lo} and {hi}")
    conditions = exp.get("conditions") or []
    if "self_healing" not in conditions or not set(conditions) <= CONDITIONS:
        problems.append("experiment.conditions must include self_healing (optionally static)")
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
            "render": str(raw.get("render") or "static").strip().lower(),
            "html_file": str(raw.get("html_file") or "").strip(),
            "prune_keep": [str(t).lower() for t in (raw.get("prune_keep") or [])],
            "notes": str(raw.get("notes") or ""),
            "fields": [],
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
        if site["render"] not in ("static", "js"):
            problems.append(f"{where}: render must be 'static' or 'js'")
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
