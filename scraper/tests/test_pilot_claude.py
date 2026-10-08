"""The two-site Claude pilot config is built from the original pilot's sites
file without changing anything but the name, the LLM and the repetitions."""

import importlib.util
import tempfile
from pathlib import Path

import yaml
from django.test import SimpleTestCase

from scraper.evaluation.config import resolve

SCRIPT = Path(__file__).resolve().parents[2] / "eval" / "pilot_claude.py"


def load():
    spec = importlib.util.spec_from_file_location("pilot_claude", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PilotConfigTests(SimpleTestCase):
    def test_config_keeps_original_settings(self):
        mod = load()
        tmp = Path(tempfile.mkdtemp())
        pages = {}
        for key in ("site_001", "site_002"):
            pages[key] = tmp / f"{key}.html"
            pages[key].write_text("<html><body><h1>x</h1></body></html>", encoding="utf-8")
        field = {"name": "title", "type": "string", "cardinality": "single", "required": True,
                 "description": "the title"}
        original = {
            "experiment": {"name": "ieee-2026", "seed": 20261004, "health_threshold": 0.8, "max_heal_attempts": 3,
                           "failure_memory_size": 10, "repetitions": 2, "planned_repetitions": 3,
                           "conditions": ["self_healing", "static", "rule_based", "rule_static"],
                           "llm": {"provider": "groq", "model": "openai/gpt-oss-120b"}},
            "sites": [{"key": k, "name": k, "url": f"https://example.com/{k}", "category": "ecommerce",
                       "fields": [field]} for k in ("site_001", "site_002", "site_003")],
        }
        src = tmp / "sites.yaml"
        src.write_text(yaml.safe_dump(original), encoding="utf-8")
        cfg = mod.build_config(src, pages)
        e = cfg["experiment"]
        self.assertEqual(e["name"], "pilot-claude-2sites")
        self.assertEqual((e["seed"], e["health_threshold"], e["max_heal_attempts"], e["failure_memory_size"]),
                         (20261004, 0.8, 3, 10))
        self.assertEqual((e["repetitions"], e["conditions"]), (1, ["self_healing", "static"]))
        self.assertEqual(e["llm"]["model"], "claude-opus-5-5")
        self.assertEqual((e["llm"]["tokens_per_minute"], e["llm"]["max_input_chars"]), (8000, 60000))
        self.assertEqual([s["key"] for s in cfg["sites"]], ["site_001", "site_002"])
        self.assertEqual(cfg["sites"][0]["fields"], [field])
        self.assertEqual(cfg["sites"][1]["html_file"], str(pages["site_002"]))
        resolve(cfg, base_dir=tmp)       # a valid configuration
