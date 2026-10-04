"""
Step 5 — check everything is ready before spending LLM quota.

    python manage.py eval_check            # offline checks + call/token estimate
    python manage.py eval_check --ping     # also verify the LLM: model list, one tiny
                                           # call, rate limits (saved to llm_ping.json)
"""

from __future__ import annotations

import json
import time

from django.core.management.base import BaseCommand
from django.utils import timezone

from scraper.evaluation.cli import add_experiment_arg, get_experiment, pct
from scraper.evaluation.protocol import baseline_snapshot, estimate, gt_status, make_provider
from scraper.llm.base import FieldSpec
from scraper.services import snapshots as snapstore
from scraper.services import tokens


class Command(BaseCommand):
    help = "Readiness check for an evaluation experiment."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--ping", action="store_true",
                            help="Verify the LLM: list models, one tiny call, rate limits.")

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        e = exp.config["experiment"]
        llm = e["llm"]
        problems, notes = [], []

        if tokens.is_estimate():
            problems.append("exact tokenizer unavailable (tiktoken) - token counts would be estimates")
        try:
            provider = make_provider(e)
        except Exception as exc:
            provider = None
            problems.append(f"LLM provider not usable: {exc}")

        self.stdout.write(
            f"Experiment {exp.name}: {len(exp.config['sites'])} sites, {e['repetitions']} repetitions, "
            f"LLM {llm['provider']} / {llm.get('model') or 'default'}"
            + (f", reasoning {llm['reasoning_effort']}" if llm.get("reasoning_effort") else "")
            + (f", TPM {llm['tokens_per_minute']}" if llm.get("tokens_per_minute") else "") + "\n")
        est = estimate(exp)
        per_site = {r["site"]: r for r in est["sites"]}
        for site in exp.config["sites"]:
            snap = baseline_snapshot(exp, site["key"])
            if snap is None:
                problems.append(f"{site['key']}: no baseline snapshot")
                self.stdout.write(f"  {site['key']:<20} NO SNAPSHOT")
                continue
            missing, labelled = gt_status(snap, site)
            flags = []
            meta = snap.fetch_meta or {}
            if snap.robots_allowed is False:
                flags.append("robots DISALLOW")
            if meta.get("suspected_block"):
                flags.append("possible bot-block page")
                notes.append(f"{site['key']}: open the labeling view to confirm it is the real page")
            check = meta.get("check") or {}
            if check and not check.get("content_ok", True):
                flags.append("content check FAILED")
                problems.append(f"{site['key']}: saved page does not contain the expected texts "
                                f"{site.get('expect_text')} (consent wall / bot check / unrendered?)")
            if snap.llm_view_truncated:
                flags.append("LLM view truncated")
            if missing:
                problems.append(f"{site['key']}: {len(missing)} unlabelled field(s): {', '.join(missing)}")
            for name, p in (snap.field_presence or {}).items():
                if p.get("labelled_absent"):
                    continue
                if p.get("in_raw") and not p.get("in_clean_dom"):
                    problems.append(f"{site['key']}.{name}: removed by pruning (add to prune_keep)")
                elif not p.get("in_raw"):
                    notes.append(f"{site['key']}.{name}: labelled value not found in page text (typo?)")
            row = per_site.get(site["key"]) or {}
            render = meta.get("render_chosen") or meta.get("render") or "file"
            self.stdout.write(
                f"  {site['key']:<10} HTTP {snap.http_status}  render {render}"
                f"  view {snap.llm_view_tokens} tok ({pct(snap.token_reduction)} reduction)"
                f"  request ~{row.get('per_call_tokens') or '?'} tok"
                f"  labels {len(labelled)}/{len(site['fields'])}"
                + (f"  [{'; '.join(flags)}]" if flags else "")
            )

        t = est["totals"]
        self.stdout.write(
            f"\nEstimated LLM usage for the full run: {t['calls_min']}-{t['calls_max']} calls, "
            f"~{t['tokens_min']:,}-{t['tokens_max']:,} tokens (per call: planned prompt + completion "
            "budget; the upper bound assumes every step breaks and every heal needs all attempts).")

        if opts["ping"] and provider is not None:
            self._ping(exp, provider, llm, problems, notes)

        for n in notes:
            self.stdout.write(self.style.WARNING(f"  NOTE {n}"))
        if problems:
            self.stdout.write(self.style.ERROR(f"\nNOT READY - {len(problems)} problem(s):"))
            for p in problems:
                self.stdout.write(self.style.ERROR(f"  - {p}"))
        else:
            self.stdout.write(self.style.SUCCESS("\nREADY. Next: python manage.py eval_run"))

    # ------------------------------------------------------------------
    def _ping(self, exp, provider, llm, problems, notes):
        record = {"at": timezone.now().isoformat(), "provider": llm["provider"],
                  "model": llm.get("model"), "options": {k: llm.get(k) for k in (
                      "reasoning_effort", "include_reasoning", "tokens_per_minute",
                      "completion_counts_in_tpm", "max_input_chars")}, "ok": False}
        ping_file = snapstore.experiment_dir(exp) / "llm_ping.json"

        if hasattr(provider, "list_models"):
            try:
                models = provider.list_models()
                record["available_models"] = models
                if getattr(provider, "model", None) not in models:
                    problems.append(f"model {provider.model!r} is not available to this API key. "
                                    f"Available: {', '.join(models)}")
                else:
                    self.stdout.write(f"\nModel {provider.model} is available to this key "
                                      f"({len(models)} models listed).")
            except Exception as exc:
                notes.append(f"could not list models: {type(exc).__name__}: {exc}")

        html = "<html><body><main><h1 class='title'>Ping</h1></main></body></html>"
        start = time.perf_counter()
        try:
            res = provider.generate_xpaths(html, [FieldSpec("title", "the page title")])
            ms = int((time.perf_counter() - start) * 1000)
            limits = getattr(provider, "last_rate_limits", {}) or {}
            record.update({"ok": bool(res.xpaths.get("title")), "latency_ms": ms,
                           "model_reported": res.model, "prompt_tokens": res.prompt_tokens,
                           "completion_tokens": res.completion_tokens,
                           "xpath": res.xpaths.get("title"), "raw": res.raw_response[:300],
                           "limits": limits})
            if not res.xpaths.get("title"):
                problems.append(f"LLM ping returned no XPath (raw: {res.raw_response[:160]!r})")
            else:
                self.stdout.write(self.style.SUCCESS(
                    f"LLM ping OK: {res.model}, {ms} ms, {res.prompt_tokens}+{res.completion_tokens} "
                    f"tokens, returned {res.xpaths['title']!r}"))
            if limits:
                self.stdout.write("Rate limits reported for this key/model:")
                for k, v in sorted(limits.items()):
                    self.stdout.write(f"  {k}: {v}")
            tpm = _int(limits.get("x-ratelimit-limit-tokens"))
            record["tokens_per_minute_reported"] = tpm
            if tpm and llm.get("tokens_per_minute") and llm["tokens_per_minute"] > tpm:
                problems.append(f"llm.tokens_per_minute is {llm['tokens_per_minute']} but the account "
                                f"limit is {tpm}; set it to {tpm} in eval/sites.yaml and re-run eval_init")
            if tpm and not llm.get("tokens_per_minute"):
                notes.append(f"the account's per-minute token limit is {tpm}; set "
                             f"llm.tokens_per_minute: {tpm} so requests are planned to fit")
            if tpm and hasattr(provider, "probe_completion_accounting"):
                try:
                    counted = provider.probe_completion_accounting(tpm)
                except Exception as exc:
                    counted = None
                    notes.append(f"TPM accounting probe failed: {type(exc).__name__}: {exc}")
                record["completion_counts_in_tpm_observed"] = counted
                if counted is not None:
                    self.stdout.write("max_completion_tokens counted in the per-minute limit: "
                                      + ("yes" if counted else "no"))
                if counted is True and llm.get("completion_counts_in_tpm") is False:
                    problems.append("the provider counts max_completion_tokens in the per-minute "
                                    "limit; set llm.completion_counts_in_tpm: true")
        except Exception as exc:
            problems.append(f"LLM ping failed: {type(exc).__name__}: {exc}")
            record["error"] = f"{type(exc).__name__}: {exc}"[:500]
        ping_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
        self.stdout.write(f"(saved to {ping_file})")


def _int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
