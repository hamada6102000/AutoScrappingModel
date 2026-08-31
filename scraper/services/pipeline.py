"""
Pipeline orchestrator — the closed self-healing loop (proposal Appendix,
Figure A.1: Overall System Architecture).

Coordinates all four layers and persists every artefact to the database:

    fetch & clean  ->  (generate XPaths if none)  ->  extract  ->
    validate + health score  ->  health OK?  --yes-->  output
                                              --no-->   change detection
                                              -> self-healing (regenerate)
                                              -> retry (bounded loop)

The loop is bounded by SCRAPER["MAX_HEAL_ATTEMPTS"] so it always terminates.
"""

from __future__ import annotations

import time

from django.conf import settings
from django.utils import timezone

from ..llm.base import FieldSpec, LLMProvider
from ..llm.factory import get_provider_safe
from ..models import (
    ChangeEvent,
    ChangeType,
    ExtractedRecord,
    FailureRecord,
    LLMCallLog,
    LLMPurpose,
    Resolution,
    RuleSource,
    RunStatus,
    ScrapeRun,
    XPathRule,
)
from .change_detector import detect_change
from .extractor import FieldOutcome, extract_fields
from .fetcher import fetch, prune_result
from .health_monitor import compute_health
from .llm_analyzer import generate
from .self_healer import build_failure_hints, regenerate


class ScrapePipeline:
    def __init__(self, provider: LLMProvider | None = None):
        self.cfg = settings.SCRAPER
        self.provider = provider or get_provider_safe()
        self.threshold = self.cfg["HEALTH_THRESHOLD"]
        self.max_attempts = self.cfg["MAX_HEAL_ATTEMPTS"]

    # ------------------------------------------------------------------
    def run(self, target, html: str | None = None,
            change_type: str = ChangeType.NATURAL) -> ScrapeRun:
        """Execute the full pipeline for one target and return the ScrapeRun."""
        t0 = time.perf_counter()
        run = ScrapeRun.objects.create(target=target, status=RunStatus.RUNNING)
        trace: list[dict] = []

        # --- Layer 1: fetch + prune -----------------------------------
        if html is None:
            fr = fetch(
                target.url,
                timeout=self.cfg["FETCH_TIMEOUT"],
                user_agent=self.cfg["USER_AGENT"],
            )
        else:
            fr = prune_result(target.url, html)

        run.http_status = fr.http_status
        run.raw_html_signature = fr.raw_html_signature
        run.dom_signature = fr.dom_signature
        run.dom_skeleton = fr.dom_skeleton
        # Stash the pruned DOM so extraction passes can reuse it within the run.
        self._pruned_html = fr.pruned_html

        if fr.error or not fr.pruned_html:
            trace.append({"step": "fetch", "ok": False, "detail": fr.error or "empty DOM"})
            run.status = RunStatus.FAILED
            run.log = trace
            self._finalize(run, t0)
            return run

        trace.append({
            "step": "fetch",
            "ok": True,
            "detail": f"HTTP {fr.http_status}; pruned DOM, "
                      f"{fr.token_reduction * 100:.1f}% token reduction "
                      f"({fr.raw_token_estimate}->{fr.pruned_token_estimate} tokens)",
        })

        fields = list(target.fields.all())
        specs = [FieldSpec(f.name, f.description, f.field_type) for f in fields]
        required = {f.name for f in fields if f.is_required}

        # --- change detection vs previous run -------------------------
        prev_run = (
            target.runs.exclude(pk=run.pk)
            .exclude(dom_skeleton="")
            .order_by("-started_at")
            .first()
        )
        prev_skeleton = prev_run.dom_skeleton if prev_run else ""
        change = detect_change(prev_skeleton, fr.dom_skeleton)
        run.structural_distance = change.structural_distance
        if change.previous_available:
            trace.append({
                "step": "change-detection",
                "ok": True,
                "detail": f"structural distance vs previous DOM = "
                          f"{change.structural_distance:.3f} "
                          f"({'CHANGED' if change.changed else 'stable'})",
            })

        # --- Layer 2: initial generation if no active rules -----------
        has_rules = XPathRule.objects.filter(
            field__target=target, is_active=True
        ).exists()
        if not has_rules:
            result, latency = generate(self.provider, fr.pruned_html, specs)
            self._store_rules(fields, result.xpaths, fr.dom_signature, run,
                              purpose=LLMPurpose.INITIAL, latency=latency,
                              provider=self.provider)
            run.token_usage += result.total_tokens
            trace.append({
                "step": "llm-generate",
                "ok": True,
                "detail": f"{self.provider.name}/{result.model}: generated "
                          f"{len(result.xpaths)}/{len(fields)} XPaths "
                          f"({latency} ms)",
            })

        # --- extract + validate + health ------------------------------
        outcomes = self._extract(target, fields)
        health = compute_health(outcomes, required, change.structural_distance)
        self._apply_health(run, health)
        trace.append({
            "step": "validate",
            "ok": health.score >= self.threshold,
            "detail": f"health={health.score:.2f} "
                      f"(threshold {self.threshold:.2f}); "
                      f"{health.fields_extracted}/{health.fields_total} fields; "
                      f"failure-rate={health.xpath_failure_rate:.2f}",
        })

        # --- healthy? then output -------------------------------------
        if health.score >= self.threshold:
            run.status = RunStatus.SUCCESS
            trace.append({"step": "output", "ok": True,
                          "detail": "health OK — emitting clean records"})
            self._persist_records(run, fields, outcomes)
            run.log = trace
            self._finalize(run, t0)
            return run

        # --- Layer 4: self-healing loop -------------------------------
        self._record_failures(run, fields, outcomes)
        failed_names = [o.field_name for o in outcomes if not o.success]
        change_event = ChangeEvent.objects.create(
            target=target,
            run=run,
            change_type=change_type,
            resolution=Resolution.UNRESOLVED,
            affected_fields=failed_names,
            structural_distance=change.structural_distance,
        )
        trace.append({
            "step": "self-heal:trigger",
            "ok": False,
            "detail": f"health below threshold — {len(failed_names)} field(s) "
                      f"affected: {', '.join(failed_names)}",
        })

        heal_t0 = time.perf_counter()
        name_to_field = {f.name: f for f in fields}
        for attempt in range(1, self.max_attempts + 1):
            run.heal_attempts = attempt

            failed_outcomes = [o for o in outcomes if not o.success]
            failed_specs = [
                FieldSpec(
                    name_to_field[o.field_name].name,
                    name_to_field[o.field_name].description,
                    name_to_field[o.field_name].field_type,
                )
                for o in failed_outcomes
                if o.field_name in name_to_field
            ]
            hints = build_failure_hints(failed_outcomes) + self._recent_memory(target)

            result, latency = regenerate(
                self.provider, fr.pruned_html, failed_specs, hints
            )
            run.heal_latency_ms += latency
            run.token_usage += result.total_tokens
            self._store_rules(
                [name_to_field[n] for n in result.xpaths if n in name_to_field],
                result.xpaths, fr.dom_signature, run,
                purpose=LLMPurpose.HEALING, latency=latency, provider=self.provider,
            )
            trace.append({
                "step": f"self-heal:attempt-{attempt}",
                "ok": True,
                "detail": f"regenerated {len(result.xpaths)} XPath(s) with "
                          f"failure-history memory ({latency} ms)",
            })

            # re-extract everything and re-score.
            outcomes = self._extract(target, fields)
            health = compute_health(outcomes, required, change.structural_distance)
            self._apply_health(run, health)

            if health.score >= self.threshold:
                run.healed = True
                run.status = RunStatus.HEALED
                recovery_ms = int((time.perf_counter() - heal_t0) * 1000)
                change_event.resolution = Resolution.AUTO_HEALED
                change_event.time_to_recovery_ms = recovery_ms
                change_event.save(update_fields=["resolution", "time_to_recovery_ms"])
                trace.append({
                    "step": "self-heal:recovered",
                    "ok": True,
                    "detail": f"recovered after {attempt} attempt(s) in "
                              f"{recovery_ms} ms; health={health.score:.2f}",
                })
                break
        else:
            run.status = RunStatus.DEGRADED
            trace.append({
                "step": "self-heal:exhausted",
                "ok": False,
                "detail": f"still degraded after {self.max_attempts} attempt(s); "
                          f"health={health.score:.2f}",
            })

        self._persist_records(run, fields, outcomes)
        run.log = trace
        self._finalize(run, t0)
        return run

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract(self, target, fields) -> list[FieldOutcome]:
        # Re-fetch active rules (they may have just been regenerated).
        rules = {
            r.field_id: r
            for r in XPathRule.objects.filter(field__target=target, is_active=True)
        }
        self._active_rules = rules  # cached for record linkage
        # need the pruned HTML — reuse the last fetch stored on the instance.
        field_rules = [
            (f.name, f.field_type, rules[f.id].xpath if f.id in rules else "")
            for f in fields
        ]
        return extract_fields(self._pruned_html, field_rules)

    def _store_rules(self, fields, xpaths: dict, dom_signature: str, run,
                     purpose: str, latency: int, provider) -> None:
        source = RuleSource.HEURISTIC if provider.name == "heuristic" else RuleSource.LLM
        for f in fields:
            xpath = xpaths.get(f.name, "")
            if not xpath:
                continue
            latest = f.rules.order_by("-version").first()
            next_version = (latest.version + 1) if latest else 1
            f.rules.filter(is_active=True).update(is_active=False)
            XPathRule.objects.create(
                field=f,
                xpath=xpath,
                version=next_version,
                is_active=True,
                source=source,
                dom_signature=dom_signature,
            )
        # Log the LLM call once per store operation.
        LLMCallLog.objects.create(
            run=run,
            provider=provider.name,
            model=getattr(provider, "model", provider.name),
            purpose=purpose,
            latency_ms=latency,
            success=bool(xpaths),
        )

    def _persist_records(self, run, fields, outcomes: list[FieldOutcome]) -> None:
        # Idempotent: clear any records already written for this run.
        run.records.all().delete()
        name_to_field = {f.name: f for f in fields}
        rules = getattr(self, "_active_rules", {})
        for o in outcomes:
            field = name_to_field.get(o.field_name)
            if field is None:
                continue
            gt = (field.ground_truth or "").strip()
            matches = None
            if gt:
                matches = o.normalized_value.strip().lower() == gt.lower()
            ExtractedRecord.objects.create(
                run=run,
                field=field,
                xpath_used=rules.get(field.id),
                raw_value=o.raw_value,
                normalized_value=o.normalized_value,
                success=o.success,
                matches_ground_truth=matches,
            )

    def _record_failures(self, run, fields, outcomes) -> None:
        name_to_field = {f.name: f for f in fields}
        for o in outcomes:
            if o.success:
                continue
            field = name_to_field.get(o.field_name)
            if field is None:
                continue
            FailureRecord.objects.create(
                field=field,
                run=run,
                failed_xpath=o.xpath,
                reason=o.reason or "extraction-failed",
            )

    def _recent_memory(self, target):
        limit = self.cfg["FAILURE_MEMORY_SIZE"]
        recent = (
            FailureRecord.objects.filter(field__target=target)
            .select_related("field")
            .order_by("-created_at")[:limit]
        )
        return build_failure_hints_from_records(recent)

    def _apply_health(self, run, health) -> None:
        run.health_score = health.score
        run.xpath_failure_rate = health.xpath_failure_rate
        run.field_count_mismatch = health.field_count_mismatch
        run.fields_total = health.fields_total
        run.fields_extracted = health.fields_extracted

    def _finalize(self, run, t0) -> None:
        run.finished_at = timezone.now()
        run.duration_ms = int((time.perf_counter() - t0) * 1000)
        run.save()

    # pruned HTML is stashed on the instance by run() via property below.
    _pruned_html: str = ""


# Small adapter so the pipeline can reuse the same FailureHint shape when the
# memory comes from persisted FailureRecord rows.
def build_failure_hints_from_records(records) -> list:
    from ..llm.base import FailureHint

    return [
        FailureHint(
            field_name=r.field.name,
            failed_xpath=r.failed_xpath,
            reason=r.reason or "extraction-failed",
        )
        for r in records
    ]
