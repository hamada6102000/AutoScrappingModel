"""
Pipeline orchestrator — the closed self-healing loop (proposal Appendix,
Figure A.1: Overall System Architecture).

    fetch & clean -> generate XPaths for fields without a rule (validated;
    invalid ones repaired before storing, §4.3 step 3) -> extract ->
    validate + health score -> health OK? -- yes --> output
                                           -- no  --> change event ->
    self-healing: regenerate failed fields with failure-history memory,
    validate, re-extract (bounded by MAX_HEAL_ATTEMPTS)

Everything the evaluation reports is recorded here: health before and after
healing, which fields failed and why, every LLM call (tokens, pure latency,
rate-limit waits, retries, fallback, failure-history length, raw response),
and a ground-truth comparison of every extracted value.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from ..llm.base import (
    FailureHint, FieldSpec, LLMProvider, LLMResult, ProviderUnavailable, RateLimitExhausted,
    RequestTooLarge,
)
from ..llm.factory import get_provider_safe
from ..llm.heuristic import HeuristicProvider
from ..llm.prompts import PROMPT_VERSION
from ..models import (
    ChangeEvent, ChangeType, ExtractedRecord, FailureRecord, LLMCallLog,
    LLMPurpose, RecordPhase, Resolution, RulePurpose, RuleSource, RunStatus,
    ScrapeRun, StepKind, XPathRule,
)
from . import snapshots as snapstore
from .change_detector import detect_change
from .extractor import FieldOutcome, extract_one, set_base_url
from .fetcher import FetchResult, fetch, profile_signature, prune_result
from .health_monitor import compute_health
from .htmlutil import decode_html, parse_html
from .matching import exact_match, list_match, token_prf
from .self_healer import build_failure_hints

_PURPOSE_TO_RULE = {
    LLMPurpose.INITIAL: RulePurpose.INITIAL,
    LLMPurpose.REPAIR: RulePurpose.REPAIR,
    LLMPurpose.HEALING: RulePurpose.HEALING,
}


@dataclass
class RunContext:
    """Evaluation metadata attached to a run (all optional for live runs)."""

    experiment: object = None
    snapshot: object = None
    condition: str = ""
    repetition: int | None = None
    step: int | None = None
    step_kind: str = StepKind.LIVE
    change_type: str = ""
    true_change: bool | None = None
    mutation: dict = field(default_factory=dict)


def spec_for(f) -> FieldSpec:
    return FieldSpec(f.name, f.description, f.field_type, f.cardinality)


class ScrapePipeline:
    def __init__(self, provider: LLMProvider | None = None, *, heal_enabled: bool = True,
                 allow_fallback: bool | None = None, threshold: float | None = None,
                 max_attempts: int | None = None, memory_size: int | None = None,
                 raise_on_quota: bool = False, save_snapshots: bool | None = None,
                 generate_missing: bool = True):
        cfg = settings.SCRAPER
        self.cfg = cfg
        self.provider = provider or get_provider_safe()
        self.heal_enabled = heal_enabled
        # False for the paired static baseline: it reuses the self-healing
        # condition's t0 rules and must never call the LLM itself.
        self.generate_missing = generate_missing
        self.allow_fallback = cfg.get("ALLOW_FALLBACK", True) if allow_fallback is None else allow_fallback
        self.threshold = cfg["HEALTH_THRESHOLD"] if threshold is None else threshold
        self.max_attempts = cfg["MAX_HEAL_ATTEMPTS"] if max_attempts is None else max_attempts
        self.memory_size = cfg["FAILURE_MEMORY_SIZE"] if memory_size is None else memory_size
        self.raise_on_quota = raise_on_quota
        self.save_snapshots = cfg.get("SAVE_SNAPSHOTS", True) if save_snapshots is None else save_snapshots
        self.pricing = cfg.get("PRICING") or {}

    # ------------------------------------------------------------------
    def run(self, target, html: str | None = None, *, snapshot=None,
            context: RunContext | None = None, change_type: str | None = None) -> ScrapeRun:
        """Execute the full pipeline for one target and return the ScrapeRun."""
        ctx = context or RunContext()
        snapshot = snapshot or ctx.snapshot
        t0 = time.perf_counter()
        run = ScrapeRun.objects.create(
            target=target,
            status=RunStatus.RUNNING,
            experiment=ctx.experiment or target.experiment,
            snapshot=snapshot,
            condition=ctx.condition or target.condition,
            repetition=ctx.repetition if ctx.repetition is not None else target.repetition,
            step=ctx.step,
            step_kind=ctx.step_kind,
            change_type=ctx.change_type or change_type or "",
            true_change=ctx.true_change,
            heal_enabled=self.heal_enabled,
            provider=self.provider.name,
            model=getattr(self.provider, "model", "") or "",
        )
        self._run = run
        self._trace: list[dict] = []
        self._mutation = ctx.mutation or (snapshot.mutation if snapshot else {})
        keep = target.prune_keep or ()

        # --- Layer 1: fetch + prune --------------------------------------
        fr = self._load(target, html, run.snapshot, keep)
        run.http_status = fr.http_status
        run.raw_html_signature = fr.raw_html_signature
        run.dom_signature = fr.dom_signature
        run.raw_tokens = fr.page.raw_tokens
        run.llm_view_tokens = fr.page.llm_view_tokens
        run.tokenizer = fr.page.tokenizer
        if run.snapshot_id is None:
            run.dom_profile = {"tag": fr.page.tag_profile, "markup": fr.page.markup_profile}
        self._page = fr.page
        self._tree = parse_html(fr.page.clean_html)
        set_base_url(self._tree, fr.url or target.url)

        if fr.error or fr.page.empty or self._tree is None:
            run.status = RunStatus.FAILED
            run.error = fr.error or "empty DOM"
            self._step("fetch", False, run.error)
            return self._finalize(t0)

        self._step("fetch", True,
                   f"HTTP {fr.http_status}; {fr.page.raw_tokens} raw -> "
                   f"{fr.page.llm_view_tokens} LLM-view tokens "
                   f"({fr.page.token_reduction * 100:.1f}% reduction, {fr.page.tokenizer})")

        fields = list(target.fields.all())
        self._fields = {f.name: f for f in fields}
        required = {f.name for f in fields if f.is_required}
        self._gt = snapstore.gt_map(run.snapshot) if run.snapshot_id else {}

        # --- change detection vs previous run ---------------------------
        prev = self._previous_profiles(target, run)
        change = detect_change(prev, (fr.page.tag_profile, fr.page.markup_profile))
        run.structural_distance = change.structural_distance
        run.markup_distance = change.markup_distance
        if change.previous_available:
            self._step("change-detection", True,
                       f"structural={change.structural_distance:.3f}, "
                       f"markup={change.markup_distance:.3f} "
                       f"({'CHANGED' if change.changed else 'stable'})")

        # --- Layer 2: generate rules for fields that have none ----------
        missing = [f for f in fields if not f.rules.filter(is_active=True).exists()]
        if missing and self.generate_missing:
            self._generate(missing, fr.dom_signature)
        elif missing:
            self._step("llm-generate:skipped", False,
                       "no rule for: " + ", ".join(f.name for f in missing)
                       + " (generation disabled for this condition)")

        # --- Layer 3: extract + validate + health -----------------------
        outcomes = self._extract(fields)
        health = compute_health(outcomes, required, change.structural_distance)
        self._apply_health(health)
        run.pre_heal_health = health.score
        run.pre_heal_failure_rate = health.xpath_failure_rate
        run.pre_heal_field_mismatch = health.field_count_mismatch
        run.pre_heal_fields_extracted = health.fields_extracted
        run.pre_heal_failed = [o.field_name for o in outcomes if not o.success]
        self._step("validate", health.score >= self.threshold,
                   f"health={health.score:.2f} (threshold {self.threshold:.2f}); "
                   f"{health.fields_extracted}/{health.fields_total} fields; "
                   f"failure-rate={health.xpath_failure_rate:.2f}")

        if health.score >= self.threshold:
            run.status = RunStatus.SUCCESS
            self._persist(outcomes, RecordPhase.FINAL)
            self._step("output", True, "health OK — emitting records")
            return self._finalize(t0)

        # --- below threshold: change event ------------------------------
        run.detected = True
        self._record_failures(outcomes)
        failed_names = [o.field_name for o in outcomes if not o.success]
        event = ChangeEvent.objects.create(
            target=target,
            run=run,
            change_type=run.change_type if run.change_type in ChangeType.values else ChangeType.NATURAL,
            mutation=self._mutation or {},
            resolution=Resolution.UNRESOLVED,
            affected_fields=failed_names,
            structural_distance=change.structural_distance,
            markup_distance=change.markup_distance,
        )
        self._step("self-heal:trigger", False,
                   f"health below threshold — {len(failed_names)} field(s) affected: "
                   f"{', '.join(failed_names) or '-'}")

        if not self.heal_enabled:
            event.resolution = Resolution.NOT_ATTEMPTED
            event.fields_unrecovered = failed_names
            event.save()
            run.status = RunStatus.DEGRADED
            self._persist(outcomes, RecordPhase.FINAL)
            self._step("self-heal:disabled", False, "healing disabled (static condition)")
            return self._finalize(t0)

        # --- Layer 4: self-healing loop ---------------------------------
        self._persist(outcomes, RecordPhase.PRE_HEAL)
        heal_t0 = time.perf_counter()
        wait_before = run.rate_limit_wait_ms
        history_len = 0
        for attempt in range(1, self.max_attempts + 1):
            run.heal_attempts = attempt
            failed = [o for o in outcomes if not o.success and o.field_name in self._fields]
            hints = self._memory(target, failed)
            history_len = len(hints)
            failed_fields = [self._fields[o.field_name] for o in failed]
            result, log = self._call([spec_for(f) for f in failed_fields], hints,
                                     LLMPurpose.HEALING, attempt)
            valid, invalid = self._validate_and_store(failed_fields, result.xpaths,
                                                      LLMPurpose.HEALING, fr.dom_signature, log)
            for f, xp, reason in invalid:
                if xp:
                    FailureRecord.objects.create(field=f, run=run, failed_xpath=xp[:2000],
                                                 reason=f"candidate {reason}")
            run.heal_latency_ms += log.latency_ms + log.validation_ms
            outcomes = self._extract(fields)
            health = compute_health(outcomes, required, change.structural_distance)
            self._apply_health(health)
            self._step(f"self-heal:attempt-{attempt}", log.success,
                       f"{log.provider} returned {len(result.xpaths)} XPath(s), "
                       f"{len(valid)} valid; memory={history_len}; "
                       f"{log.latency_ms} ms (+{log.rate_limit_wait_ms} ms rate-limit wait)"
                       + (f"; error: {log.error}" if log.error else ""))
            if health.score >= self.threshold:
                run.healed = True
                run.status = RunStatus.HEALED
                break
        else:
            run.status = RunStatus.DEGRADED

        wall_ms = int((time.perf_counter() - heal_t0) * 1000)
        waits = run.rate_limit_wait_ms - wait_before
        succeeded = {o.field_name for o in outcomes if o.success}
        event.attempts = run.heal_attempts
        event.failure_history_len = history_len
        event.fields_recovered = [n for n in failed_names if n in succeeded]
        event.fields_unrecovered = [n for n in failed_names if n not in succeeded]
        if run.healed:
            event.resolution = Resolution.AUTO_HEALED
            event.time_to_recovery_ms = wall_ms
            event.time_to_recovery_excl_wait_ms = max(0, wall_ms - waits)
        event.save()
        self._step("self-heal:recovered" if run.healed else "self-heal:exhausted", run.healed,
                   f"after {run.heal_attempts} attempt(s), {wall_ms} ms; health={health.score:.2f}")
        self._persist(outcomes, RecordPhase.FINAL)
        return self._finalize(t0)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self, target, html, snapshot, keep) -> FetchResult:
        if html is not None:
            return prune_result(target.url, html, keep_tags=keep)
        if snapshot is not None:
            page = snapstore.pruned_page(snapshot, keep)
            return FetchResult(
                url=snapshot.url or target.url,
                http_status=snapshot.http_status or 200,
                raw_html="",
                page=page,
                raw_html_signature=snapshot.sha256,
                dom_signature=profile_signature(page),
            )
        if target.html_file:
            raw, _ = decode_html(Path(target.html_file).read_bytes())
            fr = prune_result(target.url, raw, keep_tags=keep)
        else:
            fr = fetch(target.url, timeout=self.cfg["FETCH_TIMEOUT"],
                       user_agent=self.cfg["USER_AGENT"], render=target.render or "static",
                       keep_tags=keep)
        if self.save_snapshots and fr.raw_html and not fr.error:
            self._run.snapshot = snapstore.save_live(target, fr)
        return fr

    def _previous_profiles(self, target, run):
        prev = (target.runs.exclude(pk=run.pk).filter(completed=True)
                .order_by("-id").first())
        if prev is None:
            return None
        try:
            if prev.snapshot_id:
                page = snapstore.pruned_page(prev.snapshot, target.prune_keep or ())
                return page.tag_profile, page.markup_profile
        except OSError:
            return None
        if prev.dom_profile:
            return prev.dom_profile.get("tag", {}), prev.dom_profile.get("markup", {})
        return None

    # ------------------------------------------------------------------
    # LLM calls, generation and validation
    # ------------------------------------------------------------------
    def _call(self, specs, hints, purpose, attempt) -> tuple[LLMResult, LLMCallLog]:
        run = self._run
        used = self.provider
        error = ""
        result = None
        start = time.perf_counter()
        try:
            result = self.provider.generate_xpaths(self._page.llm_view, specs,
                                                   failure_history=hints or None)
        except RateLimitExhausted as exc:
            if self.raise_on_quota:
                raise
            error = f"RateLimitExhausted: {exc}"[:300]
        except Exception as exc:  # network, HTTP, parsing, bad key
            if self.raise_on_quota:
                # Evaluation mode: the provider being unreachable is not a
                # property of the method; stop and roll the run back.
                fatal = isinstance(exc, RequestTooLarge) or re.search(r"HTTP 4\d\d", str(exc))
                raise ProviderUnavailable(f"{type(exc).__name__}: {exc}"[:400],
                                          retryable=not fatal) from exc
            error = f"{type(exc).__name__}: {exc}"[:300]
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        if result is None:
            run.llm_errors += 1
            if self.allow_fallback and self.provider.name != "heuristic":
                used = HeuristicProvider()
                result = used.generate_xpaths(self._page.llm_view, specs,
                                              failure_history=hints or None)
                run.fallback_used = True
            else:
                result = LLMResult(xpaths={}, model=getattr(self.provider, "model", ""))

        latency = result.latency_ms or max(0, elapsed_ms - result.rate_limit_wait_ms)
        is_llm = used.name != "heuristic"
        cost = self._cost(result.model, result.prompt_tokens, result.completion_tokens) if is_llm else None
        log = LLMCallLog.objects.create(
            run=run,
            provider=used.name,
            requested_provider=self.provider.name,
            fallback_used=used is not self.provider,
            model=result.model or getattr(used, "model", ""),
            purpose=purpose,
            attempt=attempt,
            failure_history_len=len(hints or []),
            fields_requested=len(specs),
            xpaths_returned=len(result.xpaths),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=cost,
            latency_ms=latency,
            rate_limit_wait_ms=result.rate_limit_wait_ms,
            retries=result.retries,
            input_chars=result.input_chars,
            input_truncated=result.input_truncated,
            prompt_sha=result.prompt_sha,
            prompt_version=PROMPT_VERSION if is_llm else "",
            raw_response=(result.raw_response or "")[:20000],
            success=not error,
            error=error,
        )
        run.llm_calls_count += 1
        run.prompt_tokens += result.prompt_tokens
        run.completion_tokens += result.completion_tokens
        run.token_usage = run.prompt_tokens + run.completion_tokens
        run.rate_limit_wait_ms += result.rate_limit_wait_ms
        run.llm_input_truncated = run.llm_input_truncated or result.input_truncated
        if cost is not None:
            run.cost_usd = (run.cost_usd or 0.0) + cost
        return result, log

    def _generate(self, fields, dom_signature) -> None:
        """Initial generation with validation repair (proposal §4.3 step 3)."""
        pending = list(fields)
        hints = None
        for attempt in range(0, self.max_attempts + 1):
            if not pending:
                break
            purpose = LLMPurpose.INITIAL if attempt == 0 else LLMPurpose.REPAIR
            result, log = self._call([spec_for(f) for f in pending], hints, purpose, attempt)
            valid, invalid = self._validate_and_store(pending, result.xpaths, purpose,
                                                      dom_signature, log)
            self._step("llm-generate" if attempt == 0 else f"llm-repair-{attempt}",
                       bool(valid) and not log.error,
                       f"{log.provider}/{log.model}: {len(valid)}/{len(pending)} valid XPaths "
                       f"({log.latency_ms} ms)" + (f"; error: {log.error}" if log.error else ""))
            hints = [FailureHint(f.name, xp or "(no XPath returned)", reason)
                     for f, xp, reason in invalid]
            pending = [f for f, _, _ in invalid]
        if pending:
            self._step("llm-generate:incomplete", False,
                       "no valid XPath for: " + ", ".join(f.name for f in pending))

    def _validate_and_store(self, fields, xpaths, purpose, dom_signature, log):
        start = time.perf_counter()
        source = RuleSource.HEURISTIC if log.provider == "heuristic" else RuleSource.LLM
        valid, invalid = {}, []
        for f in fields:
            xp = (xpaths.get(f.name) or "").strip()
            if len(xp) > 2000:
                invalid.append((f, xp[:2000], "xpath-too-long"))
                continue
            out = extract_one(self._tree, f.name, f.field_type, f.cardinality, xp)
            if out.success:
                self._store_rule(f, xp, source, _PURPOSE_TO_RULE[purpose], dom_signature)
                valid[f.name] = xp
            else:
                invalid.append((f, xp, out.reason or "no-rule"))
        log.validation_ms = int((time.perf_counter() - start) * 1000)
        log.xpaths_valid = len(valid)
        log.save(update_fields=["validation_ms", "xpaths_valid"])
        return valid, invalid

    def _store_rule(self, f, xpath, source, purpose, dom_signature) -> None:
        latest = f.rules.order_by("-version").first()
        f.rules.filter(is_active=True).update(is_active=False)
        XPathRule.objects.create(
            field=f, xpath=xpath, version=(latest.version + 1) if latest else 1,
            is_active=True, source=source, purpose=purpose,
            created_in_run=self._run, dom_signature=dom_signature,
        )

    # ------------------------------------------------------------------
    # Extraction, memory, persistence
    # ------------------------------------------------------------------
    def _extract(self, fields) -> list[FieldOutcome]:
        rules = {r.field_id: r for r in XPathRule.objects.filter(field__in=fields, is_active=True)}
        self._active_rules = rules
        return [
            extract_one(self._tree, f.name, f.field_type, f.cardinality,
                        rules[f.id].xpath if f.id in rules else "")
            for f in fields
        ]

    def _memory(self, target, failed: list[FieldOutcome]) -> list[FailureHint]:
        names = {o.field_name for o in failed}
        candidates = build_failure_hints(failed)
        recent = (FailureRecord.objects.filter(field__target=target, field__name__in=names)
                  .exclude(failed_xpath="").select_related("field")
                  .order_by("-created_at", "-id")[: max(1, self.memory_size) * 3])
        candidates += [FailureHint(r.field.name, r.failed_xpath, r.reason) for r in recent]
        seen, hints = set(), []
        for h in candidates:
            key = (h.field_name, h.failed_xpath)
            if not h.failed_xpath or key in seen:
                continue
            seen.add(key)
            hints.append(h)
            if len(hints) >= self.memory_size:
                break
        return hints

    def _record_failures(self, outcomes) -> None:
        for o in outcomes:
            if o.success or o.field_name not in self._fields:
                continue
            FailureRecord.objects.create(
                field=self._fields[o.field_name], run=self._run,
                failed_xpath=(o.xpath or "")[:2000], reason=o.reason or "extraction-failed",
            )

    def _persist(self, outcomes, phase) -> None:
        run = self._run
        run.records.filter(phase=phase).delete()
        rules = getattr(self, "_active_rules", {})
        for o in outcomes:
            f = self._fields.get(o.field_name)
            if f is None:
                continue
            rec = ExtractedRecord(
                run=run, field=f, phase=phase, xpath_used=rules.get(f.id),
                raw_value=o.raw_value, normalized_value=o.normalized_value,
                values=o.values, success=o.success, reason=o.reason,
            )
            gt = self._gt.get(f.name)
            if gt is not None:
                self._score(rec, o, f, gt.values, gt.display_value)
            elif f.ground_truth and run.experiment_id is None and not self._gt:
                # Legacy static label — only for ordinary live/demo runs whose
                # page has no per-snapshot labels at all (never mixed).
                self._score(rec, o, f, [f.ground_truth], f.ground_truth)
            rec.save()

    @staticmethod
    def _score(rec, o, f, gt_values, gt_text) -> None:
        rec.gt_value = gt_text
        rec.gt_available = bool(gt_values)
        if not gt_values:
            return
        if f.cardinality == "many":
            preds = o.values if o.success else []
            tp, fp, fn = list_match(preds, gt_values, f.field_type)
            rec.list_tp, rec.list_fp, rec.list_fn = tp, fp, fn
            rec.exact_match = fp == 0 and fn == 0
            prf = token_prf(" ".join(preds), " ".join(gt_values))
        else:
            # Several acceptable forms (alternatives): any exact match counts;
            # token F1 is the best over the forms.
            pred = o.raw_value if o.success else ""
            rec.exact_match = bool(o.success and any(
                exact_match(pred, g, f.field_type) for g in gt_values))
            scores = [p for p in (token_prf(pred, g) for g in gt_values) if p is not None]
            prf = max(scores, key=lambda t: t[2]) if scores else None
        rec.matches_ground_truth = rec.exact_match
        if prf is not None:
            rec.token_precision, rec.token_recall, rec.token_f1 = prf

    # ------------------------------------------------------------------
    def _cost(self, model, prompt_tokens, completion_tokens):
        price = self.pricing.get(model or "")
        if not price:
            return None
        return round(prompt_tokens * float(price.get("input", 0)) / 1e6
                     + completion_tokens * float(price.get("output", 0)) / 1e6, 8)

    def _apply_health(self, health) -> None:
        run = self._run
        run.health_score = health.score
        run.xpath_failure_rate = health.xpath_failure_rate
        run.field_count_mismatch = health.field_count_mismatch
        run.fields_total = health.fields_total
        run.fields_extracted = health.fields_extracted

    def _step(self, name, ok, detail) -> None:
        self._trace.append({"step": name, "ok": bool(ok), "detail": detail})

    def _finalize(self, t0) -> ScrapeRun:
        run = self._run
        run.finished_at = timezone.now()
        run.duration_ms = int((time.perf_counter() - t0) * 1000)
        run.completed = True
        run.log = self._trace
        run.save()
        return run
