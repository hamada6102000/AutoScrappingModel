"""
Evaluation protocol (proposal §4.1 and §5: 20+ sites, self-healing vs. a
static LLM scraper, layout-change events, recovery metrics).

Design — paired and cumulative, per site and repetition r:

  t0  baseline snapshot (human ground truth). The self_healing condition
      generates XPaths (with validation repair); the static condition gets an
      exact copy of those rules, so both start identically.
  t1..tK  a seeded sequence of simulated changes applied cumulatively
      (structural / content-only / control). Both conditions run on the same
      snapshot at every step. Only self_healing may call the LLM afterwards.

Optional natural chain (repetition 0): the baseline followed by pages re-fetched
later (``eval_snapshot --kind natural``), labelled by a human.

Each run executes in one DB transaction. If the LLM quota is exhausted the run
is rolled back and the runner stops; re-running resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import random
import sys

from django.conf import settings
from django.db import transaction

from ..llm.base import RateLimitExhausted
from ..llm.factory import get_provider
from ..models import (
    ChangeType, Experiment, GroundTruth, GTSource, LLMCallLog, RulePurpose,
    ScrapeRun, ScrapeTarget, Snapshot, SnapshotKind, StepKind, TargetField, XPathRule,
)
from ..services import snapshots as snapstore
from ..services.fetcher import fetch_raw, robots_allowed
from ..services.htmlutil import decode_html
from ..services.mutations import apply_mutation, family
from ..services.pipeline import RunContext, ScrapePipeline
from .config import site_by_key
from .environment import capture

BASELINE_LABEL = "baseline"
NATURAL_REP = 0


class EvalError(Exception):
    pass


class QuotaStop(Exception):
    pass


class BudgetStop(Exception):
    pass


# ---------------------------------------------------------------------------
# Experiment / target setup
# ---------------------------------------------------------------------------
def init_experiment(config: dict, *, notes: str = "", force: bool = False) -> Experiment:
    name = config["experiment"]["name"]
    exp = Experiment.objects.filter(name=name).first()
    if exp and exp.runs.exists():
        if not force:
            raise EvalError(
                f"experiment '{name}' already has {exp.runs.count()} runs. Use a new "
                "experiment name, or --force to delete its runs and start over."
            )
        exp.runs.all().delete()
        XPathRule.objects.filter(field__target__experiment=exp).delete()
    if exp is None:
        exp = Experiment.objects.create(name=name)
    exp.config = config
    exp.environment = capture()
    if notes:
        exp.notes = notes
    exp.save()

    reps = config["experiment"]["repetitions"]
    wanted = set()
    for site in config["sites"]:
        for cond in config["experiment"]["conditions"]:
            for rep in range(1, reps + 1):
                target = ensure_target(exp, site, cond, rep)
                wanted.add(target.pk)
    # Remove targets of sites/conditions no longer in the config (no runs yet).
    exp.targets.exclude(pk__in=wanted).exclude(repetition=NATURAL_REP).delete()
    return exp


def ensure_target(exp, site: dict, condition: str, rep: int) -> ScrapeTarget:
    target, _ = ScrapeTarget.objects.update_or_create(
        experiment=exp, site_key=site["key"], condition=condition, repetition=rep,
        defaults={
            "name": f"{site['name']} [{condition} r{rep}]",
            "url": site["url"] or "http://local.invalid/",
            "category": site["category"],
            "render": site["render"],
            "html_file": site["html_file"],
            "prune_keep": site["prune_keep"],
            "is_active": False,  # never scraped live by run_scrape --all
        },
    )
    names = []
    for order, f in enumerate(site["fields"], start=1):
        TargetField.objects.update_or_create(
            target=target, name=f["name"],
            defaults={
                "description": f["description"],
                "field_type": f["type"],
                "cardinality": f["cardinality"],
                "is_required": f["required"],
                "order": order,
            },
        )
        names.append(f["name"])
    target.fields.exclude(name__in=names).delete()
    return target


def field_types(site: dict) -> dict:
    return {f["name"]: f["type"] for f in site["fields"]}


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------
def capture_snapshot(exp, site: dict, *, kind: str = SnapshotKind.BASELINE,
                     label: str = BASELINE_LABEL, refresh: bool = False,
                     write=print) -> Snapshot:
    existing = Snapshot.objects.filter(experiment=exp, site_key=site["key"],
                                       kind=kind, label=label).first()
    if existing and not refresh:
        return existing
    if existing:
        if ScrapeRun.objects.filter(snapshot__in=[existing, *existing.children.all()]).exists():
            raise EvalError(f"{site['key']}: snapshot '{label}' already used by runs; cannot refresh")
        existing.delete()

    fetch_cfg = exp.config["experiment"].get("fetch") or {}
    ua = fetch_cfg.get("user_agent") or settings.SCRAPER["USER_AGENT"]
    timeout = int(fetch_cfg.get("timeout") or 30)
    robots = None
    if site["html_file"]:
        from pathlib import Path

        raw, enc = decode_html(Path(site["html_file"]).read_bytes())
        status, meta = 200, {"source": "html_file", "path": site["html_file"], "encoding": enc}
    else:
        if fetch_cfg.get("respect_robots", True):
            robots = robots_allowed(site["url"], ua, timeout)
            if robots is False:
                raise EvalError(
                    f"{site['key']}: robots.txt disallows {site['url']} for this user agent. "
                    "Save the page manually and set html_file, or document why you override it."
                )
        status, raw, meta = fetch_raw(site["url"], timeout, ua, site["render"])
        if meta.get("error") or not raw:
            raise EvalError(f"{site['key']}: fetch failed ({meta.get('error') or 'empty page'})")
    snap = snapstore.create_snapshot(
        html=raw, site_key=site["key"], kind=kind, experiment=exp, url=site["url"],
        http_status=status, label=label, fetch_meta=meta, robots_allowed=robots,
        keep_tags=site["prune_keep"], label_view=True,
    )
    if meta.get("suspected_block"):
        write(f"  WARNING {site['key']}: page looks like a bot-check/blocked page "
              f"(HTTP {status}). Open its labeling view; if so, save the page from your "
              "browser and use html_file.")
    if snap.llm_view_truncated:
        write(f"  NOTE {site['key']}: LLM view ({snap.llm_view_tokens} tokens) exceeds "
              "LLM_MAX_INPUT_CHARS and will be truncated in prompts.")
    return snap


def baseline_snapshot(exp, site_key: str):
    return Snapshot.objects.filter(experiment=exp, site_key=site_key,
                                   kind=SnapshotKind.BASELINE, label=BASELINE_LABEL).first()


def gt_status(snapshot, site: dict) -> tuple[list, list]:
    """(missing_fields, labelled_fields) for a snapshot."""
    labelled = set(snapshot.ground_truth.values_list("field_name", flat=True))
    names = [f["name"] for f in site["fields"]]
    return [n for n in names if n not in labelled], [n for n in names if n in labelled]


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def _seed(*parts) -> int:
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def build_schedule(exp_cfg: dict, site_key: str, rep: int) -> list[dict]:
    protocol = exp_cfg["protocol"]
    rng = random.Random(_seed(exp_cfg["seed"], site_key, rep))
    steps = [dict(s) for s in protocol["steps"]]
    if protocol.get("shuffle", True):
        rng.shuffle(steps)
    severities = protocol.get("severities") or [0.5]
    for s in steps:
        s.setdefault("focus", family(s["op"]) == "structural" and s["op"] != "insert_banner")
        if "severity" not in s:
            s["severity"] = rng.choice(severities)
        s["seed"] = rng.getrandbits(32)
    return steps


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class EvaluationRunner:
    def __init__(self, exp: Experiment, *, write=None, max_llm_calls: int | None = None,
                 sites: list[str] | None = None, provider=None):
        self.exp = exp
        self.cfg = exp.config
        self.ecfg = self.cfg["experiment"]
        self.write = write or (lambda msg: print(msg, file=sys.stdout))
        self.max_llm_calls = max_llm_calls
        self.calls_made = 0
        self.site_keys = sites or [s["key"] for s in self.cfg["sites"]]
        llm = self.ecfg["llm"]
        self.provider = provider or get_provider(llm["provider"], llm.get("model"))
        common = dict(
            provider=self.provider,
            allow_fallback=False,          # never mix heuristic into LLM results
            threshold=self.ecfg["health_threshold"],
            max_attempts=self.ecfg["max_heal_attempts"],
            memory_size=self.ecfg["failure_memory_size"],
            raise_on_quota=True,
            save_snapshots=False,
        )
        self.pipelines = {
            "self_healing": ScrapePipeline(heal_enabled=True, **common),
            "static": ScrapePipeline(heal_enabled=False, generate_missing=False, **common),
        }
        self.stats = {"runs": 0, "skipped": 0, "llm_calls": 0}

    # -- public -----------------------------------------------------------
    def run(self, natural: bool = False) -> dict:
        problems = self.readiness()
        if problems:
            raise EvalError("not ready:\n  - " + "\n  - ".join(problems))
        try:
            for key in self.site_keys:
                site = site_by_key(self.cfg, key)
                if natural:
                    self.run_natural(site)
                else:
                    for rep in range(1, self.ecfg["repetitions"] + 1):
                        self.run_site_rep(site, rep)
        except QuotaStop as exc:
            self.stats["stopped"] = f"LLM quota exhausted: {exc}"
        except BudgetStop as exc:
            self.stats["stopped"] = str(exc)
        return self.stats

    def readiness(self) -> list[str]:
        problems = []
        for key in self.site_keys:
            try:
                site = site_by_key(self.cfg, key)
            except KeyError:
                problems.append(f"unknown site '{key}'")
                continue
            snap = baseline_snapshot(self.exp, key)
            if snap is None:
                problems.append(f"{key}: no baseline snapshot (run eval_snapshot)")
                continue
            missing, _ = gt_status(snap, site)
            if missing:
                problems.append(f"{key}: ground truth missing for {', '.join(missing)} (run eval_gt_import)")
        return problems

    def run_site_rep(self, site: dict, rep: int) -> None:
        conditions = self.ecfg["conditions"]
        targets = {c: ensure_target(self.exp, site, c, rep) for c in conditions}
        base = baseline_snapshot(self.exp, site["key"])

        self._run_step(targets["self_healing"], base, 0, StepKind.BASELINE, None, {})
        if "static" in targets:
            self._copy_t0_rules(targets["self_healing"], targets["static"])
            self._run_step(targets["static"], base, 0, StepKind.BASELINE, None, {})

        parent = base
        for k, spec in enumerate(build_schedule(self.ecfg, site["key"], rep), start=1):
            snap = self._simulated_snapshot(site, rep, k, parent, spec)
            fam = family(spec["op"])
            kind = {"structural": StepKind.STRUCTURAL, "content": StepKind.CONTENT,
                    "control": StepKind.CONTROL}[fam]
            for cond in conditions:
                self._run_step(targets[cond], snap, k, kind, fam == "structural", snap.mutation)
            parent = snap

    def run_natural(self, site: dict) -> None:
        base = baseline_snapshot(self.exp, site["key"])
        naturals = list(Snapshot.objects.filter(
            experiment=self.exp, site_key=site["key"], kind=SnapshotKind.NATURAL
        ).order_by("fetched_at", "id"))
        if not naturals:
            self.write(f"  {site['key']}: no natural snapshots; skipped")
            return
        conditions = self.ecfg["conditions"]
        targets = {c: ensure_target(self.exp, site, c, NATURAL_REP) for c in conditions}
        self._run_step(targets["self_healing"], base, 0, StepKind.BASELINE, None, {})
        if "static" in targets:
            self._copy_t0_rules(targets["self_healing"], targets["static"])
            self._run_step(targets["static"], base, 0, StepKind.BASELINE, None, {})
        for k, snap in enumerate(naturals, start=1):
            missing, _ = gt_status(snap, site)
            if missing:
                self.write(f"  {site['key']} {snap.label}: ground truth missing for "
                           f"{', '.join(missing)}; natural chain stopped here")
                return
            changed = (snap.mutation or {}).get("layout_changed")
            for cond in conditions:
                self._run_step(targets[cond], snap, k, StepKind.NATURAL, changed,
                               {"op": "natural", "label": snap.label}, natural=True)

    # -- internals ---------------------------------------------------------
    def _copy_t0_rules(self, src: ScrapeTarget, dst: ScrapeTarget) -> None:
        if XPathRule.objects.filter(field__target=dst).exists():
            return
        t0 = src.runs.filter(step=0, completed=True).first()
        if t0 is None:
            return
        for f in src.fields.all():
            rule = (XPathRule.objects.filter(field=f, created_in_run=t0)
                    .order_by("-version").first())
            if rule is None:
                continue
            XPathRule.objects.create(
                field=dst.fields.get(name=f.name), xpath=rule.xpath, version=1,
                is_active=True, source=rule.source, purpose=RulePurpose.COPIED,
                copied_from=rule, dom_signature=rule.dom_signature,
            )

    def _simulated_snapshot(self, site: dict, rep: int, step: int, parent, spec: dict):
        existing = Snapshot.objects.filter(experiment=self.exp, site_key=site["key"],
                                           kind=SnapshotKind.SIMULATED, repetition=rep,
                                           step=step).first()
        if existing:
            return existing
        types = field_types(site)
        gt_rows = list(parent.ground_truth.all())
        gt_fields = {g.field_name: (g.values, types.get(g.field_name, "string")) for g in gt_rows}
        result = apply_mutation(snapstore.read_html(parent), spec["op"], spec["seed"],
                                float(spec["severity"]), bool(spec["focus"]), gt_fields)
        snap = snapstore.create_snapshot(
            html=result.html, site_key=site["key"], kind=SnapshotKind.SIMULATED,
            experiment=self.exp, url=parent.url, http_status=parent.http_status,
            repetition=rep, step=step, parent=parent, mutation=result.spec,
            keep_tags=site["prune_keep"],
        )
        preserved = result.spec.get("gt_preserved") or {}
        for g in gt_rows:
            new_value = result.gt_updates.get(g.field_name)
            notes = ""
            if preserved.get(g.field_name) is False:
                notes = "exclude: text not preserved by mutation"
            GroundTruth.objects.create(
                snapshot=snap, field_name=g.field_name, cardinality=g.cardinality,
                value=new_value if new_value is not None else g.value,
                source=GTSource.PERTURBED if new_value is not None else GTSource.DERIVED,
                verified_by=g.verified_by, notes=notes,
            )
        return snap

    def _run_step(self, target, snapshot, step, kind, true_change, mutation, natural=False):
        if ScrapeRun.objects.filter(target=target, step=step, completed=True).exists():
            self.stats["skipped"] += 1
            return
        ScrapeRun.objects.filter(target=target, step=step, completed=False).delete()
        if self.max_llm_calls is not None and self.calls_made >= self.max_llm_calls:
            raise BudgetStop(f"stopped after the --max-llm-calls budget ({self.max_llm_calls})")
        ctx = RunContext(
            experiment=self.exp, snapshot=snapshot, condition=target.condition,
            repetition=target.repetition, step=step, step_kind=kind,
            change_type=(ChangeType.NATURAL if natural else ChangeType.SIMULATED) if step else "",
            true_change=true_change, mutation=mutation or {},
        )
        try:
            with transaction.atomic():
                run = self.pipelines[target.condition].run(target, snapshot=snapshot, context=ctx)
        except RateLimitExhausted as exc:
            raise QuotaStop(str(exc)) from exc
        self.calls_made += run.llm_calls_count
        self.stats["runs"] += 1
        self.stats["llm_calls"] += run.llm_calls_count
        op = (mutation or {}).get("op", "baseline")
        pre = run.pre_heal_health if run.pre_heal_health is not None else run.health_score
        self.write(
            f"  {target.site_key:<18} r{target.repetition} s{step:02d} {target.condition:<12} "
            f"{op:<16} {run.status:<9} health {pre:.2f}->{run.health_score:.2f}  "
            f"LLM calls {run.llm_calls_count}"
        )


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def estimate(exp: Experiment) -> dict:
    """Rough upper/lower bounds on LLM calls and tokens before spending quota."""
    ecfg = exp.config["experiment"]
    reps, attempts = ecfg["repetitions"], ecfg["max_heal_attempts"]
    structural = sum(1 for s in ecfg["protocol"]["steps"] if family(s["op"]) == "structural")
    rows, totals = [], {"calls_min": 0, "calls_max": 0, "tokens_min": 0, "tokens_max": 0}
    for site in exp.config["sites"]:
        snap = baseline_snapshot(exp, site["key"])
        view = snap.llm_view_tokens if snap and snap.llm_view_tokens else None
        calls_min = reps
        calls_max = reps * ((1 + attempts) + structural * attempts)
        per_call = (view or 0) + 700  # prompt instructions + response
        row = {"site": site["key"], "llm_view_tokens": view, "calls_min": calls_min,
               "calls_max": calls_max,
               "tokens_min": calls_min * per_call if view else None,
               "tokens_max": calls_max * per_call if view else None}
        rows.append(row)
        totals["calls_min"] += calls_min
        totals["calls_max"] += calls_max
        if view:
            totals["tokens_min"] += row["tokens_min"]
            totals["tokens_max"] += row["tokens_max"]
    return {"sites": rows, "totals": totals}


def llm_calls_for(exp) -> int:
    return LLMCallLog.objects.filter(run__experiment=exp).count()
