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
import json
import random
import sys

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from ..llm.base import ProviderUnavailable, RateLimitExhausted
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
from .config import COND_INFO, CONDITION_ORDER, site_by_key
from .environment import capture

BASELINE_LABEL = "baseline"
NATURAL_REP = 0


class EvalError(Exception):
    pass


class QuotaStop(Exception):
    def __init__(self, message: str = "", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def configure(exp) -> None:
    """Apply experiment-wide settings that live outside the pipeline objects."""
    from ..services.normalizer import set_date_order

    set_date_order((exp.config.get("experiment") or {}).get("date_order") or "dmy")


LLM_OPTIONS = ("max_input_chars", "reasoning_effort", "include_reasoning", "tokens_per_minute",
               "completion_counts_in_tpm")


def llm_options(llm: dict) -> dict:
    return {k: llm[k] for k in LLM_OPTIONS if llm.get(k) is not None}


def make_provider(exp_cfg: dict):
    """The experiment's LLM provider with every option from its llm section."""
    llm = exp_cfg["llm"]
    return get_provider(llm["provider"], llm.get("model"), **llm_options(llm))


def request_plan(exp_cfg: dict, view: str, specs: list, failure_history=None) -> tuple[str, dict, int]:
    """(prompt, meta, max_completion_tokens) exactly as the provider will build
    one request — used to audit prompt sizes before spending quota."""
    from ..llm.prompts import build_prompt, completion_budget, prompt_token_budget

    llm = exp_cfg["llm"]
    if llm.get("provider") == "anthropic":
        from ..llm.anthropic_provider import completion_tokens_for

        max_tokens = completion_tokens_for(len(specs))
    else:
        max_tokens = completion_budget(len(specs), llm.get("reasoning_effort"))
    cap = int(llm.get("max_input_chars") or settings.SCRAPER.get("LLM_MAX_INPUT_CHARS", 60000))
    budget = prompt_token_budget(llm.get("tokens_per_minute"), max_tokens,
                                 llm.get("completion_counts_in_tpm", True))
    prompt, meta = build_prompt(view, specs, failure_history, cap, max_prompt_tokens=budget)
    return prompt, meta, max_tokens


def input_cap(exp) -> int:
    llm = exp.config["experiment"].get("llm") or {}
    return int(llm.get("max_input_chars") or settings.SCRAPER.get("LLM_MAX_INPUT_CHARS", 60000))


class BudgetStop(Exception):
    pass


class ProviderStop(Exception):
    def __init__(self, message: str = "", retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


# ---------------------------------------------------------------------------
# Experiment / target setup
# ---------------------------------------------------------------------------
def init_experiment(config: dict, *, notes: str = "", force: bool = False) -> Experiment:
    name = config["experiment"]["name"]
    exp = Experiment.objects.filter(name=name).first()
    if exp and exp.runs.exists() and not force:
        change = config_change(exp.config, config, exp)
        if change == "same":
            return exp                      # re-running eval_init is harmless
        if change not in ("llm-only", "additive"):
            raise EvalError(
                f"experiment '{name}' already has {exp.runs.count()} runs and the new config "
                f"changes {change}. Use a new experiment name, or --force to delete its runs "
                "and start over."
            )
        # Only LLM settings changed with no LLM-module run yet, and/or more
        # repetitions were added: completed runs are unaffected, so the update
        # is allowed; targets of new repetitions are created.
        exp.config = config
        exp.save(update_fields=["config"])
        if change == "additive":
            for site in config["sites"]:
                for cond in config["experiment"]["conditions"]:
                    for rep in range(1, config["experiment"]["repetitions"] + 1):
                        ensure_target(exp, site, cond, rep)
        return exp
    if exp and exp.runs.exists():
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


def config_change(old: dict, new: dict, exp=None) -> str:
    """'same'; 'llm-only' (only experiment.llm differs and no LLM-module run
    exists yet); 'additive' (repetitions increased, possibly with an llm-only
    change; existing runs stay valid); or a short description of what else
    differs (refused once runs exist)."""
    if old == new:
        return "same"
    o, n = json.loads(json.dumps(old)), json.loads(json.dumps(new))
    o_llm, n_llm = o["experiment"].pop("llm", None), n["experiment"].pop("llm", None)
    additive = False
    if o != n:
        diff = sorted(k for k in set(o["experiment"]) | set(n["experiment"])
                      if o["experiment"].get(k) != n["experiment"].get(k))
        if o.get("sites") != n.get("sites"):
            diff.append("sites/fields")
        more_reps = (diff == ["repetitions"]
                     and int(n["experiment"]["repetitions"]) > int(o["experiment"]["repetitions"]))
        if not more_reps:
            return ", ".join(diff) or "the configuration"
        additive = True
    if o_llm != n_llm:
        llm_runs = exp is not None and exp.runs.filter(condition__in=["self_healing", "static"]).exists()
        if llm_runs:
            return "llm settings (LLM-module runs already exist)"
    if additive:
        return "additive"
    return "same" if o_llm == n_llm else "llm-only"


def ensure_target(exp, site: dict, condition: str, rep: int) -> ScrapeTarget:
    target, _ = ScrapeTarget.objects.update_or_create(
        experiment=exp, site_key=site["key"], condition=condition, repetition=rep,
        defaults={
            "name": f"{site['name']} [{condition} r{rep}]",
            "url": site["url"] or "http://local.invalid/",
            "category": site["category"],
            "render": site["render"] if site["render"] in ("static", "js") else "static",
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
                     write=print, reuse: bool = False) -> Snapshot:
    """Fetch and save one site's page. ``reuse`` rebuilds the snapshot from
    the pages already downloaded by an earlier attempt (no new request)."""
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
        meta["check"] = assess_page(raw, site, site["url"])
    else:
        if fetch_cfg.get("respect_robots", True):
            robots = robots_allowed(site["url"], ua, timeout)
            if robots is False:
                raise EvalError(
                    f"{site['key']}: robots.txt disallows {site['url']} for this user agent. "
                    "Save the page manually and set html_file, or document why you override it."
                )
        status, raw, meta = _fetch_best(site, ua, timeout, exp, kind, label, write, reuse=reuse)
    snap = snapstore.create_snapshot(
        html=raw, site_key=site["key"], kind=kind, experiment=exp, url=site["url"],
        http_status=status, label=label, fetch_meta=meta, robots_allowed=robots,
        keep_tags=site["prune_keep"], label_view=True,
    )
    check = meta.get("check") or {}
    if meta.get("suspected_block"):
        write(f"  WARNING {site['key']}: page looks like a bot-check/blocked page "
              f"(HTTP {status}). Open its labeling view; if so, save the page from your "
              "browser and use html_file.")
    if check and not check.get("content_ok"):
        write(f"  WARNING {site['key']}: none of the expected texts {site['expect_text']} is on "
              "the saved page (consent wall, bot check or unrendered page?).")
    if snap.llm_view_truncated:
        write(f"  NOTE {site['key']}: LLM view ({snap.llm_view_tokens} tokens) exceeds "
              "LLM_MAX_INPUT_CHARS and will be truncated in prompts.")
    return snap


def assess_page(raw: str, site: dict, url: str) -> dict:
    """How much of the annotated dataset a fetched page contains: the expected
    texts (content check) and the recorded field values found literally."""
    from ..services.fetcher import prune
    from .dataset import PageIndex, locate

    page = prune(raw, site["prune_keep"])
    idx = PageIndex(page.clean_html, url)
    expect = site.get("expect_text") or []
    expect_found = [t for t in expect if idx.where(t, "string")]
    fields = list(site["fields"]) + list(site.get("excluded_fields") or [])
    checkable = found = 0
    for f in fields:
        if not f.get("dataset_value"):
            continue
        loc = locate(f, idx)
        if loc.status in ("not-literal", "boolean"):
            continue
        checkable += 1
        found += loc.status != "not-found"
    return {
        "content_ok": bool(expect_found) or not expect,
        "expect_found": len(expect_found), "expect_total": len(expect),
        "values_found": found, "values_checkable": checkable,
        "clean_chars": page.clean_chars, "llm_view_tokens": page.llm_view_tokens,
    }


def _fetched_paths(exp, site, kind, label, mode):
    rel = snapstore._rel(exp, site["key"], f"{kind}_{snapstore.slug(label)}.fetched-{mode}.html")
    path = snapstore.data_dir() / rel
    return path, path.with_suffix(".json")


def _fetch_best(site, ua, timeout, exp, kind, label, write, reuse=False):
    """Fetch with the site's render mode; with 'auto', fetch both the static
    HTML and the browser-rendered page and keep the one that contains more of
    the annotated values (the annotators read the rendered page). Every
    candidate's check is stored in fetch_meta; rejected HTML is kept on disk."""
    modes = ["static", "js"] if site["render"] == "auto" else [site["render"]]
    candidates, errors = [], {}
    for mode in modes:
        fetched = _fetched_paths(exp, site, kind, label, mode)
        if reuse and fetched[0].exists() and fetched[1].exists():
            raw = fetched[0].read_text(encoding="utf-8")
            status, meta = 200, json.loads(fetched[1].read_text(encoding="utf-8"))
            status = int(meta.pop("_http_status", 200))
            meta["reused_fetch"] = True
        else:
            status, raw, meta = fetch_raw(site["url"], timeout, ua, mode)
            if meta.get("error") or not raw:
                errors[mode] = meta.get("error") or "empty page"
                continue
            # Keep exactly what was downloaded before any processing, so a
            # processing problem never costs a second request to the site.
            fetched[0].parent.mkdir(parents=True, exist_ok=True)
            fetched[0].write_text(raw, encoding="utf-8", newline="")
            meta["fetched_at"] = timezone.now().isoformat()
            fetched[1].write_text(json.dumps({**meta, "_http_status": status}, indent=1), encoding="utf-8")
        meta["check"] = assess_page(raw, site, meta.get("final_url") or site["url"])
        candidates.append((mode, status, raw, meta))
    if not candidates:
        raise EvalError(f"{site['key']}: fetch failed ({'; '.join(f'{m}: {e}' for m, e in errors.items())})")

    def score(c):
        mode, status, _, meta = c
        chk = meta["check"]
        return (chk["content_ok"], not meta.get("suspected_block"), status < 400,
                chk["values_found"], mode == "static")

    candidates.sort(key=score, reverse=True)
    mode, status, raw, meta = candidates[0]
    meta["render_chosen"] = mode
    meta["render_candidates"] = {
        m: {"http_status": st, "suspected_block": bool(mt.get("suspected_block")), **mt["check"]}
        for m, st, _, mt in candidates
    }
    if errors:
        meta["render_errors"] = errors
    for m, _, other_raw, _ in candidates[1:]:
        snapstore.write_text(
            snapstore._rel(exp, site["key"], f"{kind}_{snapstore.slug(label)}.rejected-{m}.html"),
            other_raw,
        )
    if len(modes) > 1:
        summary = ", ".join(
            f"{m}: {v['values_found']}/{v['values_checkable']} values"
            + ("" if v["content_ok"] else " (content check failed)")
            for m, v in meta["render_candidates"].items())
        write(f"  {site['key']}: render auto -> {mode} ({summary}"
              + (f"; failed: {', '.join(errors)}" if errors else "") + ")")
    return status, raw, meta


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
                 sites: list[str] | None = None, provider=None, conditions: list[str] | None = None,
                 allow_unlabelled: bool = False, skip_unavailable: bool = False,
                 reps: list[int] | None = None):
        self.exp = exp
        self.cfg = exp.config
        self.ecfg = self.cfg["experiment"]
        self.write = write or (lambda msg: print(msg, file=sys.stdout))
        self.max_llm_calls = max_llm_calls
        self.calls_made = 0
        self.allow_unlabelled = allow_unlabelled
        all_reps = list(range(1, self.ecfg["repetitions"] + 1))
        bad = [r for r in (reps or []) if r not in all_reps]
        if bad:
            raise EvalError(f"repetitions {bad} are not in the experiment (1..{self.ecfg['repetitions']})")
        self.reps = sorted(reps) if reps else all_reps
        self.site_keys = sites or [s["key"] for s in self.cfg["sites"]]
        self.skipped_sites = []
        if skip_unavailable:
            self.skipped_sites = [k for k in self.site_keys if baseline_snapshot(exp, k) is None]
            self.site_keys = [k for k in self.site_keys if k not in self.skipped_sites]
        configured = self.ecfg["conditions"]
        wanted = conditions or configured
        unknown = [c for c in wanted if c not in configured]
        if unknown:
            raise EvalError(f"conditions {unknown} are not in the experiment ({', '.join(configured)})")
        self.conditions = [c for c in CONDITION_ORDER if c in wanted]
        configure(exp)
        # The LLM is only needed when the LLM-based module generates or heals.
        self.provider = None
        if "self_healing" in self.conditions:
            self.provider = provider or make_provider(self.ecfg)
        from ..llm.rule_based import RuleBasedProvider

        rules = RuleBasedProvider()
        common = dict(
            allow_fallback=False,          # never mix providers within a condition
            threshold=self.ecfg["health_threshold"],
            max_attempts=self.ecfg["max_heal_attempts"],
            memory_size=self.ecfg["failure_memory_size"],
            raise_on_quota=True,
            save_snapshots=False,
        )
        self.pipelines = {}
        for cond in self.conditions:
            info = COND_INFO[cond]
            prov = rules if info["generator"] == "rule_based" else (self.provider or rules)
            self.pipelines[cond] = ScrapePipeline(
                provider=prov, heal_enabled=info["heal"],
                generate_missing=info["copy_of"] is None, **common)
        self.stats = {"runs": 0, "skipped": 0, "llm_calls": 0, "rule_calls": 0, "waiting": [],
                      "sites_without_snapshot": self.skipped_sites}
        # Recorded LLM-scope decision: LLM-based modules only on these sites.
        from . import llm_scope

        scope = llm_scope.load(exp)
        self.llm_sites = set(scope["llm_sites"]) if scope else None
        if self.llm_sites is not None:
            outside = [k for k in self.site_keys if k not in self.llm_sites]
            llm_conds = [c for c in self.conditions if COND_INFO[c]["generator"] == "llm"]
            if outside and llm_conds:
                self.write(f"  LLM scope: {', '.join(llm_conds)} run on {', '.join(sorted(self.llm_sites))} "
                           f"only; not run (no LLM call) on {len(outside)} other site(s).")

    def conditions_for(self, site_key: str) -> list[str]:
        """The selected conditions that may run on this site (the recorded
        LLM scope keeps LLM-based modules off the other sites)."""
        if self.llm_sites is None or site_key in self.llm_sites:
            return list(self.conditions)
        return [c for c in self.conditions if COND_INFO[c]["generator"] != "llm"]

    # -- public -----------------------------------------------------------
    def run(self, natural: bool = False) -> dict:
        problems = self.readiness()
        if problems:
            raise EvalError("not ready:\n  - " + "\n  - ".join(problems))
        try:
            if natural:
                for key in self.site_keys:
                    self.run_natural(site_by_key(self.cfg, key))
            else:
                # Repetition-major: every site finishes repetition 1 before any
                # site starts repetition 2, so a run stopped by quota leaves
                # complete repetitions behind.
                for rep in self.reps:
                    for key in self.site_keys:
                        self.run_site_rep(site_by_key(self.cfg, key), rep)
        except QuotaStop as exc:
            self.stats["stopped"] = f"LLM quota exhausted: {exc}"
            self.stats["retry_after"] = exc.retry_after
        except BudgetStop as exc:
            self.stats["stopped"] = str(exc)
        except ProviderStop as exc:
            self.stats["stopped"] = f"LLM provider error: {exc}"
            self.stats["retryable"] = exc.retryable
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
            if missing and not self.allow_unlabelled:
                problems.append(f"{key}: ground truth missing for {', '.join(missing)} (run eval_gt_import)")
        return problems

    def _start(self, site: dict, rep: int, base, natural: bool = False) -> dict:
        """t0 for every selected condition: generators first, then the static
        conditions start from an exact copy of their generator's t0 rules."""
        conds = self.conditions_for(site["key"])
        targets = {c: ensure_target(self.exp, site, c, rep) for c in conds}
        for cond in conds:
            if COND_INFO[cond]["copy_of"] is None:
                self._run_step(targets[cond], base, 0, StepKind.BASELINE, None, {})
        for cond in conds:
            src = COND_INFO[cond]["copy_of"]
            if src is None:
                continue
            src_target = targets.get(src) or ensure_target(self.exp, site, src, rep)
            if not src_target.runs.filter(step=0, completed=True).exists():
                note = f"{site['key']} r{rep}: {cond} waits for the t0 selectors of {src}"
                self.stats["waiting"].append(note)
                self.write(f"  {note}")
                targets.pop(cond)
                continue
            self._copy_t0_rules(src_target, targets[cond])
            self._run_step(targets[cond], base, 0, StepKind.BASELINE, None, {})
        return targets

    def run_site_rep(self, site: dict, rep: int) -> None:
        base = baseline_snapshot(self.exp, site["key"])
        targets = self._start(site, rep, base)
        if not targets:
            return
        parent = base
        for k, spec in enumerate(build_schedule(self.ecfg, site["key"], rep), start=1):
            snap = self._simulated_snapshot(site, rep, k, parent, spec)
            fam = family(spec["op"])
            kind = {"structural": StepKind.STRUCTURAL, "content": StepKind.CONTENT,
                    "control": StepKind.CONTROL}[fam]
            for cond, target in targets.items():
                self._run_step(target, snap, k, kind, fam == "structural", snap.mutation)
            parent = snap

    def run_natural(self, site: dict) -> None:
        base = baseline_snapshot(self.exp, site["key"])
        naturals = list(Snapshot.objects.filter(
            experiment=self.exp, site_key=site["key"], kind=SnapshotKind.NATURAL
        ).order_by("fetched_at", "id"))
        if not naturals:
            self.write(f"  {site['key']}: no natural snapshots; skipped")
            return
        targets = self._start(site, NATURAL_REP, base, natural=True)
        for k, snap in enumerate(naturals, start=1):
            missing, _ = gt_status(snap, site)
            if missing and not self.allow_unlabelled:
                self.write(f"  {site['key']} {snap.label}: ground truth missing for "
                           f"{', '.join(missing)}; natural chain stopped here")
                return
            changed = (snap.mutation or {}).get("layout_changed")
            for cond, target in targets.items():
                self._run_step(target, snap, k, StepKind.NATURAL, changed,
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
            if (g.notes or "").startswith("exclude"):
                notes = g.notes  # excluded at the parent stays excluded
            elif preserved.get(g.field_name) is False:
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
            raise QuotaStop(str(exc), getattr(exc, "retry_after", None)) from exc
        except ProviderUnavailable as exc:
            raise ProviderStop(str(exc), exc.retryable) from exc
        uses_llm = COND_INFO[target.condition]["generator"] == "llm"
        if uses_llm:
            self.calls_made += run.llm_calls_count
            self.stats["llm_calls"] += run.llm_calls_count
        else:
            self.stats["rule_calls"] += run.llm_calls_count
        self.stats["runs"] += 1
        op = (mutation or {}).get("op", "baseline")
        pre = run.pre_heal_health if run.pre_heal_health is not None else run.health_score
        self.write(
            f"  {target.site_key:<10} r{target.repetition} s{step:02d} {target.condition:<12} "
            f"{op:<16} {run.status:<9} health {pre:.2f}->{run.health_score:.2f}  "
            f"{'LLM' if uses_llm else 'rule'} calls {run.llm_calls_count}"
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
    from ..llm.base import FieldSpec
    from ..llm.prompts import SYSTEM_PROMPT
    from ..services import tokens as tok

    for site in exp.config["sites"]:
        snap = baseline_snapshot(exp, site["key"])
        view = None
        out_tokens = 0
        if snap is not None:
            page = snapstore.pruned_page(snap, site["prune_keep"])
            specs = [FieldSpec(f["name"], f["description"], f["type"], f["cardinality"])
                     for f in site["fields"]]
            prompt, _, out_tokens = request_plan(ecfg, page.llm_view, specs)
            view = tok.count_tokens(SYSTEM_PROMPT + prompt)
        calls_min = reps
        calls_max = reps * ((1 + attempts) + structural * attempts)
        # planned prompt + the completion budget (an upper bound on output)
        per_call = (view or 0) + out_tokens
        row = {"site": site["key"], "llm_view_tokens": view, "per_call_tokens": per_call if view else None,
               "calls_min": calls_min,
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
