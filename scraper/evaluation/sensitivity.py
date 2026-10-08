"""
Post hoc sensitivity analysis.

The main metrics are recomputed without the sites whose saved page lacks a
large share of the values the annotator recorded. The rule uses only the
configuration audit (saved page versus annotated dataset), which is fixed
before any LLM call and does not depend on any module's result:

    share(site) = evaluated fields whose recorded value does not occur in the
                  saved HTML (audit status "not-found") / evaluated fields
    a site is set aside when share(site) >= threshold (default 0.25)

The analysis is reported next to the primary results on all sites, never
instead of them.
"""

from __future__ import annotations

from . import report as R
from .audit import run_audit

DEFAULT_THRESHOLD = 0.25
RULE = ("share of a site's evaluated fields whose recorded value does not occur in the saved HTML "
        "(audit status 'not-found') is at least the threshold")


def not_found_shares(exp, audit_rows: list[dict] | None = None) -> dict[str, dict]:
    rows = audit_rows if audit_rows is not None else run_audit(exp)
    out = {}
    for row in rows:
        fields = [f for f in row["fields"] if not f.get("excluded")]
        nf = sum(1 for f in fields if f.get("status") == "not-found")
        out[row["site"]] = {"not_found": nf, "fields": len(fields),
                            "share": nf / len(fields) if fields else 0.0}
    return out


def set_aside(shares: dict[str, dict], threshold: float) -> list[str]:
    return sorted(s for s, v in shares.items() if v["share"] >= threshold)


def summarize(data: dict, conds: list[str], own: dict) -> dict:
    """Module rows, paired tests, per-site sustained F1 and the target metrics
    of LLM-based self-healing on the given view of the data."""
    modules = []
    for c in conds:
        rows = R.module_rows(R.restrict(data, own[c]), [c])
        if rows:
            rows[0]["chains"] = len(own[c])
            modules.append(rows[0])
    sites = R.per_site(data, conds)
    tests = R.paired_tests(data, conds, own)
    sh = next((m for m in modules if m["condition"] == "self_healing"), None)
    sustained_80 = {c: sum(1 for r in sites if (r.get(f"{c}_sustained_f1") or 0) >= 0.8) for c in conds}
    targets = None
    if sh:
        targets = {
            "initial_f1": sh["initial_f1"],
            "healing_success_rate_gt": sh["healing_success_rate"],
            "time_to_recovery_median_s": sh["ttr_median_s"],
            "regeneration_latency_median_s": sh["regeneration_latency_median_s"],
            "sustained_f1": sh["overall_f1"],
            "false_alarms_per_monitoring_run": sh["false_alarms_per_monitoring_run"],
        }
    return {"sites": sorted({r["site"] for r in sites}), "modules": modules, "paired_tests": tests,
            "sites_sustained_f1_ge_0_80": sustained_80, "targets": targets}


def build(exp, threshold: float = DEFAULT_THRESHOLD, audit_rows: list[dict] | None = None) -> dict:
    shares = not_found_shares(exp, audit_rows)
    aside = set_aside(shares, threshold)
    data = R.collect(exp)
    R.label_breaking(data)
    ecfg = exp.config["experiment"]
    conds = R.present_conditions(data, ecfg)
    n_steps = len(ecfg["protocol"]["steps"])
    own = {c: R.complete_chains(data, c, n_steps) for c in conds}
    data = R.restrict_own(data, own)
    full = summarize(data, conds, own)
    kept_own = {c: {ch for ch in own[c] if ch[0] not in aside} for c in conds}
    kept = set().union(*kept_own.values()) if kept_own else set()
    view = R.restrict(data, kept)
    view["site_fields"] = {s: f for s, f in data["site_fields"].items() if s not in aside}
    view["baselines"] = [b for b in data["baselines"] if b["site"] not in aside]
    subset = summarize(view, conds, kept_own)
    return {
        "experiment": exp.name,
        "kind": "post hoc sensitivity analysis (primary results: all sites)",
        "rule": RULE,
        "threshold": threshold,
        "site_shares": shares,
        "sites_set_aside": aside,
        "all_sites": full,
        "without_set_aside": subset,
    }


def _p(v):
    return "n/a" if v is None else ("< 0.0001" if v < 0.0001 else f"{v:.4f}")


def _pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def markdown(rep: dict) -> str:
    full, sub = rep["all_sites"], rep["without_set_aside"]
    n_all, n_sub = len(full["sites"]), len(sub["sites"])
    lines = [
        f"# Sensitivity analysis — experiment `{rep['experiment']}`",
        "",
        "**Post hoc analysis, reported next to the primary results on all sites, never instead of them.**",
        "",
        f"Rule: a site is set aside when the {rep['rule']} (threshold {rep['threshold']:.2f}). The audit "
        "compares the saved page with the annotated dataset; it is fixed before any LLM call and does not "
        "depend on any module's result.",
        "",
        "| Site | Fields | Not found | Share | Set aside |",
        "|---|---|---|---|---|",
    ]
    for site, v in sorted(rep["site_shares"].items()):
        lines.append(f"| {site} | {v['fields']} | {v['not_found']} | {_pct(v['share'])} | "
                     f"{'yes' if site in rep['sites_set_aside'] else ''} |")
    lines += ["", f"Set aside: {', '.join(rep['sites_set_aside']) or 'none'}.", "",
              f"## Modules ({n_all} sites vs {n_sub} sites)", "",
              "| Module | F1 t0 | F1 changed | Sustained F1 | Healing success | Alarm FPR | False alarms / run |",
              "|---|---|---|---|---|---|---|"]
    by_sub = {m["condition"]: m for m in sub["modules"]}
    for m in full["modules"]:
        s = by_sub.get(m["condition"], {})
        lines.append(
            f"| {m['module']} | {_pct(m['initial_f1'])} → {_pct(s.get('initial_f1'))} "
            f"| {_pct(m['changed_f1'])} → {_pct(s.get('changed_f1'))} "
            f"| {_pct(m['overall_f1'])} → {_pct(s.get('overall_f1'))} "
            f"| {_pct(m['healing_success_rate'])} → {_pct(s.get('healing_success_rate'))} "
            f"| {_pct(m['alarm_false_positive_rate'])} → {_pct(s.get('alarm_false_positive_rate'))} "
            f"| {_pct(m['false_alarms_per_monitoring_run'])} → {_pct(s.get('false_alarms_per_monitoring_run'))} |")
    lines += ["", "## Paired per-site tests of sustained F1", "",
              "| A vs B | Sites | Mean diff | W/T/L | Sign p | Wilcoxon p |", "|---|---|---|---|---|---|"]
    for label, view in ((f"{n_all} sites", full), (f"{n_sub} sites", sub)):
        for t in view["paired_tests"]:
            lines.append(f"| {t['a']} vs {t['b']} ({label}) | {t['sites']} | {_pct(t['mean_diff'])} | "
                         f"{t['wins']}/{t['ties']}/{t['losses']} | {_p(t['sign_test_p'])} | "
                         f"{_p(t['wilcoxon_p'])} |")
    lines += ["", "## Target metrics, LLM-based self-healing", "",
              "| Metric | All sites | Without set-aside sites |", "|---|---|---|"]
    if full["targets"]:
        for k, v in full["targets"].items():
            w = sub["targets"][k] if sub["targets"] else None
            fmt = (lambda x: "n/a" if x is None else f"{x:.2f} s") if k.endswith("_s") else _pct
            lines.append(f"| {k} | {fmt(v)} | {fmt(w)} |")
    lines += ["", "Sites with sustained F1 >= 80%: "
              + ", ".join(f"{c} {full['sites_sustained_f1_ge_0_80'][c]}/{n_all} → "
                          f"{sub['sites_sustained_f1_ge_0_80'].get(c, 0)}/{n_sub}" for c in full["sites_sustained_f1_ge_0_80"]),
              ""]
    return "\n".join(lines) + "\n"
