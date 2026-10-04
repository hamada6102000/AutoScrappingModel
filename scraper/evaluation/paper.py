"""
Experimental-setup and results text for the paper, generated from the report
(``eval_paper``). Every number is read from summary.json; statements that
compare modules are derived from those numbers (e.g. which module has the
highest F1), never written in advance. Whatever was not run is said to be not
run.
"""

from __future__ import annotations

from .config import COND_INFO


def _p(v, digits=1):
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


def _n(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _tex(v, digits=1):
    return "--" if v is None else f"{v * 100:.{digits}f}"


def scope_sentence(rep) -> str:
    c = rep["counts"]
    run = rep.get("conditions_run") or []
    planned = c.get("planned_repetitions")
    reps = c.get("repetitions_completed")
    mods = ", ".join(COND_INFO[x]["label"] for x in run) or "no module"
    evaluated = c.get("sites_evaluated", c["sites"])
    s = (f"The results below cover {evaluated} of the {c['sites']} configured website(s) "
         f"({c.get('fields', 'n/a')} target fields configured), "
         f"{reps} repetition(s) of the change sequence"
         + (f" ({planned} were planned)" if planned and planned != reps else "")
         + f", and the following module(s): {mods}. ")
    missing = [COND_INFO[x]["label"] for x in ("static", "rule_static", "rule_based", "self_healing")
               if x in rep["config"]["conditions"] and x not in run]
    if missing:
        s += ("Not run in this build of the results: " + ", ".join(missing) + ". ")
    mreps = rep.get("module_repetitions") or {}
    if mreps and len({tuple(v) for v in mreps.values()}) > 1:
        s += "Completed repetitions per module: " + "; ".join(
            f"{COND_INFO[k]['label']} {', '.join(str(x) for x in v) or 'none'}" for k, v in mreps.items()) + ". "
    ch = rep.get("comparison_chains")
    if ch is not None:
        s += (f"Modules are compared on the {len(ch)} site-repetition chains completed by every module "
              f"({len({x[0] for x in ch})} sites, repetitions "
              f"{', '.join(str(r) for r in sorted({x[1] for x in ch}))}). ")
    src = c.get("baseline_labels_by_source") or {}
    if src:
        s += ("Ground-truth labels at t0: " + ", ".join(f"{v} {k}" for k, v in sorted(src.items()))
              + f"; field decisions without a label are not scored ({c.get('unscored_decisions', 0)}). ")
    return s


def best(rep, key) -> tuple[str, float] | None:
    vals = [(m["module"], m[key]) for m in rep.get("modules") or [] if m.get(key) is not None]
    return max(vals, key=lambda t: t[1]) if vals else None


def results_markdown(rep: dict) -> str:
    cfg = rep["config"]
    L = ["# V. EXPERIMENTAL RESULTS (generated from the evaluation data)", "",
         "## A. Setup", ""]
    L.append(scope_sentence(rep))
    L.append("")
    L.append(f"Each repetition applies {len(cfg['protocol']['steps'])} cumulative, seeded changes to the "
             "saved page (class renaming, wrapper insertion, tag changes, a banner, a content-value change, "
             "a no-change control, a combined redesign and an inserted sibling block); every module runs on "
             f"the same pages. Self-healing is triggered when the health score falls below "
             f"{cfg['health_threshold']}; at most {cfg['max_heal_attempts']} regeneration attempts are made, "
             f"with a failure memory of {cfg['failure_memory_size']} selectors. LLM: {cfg['llm']['provider']} / "
             f"{cfg['llm'].get('model')}"
             + (f", reasoning effort {cfg['llm']['reasoning_effort']}" if cfg['llm'].get('reasoning_effort') else "")
             + ". The rule-based module uses no LLM: value anchoring, label anchoring, semantic attributes, "
             "type conventions and structured metadata.")
    L += ["", "## B. Module comparison", ""]
    mods = rep.get("modules") or []
    if not mods:
        L.append("No module has been run yet.")
    else:
        L.append("| Module | Acc. (t0) | F1 (t0) | Acc. (changed) | Precision (changed) | Recall (changed) | "
                 "F1 (changed) | Alarm FPR | Healing success | Median TTR (s) | Median run latency (ms) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for m in mods:
            L.append(f"| {m['module']} | {_p(m['initial_accuracy'])} | {_p(m['initial_f1'])} | "
                     f"{_p(m['changed_accuracy'])} | {_p(m['changed_precision'])} | {_p(m['changed_recall'])} | "
                     f"{_p(m['changed_f1'])} | {_p(m['alarm_false_positive_rate'])} | "
                     f"{_p(m['healing_success_rate'])} | {_n(m['ttr_median_s'])} | "
                     f"{_n(m['run_latency_median_ms'], 0)} |")
        L.append("")
        top = best(rep, "changed_f1")
        if top:
            L.append(f"Under layout changes the highest F1 was obtained by the {top[0]} module "
                     f"({_p(top[1])}).")
        for m in mods:
            if m["initial_f1"] is not None and m["changed_f1"] is not None:
                delta = m["changed_f1"] - m["initial_f1"]
                L.append(f"- {m['module']}: F1 {_p(m['initial_f1'])} on the original pages and "
                         f"{_p(m['changed_f1'])} after the changes ({'+' if delta >= 0 else ''}{delta * 100:.1f} points); "
                         f"{m['scored_decisions']} scored field decisions.")
        tests = rep.get("paired_tests") or []
        if tests:
            L += ["", "Paired per-site comparison of sustained F1:", ""]
            for t in tests:
                L.append(f"- {COND_INFO[t['a']]['label']} vs {COND_INFO[t['b']]['label']}: mean difference "
                         f"{_p(t['mean_diff'])} over {t['sites']} site(s), wins/ties/losses "
                         f"{t['wins']}/{t['ties']}/{t['losses']}, sign-test p = {_n(t['sign_test_p'], 3)}"
                         + (f", Wilcoxon p = {_n(t['wilcoxon_p'], 3)}" if t.get('wilcoxon_p') is not None else "")
                         + ".")
    L += ["", "## C. Self-healing", ""]
    heal_mods = [m for m in mods if COND_INFO[m["condition"]]["heal"]]
    if not heal_mods:
        L.append("No self-healing module has been run yet.")
    for m in heal_mods:
        L.append(f"- {m['module']}: {m['breaking_events']} breaking changes (a field correct at the "
                 f"previous step became wrong), {m['detected_breaking_events']} of them detected by the "
                 f"health monitor; {m['healed_events_gt']} detected events fully repaired (GT-verified "
                 f"success {_p(m['healing_success_rate'])} of detected, "
                 f"{_p(m['healing_success_all_breaking'])} of all breaking changes); field-level recovery "
                 f"{_p(m['field_recovery_rate'])}; median time-to-recovery {_n(m['ttr_median_s'])} s; "
                 f"median regeneration latency {_n(m['regeneration_latency_median_s'])} s.")
    L += ["", "## D. Failure detection", ""]
    for m in mods:
        L.append(f"- {m['module']}: degraded monitoring runs {_p(m['degraded_run_share'])} (a field correct "
                 f"at t0 wrong before healing); alarm false-positive rate {_p(m['alarm_false_positive_rate'])}; "
                 f"change-detection recall {_p(m['change_detection_recall'])}; silent failures (degraded, "
                 f"no alarm) {m['silent_failures'] if m['silent_failures'] is not None else 'n/a'}.")
    tk = rep.get("tokens") or {}
    if tk.get("sites"):
        L += ["", "## E. Pruning", ""]
        L.append(f"Across {tk['sites']} saved page(s) the LLM view has a median of "
                 f"{_n(tk['llm_view_tokens']['median'], 0)} tokens versus {_n(tk['raw_tokens']['median'], 0)} "
                 f"for the raw HTML (mean reduction {_p(tk['reduction']['mean'])}; "
                 f"{tk['sites_at_or_above_90pct']} of {tk['sites']} pages reach >= 90%).")
    figs = {k: v for k, v in (rep.get("figures") or {}).items() if str(v).endswith(".png")}
    if figs:
        L += ["", "## Figures", ""]
        for k, v in figs.items():
            L.append(f"- {k}: `figures/{k}.png`")
    return "\n".join(L) + "\n"


def results_latex(rep: dict) -> str:
    mods = rep.get("modules") or []
    rows = "\n".join(
        f"{m['module']} & {_tex(m['initial_f1'])} & {_tex(m['changed_precision'])} & {_tex(m['changed_recall'])} "
        f"& {_tex(m['changed_f1'])} & {_tex(m['changed_accuracy'])} & {_tex(m['alarm_false_positive_rate'])} "
        f"& {_tex(m['healing_success_rate'])} & {_n(m['ttr_median_s']) if m['ttr_median_s'] is not None else '--'} \\\\"
        for m in mods)
    return (
        "% Generated by eval_paper from summary.json - do not edit numbers by hand.\n"
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Comparison of the scraping modules (field-level, \\%)}\n\\label{tab:modules}\n"
        "\\begin{tabular}{lrrrrrrrr}\n\\hline\n"
        "Module & F1 $t_0$ & P & R & F1 & Acc. & Alarm FPR & Heal. succ. & TTR (s) \\\\\n\\hline\n"
        f"{rows}\n\\hline\n\\end{{tabular}}\n\\end{{table*}}\n"
    )
