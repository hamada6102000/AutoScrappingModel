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
    s += scope_text(rep)
    src = c.get("baseline_labels_by_source") or {}
    if src:
        s += ("Ground-truth labels at t0: " + ", ".join(f"{v} {k}" for k, v in sorted(src.items()))
              + f"; field decisions without a label are not scored ({c.get('unscored_decisions', 0)}). ")
    return s


def _sites(keys) -> str:
    keys = list(keys)
    return " and ".join(keys) if len(keys) <= 2 else ", ".join(keys[:-1]) + " and " + keys[-1]


def scope_text(rep) -> str:
    """Where each module was executed, stated explicitly."""
    scope = rep.get("llm_scope") or {}
    ch = rep.get("comparison_chains")
    mods = rep.get("modules") or []
    if not scope.get("reduced"):
        if ch is None:
            return ""
        return (f"Modules are compared on the {len(ch)} site-repetition chains completed by every module "
                f"({len({x[0] for x in ch})} sites, repetitions "
                f"{', '.join(str(r) for r in sorted({x[1] for x in ch}))}). ")
    llm = [m for m in mods if COND_INFO[m["condition"]]["generator"] == "llm"]
    rule = [m for m in mods if COND_INFO[m["condition"]]["generator"] != "llm"]
    reps = sorted({r for m in llm for r in m.get("repetitions") or []})
    t = (f"The LLM-based modules ({', '.join(m['module'] for m in llm)}) were executed on "
         f"{len(scope['llm_sites'])} of the {scope['configured_sites']} websites ({_sites(scope['llm_sites'])}; "
         f"repetition {', '.join(str(r) for r in reps)})")
    t += f", {scope['reason']}" if scope.get("reason") else ""
    repl = rep.get("llm_replication")
    if repl:
        n = len(repl["replicated_sites"])
        t += (f". The values measured on {'this website' if len(repl['source_sites']) == 1 else 'these websites'} "
              f"were replicated to the remaining {n} websites: each of them is assigned the LLM-module values "
              f"measured on {_sites(repl['source_sites'])} (pooled over their field decisions), so the LLM rows "
              f"over all {scope['configured_sites']} websites equal the measured "
              f"{len(repl['source_sites'])}-website result. The replicated values are not measurements: no "
              "LLM call was made for those websites, and they are excluded from all statistical tests. "
              f"The rule-based modules were executed and measured on {len(scope['rule_sites'])} websites")
        if rule:
            rr = sorted({r for m in rule for r in m.get("repetitions") or []})
            t += f" (repetitions {', '.join(str(r) for r in rr)})"
        t += ". "
        if ch:
            t += (f"For a like-for-like comparison, all modules are also reported on the {len(ch)} "
                  f"site-repetition chain(s) every module completed ({_sites(sorted({x[0] for x in ch}))}). ")
        return t
    t += (". The result measured on these websites is reported as the LLM-module result next to the "
          f"rule-based modules, which were executed on {len(scope['rule_sites'])} websites")
    if rule:
        rr = sorted({r for m in rule for r in m.get("repetitions") or []})
        t += f" (repetitions {', '.join(str(r) for r in rr)})"
    t += (". The LLM figures are therefore measurements on "
          f"{len(scope['llm_sites'])} website(s), not on the other "
          f"{scope['configured_sites'] - len(scope['llm_sites'])}; no LLM output exists for those websites and "
          "none was copied to them. ")
    if ch:
        t += (f"For a like-for-like comparison, all modules are also reported on the {len(ch)} "
              f"site-repetition chain(s) every module completed ({_sites(sorted({x[0] for x in ch}))}). ")
    return t


def best(rows, key) -> tuple[str, float] | None:
    vals = [(m["module"], m[key]) for m in rows or [] if m.get(key) is not None]
    return max(vals, key=lambda t: t[1]) if vals else None


def _module_table(L, rows):
    L.append("| Module | Sites | Chains | Acc. (t0) | F1 (t0) | Acc. (changed) | Precision (changed) | "
             "Recall (changed) | F1 (changed) | Alarm FPR | Healing success | Median TTR (s) | "
             "Median run latency (ms) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in rows:
        sites = (f"{m['sites_measured']} measured + {m['sites_replicated']} replicated"
                 if m.get("sites_replicated") else str(m["sites"]))
        L.append(f"| {m['module']} | {sites} | {m.get('chains', 'n/a')} | {_p(m['initial_accuracy'])} | "
                 f"{_p(m['initial_f1'])} | {_p(m['changed_accuracy'])} | {_p(m['changed_precision'])} | "
                 f"{_p(m['changed_recall'])} | {_p(m['changed_f1'])} | {_p(m['alarm_false_positive_rate'])} | "
                 f"{_p(m['healing_success_rate'])} | {_n(m['ttr_median_s'])} | "
                 f"{_n(m['run_latency_median_ms'], 0)} |")


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
    like = rep.get("modules_like_for_like") or []
    reduced = (rep.get("llm_scope") or {}).get("reduced")
    if not mods:
        L.append("No module has been run yet.")
    else:
        repl = rep.get("llm_replication")
        if repl:
            L.append("Sites = websites with measured values + websites with replicated LLM values; Chains = "
                     "measured site-repetition chains. " + (rep.get("llm_scope") or {}).get("note", ""))
            L.append("")
        elif reduced:
            L.append("Each module on the websites it was executed on (Sites, Chains = site-repetition "
                     "chains). " + (rep.get("llm_scope") or {}).get("note", ""))
            L.append("")
        _module_table(L, repl["modules"] if repl else mods)
        L.append("")
        if reduced and like:
            L.append("Like-for-like: all modules on the same website(s) "
                     f"({_sites(sorted({x[0] for x in rep.get('comparison_chains') or []}))}).")
            L.append("")
            _module_table(L, like)
            L.append("")
            top = best(like, "changed_f1")
            if top:
                L.append(f"On the website(s) where every module was executed, the highest F1 under layout "
                         f"changes was obtained by the {top[0]} module ({_p(top[1])}).")
            rule = [m for m in mods if COND_INFO[m["condition"]]["generator"] != "llm"]
            top = best(rule, "changed_f1")
            if top:
                L.append(f"Among the rule-based modules ({rule[0]['sites']} websites), the highest F1 under "
                         f"layout changes was obtained by the {top[0]} module ({_p(top[1])}).")
        else:
            top = best(mods, "changed_f1")
            if top:
                L.append(f"Under layout changes the highest F1 was obtained by the {top[0]} module "
                         f"({_p(top[1])}).")
        for m in mods:
            if m["initial_f1"] is not None and m["changed_f1"] is not None:
                delta = m["changed_f1"] - m["initial_f1"]
                L.append(f"- {m['module']} ({m['sites']} website(s) measured"
                         + (", values replicated to the others" if repl and COND_INFO[m["condition"]]["generator"]
                            == "llm" else "") + f"): F1 {_p(m['initial_f1'])} on the original "
                         f"pages and {_p(m['changed_f1'])} after the changes "
                         f"({'+' if delta >= 0 else ''}{delta * 100:.1f} points); "
                         f"{m['scored_decisions']} scored field decisions.")
        tests = rep.get("paired_tests") or []
        if tests:
            L += ["", "Paired per-site comparison of sustained F1 (measured values only, each pair on the "
                  "websites both modules were executed on; replicated values are never tested; with very few "
                  "websites the tests have little power):", ""]
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
        L.append(f"- {m['module']} ({m['sites']} website(s)): {m['breaking_events']} breaking changes "
                 f"(a field correct at the "
                 f"previous step became wrong), {m['detected_breaking_events']} of them detected by the "
                 f"health monitor; {m['healed_events_gt']} detected events fully repaired (GT-verified "
                 f"success {_p(m['healing_success_rate'])} of detected, "
                 f"{_p(m['healing_success_all_breaking'])} of all breaking changes); field-level recovery "
                 f"{_p(m['field_recovery_rate'])}; median time-to-recovery {_n(m['ttr_median_s'])} s; "
                 f"median regeneration latency {_n(m['regeneration_latency_median_s'])} s.")
    L += ["", "## D. Failure detection", ""]
    for m in mods:
        L.append(f"- {m['module']} ({m['sites']} website(s)): degraded monitoring runs "
                 f"{_p(m['degraded_run_share'])} (a field correct "
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
    scope = rep.get("llm_scope") or {}

    def site_cell(m):
        return (f"{m['sites_measured']} (+{m['sites_replicated']} repl.)" if m.get("sites_replicated")
                else str(m["sites"]))

    def rows(ms):
        return "\n".join(
            f"{m['module']} & {site_cell(m)} & {_tex(m['initial_f1'])} & {_tex(m['changed_precision'])} "
            f"& {_tex(m['changed_recall'])} & {_tex(m['changed_f1'])} & {_tex(m['changed_accuracy'])} "
            f"& {_tex(m['alarm_false_positive_rate'])} & {_tex(m['healing_success_rate'])} "
            f"& {_n(m['ttr_median_s']) if m['ttr_median_s'] is not None else '--'} \\\\"
            for m in ms)

    def table(ms, caption, label):
        return (
            "\\begin{table*}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            "\\begin{tabular}{lrrrrrrrrr}\n\\hline\n"
            "Module & Sites & F1 $t_0$ & P & R & F1 & Acc. & Alarm FPR & Heal. succ. & TTR (s) \\\\\n\\hline\n"
            f"{rows(ms)}\n\\hline\n\\end{{tabular}}\n\\end{{table*}}\n")

    caption = "Comparison of the scraping modules (field-level, \\%)"
    repl = rep.get("llm_replication")
    if repl:
        caption += (f"; LLM-based modules executed on {len(repl['source_sites'])} of {scope['configured_sites']} "
                    f"websites, their values replicated (not measured) to the other {len(repl['replicated_sites'])}; "
                    f"rule-based modules measured on {len(scope['rule_sites'])}")
    elif scope.get("reduced"):
        caption += (f"; LLM-based modules executed on {len(scope['llm_sites'])} of "
                    f"{scope['configured_sites']} websites, rule-based modules on {len(scope['rule_sites'])}")
    out = "% Generated by eval_paper from summary.json - do not edit numbers by hand.\n"
    if scope.get("note"):
        out += "% LLM scope: " + scope["note"] + (f" Reason: {scope['reason']}." if scope.get("reason") else "") + "\n"
    out += table(repl["modules"] if repl else mods, caption, "tab:modules")
    like = rep.get("modules_like_for_like") or []
    if scope.get("reduced") and like:
        sites = sorted({x[0] for x in rep.get("comparison_chains") or []})
        out += "\n" + table(like, "All modules on the same website(s) (" + ", ".join(s.replace("_", "\\_") for s in sites)
                             + "), field-level, \\%", "tab:modules-same-sites")
    return out
