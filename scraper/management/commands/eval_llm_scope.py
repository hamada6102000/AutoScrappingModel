"""
Record (or show) on which sites the LLM-based modules are executed, and
whether the report replicates their values to the other sites.

    python manage.py eval_llm_scope                       # show the recorded decision
    python manage.py eval_llm_scope --llm-sites site_001,site_002 --replicate-to-all-sites ^
        --reason "decision of the project supervisor (Groq free-tier limits)"
    python manage.py eval_llm_scope --clear               # LLM modules on all sites again

Written to eval_data/<exp>/llm_scope.json and copied into the results. With
it, eval_run never executes an LLM-based module on another site (no LLM call
for those sites).
"""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from scraper.evaluation import llm_scope
from scraper.evaluation.cli import add_experiment_arg, get_experiment, site_list


class Command(BaseCommand):
    help = "Record the sites the LLM-based modules run on (and replication of their values in the report)."

    def add_arguments(self, parser):
        add_experiment_arg(parser)
        parser.add_argument("--llm-sites", help="Comma-separated site keys the LLM modules run on.")
        parser.add_argument("--replicate-to-all-sites", action="store_true",
                            help="Report: copy the LLM values measured on these sites to every other site "
                                 "(labelled 'replicated').")
        parser.add_argument("--reason", default="", help="Why (written into the report and paper text).")
        parser.add_argument("--clear", action="store_true", help="Remove the recorded decision.")

    def handle(self, *args, **opts):
        exp = get_experiment(opts["experiment"])
        if opts["clear"]:
            removed = llm_scope.clear(exp)
            self.stdout.write("LLM-scope decision removed." if removed else "No LLM-scope decision recorded.")
            return
        if opts["llm_sites"]:
            try:
                data = llm_scope.save(exp, site_list(opts["llm_sites"]), opts["replicate_to_all_sites"],
                                      opts["reason"])
            except ValueError as exc:
                raise CommandError(str(exc)) from exc
        else:
            data = llm_scope.load(exp)
            if data is None:
                self.stdout.write("No LLM-scope decision recorded: the LLM modules run on all sites.")
                return
        n = len(exp.config["sites"])
        others = n - len(data["llm_sites"])
        self.stdout.write(self.style.WARNING(
            f"LLM SCOPE: the LLM-based modules are executed on {', '.join(data['llm_sites'])} only "
            f"({len(data['llm_sites'])} of {n} sites); no LLM call is made for the other {others}. "
            + (f"The report replicates the LLM values measured on these sites to the other {others} sites, "
               "labelled 'replicated' (not measured). " if data["replicate_to_all_sites"] else "")
            + (f"Reason: {data['reason']}. " if data.get("reason") else "")
            + f"Recorded {data['recorded_at']} in {llm_scope.path(exp)}"))
