# Self-Healing Web Scraper — LLM-Driven XPath Regeneration

A reference implementation of the master's thesis
**“Self-Healing Web Scraping via LLM-Driven XPath Regeneration — A Novel
Approach to Autonomous, Maintenance-Free Data Extraction”**
(Fayoum University, Faculty of Computers and Artificial Intelligence).

It implements the proposal's four-layer, closed-loop architecture as a Django
(MVC) web application with a dashboard. Conventional scrapers rely on hard-coded
XPath selectors that silently break when a site changes its HTML. This system
**detects** the breakage from a continuous health score, **diagnoses** which
selectors failed, and **recovers** by regenerating XPaths with an LLM — with no
human intervention.

> **Runs offline out of the box.** The default `heuristic` provider re-derives
> XPaths from the live DOM, so the entire pipeline — including a real
> self-healing demo — works with **zero configuration and no API keys**. Plug in
> Claude / GPT-4 / LLaMA behind the same interface whenever you want stronger
> generalisation.

---

## The four layers (proposal §1.4)

| Layer | Responsibility | Code |
|------|-----------------|------|
| **1. Fetcher** | Fetch (or load a saved snapshot), strip scripts/styles/comments/hidden/ad blocks and page chrome (keeping an article's own header), whitelist attributes → a clean extraction DOM plus a compressed LLM view | `scraper/services/fetcher.py` |
| **2. LLM Analysis** | Prompt the LLM with the LLM view + field schema → one XPath per field; each XPath is validated and invalid ones repaired before storing (§4.3 step 3) | `scraper/services/pipeline.py`, `scraper/llm/` |
| **3. Validation & Change-Detection** | Apply XPaths (single values or lists), type-check values, compute a health score (failure rate, required-field mismatch, structural distance) and two change signals (tag-path and markup distance) | `scraper/services/{extractor,health_monitor,change_detector}.py` |
| **4. Self-Healing** | On alarm, re-invoke the LLM with the fresh page + failure-history memory → validated corrected XPaths; bounded retry loop; `--no-heal` for the static baseline | `scraper/services/{self_healer,pipeline}.py` |

The orchestrator (`scraper/services/pipeline.py`) wires these into the closed
loop from **Figure A.1** of the proposal:

```
fetch & clean → (generate XPaths if none) → extract → validate + health score
     → health OK?  ── yes ──▶  output (clean records)
                   └─ no ──▶  change detection → self-heal (regenerate) → retry
```

---

## Quick start

```bash
# 1. Create a virtualenv and install dependencies
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Set up the database
python manage.py migrate

# 3. Seed the self-contained demo target (an e-commerce product page)
python manage.py seed_demo

# 4. Start the server
python manage.py runserver
```

Open **http://127.0.0.1:8000/** for the dashboard.

### See self-healing happen (2 minutes)

From the dashboard:

1. Click **Run scrape** on the demo target → first run **succeeds** (health 100%,
   5/5 fields), generating XPaths against the original layout (v1).
2. Click **Simulate layout change** → the demo site now serves a **redesigned**
   layout (v2): same data, new DOM structure and renamed CSS classes.
3. Click **Run scrape** again → the pipeline detects the structural change, all
   stale selectors fail, and the **self-healing loop regenerates working XPaths**
   → status becomes **Healed**, health returns to ~98%, values still match ground
   truth.

Open the resulting **run detail** page to see the full pipeline trace, the old
(broken) selectors in *failure memory*, and the new regenerated selectors.

The same flow on the command line:

```bash
python manage.py run_scrape --target 1                 # v1 → success
python manage.py simulate_change v2                    # redesign the site
python manage.py run_scrape --target 1 --no-heal       # static baseline: detects, does not repair
python manage.py run_scrape --target 1                 # → self-heals
python manage.py simulate_change v1                    # revert
```

`run_scrape` options: `--provider groq`, `--model <id>`, `--no-heal`,
`--no-fallback`. Every live run saves its page as a snapshot under `eval_data/live/`.

---

## Choosing an LLM provider

Set `SCRAPER_LLM_PROVIDER` (env var or `.env` — copy from `.env.example`):

| Provider | Value | Notes |
|----------|-------|-------|
| Heuristic (default) | `heuristic` | Offline, no keys. Re-derives XPaths from the DOM. |
| **Groq (free)** | `groq` | **Free**, fast LLaMA inference. Needs a free `GROQ_API_KEY`. |
| Claude (Anthropic) | `anthropic` | Paid API. Needs `ANTHROPIC_API_KEY`. |
| OpenAI GPT | `openai` | Paid. Needs `OPENAI_API_KEY` and `pip install openai`. |
| Ollama / LLaMA | `ollama` | Free/local. Needs a running Ollama server + a pulled model. |

### Use the free Groq LLM (recommended)

Groq serves LLaMA models for free (rate-limited, no cost) via an OpenAI-compatible
API — no extra Python package required.

1. Sign in at **https://console.groq.com** and create a free API key at
   **https://console.groq.com/keys**.
2. Configure it (either export the vars, or put them in `.env`):

   ```bash
   export SCRAPER_LLM_PROVIDER=groq
   export GROQ_API_KEY=gsk_...your_key...
   # optional: export GROQ_MODEL=llama-3.1-8b-instant   # faster/lighter
   ```
3. Run it:

   ```bash
   python manage.py run_scrape --target 1 --provider groq
   ```

Default model is `llama-3.3-70b-versatile`; `llama-3.1-8b-instant` is faster and
lighter on the free rate limits. Rate limits (HTTP 429) are handled by waiting
and retrying; the waiting time is logged separately from latency. For ordinary
runs, an LLM failure falls back to the heuristic for that call and is flagged
(`fallback_used`); the evaluation never falls back, so LLM results are never
mixed with heuristic ones.

> ℹ️ **About Claude access.** The `anthropic` provider uses the **Anthropic API**
> (a key from **console.anthropic.com**, billed separately). A **claude.ai
> Pro/Team/Enterprise chat subscription is _not_ the same thing** and cannot
> authenticate API calls — which is why Groq is wired in as the free option.

All providers implement the same `LLMProvider` interface
(`scraper/llm/base.py`), so swapping one in changes nothing else in the pipeline.

---

## Project structure

```
config/                 Django project (settings, urls) — thresholds live in SCRAPER{}
scraper/                Core app (Models + services + LLM providers)
  models.py             ScrapeTarget, TargetField, XPathRule, ScrapeRun, ExtractedRecord,
                        ChangeEvent, FailureRecord, LLMCallLog, Experiment, Snapshot, GroundTruth
  llm/                  Pluggable provider layer (base, heuristic, groq, anthropic,
                        openai, ollama, prompts, factory, errors)
  services/             The four layers + orchestrator
    fetcher.py          cleaning / pruning, clean DOM + LLM view, fetching
    extractor.py        XPath evaluation, list fields, type validation
    normalizer.py       value normalisation (US/EU/Arabic numbers, dates)
    matching.py         ground-truth matching (exact, token F1, lists)
    change_detector.py  structural and markup distance
    health_monitor.py   health score
    mutations.py        seeded simulated layout changes
    snapshots.py        snapshot storage + labeling views
    tokens.py           token counting (tiktoken)
    pipeline.py         closed-loop orchestrator
  evaluation/           20-site evaluation harness (config, protocol, ground-truth
                        sheets, metrics, report, export, external baselines)
  management/commands/  seed_demo, run_scrape, simulate_change, eval_import_dataset,
                        eval_init, eval_snapshot, eval_audit, eval_gt_template,
                        eval_gt_import, eval_check, eval_run, eval_report,
                        eval_import_predictions, export_results
  tests/                Unit + integration tests (94)
dashboard/              Web UI (Controllers + Templates) — the "V" + "C" of MVC
demopages/              Self-contained demo product page (v1 / v2 fixtures)
eval/sites.example.yaml Template for your evaluation sites
docs/                   ARCHITECTURE.md, EVALUATION.md (runbook + metric definitions)
```

The research core (`scraper/services` + `scraper/llm`) is deliberately
**framework-agnostic** — it has no Django dependency beyond settings, so it can
be unit-tested and reused outside the web app.

---

## Tests

```bash
python manage.py test scraper
```

Covers pruning, tokens, normalisation, ground-truth matching, extraction, change
detection, mutations, the Groq provider (rate limits, quota, oversized prompts),
the pipeline (validation repair, `--no-heal`, fallback tracking, cost) and the
full evaluation harness (paired runs, quota stop and resume, report, export).
All tests run offline.

---

## Evaluation on real websites (the paper's results)

The harness runs the proposal's evaluation on your own sites and produces every
metric in Table 5.1 and RQ1–RQ5 from stored runs. Nothing is estimated. Full
runbook: **[docs/EVALUATION.md](docs/EVALUATION.md)**.

```bat
python manage.py eval_import_dataset --dataset eval\dataset\dataset_20_sites.json
python manage.py eval_init --config eval\sites.yaml
python manage.py eval_snapshot                    &  rem save every page
python manage.py eval_audit                       &  rem check pages vs. dataset, pre-fill labels
python manage.py eval_gt_template                 &  rem Excel sheet to label
python manage.py eval_gt_import --file eval_data\ieee-2026\ground_truth_baseline.xlsx
python manage.py eval_check --ping                &  rem readiness + key test
python manage.py eval_run                         &  rem resumable
python manage.py eval_report
python manage.py export_results                   &  rem one zip with everything
```

Design: for each site and repetition, the self-healing system and a static
copy of its t0 rules (no healing) run on the same seeded sequence of simulated
layout changes; optional natural re-fetches add real changes. Every run records
values before and after healing, health, change signals, and every LLM call
(tokens, latency, rate-limit waits, failure-memory length, raw response), all
scored against human labels.

---

*Fayoum University — Faculty of Computers and Artificial Intelligence, 2026.*
