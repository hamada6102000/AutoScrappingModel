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
| **1. Fetcher** | Fetch raw HTML, strip scripts/styles/comments, prune nav/footer/ads → a token-reduced DOM | `scraper/services/fetcher.py` |
| **2. LLM Analysis** | Prompt the LLM with the pruned DOM + field schema → one XPath per field | `scraper/services/llm_analyzer.py`, `scraper/llm/` |
| **3. Validation & Change-Detection** | Apply XPaths, compute a health score (failure rate, field-count mismatch, structural distance), detect layout change | `scraper/services/{extractor,health_monitor,change_detector}.py` |
| **4. Self-Healing** | On failure, re-invoke the LLM with fresh HTML + failure-history memory → corrected XPaths; bounded retry loop | `scraper/services/self_healer.py`, `scraper/services/pipeline.py` |

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
python manage.py run_scrape --target 1     # v1 → success
python manage.py simulate_change v2        # redesign the site
python manage.py run_scrape --target 1     # → self-heals
python manage.py simulate_change v1        # revert
```

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
lighter on the free rate limits. If a key is missing or invalid the pipeline
safely falls back to the heuristic provider (it never crashes).

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
  models.py             ScrapeTarget, TargetField, XPathRule, ScrapeRun,
                        ExtractedRecord, ChangeEvent, FailureRecord, LLMCallLog
  llm/                  Pluggable provider layer
    base.py             LLMProvider interface + data structures
    heuristic.py        Offline provider (default)
    groq_provider.py    Free LLaMA provider (Groq, OpenAI-compatible)
    anthropic_provider.py / openai_provider.py / ollama_provider.py
    prompts.py          Prompt construction + JSON parsing
    factory.py          Resolve provider from settings
  services/             The four layers + orchestrator
    fetcher.py normalizer.py extractor.py
    llm_analyzer.py health_monitor.py change_detector.py self_healer.py
    pipeline.py         Closed-loop orchestrator
  management/commands/  seed_demo, run_scrape, simulate_change
  tests/                Unit + integration tests (incl. the full heal loop)
dashboard/              Web UI (Controllers + Templates) — the "V" + "C" of MVC
demopages/              Self-contained demo product page (v1 / v2 fixtures)
docs/ARCHITECTURE.md    Deeper design notes + thesis mapping
```

The research core (`scraper/services` + `scraper/llm`) is deliberately
**framework-agnostic** — it has no Django dependency beyond settings, so it can
be unit-tested and reused outside the web app.

---

## Tests

```bash
python manage.py test scraper
```

Covers DOM pruning, the heuristic generator, value normalization, health scoring,
change detection, and the **end-to-end self-healing loop** (v1 success → v2
detection → regeneration → recovery).

---

## How the metrics map to the proposal

| Proposal metric (Table 5.1) | Where it appears |
|-----------------------------|------------------|
| Initial extraction accuracy (F1 vs ground truth) | `ExtractedRecord.matches_ground_truth`, run detail |
| Self-healing success rate | `ChangeEvent.resolution == auto_healed` |
| Time-to-recovery | `ChangeEvent.time_to_recovery_ms` |
| XPath regeneration latency | `ScrapeRun.heal_latency_ms`, `LLMCallLog.latency_ms` |
| Health score / failure rate | `ScrapeRun.health_score`, `xpath_failure_rate` |
| Structural (tree-edit) distance | `ScrapeRun.structural_distance` |
| LLM token usage | `ScrapeRun.token_usage`, `LLMCallLog` |

---

## Extending toward the full thesis evaluation

This scaffold implements the complete closed loop on a controlled demo. Natural
next steps for the thesis experiments:

- Add the 20+ real-world targets (e-commerce, news, finance, jobs, academic) via
  the admin (`/admin/`) or a fixture, each with its field schema + ground truth.
- Wire a real LLM provider and compare against the heuristic baseline.
- Add a scheduler (e.g. Celery beat / `cron` calling `run_scrape --all`) for the
  continuous-monitoring evaluation period.
- Replace the skeleton-based structural distance with a full tree-edit-distance
  implementation if finer change characterisation is needed.
- Export runs/records to compute F1-over-time and the baseline comparison tables.

---

*Fayoum University — Faculty of Computers and Artificial Intelligence, 2026.*
