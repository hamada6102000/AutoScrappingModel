# Running the 20-site evaluation

This is the step-by-step procedure for producing the paper's results on your
own machine. Every number in the report comes from runs you execute, on page
snapshots you saved, scored against labels you entered. Nothing is estimated
or filled in.

Commands are for **Windows Command Prompt**, run inside the project folder
(the one that contains `manage.py`) with the virtual environment active.

---

## 0. Quick path for the annotated 20-site dataset

Your dataset (`eval\dataset\dataset_20_sites.json`, with the same content as
`sites.csv` + `fields.csv` and the workbook) is converted to the config
without changing any URL, category or field:

```bat
python manage.py eval_import_dataset --dataset eval\dataset\dataset_20_sites.json --force
python manage.py eval_init --config eval\sites.yaml
python manage.py eval_snapshot
python manage.py eval_audit
```

`eval\prepare.bat` does the whole preparation in one go (virtual environment,
packages, headless browser, migrate, tests, eval_init, eval_snapshot,
eval_audit, eval_check --ping) and logs everything to
`eval_data\prepare_log.txt`. It makes no evaluation runs.

- `eval_import_dataset` maps types (text->string, other->mixed; number, price,
  date, url and boolean unchanged), Single/List -> single/many and
  Yes/No -> required. The recorded value, source, dynamic flag and original
  type stay on every field for the record; the LLM only sees the name, type,
  cardinality and description.
- `eval_snapshot` uses `render: auto`: it saves both the plain HTML and the
  page rendered by a headless browser and keeps the one that contains more of
  your recorded values (the losing version is kept on disk as
  `*.rejected-<mode>.html`). It warns when none of a site's expected texts
  (e.g. its title) is on the page — a consent wall, bot check or empty shell.
- `eval_audit` (no LLM calls) writes `eval_data\ieee-2026\audit\`:
  `audit.md` (per site: render mode, robots.txt, content check, prompt size;
  per field: whether your recorded value is on the saved page, where — visible
  text, an attribute, or only the meta tags / JSON-LD block — and whether it is
  inside the LLM prompt) and **`ground_truth_prefilled.xlsx`**: labels
  pre-filled only where your recorded value is literally on the saved page
  (abridged texts/lists completed from the page element that holds them, and
  marked as such). Empty rows are the ones you must label. Review it and import
  it as in Step 3.

Snapshot pages **as soon as possible after annotation**: dynamic values
(prices, quotes, rates) and job postings change or disappear.

**Structured metadata.** Several dataset fields live only in JSON-LD, `<meta>`
tags or the canonical URL, which the cleaner removes with `<head>` and
`<script>`. With `SCRAPER_STRUCTURED_DATA=true` (default) they are re-inserted
as `<section id="page-metadata">` (JSON-LD as nested `div[@data-key]`), so an
XPath can address them. The simulated layout changes do not touch this block,
so the report also gives accuracy split by where values were read from.

## 1. What the evaluation does

For each site and each repetition (3 by default):

1. **t0 — baseline.** The saved page is cleaned and the LLM (Groq / LLaMA)
   generates one XPath per field. Each XPath is validated and, if invalid,
   repaired before it is stored (proposal §4.3 step 3). The values are compared
   with your labels: this is the **initial F1**.
2. **Two conditions start from identical rules.**
   - `self_healing`: the proposed system.
   - `static`: an exact copy of the t0 rules, with healing switched off. This
     is the static-LLM-scraper baseline of RQ3; it never calls the LLM again.
3. **t1 … t8 — simulated layout changes**, applied cumulatively in a seeded
   random order (proposal §4.1, "controlled structural modifications"):
   class renames, extra wrappers, changed tags, reordered blocks, inserted
   banners, a full redesign, a price change, and one no-change control. Both
   conditions run on the *same* changed page at every step.
4. Every run records: values before and after healing, health, both change
   signals, every LLM call (tokens, latency, rate-limit waits, failure-memory
   length, raw response), and the comparison with the labels.

Optionally you can add **natural changes**: re-fetch the same pages days or
weeks later and label them again (section 6).

## 2. What you need to provide

| Item | Where |
|---|---|
| Your 20 page URLs, categories and fields | `eval\sites.yaml` |
| The correct value of every field on every saved page | the ground-truth sheet (step 4) |
| A Groq API key | `.env` |
| Enough Groq quota, or time to resume over several days | see section 5 |

## 3. One-time setup

```bat
cd C:\Users\hm245\Downloads\AutoScrappingModel\AutoScrappingModel
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m playwright install chromium
copy .env.example .env
notepad .env
```

In `.env` set `GROQ_API_KEY=gsk_...`. Then:

```bat
python manage.py migrate
python manage.py test scraper
```

All tests should pass (they run offline).

## 4. Step by step

### Step 1 — list your sites

```bat
copy eval\sites.example.yaml eval\sites.yaml
notepad eval\sites.yaml
```

Replace the example sites with your 20 pages. One specific page per site.
Keep the experiment settings unless you have a reason to change them, and
**never change them after you start running** — use a new experiment name.

```bat
python manage.py eval_init --config eval\sites.yaml
```

It reports every problem in the file at once. It also warns if the exact
tokenizer cannot load (it downloads its vocabulary once, so you need to be
online the first time).

### Step 2 — save the pages

```bat
python manage.py eval_snapshot
```

For each site it prints the HTTP status, the raw vs. LLM-view token counts,
the robots.txt result, and the path of a **labeling view** (the saved page
without scripts). Act on any warning:

- **"looks like a bot-check / blocked page"**: open the labeling view. If it
  is a captcha or "access denied" page, open the real page in your browser,
  press `Ctrl+S` and choose **"Webpage, HTML only"**. Put the file in
  `eval\saved_pages\`, then add `html_file: saved_pages/<name>.html` to that
  site and re-run: `python manage.py eval_snapshot --sites <key> --refresh`
- **Content missing because it loads with JavaScript**: either save the page
  from the browser as above, or set `render: js` (or `auto`) for that site.
  Playwright is in requirements.txt; run `python -m playwright install chromium`
  once.
- **robots.txt disallows the page**: the page is not fetched. Use a different
  page, or save it manually and state this in the paper's ethics section.

### Step 3 — label the ground truth

```bat
python manage.py eval_gt_template
```

This writes `eval_data\<experiment>\ground_truth_baseline.xlsx`. Open it in
Excel and read the README sheet. For each row:

- open the file named in `labeling_view` (double-click opens it in your
  browser);
- type or paste the field's value **exactly as shown there**. Do not use the
  live website: prices and text may have changed since the snapshot;
- list fields: one item per line (Alt+Enter inside the cell);
- field not on that page: write `<ABSENT>`;
- put your initials in `verified_by`.

Then import it:

```bat
python manage.py eval_gt_import --file eval_data\ieee-2026\ground_truth_baseline.xlsx --verified-by HM
```

Read every `CHECK` line it prints:

- *value not found in the saved page text*: usually a typo. Fix it in the
  sheet and import again (re-importing replaces the labels).
- *value is on the page but removed by pruning*: add the container tag (for
  example `aside`, `nav` or `header`) to that site's `prune_keep`, run
  `eval_init` again, then re-import.

These fixes are allowed because they happen **before** any results exist.
Do not change labels after you have seen results.

### Step 4 — check readiness

```bat
python manage.py eval_check --ping
```

`--ping` makes one tiny LLM call to prove the key works. The command also
prints an estimate of the LLM calls and tokens the full run needs. Continue
only when it says **READY**.

### Step 5 — run

```bat
python manage.py eval_run --dry-run
python manage.py eval_run
```

The run prints one line per run. It is **resumable**: if you close the
window, lose the connection, or run out of quota, just run the same command
again and it continues from the last completed run. An interrupted run is
rolled back completely, so nothing half-finished is ever counted.

Useful options:

- `--max-llm-calls 300` — stop after a budget (for example, to stay within a
  daily quota);
- `--wait` — when the quota is exhausted, sleep until Groq says the limit
  resets and continue automatically (leave the window open; `--max-hours`
  bounds it);
- `--sites shop-a,news-a` — run a subset.

### Step 6 — report and export

```bat
python manage.py eval_report
python manage.py export_results
```

`eval_report` prints the Table 5.1 metrics and writes
`eval_data\<experiment>\results\` (`report.md`, `summary.json` and CSV
tables). `export_results` writes **`eval_data\<experiment>\<experiment>_results.zip`**
containing every raw table, the report, the baseline snapshots, labeling
views and the database. **That zip is what you send back for the paper.** It
contains no API keys.

## 5. Groq free-tier limits

Free-tier limits are per model and per day, and Groq changes them over time;
check yours at https://console.groq.com/settings/limits. `eval_check` and
`eval_run --dry-run` print the expected number of calls and tokens. If the
quota runs out mid-run, the run pauses cleanly (`Stopped early: LLM quota
exhausted`) and you resume later. Time spent waiting on rate limits is
recorded separately and is **excluded** from the latency metrics.

If the run would take too many days, the options are, in order of preference:

1. continue over several days (results are identical, because the design is seeded);
2. reduce `repetitions` to 2 (new experiment name, before any run);
3. use Groq's paid Developer tier (same model, so comparable results);
4. switch `model` to `llama-3.1-8b-instant` (a different model: report it as
   such, and never mix models within one experiment).

## 6. Optional extras

**Natural changes.** Some days or weeks later, re-fetch the same pages:

```bat
python manage.py eval_snapshot --kind natural --label day14
python manage.py eval_gt_template --kind natural --label day14
```

Fill in the sheet as before. The extra `natural_changes` sheet asks whether
each page's *layout* changed (yes/no). Then:

```bat
python manage.py eval_gt_import --file eval_data\ieee-2026\ground_truth_day14.xlsx
python manage.py eval_run --natural
```

The report gains a "Natural changes" section.

**External baselines (proposal O4).** To compare against AutoScraper, AXE or
another system, run it on the exported snapshot files. Then write a CSV with
the columns `snapshot_id,field,value` (snapshot IDs are in
`tables\snapshots.csv`) and import it:

```bat
python manage.py eval_import_predictions --name autoscraper --file autoscraper.csv
python manage.py eval_report
```

It is scored with the same labels and matching rules, on the same snapshots.

## 7. Metric definitions (for the methodology section)

- **Field decision** — one field in one run. TP: extracted and equal to the
  label after normalisation (numbers compared as decimals, so `$1,149.99`,
  `1.149,99` and `١٬١٤٩٫٩٩` are equal; dates compared in ISO form; text is
  case- and whitespace-insensitive). A wrong value counts as FP and FN; a
  missing value as FN; a value for a field labelled `<ABSENT>` as FP. List
  fields use item-level multiset matching. Precision, recall and F1 are micro-averaged.
- **Initial F1** — the t0 decisions. **Sustained F1** — all steps t0…t8.
- **Degraded run** — at least one field is wrong before healing.
  **Alarm** — health before healing is below the threshold.
  **False-positive rate (Table 5.1)** — alarms on runs that are not degraded,
  divided by all monitoring runs (steps ≥ 1).
- **Breaking change** — a step after which a field that was correct is wrong
  (before healing). Change detection (RQ1) is scored only on runs that started
  with every field correct.
- **Self-healing success (GT-verified)** — every field broken by a change is
  correct again after healing, divided by the detected breaking changes. The
  report also gives the system's own view (health back above the threshold).
- **Time-to-recovery** — from the alarm to valid XPaths, excluding rate-limit waits.
- **Regeneration latency** — one healing LLM call plus validation.
- **Token reduction** — 1 − (LLM-view tokens / raw-page tokens) on baseline
  pages, with the tokenizer recorded (tiktoken `cl100k_base`).
- **Silent failures** — degraded runs without an alarm (wrong but non-empty
  values the health monitor cannot see). Reported, never hidden.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `NOT READY ... unlabelled fields` | Fill the empty `value` cells and import again |
| `exact tokenizer unavailable` | Be online once, then run `eval_init` again |
| `LLM provider not usable: GROQ_API_KEY is not set` | Add the key to `.env` |
| `LLM ping failed ... 401` | The key is wrong or revoked; create a new one |
| `Stopped early: LLM quota exhausted` | Wait for the quota to reset, then re-run `eval_run` |
| A site's fetch fails | Save the page in your browser and use `html_file` |
