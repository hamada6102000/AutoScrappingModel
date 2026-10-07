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

## 0b. Modules compared and the one-command run

| Condition | Module | Selectors at t0 | Repair |
|---|---|---|---|
| `static` | Static baseline (LLM-generated selectors) | LLM | never |
| `rule_static` | Static baseline (rule-generated selectors) | rule-based | never |
| `rule_based` | Rule-based self-healing | rule-based | rule-based (value anchoring, label anchoring, attributes, type rules, structured data) |
| `self_healing` | LLM-based self-healing (proposed) | LLM | LLM with failure memory |

All four run on the same saved pages and the same seeded change sequences; the
two static baselines start from an exact copy of their generator's t0
selectors, so each self-healing module has a paired control. Detection,
threshold, attempt limit and failure memory are identical for both healing
modules; only the selector writer differs.

```bat
eval\evaluate.bat offline   &  rem rule-based modules only: no LLM, no quota
eval\evaluate.bat           &  rem everything (LLM modules sleep through rate limits)
```

`evaluate.bat` imports, as labels, only your recorded dataset values that are
found verbatim on the saved pages (label source `dataset`; labels you imported
yourself are never overwritten). Fields without a label are scraped but not
scored, and the report states how many. It ends with `eval_report` (tables
and `figures/*.png`), `eval_verify` (independent re-computation of every
metric, snapshot hashes, byte-identical re-creation of every simulated page),
`eval_paper` (`paper_results.md`, `paper_table_modules.tex`) and
`export_results`.

**LLM model: Claude Opus 5.5 on all 20 sites, through Claude Code with the
Claude subscription (decision of 2026-10-07).** The LLM-based modules
(`self_healing`, `static`) use `claude-opus-5-5` on all 20 sites with 3
repetitions, the same design as the rule-based modules. Requests go through
the installed **Claude Code CLI**, signed in with your Claude subscription; no
Anthropic API key is used. `evaluate.bat` applies this with

```bat
python manage.py eval_switch_llm --config eval\sites.yaml --provider claude_code ^
    --model claude-opus-5-5 --reasoning-effort medium --max-input-chars 200000 --repetitions 3
```

- The first time, it backs up the database and the LLM files to
  `eval_data\<exp>\backups\<time>_before_claude-opus-5-5\`, deletes the runs of
  the LLM-based modules made with the earlier model (gpt-oss-120b on Groq) and
  removes the earlier two-site scope with replicated values (`llm_scope.json`).
  Saved pages, labels and the rule-based runs are kept; the rule-based modules
  get a 3rd repetition. Later calls change nothing, so resuming keeps the
  Claude runs.
- Nothing is replicated: every LLM value in the report is measured.

**How a request is made** (`scraper/llm/claude_code_provider.py`). Each LLM
call starts one Claude Code process:

```
claude -p --output-format stream-json --verbose --model claude-opus-5-5 --effort medium
       --system-prompt-file <file with the project's system prompt> --tools "" --max-turns 1
       --no-session-persistence --safe-mode --strict-mcp-config --disable-slash-commands
stdin = the project's user prompt;  cwd = an empty folder in the system temp folder
```

- The project's system prompt and user prompt are sent unchanged. No tools,
  no MCP servers, no CLAUDE.md memory, no saved session. `max_tokens` is set
  with `CLAUDE_CODE_MAX_OUTPUT_TOKENS` (12,000 + 150 per field, at most
  32,000, as for the API provider); thinking is adaptive, depth `effort: medium`;
  no temperature (the 5.5 models use their defaults). Because sampling cannot
  be fixed, the 3 repetitions measure the run-to-run variation.
- Claude Code itself adds, and this cannot be switched off without losing the
  subscription login: a billing-header block and the line "You are a Claude
  agent, built on Anthropic's Claude Agent SDK." before the system prompt, and
  an environment note (working folder, platform, model id, date) after the
  user prompt. (Checked by running Claude Code 2.1.293 against a local mock
  endpoint and comparing the request with the project's prompts.)
- Every answer must be a single turn. If Claude Code had to continue a reply
  (token cap reached, or a thinking-only reply nudged), that answer is
  discarded and the request is sent again once with twice the cap; both calls
  are counted. A second continued reply counts as no answer.
- The answer is parsed exactly as for the other providers (`parse_xpath_json`).
- Latency is Claude Code's API time (`duration_api_ms`), without the CLI
  start-up and without waiting; tokens include cached input.

**Authentication.** The project never reads, stores or logs a token. Claude
Code uses its own login: the long-lived token from `claude setup-token`, which
it reads from the Windows user environment variable `CLAUDE_CODE_OAUTH_TOKEN`,
or the login saved by `claude` → `/login`. Before starting Claude Code the
project removes `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`
and the Bedrock / Vertex switches from its environment, and every reply must
report `apiKeySource: none` (subscription login) and the configured model,
otherwise the call fails. `eval\claude_smoke.bat` (`manage.py
eval_claude_smoke`) shows the exact command, the login method and the result
of one tiny request, and saves them to `eval_data\claude_code_smoke_log.txt`.
If Claude Code is not on `PATH`, set `CLAUDE_CODE_CLI` (in `.env`) to the full
path of `claude.exe`.

**Cost.** The subscription is not billed per token. The report's cost column is
the API-equivalent cost (USD 4 / 20 per million input / output tokens,
`SCRAPER_PRICING` set by `evaluate.bat`; Claude Code reports the same basis).

The earlier two-site decision can still be recorded with `eval_llm_scope`
(see `python manage.py eval_llm_scope --help`), but it is not used.

## 1. What the evaluation does

For each site and each repetition (3 by default):

1. **t0 — baseline.** The saved page is cleaned and the LLM (Claude Opus 5.5)
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
| Claude Code installed and signed in with your Claude subscription | `claude setup-token` (token in the user variable `CLAUDE_CODE_OAUTH_TOKEN`) or `claude` → `/login` |
| Time for the LLM runs (several hours; resumable) | see section 5 |

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

No API key is needed for the evaluation (Claude Code signs in with your subscription). Then:

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
- `--wait` — when the rate limit is exhausted, sleep until the provider says
  the limit resets and continue automatically (leave the window open; `--max-hours`
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

## 5. Claude subscription limits

The subscription meters usage in a 5-hour window and a weekly window, not in
tokens per minute. When a window is used up, Claude Code reports the reset
time; the run stops cleanly (`LLM quota exhausted: Claude subscription usage
limit reached (five_hour) ...`) and `eval_run --wait` sleeps until the reset
and continues (`evaluate.bat` allows up to 168 hours). Short 429 / 529 / 5xx
answers are retried inside Claude Code; their waiting time is recorded
separately and is **excluded** from the latency metrics. Login, model and
request errors stop the run (`HTTP 401 / 403 / 404 / 400 ...`): fix the cause
and run `evaluate.bat` again; completed runs are kept.

`eval_check` and `eval_run --dry-run` print the expected number of calls and
tokens.

**Earlier model (2026-10-04, replaced).** Before the switch the LLM-based modules
ran `openai/gpt-oss-120b` on the Groq free tier (8,000 tokens per minute), which
forced shortened prompts and, by supervisor decision, a two-site scope with
replicated values. Those runs are kept only in the backup made by
`eval_switch_llm`; they are not part of the results.

`eval_audit` shows, per field, whether the recorded value is inside the HTML
the model actually receives.

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
| `Claude Code (the 'claude' command) was not found` | Install Claude Code, or set `CLAUDE_CODE_CLI` to the full path of `claude.exe` |
| `Claude Code is not signed in` / `HTTP 401 Claude Code authentication_failed` | Add the `claude setup-token` token as the user variable `CLAUDE_CODE_OAUTH_TOKEN`, or run `claude` and `/login`; check with `eval\claude_smoke.bat` |
| `HTTP 403 Claude Code is using an API key` | Remove the API key / `apiKeyHelper` from Claude Code's settings |
| `HTTP 404 Claude Code model_not_found` | The model is not available to your plan; check with `eval\claude_smoke.bat` |
| `Claude subscription usage limit reached` | Nothing to do: the run waits for the reset and continues |
| `LLM ping failed ... 401` | The key is wrong or revoked; create a new one |
| `Stopped early: LLM quota exhausted` | Wait for the quota to reset, then re-run `eval_run` |
| A site's fetch fails | Save the page in your browser and use `html_file` |
