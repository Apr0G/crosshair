# CLAUDE.md — Crosshair

**RULE 0 — NEVER BREAK:** start every reply with my name. e.g. `Faig, ...`

CS2 impact-attribution engine. Scrapes FACEIT demos → parses with awpy → SQLite →
LightGBM win-probability model → per-action impact (WP delta). Treat the existing
pipeline as working software, not a draft.

---

## How to respond
- Terse. No preamble, no "Great question", no restating my request.
- Don't narrate what you're about to do. Do it, then 2–4 line summary.
- Show **diffs / changed blocks**, not whole files. Never paste unchanged code.
- Don't re-read files you've already read this session.
- One blocking question at a time. If not blocked, proceed.
- No filler closers ("let me know if..."). End when done.

## How to change code
- **Minimal diffs.** No drive-by refactors, renames, or reformatting of code you
  weren't asked to touch.
- Don't rewrite working code to "improve" it. Leave it.
- Match existing style: flat `src/` modules, `import db` (not `from src import db`),
  `# ── section ──` comment banners, aligned assignments, f-string prints.
- Don't introduce new patterns, frameworks, or abstraction layers unprompted.

## Ask before
- Changing DB schema (and DB changes are **additive-only** unless I approve otherwise
  — ship a migration, see *Schema traps*).
- Deleting or disabling code.
- Adding a dependency.
- Touching the Playwright downloader (`src/scraper.py`) — it is load-bearing and
  hard to re-verify without a live browser login.

## Git
- **Run `/security-review` before every commit.** No exceptions. Invoke the skill,
  read the findings, and fix or explicitly dismiss each one *before* `git commit`.
  If the review surfaces anything unresolved, stop and tell me — don't commit past it.
  Report the result in the commit summary (e.g. `security-review: clean`).
  This repo handles a FACEIT API key, browser session state, and `.env` — the review
  is mainly there to catch a secret or a session cookie leaking into a diff.
- Only commit or push when I ask.

---

## Environment
- **Always use `.venv/bin/python`** (3.11.15, arm64). Bare `python` on PATH is not
  guaranteed to be this venv. The venv was rebuilt once because x86_64 wheels under
  an arm64 host caused numpy/requests circular-import errors — don't `pip install`
  outside it.
- **BROKEN as of 2026-08-01:** `.venv` has no `lib/`, no site-packages and no pip
  (108K total). Only stdlib runs — `main.py status` and `test/test_invariants.py`
  work; everything else dies at import. Recovery:
  `/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv --clear .venv` then
  `.venv/bin/python -m pip install -r requirements.txt`. **Pin the deps first** — a
  fresh unpinned resolve will not reproduce the awpy that parsed the 710-match corpus.
  Until this is fixed, "verify before declaring done" is not achievable for most
  changes; say so rather than claiming a change is tested.
- `requirements.txt` is the **authoritative** dep list.
- `pyproject.toml` is stale and untracked: its deps are incomplete (no lightgbm /
  scikit-learn / requests) and `[project.scripts] crosshair = "crosshair.main:main"`
  points at a package that doesn't exist. Don't rely on it; don't "fix" it unasked.
- Secrets via env + `.env` (see `.env.example`). Never hardcode keys or commit secrets.
  `FACEIT_API_KEY` is used by both ingestion paths; the API path also needs Downloads
  access granted on that key.

## Run
```bash
.venv/bin/python src/main.py status                      # DB stats
.venv/bin/python src/main.py demo demos/mirage.dem       # process one local .dem (cheapest smoke test)
.venv/bin/python src/main.py scrape --source playwright  # or --source api
.venv/bin/python src/main.py train --eval                # → data/win_prob.lgb
.venv/bin/python src/main.py score                       # fill p_before/p_after/impact
.venv/bin/python src/main.py top -n 10
.venv/bin/python src/main.py round <match_id> <n> --engagements-only
```
Tests are **plain scripts, not pytest** (pytest isn't installed):
```bash
.venv/bin/python test/test_invariants.py           # stdlib only — runs even with the venv broken
.venv/bin/python test/test_win_probability.py      # needs a trained model; exit 2 = no model
.venv/bin/python test/test_local_demo.py demos/mirage.dem   # inspection; --write-db to store
.venv/bin/python test/test_scraper.py <match_id>   # opens real Chrome
```
`test_invariants.py` is the cheap regression guard for the schema traps below — run it
after any change to `db.py`, `state_sampler.py`, or `win_probability.FEATURES`.
**Decided (2026-08-01): stay on plain scripts. Don't propose pytest again** unless we
start testing `feature_extractor` / `state_sampler` / `score_impact` — only then do
shared fixtures (parse one demo once, reuse across many tests) pay for the dep.
Note `test_local_demo.py` and `test_scraper.py` are manual inspection scripts despite
the `test_` prefix. Both are now importable (body inside `main()`) and
`test_local_demo.py` no longer writes to `data/crosshair.db` unless given `--write-db`.

## Module map
```
src/main.py             CLI: status | scrape | demo | train | score | top | player | round
src/scraper.py          Playwright path: FACEIT API player discovery + browser demo-URL
                        interception + download/decompress (.zst/.gz)
src/faceit_api.py       API path: same discovery, then 2-step Downloads API
                        (GET /matches/{id} → POST /download/v2/demos/download → signed URL);
                        reuses scraper.download_demo for the fetch
src/pipeline.py         run() picks the source module, loops matches → process_one()
src/extract.py          awpy Demo.parse → dict of pandas DataFrames (+ map_name)
src/feature_extractor.py  decision events (buy / engagement / utility / rotation / bomb),
                        visibility + sound-cue precompute
src/state_sampler.py    ~1 Hz round_states snapshots (WP training rows)
src/win_probability.py  LightGBM train/load/predict; FEATURES list lives here
src/score_impact.py     brackets each event between two states → p_before/p_after/impact
src/db.py               SQLite schema + insert helpers + dedup
```

**THE SEAM** — both ingestion sources converge here; anything new that produces a
local `.dem` must plug in here rather than fork the pipeline:
`pipeline.process_one()` → `db.insert_events` / `db.insert_round_states` / `db.mark_processed`.
A source module only has to expose `iter_unprocessed_demos(...)` and `download_demo(url, dest)`.

## Invariants
- **Idempotency:** never write logic that can double-insert a match. Dedup is
  `db.is_processed(match_id)`, checked for both BO1 (`<mid>`) and BO-series
  (`<mid>_m1`, `_m2`, …) id forms. Preserve both checks in any new source.
- Both ingestion paths must produce identical downstream rows. Differences belong
  only in how the `.dem` reaches disk.
- Impact sign convention: `impact = p_after - p_before` for a CT actor, negated for T.
  Don't "normalize" this away.
- WP train/val split is `GroupShuffleSplit(groups=match_id)`. Never switch to a random
  row split — it leaks future round state and inflates AUC substantially.
- **Grouping the split is not sufficient.** A feature can encode the outcome inside
  *every* row, which no split can fix. Two did (fixed 2026-08-01): `time_remaining_s`
  was derived from the round's actual end tick, and `_info_state` snapped to the
  nearest cached visibility tick in either direction. Before adding any feature, ask:
  *could this be computed live, mid-round, without knowing how the round ends?* If not,
  it is leakage. Post-decision rows (`alive_ct = 0 or alive_t = 0`) are excluded in
  `load_training_data`, not in the sampler — `score_impact` needs them for `p_after`.
- Impact scoring needs **calibrated** probabilities, not just good ranking. Changes to
  the model/objective must keep `test/test_win_probability.py` passing.
- Adding a WP feature means: `state_sampler` emits it → `db` column + insert → 
  `win_probability.FEATURES` → retrain. Missing one of these fails silently or at insert.
- Don't invent FACEIT API fields/endpoints — verify against a real response or the docs.
  If unsure, say so.
- **Never use `0` or `"?"` as a sentinel for "unknown"** in a field where they are
  legitimate values. `ct_spread=0.0` meant both "1 player left" and "team stacked
  together"; `player_side="?"` silently took the T branch in impact scoring; a failed
  distance returning `0.0` read as "standing on top of you". Emit `None` — SQL NULL,
  and LightGBM handles missing natively.
- **`or` does not work as a null-coalesce over numerics.** `NaN` is truthy, so
  `official_end or end` never falls through, and `0` is a legitimate value. Use the
  `_first_valid` helper in `state_sampler` / `feature_extractor`.
- Credential-bearing URLs must go through `pipeline.redact_url` before any `print`,
  exception message, or DB write. Never truncate a URL by character count — FACEIT
  signed URLs carry the signature in the query string and truncation length is luck.

## Schema traps (verified 2026-08-01 — re-check before trusting)
The schema is all `CREATE TABLE IF NOT EXISTS`, so re-running `_init_schema` — which
`init_db()` does on **every** run, not just on file creation — can never add a column
to an existing table. There is no `ALTER TABLE` and no `schema_version` anywhere, so
editing the `CREATE TABLE` alone will silently do nothing to the live DB.
Two live consequences:

1. **Live DB is behind `db.py`.** `data/crosshair.db` (3.6 GB, 710 matches, 2.80 M events,
   1.18 M states) has 37 data columns in `round_states`; `insert_round_states` writes 41.
   Missing: `active_smokes_xy`, `active_infernos_xy`, `site_smoked`, `site_on_fire`.
   Any new ingest into this DB fails on insert until it's migrated or recreated.
2. **`db.py` is behind the live DB.** `_init_schema`'s `events` table has no
   `p_before` / `p_after` / `impact`, but `score_impact.py`, `main.py status`, and
   `main.py top` all read/write them. The live DB has them (added ad hoc); a **fresh**
   DB does not, so `status` and `score` break on a clean start.

Both are known and unfixed. Don't silently paper over them — propose an additive
migration (`ALTER TABLE ... ADD COLUMN`) plus matching `_init_schema` columns and let
me choose migrate-vs-recreate.

Also: `data/win_prob.lgb` currently does **not** exist. `score`, `top`, and the WP tests
all require `train` to have been run first.

## Docs accuracy
`README.md` numbers (1024 matches, 2.8 M events, 3.90 GB, AUC 0.896) are from a
**reference run**, not the current DB. Don't cite them as live state, and don't
"correct" them into README unless I ask.

## Verify before declaring done
- Run it, or the smallest relevant subset (`demo demos/mirage.dem` is the cheap path).
  Never claim it works untested.
- Report: what changed, new deps, new env vars, new/changed DB columns, and the exact
  command to run it.
- If something is blocked or unverifiable (needs a browser login, needs a trained model,
  needs the API key's Downloads scope), say so explicitly instead of assuming it passes.
