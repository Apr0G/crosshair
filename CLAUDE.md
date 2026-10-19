# CLAUDE.md — Crosshair

**RULE 0 — NEVER BREAK:** start every reply with my name. e.g. `Faig, ...`

CS2 impact-attribution engine. Scrapes FACEIT demos → parses with awpy → SQLite →
LightGBM win-probability model → per-action impact (WP delta). Treat the existing
pipeline as working software, not a draft.

> **Engine v2 (2026-10-19, `docs/ENGINE_PLAN.md` + Amendments 1 and 2).** The legacy `src/`
> pipeline described in the paragraph above and in the sections marked
> *[obsolete since Amendment 1, 2026-10-19 — see **Engine v2**]* was deleted; the
> engine is now the `engine/` package. Those sections are kept for history only. Every rule that still applies is restated, with
> its v2 location, in **Engine v2** below. Rules not marked obsolete apply unchanged.

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
  *(The `src/` / `import db` part is obsolete: engine code uses absolute imports,
  `from engine import db`; banners, aligned assignments and f-strings still apply.)*
- Don't introduce new patterns, frameworks, or abstraction layers unprompted.

## Ask before
- Changing DB schema (and DB changes are **additive-only** unless I approve otherwise
  — ship a migration, see *Schema traps*).
- Deleting or disabling code.
- Adding a dependency.
- Touching the Playwright downloader (`src/scraper.py`) — it is load-bearing and
  hard to re-verify without a live browser login. *(It moved, logic unchanged, to
  `engine/sources/scraper.py`; the rule stands.)*

## Git
- **Run `/security-review` before every commit.** No exceptions. Invoke the skill,
  read the findings, and fix or explicitly dismiss each one *before* `git commit`.
  If the review surfaces anything unresolved, stop and tell me — don't commit past it.
  Report the result in the commit summary (e.g. `security-review: clean`).
  This repo handles a FACEIT API key, browser session state, and `.env` — the review
  is mainly there to catch a secret or a session cookie leaking into a diff.
- Only commit or push when I ask.
- **Never add yourself as a contributor.** No `Co-Authored-By: Claude ...` trailer, no
  "Generated with Claude Code", no assistant name/link in commit messages, PR bodies,
  or `CONTRIBUTORS`/`AUTHORS`. Commits are authored by me alone.

---

## Engine v2

**Aim since Amendment 2 (2026-10-19 evening): sell to FACEIT / ESL FACEIT Group and let
FACEIT players analyse their own matches. The baseline tier is `faceit_10` (all ten
players level 10), not pros. Pro demos are a transfer test only. No automated access to
hltv.org (its terms forbid it); `engine/sources/hltv.py` is dormant. Every "pro" below
reads "baseline tier `faceit_10`" unless it says otherwise.**

The engine since 2026-10-19: `docs/ENGINE_PLAN.md` (phases P0–P12, Amendments 1 and 2
at the end). Progress and what is deferred: `docs/ENGINE_PROGRESS.md` (Faig maintains it).
Other docs: `docs/ENGINE_ENV.md`, `docs/ENGINE_CLI.md`, `docs/ENGINE_MODELS.md`,
`docs/ENGINE_OPS.md`, `docs/DATA_SOURCES.md`, `engine/state/SCHEMA_CHANGELOG.md`.

### Environment
- One venv, `.venv`, **Python 3.12** (3.12.10 on the Windows PC):
  `.venv\Scripts\python.exe` on Windows, `.venv/bin/python` elsewhere. Never bare
  `python`, never `pip install` outside the venv. `.venv-old311` is the dead legacy
  venv.
- Install: `.venv\Scripts\python.exe -m pip install -r requirements-engine.txt`.
  `requirements-engine.txt` is authoritative; `requirements-engine.lock` is the exact
  resolution, re-frozen after any deliberate upgrade. `pyproject.toml` reads deps from
  `requirements-engine.txt` and registers the `crosshair` console script.
- **torch is not in the requirements.** On the Windows PC it comes from AMD's ROCm
  wheels for Python 3.12 (installed: torch 2.9.1+rocm7.2.1, see `docs/ENGINE_ENV.md`);
  elsewhere install the matching torch build before `pip install -r`. Modules that do
  not train or predict must not import torch at import time.
- Secrets: `.env` (see `.env.example`), `FACEIT_API_KEY` (the API path also needs
  Downloads access on the key); `engine.cli scrape` loads `.env`. Never hardcode keys
  or commit secrets. Data lives under `data/engine/` (gitignored): `engine.db`, `demos/`,
  `parsed/`, `states/`, `actions/`, `datasets/`, `models/`, `eval/`, `reports/`, `logs/`.
  `CROSSHAIR_ENGINE_DB` overrides the DB path (tests point it at a temp file).

### Commands
```
.venv\Scripts\python.exe -m engine.cli status                    # DB counts, free space
.venv\Scripts\python.exe -m engine.cli ingest <demo.dem> --tier pro
.venv\Scripts\python.exe -m engine.cli scrape --source api --min-level 10
.venv\Scripts\python.exe -m engine.cli parse <demo.dem | dir>
.venv\Scripts\python.exe -m engine.cli build-states --all        # then build-actions --all
.venv\Scripts\python.exe -m engine.cli build-dataset --split data\engine\datasets\splits_v<N>.json
.venv\Scripts\python.exe -m engine.cli train-wp -- --vocab ... --norm ...
.venv\Scripts\python.exe -m engine.cli calibrate -- --model-dir data\engine\models\wp_v<N>
.venv\Scripts\python.exe -m engine.cli sanity --model wp_v<N> --counterfactual
.venv\Scripts\python.exe -m engine.cli analyze <demo.dem> --model wp_v<N> [--q-model q_v<N> --neighbours neighbours_v<N>]
.venv\Scripts\python.exe -m engine.cli eval --model wp_v<N> --calibration
.venv\Scripts\python.exe -m engine.cli bench analyze
.venv\Scripts\python.exe -m engine.cli inventory [--assert-targets | --models]
.venv\Scripts\python.exe -m engine.evaluation.anchors | review | robustness | usage | registry
```
Exit codes everywhere: 0 ok, 1 ran and failed, 2 usage error or missing prerequisite.
Commands not built yet are listed at the end of `docs/ENGINE_OPS.md`.

### Tests
```
.venv\Scripts\python.exe test\run_all.py [path\to\demo.dem]     # every test/engine/test_*.py
.venv\Scripts\python.exe test\engine\test_<name>.py             # one script
.venv\Scripts\python.exe -m pyflakes engine test bench          # must be clean
```
- Plain scripts, never pytest (decided 2026-08-01, still in force). One harness:
  `test/engine/_harness.py` (`check`, `skip`, `section`, `summary`); exit 0 pass,
  1 fail, 2 prerequisite missing (demo, model, golden). A test that needs real data
  exits 2 and says why; it never passes silently.
- Synthetic fixtures instead of demos: `_synthetic.py` (six stories, `STORY_FACTS`),
  `_stubs.py` (`StubPredictor`), `_cf_states.py`, `_attr_states.py`,
  `_value_states.py`. Tests write only to temp dirs, never to `data/`.
- Every module has a test script; every bug fix adds a check; every fixed review item
  (P11.2.3) adds one function to `test/engine/test_review_items.py`.

### Module map (`engine/`)
```
cli.py analyze.py       CLI (`python -m engine.cli`) and the demo -> report pipeline
config.py config.yaml   every tunable (Appendix D), validated; config_hash in manifests
paths.py manifest.py log.py   data/engine layout; run manifests; JSON-lines logging with redaction
db.py ingest.py         engine.db v1 (matches, runs, action_scores, processed); THE SEAM
sources/                scraper.py (Playwright, ask before touching), faceit_api.py, hltv.py, hf_archive.py
parse/                  schema.py (RawTables contract), awpy3_parser.py, worker.py
maps/                   assets, grid, version (+ versions.yaml), callouts, plot
state/                  schema.py (tokens, Snapshot, LIVE_SAFE), sampler, players, utility,
                        context, pairwise, geometry, builder, leakage
actions/                types.py (Action, registry), kills, utility, movement, bomb, economy,
                        enablers, extractor, windows
data/                   quarantine, inventory, splits (frozen month splits), layout, dataset, qa
model/                  tokens, set_encoder, temporal, heads, wp_model, train, checkpoints,
                        predict (Predictor), calibrate, baseline_lgbm (yardstick)
counterfactual/         interventions, apply, sanity (P7 gate), ood
attribution/            types, partition, shapley, movement, clock, enablers, engine (realized V)
value/                  descriptors, q_model, neighbours, scores (V = B + D + X), shrinkage
report/                 writer, cards, charts, schema.json
evaluation/             anchors (P11.1), review (P11.2), robustness (P11.3), usage (P12.3),
                        registry (P12.1)
```
**THE SEAM (v2):** every source ends at
`engine.ingest.register_demo(match_id, demo_path, demo_url)`: copy the `.dem` under
`data/engine/demos/<tier>/`, insert `matches` + `processed`, nothing else. Parsing,
states and actions are separate CLI stages. A source exposes
`iter_unprocessed_demos(...)` and `download_demo(url, dest)`.

### Acceptance tests (Faig's conditions, plan 1.3; "pro" = baseline tier `faceit_10` since Amendment 2)
- **C1** No faulty result from two actions on one tick: a zero-effect action gets
  exactly 0 next to a teammate's −0.5 action (`test_attribution_shapley.py`
  `null_action_same_tick`). Two bounds, decided 2026-10-19: with a stub `W`,
  `V_A == 0 ± 1e-9` and `V_B == ΔW ± 1e-9` exactly; on the real model (`--real`,
  P8.2.4) `|V_A| ≤ 0.002` and `|V_B − ΔW| ≤ 0.002` over 200 real kill windows, since a
  real model is never exactly indifferent. Conservation
  `|Σ V − (W_end − W_start)| < 1e-6` on every golden round and 1,000 sampled pro rounds.
- **C2** Every valued action carries `V, Q, B, D, R, X`; `V = B + D + X` to 1e-9 (a
  plumbing guard only: `D = Q − B` and `X = V − Q`, so it cannot fail on its own); the
  checks that can fail are `R ≤ 0`, Q tracking realized V, and a neighbour panel with
  ≥ 1 entry for ≥ 95% of pro actions (`test_decision_split.py`).
- **C3** `bench/bench_analyze.py` on the golden demo: target ≤ 60 s GPU / ≤ 180 s CPU
  end to end; hard ceiling 120 s / 300 s, above which v1 is not done.

### Decision rules and gates
- **P6.7.2** (full text in `docs/ENGINE_MODELS.md`; "pro test" = `faceit_10` test months
  since Amendment 2): adopt the NN only if it beats
  LightGBM on pro test log loss by ≥ 0.01 **and** ECE ≤ 0.02, else back to features
  before any attribution; keep FACEIT only if it helps pro log loss; keep temporal
  encoder / pairwise bias only for ≥ 0.003 log loss or better drift; hazard head AUC
  ≥ 0.80 target, 0.75–0.80 warn, < 0.75 fixed-share fallback; 4 Hz only for ≥ 20% less
  movement-credit variance at ≤ 1.6× state-building time.
- **P7 gate:** no attribution is trusted until `engine sanity --counterfactual` is
  green on the adopted model; at most five P7.3 iterations, then escalate to Faig.
- A model is used only if `docs/ENGINE_MODELS.md` has an `adopted` row for it
  (`engine.evaluation.registry adopt`); `analyze` refuses others without
  `--allow-unadopted`.

### Rules
- **Never add a feature without a `LIVE_SAFE` justification** in
  `engine/state/schema.py` (could it be computed live, mid-round, without knowing how
  the round ends?). `test_leakage.py` runs the automated audit; log the change in
  `engine/state/SCHEMA_CHANGELOG.md`. Adding a feature: schema token field +
  `LIVE_SAFE` entry → builder emits it → dataset layout → retrain.
- **Never change `sample_hz` for latency.** 2 Hz is the design value; changing it is an
  ablation decision (P6.7.2). Latency knobs, in order: `los_cache`,
  `drift_threshold_wp`, `shapley_samples`, `k_neighbours`, chart resolution.
- Style: package modules under `engine/`, absolute imports (`from engine.x import y`),
  `# ── section ──` banners, aligned assignments, f-strings, type hints, a module
  docstring saying which plan step fills it.
- Reports never contain file-system paths outside the output dir or URLs; SteamIDs are
  hashed unless `--show-ids`.

### Invariants (still in force, v2 locations)
- **Idempotency:** never double-insert a match. `engine.db.is_processed_any` checks the
  BO1 id `<mid>` and the series forms `<mid>_m1`, `_m2`, …; keep both in any new source.
  The same file from two sources is caught by the unique `demo_sha256`.
- All sources produce identical downstream rows; they differ only in how the `.dem`
  reaches disk.
- **Sign convention:** CT-positive internally; a CT actor's value is `+V_raw`, a T
  actor's `−V_raw` (impact `p_after − p_before` for CT, negated for T). Don't normalize
  it away.
- **Never a random row split.** Frozen month splits (`splits_v<N>.json`, P1.5) with
  match grouping; test months are the latest. A split file is never edited, only
  superseded.
- **Leakage:** grouping the split is not enough; a feature that encodes the outcome
  leaks in every row. See the `LIVE_SAFE` rule above.
- **Calibrated probabilities**, not just ranking: ECE ≤ 0.02 on test pro
  (`calibrate --assert`); model changes must keep it.
- **Never `0` or `"?"` for unknown** where they are legitimate values: emit `None`
  (SQL NULL, model missing-mask).
- **`or` is not a null-coalesce over numerics** (`NaN` is truthy, `0` is legitimate):
  use `engine.state.players._nan` / `_first_valid`.
- **Redaction:** credential-bearing URLs go through `engine.sources.scraper.redact_url`
  (or `engine.log.redact_text` / `redact_obj`) before any print, exception, file or DB
  write. Never truncate a URL by character count.
- **128-tick:** the tick rate comes from the demo header (`meta.tick_rate`), never a
  literal; quarantine refuses anything else (64-tick demos included).
- Don't invent FACEIT API fields or endpoints; verify against a real response or the
  docs, and say so when unsure.

### Database
`data/engine/engine.db`, `engine/db.py`, `SCHEMA_VERSION = 1`. Additive migrations in
`_MIGRATIONS` keyed by `PRAGMA user_version`; a newer file is refused. Adding a column
means `_init_schema` + a `_MIGRATIONS` entry + the version bump, with `test_db.py`
green. Schema changes stay additive-only and **ask first**. `data/crosshair.db` (the
legacy DB on the Mac) is an archive the engine never opens.

---

## Environment *[obsolete since Amendment 1, 2026-10-19 — see **Engine v2**]*
Bullets 1–3 (3.11 venv, `requirements.txt` / `.lock`) are obsolete. The `.env` /
`FACEIT_API_KEY` and Playwright-is-macOS-only bullets still apply (`engine.cli scrape`
loads `.env` itself now, not `main.py`).
- **Always use the project venv**: `.venv/bin/python` on macOS/Linux (3.11, arm64 on
  the Mac), `.venv\Scripts\python.exe` on Windows. Bare `python` on PATH is not
  guaranteed to be this venv; don't `pip install` outside it. Docs below write the
  macOS form.
- Rebuild / install: `python3.11 -m venv .venv` then
  `.venv/bin/python -m pip install -r requirements.txt -c requirements.lock`.
  `requirements.txt` is the **authoritative** dep list (compatibility ceilings);
  `requirements.lock` is the exact resolution and must be re-frozen after any
  deliberate upgrade. All wheels exist for macOS arm64 and Windows x64 (verified
  2026-10-19 on Windows).
- `pyproject.toml` is metadata only and reads its deps from `requirements.txt`.
- Secrets via env + `.env` (see `.env.example`); `main.py` loads `.env` itself. Never
  hardcode keys or commit secrets. `FACEIT_API_KEY` is used by both ingestion paths;
  the API path also needs Downloads access granted on that key.
- The Playwright path is macOS-only as written (Chrome profile + binary paths in
  `scraper.py`). On any other machine use `--source api`.

## Run *[obsolete since Amendment 1, 2026-10-19 — see **Engine v2**]*
Every command and test path below was deleted. Still in force: tests are plain
scripts, not pytest; exit 0/1/2; `test_scraper.py` (now `test/engine/`) opens real
Chrome and is not run by `run_all.py`.
```bash
.venv/bin/python src/main.py status                      # DB stats
.venv/bin/python src/main.py demo demos/mirage.dem       # process one local .dem (cheapest smoke test)
.venv/bin/python src/main.py scrape --source playwright  # or --source api
.venv/bin/python src/main.py train --eval                # → data/win_prob.lgb
.venv/bin/python src/main.py score                       # fill p_before/p_after/impact
.venv/bin/python src/main.py top -n 10
.venv/bin/python src/main.py round <match_id> <n> --engagements-only
```
Tests are **plain scripts, not pytest** (pytest isn't installed). Each exits 0 on
pass, 1 on failure, 2 when a prerequisite (demo / model / DB) is missing:
```bash
.venv/bin/python test/run_all.py [demos/faceit2.dem]   # runs everything below, tallies
.venv/bin/python test/test_invariants.py        # stdlib only: schema/feature parity, redaction
.venv/bin/python test/test_db.py                # stdlib only: fresh schema, migrations, store_match
.venv/bin/python test/test_attribution.py       # pure: conservation + Shapley on synthetic paths
.venv/bin/python test/test_feature_extractor.py # synthetic demo tables, no awpy
.venv/bin/python test/test_state_sampler.py     # synthetic demo tables, no awpy
.venv/bin/python test/test_score_impact.py      # model stubbed; grid + event scoring on sqlite
.venv/bin/python test/test_pipeline.py          # the seam + main.py commands, parser stubbed
.venv/bin/python test/test_download.py          # download_demo against a local HTTP server
.venv/bin/python test/test_attribution_windows.py [demo]   # + contract checks on a real demo
.venv/bin/python test/test_build_state.py [demo]           # needs a real demo (awpy)
.venv/bin/python test/test_win_probability.py   # needs a trained model
.venv/bin/python test/test_attribution_integration.py      # needs DB + model
.venv/bin/python test/test_attribution_compare.py          # needs DB + model; grid vs event
```
Shared bits live in `test/_harness.py` (check/summary) and `test/_synthetic.py` (a
hand-built 2v2 round as the dict of tables `extract.extract()` returns — the thing
that lets the extractor, sampler and seam be tested without awpy or a `.dem`).
**Decided (2026-08-01): stay on plain scripts. Don't propose pytest.**
`test_local_demo.py` is an inspection script (prints what a demo yields, writes
nothing — store with `main.py demo`); `test_scraper.py` opens real Chrome. Neither is
run by `run_all.py`.

## Module map *[obsolete since Amendment 1, 2026-10-19 — see **Engine v2**]*
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
src/attribution.py      pure event-boundary attribution engine (partition + conservation)
src/score_impact.py     scores events: grid bracketing (default) or event-boundary (attribution.py)
src/db.py               SQLite schema + migrations + store_match (one transaction per match)
                        also owns the shared paths: ROOT, DATA_DIR, DB_PATH, MODEL_PATH
```

**THE SEAM** *[obsolete since Amendment 1, 2026-10-19 — see **Engine v2**]* (v2 seam: `engine.ingest.register_demo`) — every ingestion source converges here; anything new that produces a
local `.dem` must plug in here rather than fork the pipeline:
`pipeline.ingest_demo(match_id, path)` → extract → events + grid states + boundary
states → `db.store_match` (one transaction). `process_one` (scrape) downloads then
calls it; `main.py demo` calls it directly. A source module only has to expose
`iter_unprocessed_demos(...)` and `download_demo(url, dest)`.

## Invariants
*(Each rule below still applies unless marked; the v2 module names are in **Engine v2 ›
Invariants**. Marked obsolete: the `GroupShuffleSplit` mechanism, the
`load_training_data` / `score_impact` note, `test/test_win_probability.py`, the
"adding a WP feature" chain, and the awpy 2 `events=EVENTS` item.)*
- **Idempotency:** never write logic that can double-insert a match. Dedup is
  `db.is_processed(match_id)`, checked for both BO1 (`<mid>`) and BO-series
  (`<mid>_m1`, `_m2`, …) id forms. Preserve both checks in any new source.
- Both ingestion paths must produce identical downstream rows. Differences belong
  only in how the `.dem` reaches disk.
- Impact sign convention: `impact = p_after - p_before` for a CT actor, negated for T.
  Don't "normalize" this away.
- WP train/val split is `GroupShuffleSplit(groups=match_id)` *(mechanism obsolete: v2
  uses frozen month splits with match grouping; the rule below stands)*. Never switch to a random
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
  *(Obsolete chain; v2 chain in **Engine v2 › Rules**.)*
- Don't invent FACEIT API fields/endpoints — verify against a real response or the docs.
  If unsure, say so.
- **FACEIT demos are 128-tick, not 64.** `extract.py` reads the real tickrate off the
  demo and passes it through `tables["tick_rate"]`; `feature_extractor` and
  `state_sampler` must use it, never a literal. The old hardcoded `64` made every
  time value in the pre-2026-08-01 corpus wrong by exactly 2×.
- *(Legacy awpy 2 only; v2 uses awpy 3, calls in `docs/ENGINE_ENV.md`.)*
  **Always pass `events=EVENTS` to `awpy Demo.parse()`.** awpy 2.0.2's default list
  asks for `player_sound`, which CS2 renamed to `player_footstep`, so the default
  fails on every modern demo — and the failure mode depends on the demoparser2
  version (KeyError on 0.41.2/.3, EntityNotFound on 0.40.x/0.41.0/.1, and on 0.41.4
  a hard crash: an unsigned underflow that tries to allocate 2**64-1 bytes).
  `demo.footsteps` reads the old key too, so build footsteps from
  `demo.events["player_footstep"]` directly. Also: demoparser2's `parse_ticks(ticks=…)`
  subset argument crashes on 0.41.4 — don't reach for it as an optimisation.
- **Never use `0` or `"?"` as a sentinel for "unknown"** in a field where they are
  legitimate values. `ct_spread=0.0` meant both "1 player left" and "team stacked
  together"; `player_side="?"` silently took the T branch in impact scoring; a failed
  distance returning `0.0` read as "standing on top of you". Emit `None` — SQL NULL,
  and LightGBM handles missing natively.
- **`or` does not work as a null-coalesce over numerics.** `NaN` is truthy, so
  `official_end or end` never falls through, and `0` is a legitimate value. Use the
  `_nan` / `_first_valid` helpers in `feature_extractor` (`state_sampler` imports them).
- Credential-bearing URLs must go through `scraper.redact_url` (re-exported by
  `pipeline`) before any `print`, exception message, or DB write. Never truncate a URL by character count — FACEIT
  signed URLs carry the signature in the query string and truncation length is luck.

## Schema and migrations *[obsolete since Amendment 1, 2026-10-19 — see **Engine v2**]*
The v2 database and its migration rule are in **Engine v2 › Database**; the rule
"DB changes remain additive-only and ask-first" stands.

`CREATE TABLE IF NOT EXISTS` can never add a column to an existing table, so `db.py`
carries ordered additive migrations (`_MIGRATIONS`, keyed by `PRAGMA user_version`,
current `SCHEMA_VERSION = 2`). `db._conn()` creates the schema on a fresh file and
migrates an old one on every open; `init_db()` just does that up front.
**Adding a column means three edits**: `_init_schema`, a new `_MIGRATIONS` entry with
`SCHEMA_VERSION` bumped, and (for a state field) `_STATES_SQL`. `test/test_db.py`
migrates a v0 database and `test/test_invariants.py` cross-checks the three lists;
both must stay green. DB changes remain additive-only and ask-first.

`data/` is gitignored: `crosshair.db` and `win_prob.lgb` exist only where the pipeline
has been run. `score`, `top`, and the WP tests need `train` to have run first.

## Docs accuracy
`README.md` numbers (1024 matches, 2.8 M events, 3.90 GB, AUC 0.896) are from a
**reference run**, not the current DB. Don't cite them as live state, and don't
"correct" them into README unless I ask.

## Verify before declaring done
- Run it, or the smallest relevant subset (`demo demos/mirage.dem` is the cheap path
  *(obsolete: in v2 run the module's `test/engine/test_*.py` script, or
  `test/run_all.py`)*). Never claim it works untested.
- Report: what changed, new deps, new env vars, new/changed DB columns, and the exact
  command to run it.
- If something is blocked or unverifiable (needs a browser login, needs a trained model,
  needs the API key's Downloads scope), say so explicitly instead of assuming it passes.
