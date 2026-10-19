# Crosshair Engine v2 — Implementation Plan

**Status:** executed through P12 on synthetic data (Amendment 1, 2026-10-19); re-aimed at FACEIT players and FACEIT/EFG as buyer by **Amendment 2 (2026-10-19 evening)** at the end of this file. Where a section below says "pro" as the baseline, Amendment 2 reads "baseline tier `faceit_10`".
**Written:** 2026-10-19
**Supersedes for the engine:** `src/score_impact.py`, `src/win_probability.py`, the
`round_states` feature set in `src/state_sampler.py`, and the proposal in
`docs/design/impact-attribution.md` (its partition engine, `src/attribution.py`, becomes
the skeleton of the new attribution module; its grid-bracketing critique still stands).
**Keeps:** the ingestion pipeline (`scraper.py`, `faceit_api.py`, `pipeline.py`,
`db.py`, `extract.py` for the legacy path), the plain-script test convention, every rule
in `CLAUDE.md`.

---

## 0. How to read this document

- Phases are `P0` … `P12`. Subphases are `P3.2`. Steps are `P3.2.4`. Steps are written
  so that someone who has never spoken to me can execute them in order.
- Every step has four fields: **Do** (the action), **Touches** (files, tables, configs),
  **Output** (the artifact that exists afterwards), **Verify** (the command or check that
  proves it before the next step starts).
- Every subphase ends with a **Test loop**. A test loop is: run the named tests, read every
  failure, fix the code (never the test, unless the test is wrong and the reason is
  written down), rerun, repeat until green, then run the full suite once more. No step
  of the next subphase starts while a test loop is red.
- Every phase ends with **Definition of done**, **Risks and fallbacks**, and **Effort**
  (working days for one person, assuming the machine does the waiting).
- Tests are plain Python scripts with exit codes `0` pass, `1` fail, `2` prerequisite
  missing (decided 2026-08-01, reaffirmed today; no pytest).
- "Ask-before" items from `CLAUDE.md` (schema changes, deletions, dependencies, touching
  `scraper.py`) are listed explicitly in each phase. Approving this plan approves exactly
  those listed items and nothing else; anything new found during execution is asked at
  the end of the work session, not silently done.
- Numbers marked **target** are goals we test against. Numbers marked **estimate** are
  my expectations and may be wrong; the plan says what to do if they are.

---

## 1. Goal, audience, acceptance criteria

### 1.1 Goal

*[Superseded by Amendment 2, section 1: the buyer is FACEIT/EFG and the users are FACEIT players; the engine is unchanged.]*

Turn Crosshair into an engine that assigns every catalogued action in a CS2 demo a
realized value, an expected value, a decision score, and an execution score, from a
calibrated neural win-probability model, in under two minutes per map, at a quality that
can be sold to HLTV, ESL FACEIT Group, Bayes Esports, or a tier-one organization.

### 1.2 Primary audience

*[Superseded by Amendment 2, section 1.]*

Professional analysts and the companies that serve them. Every number shown to them is
computed and evaluated on professional demos. FACEIT data is used to train, never to
set the professional baseline (decided today).

### 1.3 Faig's three conditions, as acceptance tests

| # | Condition (Faig, 2026-10-19) | Acceptance test (must pass before v1 is "done") |
|---|---|---|
| C1 | No faulty result caused by two actions at the same tick. An action with zero effect must get exactly zero even if a teammate's −0.5 action lands on the same tick. | `test/engine/test_attribution_shapley.py::null_action_same_tick` builds the exact A/B scenario and asserts `V_A == 0 ± 1e-9` and `V_B == ΔW ± 1e-9` with a stub `W`; on the real model (`--real`, P8.2.4) `|V_A| ≤ 0.002` and `|V_B − ΔW| ≤ 0.002` over 200 real kill windows (one bound per variant, decided 2026-10-19). Also: conservation per round `|Σ V − (W_end − W_start)| < 1e-6` on every round of the golden demo and on 1,000 sampled pro rounds. |
| C2 | Every action is split into decision (good/bad) and execution (good/bad), with "what people usually do here". | For every valued action the report carries `V, Q, B, D, R, X`, the identity `V = B + D + X` holds to 1e-9 (a plumbing guard: `D = Q − B`, `X = V − Q`; the informative checks are `R ≤ 0`, Q tracking realized V and the panel coverage), and a neighbour panel with ≥ 1 entry exists for ≥ 95% of pro actions (`test/engine/test_decision_split.py`). |
| C3 | Analysing one demo takes a reasonable time. | `bench/bench_analyze.py` on the golden demo: **target** ≤ 60 s wall-clock on the Windows PC with GPU, ≤ 180 s CPU-only, measured end to end including parsing. Hard ceiling 120 s GPU / 300 s CPU; above the ceiling v1 is not done. |

### 1.4 Non-goals for v1

- No web UI, no API. CLI and report files only (decided today).
- No live/broadcast mode. The model is causal so live is possible later, but nothing in
  v1 streams.
- No per-player ratings product beyond the aggregates the report needs. Rating design is
  a later project built on these outputs.
- No extrapolated "the model thinks you should have …" suggestions in the regret score
  (decided today: regret uses seen actions only).

---

## 2. Locked decisions and assumptions

### 2.1 Decisions made with Faig on 2026-10-19

| ID | Decision | Consequence in the plan |
|---|---|---|
| D1 | Start the engine from scratch; keep the data pipeline. | New package `engine/`; `src/` ingestion untouched except where listed. |
| D2 | Neural win-probability model over per-player state, not LightGBM over team aggregates. LightGBM stays only as the yardstick baseline. | P6.1 trains the baseline first; P6.7 must beat it on pro held-out months or the NN is not adopted. |
| D3 | Simultaneity handled by Shapley over state-editing counterfactuals. | P7 (intervention library) and P8 (attribution engine). |
| D4 | Enabler credit (flash, smoke, trade, HE, molotov) via the hazard heads, not fixed percentages. Trades included, clamped to a 5–40% band; total enabler share capped at 50%. | P6.2 heads, P8.5 clamps. |
| D5 | Decision/execution split: `V = B + D + X`, regret `R` from actions seen ≥ `n_min` times in similar states only. Extrapolated alternatives shown only as a separately flagged suggestion, and not in v1. | P9. |
| D6 | Sampling rate 2 Hz, as a single config value `sample_hz`. Boundary samples at every action tick ±1 regardless of rate. | P0.4, P3.1. 4 Hz is tested on a subset in P6.7. |
| D7 | *[Replaced by Amendment 2, section 2.]* Train on pro + FACEIT level 10 with a coarse tier token; oversample pro; final pro-only fine-tune; evaluate and calibrate on pro only. Never feed per-player external ratings. | P1, P5.3, P6.3, P6.5. |
| D8 | Counterfactual sanity suite is a hard gate before any attribution is trusted. | P7 is a gate for P8. |
| D9 | CLI only for v1. | P10. |
| D10 | Plan saved to `docs/ENGINE_PLAN.md`, uncommitted until asked. | This file. |
| D11 | `requirements.txt` stays authoritative for the ingestion venv; the engine gets its own venv (Python 3.12) because AMD's Windows PyTorch build pairs with 3.12. | P0.1. |
| D12 | awpy moves to version 3 for the Rust line-of-sight and nav-mesh, behind a parity test against the current awpy 2.0.2 parse. | P2.1. |
| D13 | Training tensors in parquet shards; SQLite keeps metadata, run registry, and scored results. Schema changes additive with migrations. | P5. |
| D14 | Tests stay plain scripts. Many of them, with test loops at every subphase. | Everywhere; index in Appendix C. |
| D15 | Shrink per-player execution aggregates with empirical Bayes `n/(n+k)`. | P9.4. |
| D16 | Never add the assistant as contributor; `/security-review` or its manual equivalent before every commit; commit only when Faig asks. | Unchanged. |

### 2.2 Standing rules from `CLAUDE.md` that constrain this plan

- Idempotent ingestion (`is_processed` for `<mid>` and `<mid>_m1` forms).
- Impact sign: `p_after − p_before` for CT actor, negated for T. Kept for `V`.
- Never a random row split; time split by month plus match grouping.
- No feature that cannot be computed live mid-round (leakage rule). The leakage audit is
  automated in P3.6.
- Never `0` or `"?"` as a sentinel for unknown. `None` → SQL NULL → model missing-mask.
- `or` is not a null-coalesce over numerics; use `_nan` / `_first_valid`.
- Credential-bearing URLs through `redact_url` before any print, exception, or write.
- FACEIT demos are 128-tick; the tick rate read from the demo drives every window.
- Always pass `events=EVENTS` to awpy 2 (legacy path). awpy 3 has its own API; P2.1
  documents the equivalent.

### 2.3 Assumptions (mine, stated so they can be challenged)

| ID | Assumption | If wrong |
|---|---|---|
| A1 | AMD's native Windows PyTorch (ROCm 7.2.x, PyTorch 2.9, Python 3.12) works on the RX 9060 XT for training a ~3M-parameter transformer. | Fallback 1: WSL2 + Linux ROCm. Fallback 2: CPU training with smaller batches, overnight runs. P0.1 tests all three. |
| A2 | demoparser2 / awpy 3 parse a pro demo in 10–15 s single-process on the 7800X3D. | If > 30 s, parse fewer props and compute derived values ourselves; P2.6 bench decides. |
| A3 | awpy 3's `VisibilityChecker` answers a line-of-sight query in well under 100 µs. | If slower, precompute an area-to-area visibility table per map (P3.5 fallback). |
| A4 | *[False, 2026-10-19; replaced by Amendment 2, section 3.]* A few thousand pro maps since October 2023 are obtainable (HLTV downloads, the public Hugging Face archive). | If fewer, lean harder on FACEIT with the tier token and say so in the evaluation report. |
| A5 | 300–500 GB free on a fast drive for demos, parquet, and the DB. | Shard by map and process in waves; P0.3 checks free space. |
| A6 | The win model, once it sees positions and pairwise sight, beats the LightGBM baseline on pro held-out log loss and Brier. Supported by Szmida & Toka (Sloan 2025): THGNN log loss 0.459 vs XGBoost 0.535 on 114 Inferno matches. | P6.7 is the decision point. If it does not beat the baseline, we stop and rethink features before touching attribution. |

---

## 3. Glossary

- **Demo**: a `.dem` replay file. **Map**: one played map within a match (BO1 = 1 map).
- **Tick**: server simulation step; 128 per second on FACEIT and pro servers.
- **Snapshot / state `s_t`**: everything the engine knows at time `t`: player tokens,
  utility tokens, context token, pairwise table. Built only from information available
  at `t`.
- **Grid sample**: a snapshot taken on the fixed `sample_hz` grid.
- **Boundary sample**: a snapshot taken one tick before and one tick after an action.
- **Terminal state**: the snapshot at round end; its win probability is pinned to the
  outcome (1.0 CT win, 0.0 T win).
- **W(s)**: the model's CT round-win probability for state `s`. Always CT-side.
- **Action**: a catalogued, attributable thing a player did (Section P4). Has an actor,
  a type, parameters, and either a tick (instantaneous) or a window (durative) or a
  round scope (buys).
- **Intervention**: a function that edits a state to undo one action's effect
  (`undo_kill`, `remove_utility`, `unflash`, …). Section P7.1.
- **Coalition**: a subset of the actions in one event window; `s0 + S` is the pre-state
  with exactly the actions in `S` applied.
- **φ_a (phi)**: the Shapley share of a window's WP change assigned to action `a`.
- **V**: realized value of an action for its actor, signed for the actor's side.
  Sum of its direct Shapley shares, enabler shares, and movement shares.
- **Enabler**: another player's action whose effect was active when an instantaneous
  event happened and raised that event's likelihood (flash on the victim, smoke
  blocking cover, a recent teammate death that set up a trade, HE/molotov damage).
- **Hazard head**: model output `q_p(s)` = probability player `p` dies within the next
  5 seconds. Also `plant15(s)`, `defuse15(s)`.
- **Q(s,a)**: expected realized value of action type/parameters `a` in state `s`.
- **B(s)**: expected realized value of whatever players typically do in state `s`.
- **D = Q − B**: decision score. **X = V − Q**: execution score.
- **R = Q(s,a) − max Q(s,a')** over alternatives `a'` seen ≥ `n_min` times in similar
  states: regret, always ≤ 0.
- **Neighbour panel**: the `k` nearest pro states in the encoder embedding, with the
  distribution of actions taken there and their mean `V`.
- **Tier**: coarse skill bucket token: `pro`, `faceit_10`, `faceit_8_9`, `other`.
- **Golden demo**: one pinned pro demo whose parsed counts and report are stored as
  golden files; every regression test runs against it.
- **Test loop**: run → fix → rerun until green → run full suite. Defined in Section 0.

---

## 4. Architecture overview

```
 demo (.dem)
   │  P2 parse (awpy 3 / demoparser2, multiprocess)         ~10–15 s
   ▼
 raw tables (polars)  ──►  P3 state builder                ~5 s
   │                        grid @ sample_hz + boundary samples
   │                        player / utility / context tokens
   │                        pairwise LOS + distances (awpy 3 VisibilityChecker, NavMesh)
   ▼
 states.parquet  ──►  P4 action extractor  ──►  actions.parquet
   │                                                │
   ▼                                                ▼
 P6 win model W(s) + hazard heads  ◄────────  P7 interventions (state editing)
   │  set transformer + causal temporal transformer           ~5–10 s / map on GPU
   ▼
 P8 attribution: Shapley over coalitions, enabler splits, movement, clock   → V
   ▼
 P9 expected value: Q(s,a), B(s), neighbours (FAISS over pro embeddings)   → D, R, X
   ▼
 P10 CLI `engine analyze` → report/ (JSON, Markdown, PNG)               total ≤ 60 s
```

New package layout (all new files; nothing in `src/` is deleted):

```
engine/
  __init__.py
  config.py            load + validate engine/config.yaml
  config.yaml          every tunable in one place (Appendix D)
  paths.py             data/engine/* layout, manifest helpers
  manifest.py          run manifests (git sha, config hash, inputs, versions)
  log.py               structured logging, redaction hook
  parse/
    awpy3_parser.py    demo → raw polars tables
    worker.py          multiprocess parse of a directory, retries, failure log
    parity.py          awpy2 vs awpy3 comparison for P2.1
  maps/
    assets.py          download/pin awpy-data meshes, cache
    callouts.py        position → callout, per-map callout polygons
    grid.py            position → coarse grid cell
    version.py         map version detection from demo header / map hash
    <map>/callouts.json
  state/
    sampler.py         grid + boundary tick selection
    players.py         player token features
    utility.py         utility tokens
    context.py         context token
    pairwise.py        LOS, distances, smoke-between
    builder.py         orchestrates one match → states.parquet
    schema.py          column names, dtypes, null policy, schema version
    leakage.py         automated leakage audit
  actions/
    types.py           Action dataclass, type registry
    kills.py, utility.py, movement.py, bomb.py, economy.py
    enablers.py        enabler detection for instantaneous events
    extractor.py       orchestrates one match → actions.parquet
  data/
    inventory.py       counts by map/tier/month
    splits.py          frozen time splits
    dataset.py         states → tensors, vocabularies, normalization stats
    qa.py              distribution report + thresholds
  model/
    tokens.py          token embedding MLPs
    set_encoder.py     set transformer with pairwise bias
    temporal.py        causal temporal transformer
    heads.py           win, hazard, plant15, defuse15, time-to-engagement
    wp_model.py        assembles the network
    baseline_lgbm.py   yardstick
    train.py           training loop, oversampling, fine-tune stage
    calibrate.py       temperature scaling, reliability, ECE
    predict.py         batched inference with caching
    checkpoints.py     save/load + manifest
  counterfactual/
    interventions.py   undo_kill, remove_utility, unflash, unplant, freeze_position, ...
    sanity.py          the P7 gate suite
    ood.py             off-distribution detector
  attribution/
    partition.py       breakpoints, windows, drift intervals
    shapley.py         exact + sampled Shapley over coalitions
    enablers.py        hazard-based splits with clamps
    movement.py        freeze-position movement credit
    clock.py           team clock bucket
    engine.py          round → valued actions
  value/
    descriptors.py     action descriptor encoding
    q_model.py         Q and B networks
    neighbours.py      FAISS index, panel
    scores.py          D, R, X, shrinkage
  report/
    cards.py, charts.py, writer.py
  cli.py               engine analyze | train-wp | train-q | calibrate | eval | sanity | bench | inventory | build-dataset
test/engine/           all engine tests (Appendix C)
bench/                 timing scripts
docs/ENGINE_PLAN.md    this file
```


---

## P0 — Environment, scaffolding, test infrastructure

Purpose: make the machine, the repo, and the test harness ready so every later phase
can be verified the moment it is written. Nothing here touches model logic.

### P0.1 Engine venv with GPU PyTorch

- **P0.1.1 Do:** install Python 3.12 (per-user, silent installer, same way 3.11 was
  installed on 2026-10-19). **Touches:** machine only. **Output:**
  `%LOCALAPPDATA%\Programs\Python\Python312\python.exe`. **Verify:**
  `python312 -c "import sys; print(sys.version)"` prints 3.12.x.
- **P0.1.2 Do:** create `.venv-engine` with Python 3.12 at the repo root. **Touches:**
  `.gitignore` (add `.venv-engine/`). **Output:** `.venv-engine\Scripts\python.exe`.
  **Verify:** the interpreter runs and `pip --version` points inside `.venv-engine`.
- **P0.1.3 Do:** follow AMD's "PyTorch on Windows" release notes for ROCm 7.2.x and
  install the wheel set they list for Python 3.12 (torch 2.9 + ROCm runtime wheels).
  **Touches:** `requirements-engine.txt` (new; records the exact wheel URLs or index and
  versions), `requirements-engine.lock` (frozen after install). **Output:** torch
  importable. **Verify:** `python -c "import torch; print(torch.__version__,
  torch.cuda.is_available(), torch.cuda.get_device_name(0))"` prints `True` and the
  9060 XT's name (the ROCm build exposes the GPU through the `torch.cuda` API).
- **P0.1.4 Do:** run a 60-second GPU smoke benchmark: a 3-layer transformer on random
  tokens of shape `(512, 24, 128)`, forward + backward, 200 iterations, fp32 and bf16.
  **Touches:** `bench/bench_gpu.py` (new). **Output:** iterations/s printed and saved to
  `data/engine/bench/gpu_<date>.json`. **Verify:** no NaN, no crash, bf16 ≥ fp32 speed
  or bf16 marked unsupported in the JSON.
- **P0.1.5 Do:** if P0.1.3 or P0.1.4 fails, set up WSL2 (Ubuntu 24.04) with Linux ROCm
  and PyTorch; repeat P0.1.3–P0.1.4 there. Record which path works in
  `docs/ENGINE_ENV.md` (new). If both fail, record CPU-only and continue; P6 effort
  doubles. **Verify:** `docs/ENGINE_ENV.md` states the chosen path and the benchmark
  numbers.
- **P0.1.6 Do:** install the remaining engine dependencies (Appendix E) into
  `.venv-engine`, then freeze `requirements-engine.lock`. **Verify:** `pip check` clean;
  `python -c "import awpy, polars, pyarrow, lightgbm, sklearn, faiss, matplotlib,
  yaml"` succeeds.

**Ask-before items in P0.1:** new dependencies (Appendix E lists every one with the
reason). Approving the plan approves that list.

**Test loop P0.1:** `test/engine/test_env.py` (new): asserts Python ≥ 3.12 in the engine
venv, torch imports, device string recorded in `docs/ENGINE_ENV.md` matches
`torch.cuda.is_available()`, every Appendix E package imports with the locked version.
Exit 2 (not 1) when run from the wrong venv.

### P0.2 Repo scaffolding

- **P0.2.1 Do:** create the `engine/` package tree from Section 4 with empty modules
  and one-line docstrings stating each module's job. **Touches:** `engine/**`.
  **Verify:** `python -c "import engine"` from the engine venv; `pyflakes engine` clean.
- **P0.2.2 Do:** create `test/engine/` and `bench/` with `__init__.py`-free plain
  scripts, plus `test/engine/_harness.py` that re-exports `test/_harness.py` (one
  harness, two locations). **Verify:** `python test/engine/_harness.py` exits 0.
- **P0.2.3 Do:** extend `test/run_all.py` with `--suite legacy|engine|all` (default
  `all`), discovering `test/engine/test_*.py` under the engine venv when present, and
  printing which interpreter ran each suite. **Touches:** `test/run_all.py`.
  **Verify:** `run_all.py --suite engine` lists zero tests without error on an empty
  dir; `--suite legacy` reproduces today's 9 pass / 4 skip.
- **P0.2.4 Do:** add `engine/log.py`: structured logs (JSON lines) to
  `data/engine/logs/<run_id>.jsonl` plus console; every record passes through
  `redact_text` from `pipeline.py` before writing. **Verify:**
  `test/engine/test_log_redaction.py`: a signed FACEIT URL in a log call never appears
  in the file.

**Test loop P0.2:** `run_all.py --suite all` green; pyflakes clean on `engine/`.

### P0.3 Disk layout and space

- **P0.3.1 Do:** create `data/engine/{demos/pro,demos/faceit,parsed,states,actions,
  datasets,models,eval,reports,bench,logs,cache/maps}`. All under `data/`, already
  gitignored. **Verify:** `engine/paths.py` has one constant per directory and
  `test/engine/test_paths.py` asserts they exist after `paths.ensure()`.
- **P0.3.2 Do:** `engine/paths.py::free_space_gb(path)` and a startup check in the CLI
  that warns below 100 GB and refuses batch parsing below 20 GB. **Verify:** unit test
  with a monkeypatched `shutil.disk_usage`.

### P0.4 Config

- **P0.4.1 Do:** write `engine/config.yaml` with every tunable in Appendix D, grouped
  by phase, each with a comment stating units and the step that uses it. Starting
  values: `sample_hz: 2`, `boundary_offset_ticks: 1`, `hazard_horizon_s: 5`,
  `plant_horizon_s: 15`, `defuse_horizon_s: 15`, `temporal_window_s: 10`,
  `shapley_exact_max: 6`, `shapley_samples: 200`, `enabler_total_cap: 0.5`,
  `trade_share_band: [0.05, 0.40]`, `n_min_regret: 30`, `k_neighbours: 200`,
  `drift_threshold_wp: 0.02`, `movement_min_units: 32`.
- **P0.4.2 Do:** `engine/config.py` loads YAML into a frozen dataclass, validates
  types and ranges (e.g. `sample_hz ∈ {1,2,4,8}`, bands ordered), computes
  `config_hash` (sha256 of the canonical YAML). **Verify:**
  `test/engine/test_config.py`: loads default; rejects `sample_hz: 3`; rejects band
  `[0.4, 0.05]`; hash changes when any value changes; hash stable across key order.
- **P0.4.3 Do:** every CLI command prints `config_hash` and writes it into its
  manifest. **Verify:** covered by P0.5 manifest tests.

### P0.5 Run manifests and reproducibility

- **P0.5.1 Do:** `engine/manifest.py::write(run_dir, kind, inputs, outputs, extra)`
  records: git sha (and dirty flag), `config_hash`, engine venv package versions,
  awpy-data release pin, map asset hashes, input file hashes, wall-clock per stage,
  hostname, GPU name. **Verify:** `test/engine/test_manifest.py`: round-trip JSON, all
  required keys present, hashes change when an input file changes.
- **P0.5.2 Do:** rule: every artifact directory under `data/engine/` contains a
  `manifest.json`; readers refuse artifacts whose `schema_version` they do not know.
  **Verify:** `test/engine/test_manifest.py::unknown_schema_refused`.

### P0.6 Benchmark harness (for condition C3)

- **P0.6.1 Do:** `bench/_bench.py` with `timed(stage)` context manager writing per-stage
  seconds to a dict, and `bench/bench_analyze.py` that runs the full pipeline on the
  golden demo and prints a table: parse, states, pairwise, actions, model, attribution,
  value, report, total. Writes `data/engine/bench/analyze_<date>.json`. **Verify:**
  runs end to end once P10 exists; until then runs the stages that exist and marks
  others `n/a`.
- **P0.6.2 Do:** budget file `bench/budget.yaml` with per-stage targets summing to
  60 s GPU / 180 s CPU; `bench_analyze.py --assert` exits 1 if any stage exceeds its
  budget by more than 50% or the total exceeds the hard ceiling. **Verify:** later
  phases run it in their test loops.

### P0.7 Golden demo and golden files

- **P0.7.1 Do:** choose one pro demo (tier-one event, 128-tick, standard map, full
  match, no forfeits) as the golden demo; store under `data/engine/demos/golden/` with
  its sha256 in `test/engine/golden/GOLDEN.json` (committed; the demo itself is not).
  **Verify:** `test/engine/test_golden_present.py` exits 2 with a clear message when the
  demo is absent, 0 when the hash matches, 1 when it mismatches.
- **P0.7.2 Do:** rule: every stage that produces per-match output has a golden-file test
  comparing counts and checksums for the golden demo
  (`test/engine/golden/<stage>.json`). Updating a golden file requires a one-line note
  in `test/engine/golden/CHANGELOG.md` saying why. **Verify:** the CHANGELOG exists with
  a header.

**Definition of done P0:** engine venv with working torch on the chosen device;
`run_all.py --suite all` green; config, manifest, log, paths, bench harness in place
with their tests; golden demo pinned; `docs/ENGINE_ENV.md` written.
**Risks:** AMD Windows PyTorch immaturity (fallbacks in P0.1.5); Python 3.12 wheels for a
dependency missing (Appendix E lists alternatives).
**Effort:** 2–3 days.

---

## P1 — Data acquisition

*[P1.1 and P1.2 are replaced by Amendment 2, section 3. P1.3–P1.5 stand.]*

Purpose: enough demos, labelled by tier, map, map version, and month, with frozen time
splits, so that every later experiment uses the same data and nothing leaks across time.

### P1.1 Pro demos

- **P1.1.1 Do:** write `docs/DATA_SOURCES.md` listing each source, what it offers, its
  terms, and the rate limits we will respect: HLTV match pages (demo download links,
  manual or lightly automated with ≥ 10 s spacing and a browser-like session; check the
  site terms and stop if they forbid it), the public Hugging Face archive
  `Vasy7777/cs2-demo-archive` (~113 GB; verify its README, license, and content list
  before downloading), event-organizer public demo dumps where they exist. **Verify:**
  document reviewed by Faig at the end of the session (listed in open questions).
- **P1.1.2 Do:** `engine/data/sources/hltv.py` (new, separate from `scraper.py`, which is
  not touched): given a list of HLTV match URLs or IDs, fetch the demo archive link,
  download with the existing validated downloader (`scraper.download_demo` is reused as
  a function import, not modified), unpack `.rar`/`.zip` archives of multi-map matches
  into one `.dem` per map, name files `<event>_<date>_<team1>-<team2>_<map>.dem`.
  **Verify:** `test/engine/test_hltv_source.py` against a local HTTP server serving a
  tiny zip with two fake `.dem` files; checks naming, extraction, redaction of URLs in
  errors, and that a corrupt archive leaves no partial files.
- **P1.1.3 Do:** `engine/data/sources/hf_archive.py`: list the archive's files, download
  by map and month in waves (so P0.3 free-space rules hold), checksum each file against
  the archive's listing when available. **Verify:** unit test on a fake listing;
  integration test skipped (exit 2) without network.
- **P1.1.4 Do:** metadata extraction at download time into `engine_matches` (P5.2):
  `match_key`, `source`, `event`, `event_tier` (1/2/3 per the source's label), `date`,
  `team_ct_start`, `team_t_start`, `map`, `demo_sha256`, `file_bytes`,
  `game_version` (from demo header), `tick_rate` (from header), `status`
  (`downloaded|parsed|states|actions|quarantined`). **Verify:**
  `test/engine/test_matches_meta.py` inserts and queries rows; `demo_sha256` unique
  index prevents duplicates.
- **P1.1.5 Do:** dedup: two files with the same sha256 are one match; the same match
  from two sources keeps the earlier row and logs the duplicate. **Verify:** test with a
  duplicate insert.
- **P1.1.6 Do:** target volumes (**targets**): ≥ 3,000 pro maps spanning October 2023 to
  the present, every active-duty map ≥ 200 maps, every month ≥ 50 maps. Inventory in
  P1.3 reports shortfalls. **Verify:** `engine inventory --assert-targets` exits 1 when a
  target is missed (used as a reminder, not a gate).

### P1.2 FACEIT level 10 demos

- **P1.2.1 Do:** reuse the existing API path (`main.py scrape --source api`) with a new
  optional filter `--min-level 10` implemented in `faceit_api.iter_unprocessed_demos`
  (additive parameter, default `None` = current behaviour). Players are discovered as
  today; a match qualifies if all ten players are at or above the level. **Touches:**
  `src/faceit_api.py`, `src/main.py`, `src/pipeline.py` (pass-through). `scraper.py`
  untouched. **Verify:** `test/test_pipeline.py` gains a case with the fake source
  filtering by level; legacy tests still green.
- **P1.2.2 Do:** when `--engine-copy` is set, the downloaded `.dem` is also copied to
  `data/engine/demos/faceit/` with a metadata row (tier `faceit_10`) before the legacy
  pipeline deletes its temp file. Default off, so legacy behaviour is unchanged.
  **Verify:** pipeline test asserts the copy exists and the legacy DB row is unchanged.
- **P1.2.3 Do:** target (**target**): ≥ 50,000 FACEIT 10 maps initially, grown weekly;
  per map ≥ 3,000. **Verify:** inventory report.

### P1.3 Inventory

- **P1.3.1 Do:** `engine inventory`: counts of matches by `tier × map × month × status`,
  disk usage per directory, failure counts, written to
  `data/engine/eval/inventory_<date>.md` and printed. **Verify:**
  `test/engine/test_inventory.py` on a synthetic `engine_matches` table with known
  counts.
- **P1.3.2 Do:** `--assert-targets` compares against P1.1.6 / P1.2.3 and lists
  shortfalls. **Verify:** test with a deliberately short table.

### P1.4 Quarantine and validation

- **P1.4.1 Do:** every new demo gets a parse smoke test (P2.6 worker with
  `--smoke`): header readable, tick rate 128 (else quarantine with reason), ≥ 13
  rounds (a completed MR12 map; 13-0 to 13-2 are legitimate, amended 2026-10-19), both teams have 5 players, map in the known pool, duration plausible. Failures
  move the file to `data/engine/demos/quarantine/<reason>/` and set
  `status = quarantined`. **Verify:** `test/engine/test_quarantine.py` with synthetic
  failure cases (truncated file, 64-tick header, 3 rounds).
- **P1.4.2 Do:** weekly `engine inventory` includes quarantine counts by reason.

### P1.5 Frozen time splits

- **P1.5.1 Do:** `engine/data/splits.py::make_splits()` writes
  `data/engine/datasets/splits_v<N>.json`: `train` = all months except the last four,
  `val` = the two months before the last two, `test` = the last two months; matches
  assigned by `date`; pro and FACEIT share the same month boundaries. The file lists
  every `match_key` per split and is never edited, only superseded by `v<N+1>` with a
  note. **Verify:** `test/engine/test_splits.py`: no match in two splits; every match in
  exactly one; month boundaries respected; file hash recorded in manifests.
- **P1.5.2 Do:** `make_splits --holdout-maps <map>` optional: a whole map held out for
  an out-of-map generalization check (used once in P6.7, not in the default split).

**Definition of done P1:** sources documented; downloaders tested against local
servers; metadata table populated; inventory command works; quarantine works; splits
file v1 written and tested. Volumes reached are reported, not gated.
**Risks:** source terms forbid automated download (then manual batches, documented);
archive contents differ from its README (verify before bulk download); FACEIT API
rate limits (existing backoff).
**Ask-before items:** additive parameter in `faceit_api.iter_unprocessed_demos`; new
SQLite table `engine_matches` (P5.2 migration).
**Effort:** 3–4 days of work plus download time running in the background.

---

## P2 — Parsing and map assets (awpy 3)

Purpose: a fast, verified parse of any demo into raw tables, and per-map geometry
(line of sight, nav mesh, callouts) the state builder needs.

### P2.1 awpy 3 parity

- **P2.1.1 Do:** install awpy 3.x (pinned) in the engine venv (already in P0.1.6).
  Read its `Demo` API and list in `docs/ENGINE_ENV.md` the exact calls for: rounds,
  kills, damages, grenades (throw, detonate, expire), bomb events, player per-tick
  snapshots with the props in Appendix A.1, footsteps/sounds if exposed, tick rate, map
  name, game version. **Verify:** the list exists and each call runs on the golden demo.
- **P2.1.2 Do:** `engine/parse/parity.py`: parse the golden demo with the legacy path
  (`src/extract.py`, awpy 2.0.2 in `.venv`) and the new path (awpy 3 in
  `.venv-engine`), write both to JSON summaries, compare: round count, round start/end
  ticks, winner per round, kill count and every (tick, attacker, victim) triple, damage
  sums per round, grenade counts by type, bomb event ticks, tick rate, map name, player
  names. **Verify:** `test/engine/test_parse_parity.py` exits 0 on zero differences;
  any difference is listed with both values. Tolerance is zero on counts and ticks;
  float positions compared to 0.01 units.
- **P2.1.3 Do:** if parity fails on a field, decide per field: awpy 3 is right (document
  the legacy bug), awpy 2 is right (file an awpy 3 issue, implement a workaround in
  `awpy3_parser.py`), or the field differs by definition (document the mapping). No
  silent acceptance. **Verify:** `docs/ENGINE_ENV.md` has a "parser differences" table.
- **P2.1.4 Do:** run parity on 20 more demos across maps and both tiers once P1 has
  them. **Verify:** `parity.py --batch` summary, all zero or documented.

### P2.2 Map assets

- **P2.2.1 Do:** `engine/maps/assets.py::ensure(map_name)`: downloads the collision mesh
  and `.nav` for the map via awpy's data manager, pinned to one awpy-data release
  recorded in config (`awpy_data_release`), cached under `data/engine/cache/maps/`,
  sha256 recorded. **Verify:** `test/engine/test_map_assets.py`: cache hit does not
  re-download (local file server), pin mismatch refuses.
- **P2.2.2 Do:** per map, a hand-built fixture `test/engine/fixtures/los_<map>.json`
  with 8 point pairs that must be visible and 8 that must be blocked (chosen from
  well-known positions: e.g. Mirage A ramp to jungle visible, A site to B site
  blocked). **Verify:** `test/engine/test_los_fixtures.py` runs awpy 3's
  `VisibilityChecker.is_visible` on each pair; all 16 must agree. Any disagreement is
  investigated before the map is used (the fixture may be wrong; fix and note).
- **P2.2.3 Do:** nav mesh: `NavMesh.find_area(pos)` for every kill position in the
  golden demo must return an area for ≥ 99.5% of positions. **Verify:**
  `test/engine/test_nav_coverage.py`.
- **P2.2.4 Do:** all-pairs area distance table per map (shortest path length on the nav
  graph, in units), computed once, cached as a `uint16` matrix with the area index.
  **Verify:** symmetric, zero on the diagonal, triangle inequality holds on 1,000 random
  triples within 1 unit.

### P2.3 Callouts

- **P2.3.1 Do:** CS2 `.nav` files carry no place names. Approach: for each map keep
  `engine/maps/<map>/callouts.json`: a list of `{name, polygons: [[x,y],...], z_min,
  z_max}`. Seed it from the map's `env_cs_place` entities if extractable (awpy issue
  #390 describes the approach), otherwise hand-draw from community callout maps.
  **Verify:** `test/engine/test_callouts.py`: polygons are valid (non-self-intersecting,
  ≥ 3 points), names unique per map, every map in the active pool has a file.
- **P2.3.2 Do:** `callouts.lookup(map, x, y, z)` returns the callout containing the
  point, else the nearest callout within 128 units, else `None` (never `"?"`).
  **Verify:** coverage test: ≥ 99% of player positions in the golden demo resolve to a
  callout; a random off-map point returns `None`.
- **P2.3.3 Do:** a plotting helper `engine/maps/plot.py` draws callouts over the map
  radar image for eyeballing; output saved to `data/engine/eval/callouts_<map>.png`.
  **Verify:** Faig reviews the images once per map (listed in the review loop P11.2).

### P2.4 Grid cells and map version

- **P2.4.1 Do:** `grid.cell(map, x, y)`: 128-unit square cells over the map bounds from
  the nav mesh extents; cell ids stable per map version. **Verify:** unit test on
  corner points; ids within `[0, n_cells)`.
- **P2.4.2 Do:** `version.detect(demo_header)`: map version key from the game version
  string plus the map file's hash when exposed; a table in
  `engine/maps/versions.yaml` names known reworks (e.g. a map's layout change date) so
  `map_version` is `<map>@<version>`. Position embeddings are keyed by `map_version`.
  **Verify:** test with two synthetic headers mapping to two versions; unknown versions
  fall back to `<map>@unknown` and are logged.

### P2.5 Parser output tables

- **P2.5.1 Do:** `awpy3_parser.parse(path) -> RawTables` (polars frames): `rounds`,
  `ticks` (player per-tick props at full rate or at the requested subsample; we request
  every tick for correctness of boundary samples and subsample later), `kills`,
  `damages`, `grenades` (throw, detonate, expire, with entity ids to link them),
  `smokes` (start/end tick, position), `infernos` (start/end, position, extent),
  `flashes` (detonate tick, per-player blind duration), `bomb` (plant start/complete,
  defuse start/complete/abort, drop, pickup), `footsteps`/`sounds` if available,
  `weapon_fire`, `player_blind`, `item_pickup/drop`, metadata (`map`, `tick_rate`,
  `game_version`, `server`). Column names and dtypes in Appendix A.1. **Verify:**
  `test/engine/test_raw_schema.py` checks every table's columns and dtypes on the golden
  demo and on a synthetic `RawTables` fixture.
- **P2.5.2 Do:** golden counts for the golden demo: rows per table, kills per round,
  stored in `test/engine/golden/raw.json`. **Verify:** `test/engine/test_raw_golden.py`.

### P2.6 Parse worker and bench

- **P2.6.1 Do:** `engine/parse/worker.py`: parses a directory with `N` processes
  (default `os.cpu_count() − 2`), each demo → `data/engine/parsed/<match_key>/` parquet
  per table plus `manifest.json`; retries once on failure; failures appended to
  `data/engine/parsed/_failures.jsonl` with the exception type (redacted); already
  parsed matches skipped unless `--force`. **Verify:**
  `test/engine/test_parse_worker.py` with a fake parser: idempotent rerun, failure
  logged, `--force` reparses.
- **P2.6.2 Do:** `bench/bench_parse.py`: time single-process parse of the golden demo
  (target ≤ 15 s) and 6-process throughput on 24 demos (target ≥ 0.4 demos/s).
  **Verify:** numbers recorded in `data/engine/bench/`; if the single-process target is
  missed, reduce requested props to the Appendix A.1 minimum and re-bench; if still
  missed, record it and raise the stage budget in `bench/budget.yaml` with a note.

**Test loop P2:** `run_all.py --suite engine` (parity, LOS fixtures, nav coverage,
callouts, raw schema, raw golden, worker) green; `bench_parse.py` within budget.
**Definition of done P2:** awpy 3 parity documented and zero-diff or explained; map
assets for every active-duty map with passing LOS fixtures; callouts for every map
reviewed visually; parse worker processes a directory idempotently within budget.
**Risks:** awpy 3 API churn (pin the version; parity test catches changes);
`env_cs_place` extraction not feasible (hand-drawn callouts, more hours); map rework
mid-project (versions.yaml + retrain, see P12.2).
**Effort:** 4–6 days (callouts are the long pole).

---

## P3 — State builder

Purpose: turn raw tables into the snapshots the model reads: grid samples at
`sample_hz`, boundary samples at every action tick ±1, each with player tokens, utility
tokens, a context token, and the pairwise table. Every feature must be computable live.

### P3.1 Sampling

- **P3.1.1 Do:** `sampler.grid_ticks(round, tick_rate, sample_hz)`: ticks from
  `freeze_end` to `official_end` (inclusive of the terminal tick) every
  `tick_rate / sample_hz` ticks, rounded to integers, deduplicated. **Verify:**
  `test/engine/test_sampler.py`: at 128 tick and 2 Hz the step is 64 ticks; at 64
  tick, 32; count equals `floor(duration_s * sample_hz) + 1`; first tick is
  `freeze_end`; last is the terminal tick.
- **P3.1.2 Do:** `sampler.boundary_ticks(actions, offset=1)`: for every instantaneous
  action at tick `τ`, add `τ − offset` and `τ + offset`; for durative actions add the
  same around `window_start` and `window_end`; clip into the round; dedupe. **Verify:**
  for a synthetic round with kills at 2000 and 4000: boundary set is
  `{1999, 2001, 3999, 4001}`; a kill on the round's last tick clips to the terminal tick.
- **P3.1.3 Do:** `sampler.all_ticks = sorted(grid ∪ boundary)` with a `kind` column:
  `grid`, `boundary_pre`, `boundary_post`, `terminal`. A tick can carry several kinds
  (stored as a list). **Verify:** kinds are correct for the synthetic round; no duplicate
  ticks; monotone.
- **P3.1.4 Do:** the sampler must not use the round's end tick for anything except
  knowing when to stop sampling. The clock feature (P3.4) uses the round timer, never
  the end tick. **Verify:** leakage audit P3.6.

### P3.2 Player tokens

- **P3.2.1 Do:** `players.token(tick_row, ctx) -> dict` with exactly the fields of
  Appendix A.2, in that order. Rules: unknown → `None`; dead players keep `alive = 0`
  and their last known position (for trade and movement logic) but have `health = 0`
  and all equipment `None`; `seconds_since_last_seen_by_enemy` is `None` until first
  seen; `flash_duration_remaining` is `0.0` when not blind (0 is a legitimate value
  here, documented). **Verify:** `test/engine/test_player_token.py` on synthetic rows:
  alive, dead, blind, scoped, carrying bomb, with defuser, unknown weapon → `None`.
- **P3.2.2 Do:** weapon class vocabulary `engine/state/vocab_weapons.yaml`: rifle,
  sniper, smg, shotgun, lmg, pistol, knife, grenade, zeus, c4, unknown; every weapon
  name awpy emits maps to one class; unmapped names log once and map to `unknown`.
  **Verify:** every weapon name in the golden demo is mapped (test lists unmapped).
- **P3.2.3 Do:** `seconds_since_last_shot`, `is_reloading`, `is_walking`, `is_ducking`,
  `is_scoped`, `is_defusing`, `is_planting` from per-tick props when exposed by the
  parser; otherwise derived (e.g. walking from speed band 0 < speed ≤ 100 units/s with
  the walk key unknown → `None`, not a guess). **Verify:** synthetic tests per flag.
- **P3.2.4 Do:** `spotted_by_mask`: 5-bit mask of enemies currently spotting the player
  (from the parser's spotted props), plus derived `seconds_since_last_seen_by_enemy`
  updated causally (only past ticks). **Verify:** synthetic: a player spotted at tick
  1000 has `0.0` at 1000 and `1.0` one second later; never negative; `None` before the
  first spot.
- **P3.2.5 Do:** `heard_enemy`: reuse the existing sound-cue logic from
  `feature_extractor._heard_enemy` (import, do not copy) with `tick_rate` passed
  explicitly. **Verify:** the existing test coverage in `test/test_feature_extractor.py`
  plus one engine test on the synthetic round.
- **P3.2.6 Do:** `player_id_hash`: sha1 of the SteamID truncated to 63 bits, stored as
  int; the vocabulary of ids seen in training is saved in P5.3; ids not in it map to
  `UNK` at inference. **Verify:** stability test (same id → same hash across runs).
- **P3.2.7 Do:** `tier`: from the match metadata; one of `pro`, `faceit_10`,
  `faceit_8_9`, `other`. **Verify:** test asserts the enum.

### P3.3 Utility tokens

- **P3.3.1 Do:** `utility.tokens(tick, raw) -> list[dict]` with Appendix A.3 fields:
  `kind` (smoke, fire), `x,y,z`, `time_left_s`, `owner_side`, `owner_id_hash`,
  `radius_units` (smoke 144, fire from extent). Active when `start_tick ≤ tick <
  end_tick`. Flashes and HE are instantaneous and produce no token (their effects are on
  players: blind duration, health). Decoys: token `kind = decoy` with `time_left_s`.
  **Verify:** synthetic: a smoke from 1500 to 1500 + 18·tick_rate is a token at 1600 and
  not at 1500 + 19·tick_rate; `time_left_s` decreases by 0.5 per grid step at 2 Hz.
- **P3.3.2 Do:** cap at `max_utility_tokens = 16` per snapshot (config), keeping the
  newest; log when the cap is hit. **Verify:** synthetic with 20 smokes → 16 tokens and
  one log line.

### P3.4 Context token

- **P3.4.1 Do:** `context.token(tick, round, raw, meta)` with Appendix A.4 fields.
  `time_remaining_s` is the live round clock: `115 − (tick − freeze_end)/tick_rate`
  before plant; `40 − (tick − plant_tick)/tick_rate` after plant (bomb timer 40 s on
  current servers; config `bomb_timer_s`), never below 0, never derived from the round's
  end tick. **Verify:** synthetic: at plant tick value is 40.0; 10 s later 30.0; before
  plant at freeze_end 115.0.
- **P3.4.2 Do:** `score_ct`, `score_t`, `round_num`, `half`, `is_overtime`,
  `loss_streak_ct/t` (consecutive losses entering the round), `money_ct/t`,
  `equip_ct/t` (sum of alive players' equipment at the tick), `alive_ct/t`,
  `bomb_planted`, `bomb_site` (A/B/None), `time_since_plant_s`, `defuse_in_progress`,
  `defuse_with_kit`, `bomb_carrier_alive`, `map_version`, `tier`. **Verify:**
  synthetic tests for each; `bomb_site` is `None` not `"?"` before plant.

### P3.5 Pairwise table

- **P3.5.1 Do:** `pairwise.table(tick_tokens, map_assets, utility_tokens) ->
  (10×10)` with channels: `los` (0/1/None if either dead), `dist_units`,
  `nav_dist_units` (from P2.2.4 by area pair; `None` if either area unknown),
  `smoke_between` (0/1: the segment between the two players intersects any active
  smoke sphere), `same_team`. LOS computed only for alive enemy pairs (≤ 25 queries),
  symmetric. **Verify:** `test/engine/test_pairwise.py`: symmetry, diagonal `None`,
  smoke_between true for a synthetic smoke centred on the segment midpoint and false
  when the smoke is 300 units off the segment; LOS from the P2.2.2 fixtures.
- **P3.5.2 Do:** cache LOS per `(area_a, area_b)` within a match when both players stay
  in the same areas between consecutive snapshots and no smoke changed (positions move
  within an area, so the cache is approximate; config `los_cache: true` and the exact
  mode is used for golden-file generation). **Verify:** golden-demo comparison exact vs
  cached: agreement ≥ 99% of pairs; bench shows the speedup.
- **P3.5.3 Fallback:** if `bench_states.py` shows pairwise above its budget even with
  the cache, precompute an area-to-area visibility matrix per map (sampling 5 points
  per area) and use it instead of per-query ray casts. **Verify:** same agreement test.

### P3.6 Leakage audit (automated)

- **P3.6.1 Do:** `leakage.audit(builder)`: runs the builder twice on the synthetic
  round, once with the true outcome and once with the winner flipped and the
  `official_end` moved by +5 s, and asserts every feature of every non-terminal snapshot
  is identical. **Verify:** `test/engine/test_leakage.py` passes; a deliberately leaky
  feature added in the test (round duration) is caught.
- **P3.6.2 Do:** static list `engine/state/schema.py::LIVE_SAFE` naming every feature,
  each with a one-line justification of why it is knowable mid-round; the test fails if
  a column exists without a justification. **Verify:** same test file.
- **P3.6.3 Do:** causality: a snapshot at tick `t` must not read any raw row with tick
  `> t`. Implement by slicing raw tables to `≤ t` before feature code runs (the builder
  passes the slice, not the full frame). **Verify:** test feeds a raw table with a
  poisoned future row (impossible value) and asserts no feature changes.

### P3.7 Labels and terminal states

- **P3.7.1 Do:** labels are attached after features are built, in a separate step:
  `round_won_ct` (0/1), `round_end_reason`, and for the auxiliary heads, computed from
  the future: `dies_within_5s[p]`, `plant_within_15s`, `defuse_within_15s`,
  `time_to_next_engagement_s` (capped at 30). Labels live in separate columns prefixed
  `y_` and are dropped at inference by schema rule. **Verify:**
  `test/engine/test_labels.py` on the synthetic round: A kills C at 2000 → C's
  `y_dies_within_5s` is 1 at every snapshot in `(2000 − 5·tr, 2000]` and 0 before.
- **P3.7.2 Do:** terminal snapshot flagged `is_terminal = 1`, `y_round_won_ct` set,
  and the model's training loss pins it (P6.3). **Verify:** exactly one terminal
  snapshot per round.

### P3.8 Output and schema

- **P3.8.1 Do:** `builder.build(match_key) -> states.parquet` with one row per
  snapshot, player tokens as 10 struct columns `p0..p9` ordered CT first then T, each
  sorted by SteamID within the team (a fixed, arbitrary order; the model is
  permutation-invariant so order does not matter, but a fixed order makes golden files
  stable), utility tokens as a list column, context fields flat, pairwise as a
  `10×10×5` fixed-size list, labels `y_*`, plus `match_key`, `round_num`, `tick`,
  `kinds`, `schema_version`. **Verify:** `test/engine/test_states_schema.py` checks
  every column name and dtype against `schema.py`; `schema_version` increments are
  logged in `engine/state/SCHEMA_CHANGELOG.md`.
- **P3.8.2 Do:** golden file `test/engine/golden/states.json`: row count, per-kind
  counts, column checksums for the golden demo. **Verify:**
  `test/engine/test_states_golden.py`.
- **P3.8.3 Do:** `bench/bench_states.py`: time per stage (sampling, players, utility,
  context, pairwise, write) on the golden demo; **target** ≤ 8 s total at 2 Hz on the
  7800X3D. **Verify:** within budget or P3.5.3 fallback engaged.

### P3.9 Synthetic fixtures

- **P3.9.1 Do:** extend `test/_synthetic.py` (legacy, keep working) with a new
  `test/engine/_synthetic.py::make_raw_tables(tick_rate=128, sample_hz=2)` that emits
  the awpy 3 `RawTables` shape for the same 2v2 story (A, B CT; C, D T; smoke 1500; HE
  1800; A kills C at 2000; D plants at 3000; B kills D at 4000; end 4040) plus a second
  story with a flash on D at 1900 followed by the kill at 2000, and a third with two
  kills on the same tick (A kills C and C kills B at 2000, the simultaneity case).
  **Verify:** `test/engine/test_synthetic.py` asserts the stories' counts.
- **P3.9.2 Do:** every engine test that needs states builds them from these fixtures
  through the real builder, never by hand, so the fixtures double as an integration test
  of the builder. **Verify:** grep in the test index (Appendix C) shows no hand-built
  states.

**Test loop P3:** all `test/engine/test_sampler|player_token|utility_token|context|
pairwise|leakage|labels|states_schema|states_golden|synthetic.py` green;
`bench_states.py` within budget; `run_all.py --suite all` green.
**Definition of done P3:** golden demo builds to `states.parquet` within budget, every
column justified as live-safe, leakage audit automated, golden counts stored.
**Risks:** parser does not expose a prop (document as `None`, never fake it); pairwise
too slow (fallback P3.5.3); utility radii wrong (fixture tests against known smoke
radius).
**Effort:** 5–7 days.

---

## P4 — Action extractor (the vocabulary)

Purpose: a complete, deterministic catalogue of attributable actions per round with
actor, type, parameters, timing, and the enablers active for each instantaneous event.

### P4.1 Action dataclass and registry

- **P4.1.1 Do:** `actions/types.py::Action` fields: `action_id` (deterministic:
  sha1 of `match_key|round|type|actor|tick`), `match_key`, `round_num`, `actor_id_hash`,
  `actor_side`, `kind` (`instant|durative|round`), `type`, `subtype`, `tick`
  (instant), `window_start_tick`, `window_end_tick` (durative), `params` (dict,
  JSON-serializable, keys per type listed in Appendix A.5), `enablers` (list of
  `{action_id, kind}` filled in P4.5), `state_pre_tick`, `state_post_tick`. **Verify:**
  `test/engine/test_action_types.py`: id determinism, JSON round-trip, invalid kind
  rejected.
- **P4.1.2 Do:** a registry mapping `type → extractor function` so adding a type is one
  entry plus one test file. **Verify:** registry test lists every type in Appendix A.5.

### P4.2 Instantaneous actions

- **P4.2.1 Kill.** One per row of `kills` (excluding suicides and team kills marked
  `team_kill = 1`, which become `type = team_kill` with the same fields). Params:
  `weapon`, `weapon_class`, `headshot`, `through_smoke`, `attacker_blind`,
  `victim_blind`, `wallbang` (penetrated > 0), `noscope`, `distance_units`,
  `attacker_callout`, `victim_callout`, `victim_health_before`, `assister_id_hash`,
  `flash_assister_id_hash`. **Verify:** golden count equals `len(kills)`; synthetic
  stories produce exactly the expected kills; every kill has `state_pre_tick = tick−1`,
  `state_post_tick = tick+1`.
- **P4.2.2 Damage.** Non-lethal damage grouped per `(attacker, victim)` into
  engagements separated by ≥ 2 s of no damage between the pair; one action per
  engagement at the tick of the first damage, `window_end_tick` at the last damage (kind
  stays `instant`, valued at the first tick; the window is metadata). Params:
  `damage_total`, `hits`, `weapon_class`, `armor_damage`. **Verify:** synthetic HE at
  1800 damaging C and D produces two damage actions for the thrower.
- **P4.2.3 Flash detonation.** One per flash detonate. Params: `blinded_enemies`
  (list of id hashes with durations), `blinded_teammates`, `blinded_self`,
  `land_callout`, `throw_callout`, `pop_delay_s`. **Verify:** synthetic flash story.
- **P4.2.4 HE detonation.** Params: `enemies_damaged`, `teammates_damaged`,
  `damage_total`, `land_callout`. **Verify:** synthetic HE.
- **P4.2.5 Peek.** For every alive enemy pair where `los` flips from 0 to 1 between
  consecutive snapshots and at least one of the two moved ≥ `movement_min_units` in the
  last 0.5 s: the mover (or, if both moved, the one with the larger displacement) is the
  actor; the other is `target`. Params: `from_callout`, `target_callout`,
  `target_spotted_me_first` (from spotted masks). One peek per pair per LOS onset.
  **Verify:** synthetic: B walking at 250 u/s toward D creates a peek by B; a pair that
  gains LOS because a smoke faded (no movement) creates no peek.
- **P4.2.6 Bomb events.** `plant_start`, `plant_complete` (params `site`,
  `planter_callout`, `ct_alive`, `t_alive`), `defuse_start` (params `kit`),
  `defuse_complete`, `defuse_abort`, `bomb_drop`, `bomb_pickup`. **Verify:** synthetic
  plant at 3000 produces `plant_start` and `plant_complete`; golden counts.

### P4.3 Durative actions

- **P4.3.1 Smoke.** Window `[detonate_tick, expire_tick]` (parser-supplied; nominal
  18 s if missing, with `pop_delay_s` fallback as today's `attribution_window`).
  Params: `land_callout`, `throw_callout`, `throw_tick`, `blocks_pairs_at_pop`
  (count of enemy-enemy pairs whose `smoke_between` turned 1 at pop). **Verify:**
  synthetic smoke story; windows scale with tick rate (existing test class).
- **P4.3.2 Molotov / incendiary.** Window from fire start to fire end; params
  `land_callout`, `enemies_damaged`, `enemies_displaced` (enemies inside the extent at
  start who left within 2 s). **Verify:** synthetic.
- **P4.3.3 Rotation.** For an alive player, a callout change where the new callout is
  held ≥ 2 s and the nav distance between the two callouts' centroids ≥
  `rotation_min_units` (config, 600). Window from leaving the old callout to arriving.
  Params: `from_callout`, `to_callout`, `nav_dist_units`, `duration_s`,
  `toward_bomb` (nav distance to bomb decreased). **Verify:** synthetic B walking from
  its start callout to D's area creates one rotation with the right window; jitter
  across a callout border and back creates none.
- **P4.3.4 Hold.** An alive player with speed < 20 u/s for ≥ `hold_min_s` (5) in one
  callout. Window = the stationary span. Params: `callout`, `duration_s`,
  `facing_callout` (callout at 500 units along the view direction, `None` if none).
  **Verify:** synthetic: A stands still from 1000 to 2000 → one hold of ≥ 7.8 s at 128
  tick; a player who moves every 3 s produces none.
- **P4.3.5 Save (retreat).** Post-plant or last 20 s of the round, an alive player whose
  nav distance to the bomb (or to the nearest enemy when no bomb) increases by ≥ 800
  units over 5 s with no shot fired. Window = the retreat span. Params: `from_callout`,
  `to_callout`, `teammates_alive`. **Verify:** synthetic retreat story (add a fourth
  story: C runs away after the plant).
- **P4.3.6 Decoy.** Window from land to expire; params `land_callout`. Low priority;
  include for completeness.

### P4.4 Round-level actions

- **P4.4.1 Buy.** One per player at `freeze_end`: `loadout_class` from equipment value
  bands (config: `pistol < 1000`, `eco < 2000`, `half < 3500`, `force < 4500`, else
  `full`), `equip_value`, `money_before`, `money_after`, `has_armor`, `has_helmet`,
  `has_kit`, `utility_count`, `primary_class`; plus a team-level `team_buy` action with
  `team_class` (majority class) and `total_equip`. Kind `round`, `tick = freeze_end`,
  `state_pre` = the economy-only prior state (Appendix B.6), `state_post` = the
  freeze-end snapshot. **Verify:** synthetic: known equipment values map to classes;
  bands are inclusive/exclusive exactly as config states.
- **P4.4.2 Drop / give.** A weapon dropped and picked up by a teammate within 5 s.
  Params: `weapon_class`, `receiver_id_hash`. **Verify:** synthetic drop story (fifth
  story).

### P4.5 Enabler detection

- **P4.5.1 Do:** for every instantaneous event `e` (kill, damage, plant_complete,
  defuse_complete) list enablers active at `tick(e) − 1`:
  - `flash`: victim's `flash_duration_remaining > 0`, link to the flash action that
    caused it (most recent flash blinding the victim).
  - `smoke_cover`: an active smoke with `smoke_between = 1` for the pair (actor,
    any alive enemy other than the victim) at `tick−1`, link to that smoke action.
  - `trade_setup`: a teammate of the actor died within `trade_window_s` (5) before
    `tick` within 1,200 units of the victim; link to that teammate's death (the kill
    action whose victim is the teammate); the enabler's beneficiary is the dead
    teammate.
  - `he_damage`: HE damage to the victim within 3 s; link to the HE action.
  - `fire_pressure`: victim on fire or inside a fire extent within 3 s; link to the
    molotov action.
  - `info`: the actor's team had `spotted` or `heard` the victim within 3 s via a
    teammate (link to the teammate's hold/peek if one exists, else `team_info` with no
    action link; valued to the team clock bucket).
  **Verify:** `test/engine/test_enablers.py` on the flash story (kill at 2000 has a
  `flash` enabler), the smoke story, a trade story (sixth story: C kills A at 2000, B
  kills C at 2100 → B's kill has `trade_setup` whose beneficiary is A), and negatives.
- **P4.5.2 Do:** enabler lists are deterministic and sorted by `(kind, action_id)`.
  **Verify:** two runs produce identical `actions.parquet` checksums.

### P4.6 Simultaneity grouping

- **P4.6.1 Do:** `extractor.group_windows(actions)`: instantaneous actions whose
  ticks are within `simultaneity_ticks` (config, 2) of each other form one event
  window `[min_tick − 1, max_tick + 1]`; durative actions contribute their start
  (`pop`) and end (`fade`) as instantaneous pseudo-actions `smoke_pop`, `smoke_fade`,
  `fire_start`, `fire_end` so every WP jump has an owner. Every window stores its member
  action ids. **Verify:** the two-kills-same-tick story yields one window with two
  members; kills 3 ticks apart yield two windows; a smoke popping at the same tick as a
  kill yields one window with `[kill, smoke_pop]`.
- **P4.6.2 Do:** windows never overlap; the gap between consecutive windows and grid
  samples defines drift intervals (P8.1). **Verify:** test asserts non-overlap and full
  coverage of the round timeline.

### P4.7 Output, golden, bench

- **P4.7.1 Do:** `actions.parquet` per match with the dataclass fields flattened,
  `params` and `enablers` as JSON strings, plus a `windows.parquet` with
  `window_id, start_tick, end_tick, member_action_ids`. **Verify:**
  `test/engine/test_actions_schema.py`.
- **P4.7.2 Do:** golden counts per type for the golden demo in
  `test/engine/golden/actions.json`. **Verify:** `test/engine/test_actions_golden.py`.
- **P4.7.3 Do:** invariants test `test/engine/test_actions_invariants.py`: every
  action has an actor who was alive at `tick − 1` (except the victim-side pseudo
  entries); every tick inside its round; windows ordered; each kill row produces exactly
  one kill action; each smoke exactly one smoke action and one pop and one fade
  pseudo-action; no action has a `None` type; extraction is idempotent.
- **P4.7.4 Do:** `bench/bench_actions.py`, **target** ≤ 3 s on the golden demo.

**Test loop P4:** all P4 tests green; golden counts stored; bench within budget;
`run_all.py --suite all` green.
**Definition of done P4:** every type in Appendix A.5 implemented with positive and
negative synthetic tests, golden counts, invariants test, deterministic output.
**Risks:** movement types (peek, rotation, hold, save) are heuristics and will be
noisy; mitigated by conservative thresholds in config, by the review loop P11.2, and by
the fact that their value is computed by the same Shapley machinery as everything else,
so a mis-detected hold simply gets a near-zero value rather than corrupting others.
**Effort:** 5–6 days.

---

## P5 — Storage, datasets, QA

Purpose: a stable home for states, actions, tensors, and results; vocabularies and
normalization fitted on the training split only; a QA report that fails loudly.

### P5.1 Parquet layout

- **P5.1.1 Do:** `data/engine/states/<map_version>/<yyyy-mm>/<match_key>.parquet`,
  same for `actions/` and `windows/`; each directory gets a `manifest.json` listing
  files and the builder's `schema_version` and `config_hash`. **Verify:**
  `test/engine/test_layout.py` writes a synthetic match and finds it by map and month.
- **P5.1.2 Do:** a `catalog.parquet` under `data/engine/` with one row per match:
  `match_key, tier, map_version, month, n_rounds, n_states, n_actions, states_path,
  actions_path, schema_version, built_at`. Rebuilt by `engine inventory --rebuild-catalog`.
  **Verify:** catalog rows equal files on disk.

### P5.2 SQLite additive migration (ask-before item)

- **P5.2.1 Do:** `SCHEMA_VERSION = 3` in `src/db.py`, migration adds tables:
  `engine_matches` (P1.1.4 columns), `engine_runs` (`run_id, kind, started_at,
  finished_at, config_hash, git_sha, manifest_path, status`), `engine_action_scores`
  (`action_id, match_key, round_num, run_id, V, V_direct, V_enabler, V_movement, Q, B,
  D, R, X, method, n_coalition, residual, flags`). No column of any existing table is
  changed. **Verify:** `test/test_db.py` gains: v2 → v3 migration on a copy of a v2
  file; fresh file creates v3; `test_invariants.py` cross-check extended to the new
  insert lists.
- **P5.2.2 Do:** indices on `engine_action_scores(match_key, round_num)` and
  `(run_id)`. **Verify:** `EXPLAIN QUERY PLAN` test uses the index for the report query.

### P5.3 Dataset builder

- **P5.3.1 Do:** `dataset.build(split_file, tiers, maps) -> shards`: reads the catalog,
  selects matches per split, writes tensor shards
  `data/engine/datasets/<split>/<shard>.npz` of fixed size (e.g. 50k snapshots), each
  with arrays: `player_num [N,10,F_p]`, `player_cat [N,10,C_p]`, `player_mask [N,10]`,
  `util_num [N,16,F_u]`, `util_cat`, `util_mask`, `ctx_num [N,F_c]`, `ctx_cat`,
  `pair [N,10,10,5]`, `y_win [N]`, `y_hazard [N,10]`, `y_plant15`, `y_defuse15`,
  `y_tte`, `is_terminal`, `round_key [N]`, `t_s [N]` (seconds from freeze end), `tier`,
  `tick_kinds`. Windows for the temporal model are formed at load time from `round_key`
  and `t_s`. **Verify:** `test/engine/test_dataset.py` on the synthetic matches: shapes,
  masks, labels match the states file; no snapshot from a `val` match in a `train`
  shard.
- **P5.3.2 Do:** normalization stats (mean/std per numeric feature, computed on
  `train` only, saved to `datasets/norm_v<N>.json`); categorical vocabularies (weapon
  classes, callouts per map version, grid cells per map version, player ids, tiers) with
  reserved `UNK=0`, `PAD=1`, saved to `datasets/vocab_v<N>.json`. **Verify:** vocab
  contains every train category; a val-only callout maps to `UNK` and is counted in the
  QA report; stats recomputation is deterministic.
- **P5.3.3 Do:** `--sample-hz 4` variant builds a second dataset for the P6.7 rate
  ablation on a subset (one map, 300 pro matches). **Verify:** row count roughly 2×.

### P5.4 Data QA report

- **P5.4.1 Do:** `qa.report(split)`: per feature per map: null %, min/max, mean, a
  10-bin histogram; per tier: rounds, snapshots, actions by type; outlier rules
  (speed > 320 u/s, health > 100, negative time left, > 5 alive per side) with counts.
  Written to `data/engine/eval/qa_<split>_<date>.md`. **Verify:**
  `test/engine/test_qa.py` on synthetic data with one planted outlier → reported.
- **P5.4.2 Do:** thresholds in config (`qa_max_null_pct` per feature family,
  `qa_max_outliers`); `engine build-dataset --assert-qa` exits 1 when violated. **Verify:**
  test with planted violations.

**Test loop P5:** `test_db.py` (migration), `test_invariants.py`, `test_layout.py`,
`test_dataset.py`, `test_qa.py` green; legacy suite still green.
**Definition of done P5:** catalog, shards, vocab, norm stats for the real splits;
migration v3 applied to a copy of the production DB without error; QA report reviewed.
**Ask-before items:** the v3 migration (listed here; approving the plan approves it).
**Effort:** 3–4 days.

---

## P6 — Win-probability model

Purpose: a calibrated, causal, permutation-invariant model `W(s)` with hazard heads,
proven better than the tree baseline on professional held-out months.

### P6.1 LightGBM yardstick

- **P6.1.1 Do:** `baseline_lgbm.py`: flatten each snapshot to a vector: context fields,
  per-team aggregates (alive, hp, armor, equip, utility counts, min/mean nav distance
  to bomb, spread), per-player features sorted by health within team (so the vector is
  permutation-invariant), pairwise summaries (count of enemy pairs with LOS, min enemy
  distance), tier, map_version. Train on `train` with early stopping on `val` (the frozen
  time split, no cross-validation), objective binary logloss. **Verify:**
  `test/engine/test_baseline.py` trains on the synthetic dataset and predicts shapes;
  real run writes `models/baseline_v<N>/` with manifest.
- **P6.1.2 Do:** evaluate on `test` (pro only) and record log loss, Brier, AUC, ECE,
  reliability table per map and per round phase (0–30 s, 30–75 s, post-plant, last
  15 s) in `eval/baseline_v<N>.md`. These are the numbers the NN must beat. **Verify:**
  the file exists with all tables.

### P6.2 Architecture

- **P6.2.1 Token embeddings.** `tokens.py`: player token → `d_model = 128` via: numeric
  features normalized → linear; categoricals (weapon class, callout, grid cell, tier,
  side, player id) → embeddings summed; `None` → per-feature learned missing vector
  plus a presence bit. Utility tokens and the context token get their own small MLPs
  to `d_model`. **Verify:** `test/engine/test_tokens.py`: output shapes; a `None`
  feature changes the output differently from `0`.
- **P6.2.2 Set encoder.** `set_encoder.py`: `L_set = 3` pre-norm transformer blocks over
  the sequence `[ctx, p0..p9, u0..u15]` with key-padding masks for dead players and
  absent utility; pairwise bias: the `10×10×5` table passes through a linear layer to
  `n_heads = 4` scalars added to attention logits for player-player positions (zeros
  elsewhere); pooled by attention pooling with one learned query → `h_t ∈ R^256`.
  **Verify:** `test/engine/test_set_encoder.py`: permuting players (and the pairwise
  table accordingly) changes `h_t` by < 1e-5; masking a dead player's features (keeping
  the alive bit) changes nothing; batch-independence (one sample's output does not
  depend on the batch).
- **P6.2.3 Temporal encoder.** `temporal.py`: causal transformer, `L_time = 2`, over the
  snapshots of the same round within the last `temporal_window_s = 10` seconds (variable
  count because of boundary samples), with relative time encodings from the actual
  `Δt` in seconds; output `z_t`. At inference the context window is built from the
  factual past; counterfactual edits touch only the present snapshot (and, for the
  post-state, the post snapshot), never the history. **Verify:**
  `test/engine/test_temporal.py`: changing any snapshot after `t` leaves `z_t`
  unchanged (causality); windows with 1 snapshot work; non-uniform `Δt` works.
- **P6.2.4 Heads.** `heads.py`: `win` (1 logit), `hazard` (10 logits from the per-player
  token outputs after the temporal layer is broadcast back, masked for dead players),
  `plant15` (1 logit), `defuse15` (1 logit), `tte` (1 regression on `log1p(seconds)`).
  **Verify:** shapes; dead players' hazard logits are masked to `−inf → p = 0`.
- **P6.2.5 Assembly.** `wp_model.py::WPModel(config)`; parameter count printed
  (**estimate** 2–4 M). `forward(batch) -> dict of outputs`. **Verify:**
  `test/engine/test_wp_model.py`: forward on the synthetic dataset; gradient reaches
  every parameter (no dead branches); deterministic under fixed seed; fp32 and bf16
  agree within 1e-2 on logits.
- **P6.2.6 Over-fit test.** Train 300 steps on one shard of 512 snapshots; train loss
  must fall below 0.05. **Verify:** `test/engine/test_overfit_tiny.py` (marks exit 2 if
  no GPU and `--allow-cpu` not set; on CPU uses 64 snapshots).

### P6.3 Training protocol

- **P6.3.1 Do:** `train.py`: AdamW (`lr 3e-4`, `wd 0.01`), cosine schedule with 1,000
  warm-up steps, batch = 256 round-windows, gradient clipping 1.0, bf16 autocast if the
  device supports it, seed fixed and logged, checkpoints every epoch with manifest,
  early stopping on `val` win log loss (patience 3), max 30 epochs. Loss:
  `1.0·BCE(win) + 0.5·BCE(hazard, alive-masked) + 0.2·BCE(plant15) + 0.2·BCE(defuse15)
  + 0.1·Huber(tte)` with weights in config. Terminal snapshots: win label pinned, loss
  weight ×2 so the model learns the endpoints. Post-decision snapshots
  (`alive_ct = 0 or alive_t = 0`) are kept in training with normal weight (the model
  must predict them for `p_after`), unlike the legacy loader which excluded them; the
  terminal pin makes them learnable. **Verify:** `test/engine/test_train_loop.py` runs 20
  steps on synthetic shards: loss finite and decreasing, checkpoint saved and reloaded
  reproduces outputs within 1e-6.
- **P6.3.2 Tier sampling.** Batches drawn with pro snapshots oversampled by
  `pro_oversample = 4` (config) relative to their share, FACEIT filling the rest. After
  early stopping, a fine-tune stage: `pro_finetune_epochs = 2` on pro only at `lr 3e-5`.
  **Verify:** sampler test: observed pro share in 1,000 batches within ±2% of the
  target; the fine-tune stage loads the best checkpoint, not the last.
- **P6.3.3 Augmentation for counterfactual robustness** (config, all on by default in
  training only): drop each utility token with `p = 0.1`; jitter alive players'
  positions by `N(0, 16 u)` and recompute `dist_units` (LOS unchanged); set a blind
  player's `flash_duration_remaining` to 0 with `p = 0.05`; flip a random `los` bit with
  `p = 0.02`; drop a dead player's token entirely with `p = 0.5` (so both "dead token
  present" and "absent" are seen). **Verify:** `test/engine/test_augment.py`: each
  augmentation changes only the fields it claims; off by default at eval.
- **P6.3.4 Do:** training writes `models/wp_v<N>/{best.pt, last.pt, manifest.json,
  curves.csv, config.yaml}` and a one-page `eval/wp_v<N>_train.md` with curves.
  **Verify:** files present after the synthetic run.

### P6.4 Inference engine

- **P6.4.1 Do:** `predict.py::Predictor(checkpoint, device)`: `predict(states) ->
  dict(win, hazard, plant15, defuse15)` for a list of snapshot batches with their
  factual history; builds temporal windows internally; caches `h_t` per
  `(match, round, tick, edit_hash)` so attribution's repeated calls on the same factual
  history reuse the history encoding and only recompute the edited present snapshot.
  **Verify:** `test/engine/test_predict.py`: cached and uncached outputs equal; batch of
  1 equals batch of 512 element-wise; `edit_hash` differs for different edits.
- **P6.4.2 Do:** `bench/bench_predict.py`: snapshots/s for batch 512 on GPU and CPU;
  **target** ≥ 5,000/s GPU, ≥ 300/s CPU. **Verify:** recorded; budget file updated if
  the estimate was wrong.

### P6.5 Calibration

- **P6.5.1 Do:** `calibrate.py`: fit a single temperature `T` on `val` pro snapshots
  (minimize NLL); also fit per-`(map_version, phase)` temperatures and keep them only
  if val ECE improves by ≥ 0.002 (else the single `T`). Saved into the model directory
  as `calibration.json`; `Predictor` applies it. **Verify:**
  `test/engine/test_calibrate.py` on synthetic logits with known miscalibration: `T`
  recovered within 5%.
- **P6.5.2 Do:** reliability report on `test` pro: 20 bins, every bin with `n ≥ 200`
  must have `|mean_pred − observed| ≤ 0.03`; ECE ≤ 0.02 (**targets**); broken out by
  map, phase, alive-difference; plot saved. **Verify:** `engine eval --calibration`
  exits 1 when a target is missed.
- **P6.5.3 Do:** terminal pin check: for terminal snapshots the calibrated `W` must be
  ≥ 0.99 for CT wins and ≤ 0.01 for T wins on ≥ 99% of test rounds. **Verify:** part of
  `engine eval`.

### P6.6 Statistical sanity on the real model

- **P6.6.1 Do:** on 10,000 sampled test snapshots: `W` increases when `alive_ct`
  increases by one (synthetic edit: revive a dead CT at a teammate's position) in ≥ 97%
  of cases; decreases when a T is revived in ≥ 97%; `W` decreases after a plant edit in
  ≥ 95%; more time left pre-plant decreases `W` (CT) in ≥ 90% when `alive_t ≥
  alive_ct`. **Verify:** `engine sanity --model` writes `eval/wp_v<N>_sanity.md` and
  exits 1 below thresholds. (This is the model-level sanity; P7 is the
  counterfactual-intervention gate.)
- **P6.6.2 Do:** the martingale check: on test rounds, group snapshots by predicted
  `W` bin and by seconds ahead `k ∈ {1, 5, 10}`; mean of `W(t+k) − W(t)` within ±0.01
  of zero per bin with `n ≥ 500`. A calibrated process has zero expected drift; large
  drift means the model is systematically late or early. **Verify:** report table;
  threshold in config; exit 1 if violated.

### P6.7 Ablation loop and adoption decision

- **P6.7.1 Do:** run and record on the same `test` pro months: (a) baseline LightGBM;
  (b) NN pro-only training; (c) NN pro+FACEIT with tier token (the default); (d) (c)
  without the temporal encoder; (e) (c) without pairwise bias; (f) (c) without
  auxiliary heads; (g) (c) at 4 Hz on the subset from P5.3.3 versus (c) at 2 Hz on the
  same subset. Each writes `eval/ablation_<label>.md`; a summary table in
  `eval/ablations_v<N>.md`. **Verify:** every row has log loss, Brier, ECE, and the
  movement-credit variance metric from P8.4 for (g).
- **P6.7.2 Decision rules** (written here so they are not negotiated later):
  - Adopt the NN only if (c) or (b) beats (a) on pro test log loss by ≥ 0.01 **and**
    ECE ≤ 0.02. Otherwise stop, write up, and return to features (P3) before any
    attribution work.
  - Keep FACEIT in training only if (c) beats (b) on pro test log loss. Otherwise
    pro-only.
  - Keep the temporal encoder only if (c) beats (d) by ≥ 0.003 log loss **or** improves
    P6.6.2 drift; same rule for pairwise bias against (e) and heads against (f)
    (the heads are needed by P8 regardless, so the criterion is the hazard head's own
    quality on test pro: AUC ≥ 0.80 is the target; between 0.75 and 0.80 the hazard
    mode is used and every report carries a "hazard head below target" warning; below
    0.75 enabler splits fall back to the fixed shares of P8.5.3 until it improves, and
    the report says so).
  - Move to 4 Hz only if (g) reduces movement-credit variance by ≥ 20% at ≤ 1.6× the
    state-building time.
- **P6.7.3 Do:** optional ensemble `W = α·NN + (1−α)·LGBM` with `α` fit on val; adopt
  only if test log loss improves ≥ 0.002. **Verify:** recorded in the ablation table.

**Test loop P6:** unit tests (tokens, set encoder, temporal, heads, model, train loop,
augment, predict, calibrate) green on synthetic data; overfit test passes; real
training run completes with curves; `engine eval` and `engine sanity --model` pass
thresholds; ablation table filled; decision recorded in `eval/DECISION_wp_v<N>.md`.
**Definition of done P6:** an adopted, calibrated `wp_v<N>` with manifest, passing
P6.5–P6.6, beating the baseline per P6.7.2, with `Predictor` within its bench budget.
**Risks:** ROCm instability (checkpoints every epoch; resume supported); overfitting to
pro (early stopping on val; FACEIT regularizes); hazard head too weak (fallback shares
in P6.7.2); data volume short (A4).
**Effort:** 8–12 days including training time.

---

## P7 — Counterfactual interventions and the sanity gate

Purpose: the state-editing operations the Shapley split needs, each tested alone, and a
suite that proves the model answers "what if this had not happened" sensibly. Nothing
in P8 runs until this gate passes.

### P7.1 Intervention library

Each intervention is a pure function `state -> state'` on the snapshot dict/struct (not
on tensors), followed by a `recompute_derived(state')` that fixes every dependent
field (alive counts, team equip sums, pairwise LOS for moved players, `smoke_between`,
hazard masks). Every intervention has: a docstring stating what it undoes, its
preconditions, and a unit test with a positive case, a no-op case, and an invalid
case.

- **P7.1.1 `undo_kill(state_post, kill)`:** victim restored with `alive = 1`,
  health/armor/equipment from `state_pre`, position from `state_pre`; attacker's
  in-round kill counter decremented; alive counts, equip sums, pairwise recomputed.
  Precondition: victim dead in `state_post`. **Verify:** `test/engine/test_interventions.py`.
- **P7.1.2 `undo_damage(state_post, damage)`:** victim's health/armor restored by the
  damage amounts, capped at `state_pre` values.
- **P7.1.3 `remove_utility(state, utility_action)`:** drop the smoke/fire/decoy token;
  recompute `smoke_between`, `in_smoke`, `on_fire`. Also used for `smoke_pop` /
  `smoke_fade` pseudo-actions (`add_utility` is its inverse, used to build `s0 + S`
  when the member is a pop).
- **P7.1.4 `unflash(state, flash_action)`:** set `flash_duration_remaining = 0` for
  every player that flash blinded; inverse `reflash` restores durations from `state_pre`.
- **P7.1.5 `unplant(state, plant_action)` / `replant`:** bomb state, site, timer, clock
  mode, `is_planting` reset; the carrier gets the bomb back at the planter's position.
- **P7.1.6 `undefuse_start/complete` and inverses.**
- **P7.1.7 `freeze_position(state, player, state_ref)`:** the player's position, area,
  callout, grid cell, velocity set from `state_ref` (the interval's start); pairwise
  rows/cols for that player recomputed; `spotted` fields left as is (information is
  not undone, only movement; documented approximation).
- **P7.1.8 `revert_buy(state_freeze_end, player)`:** equipment of that player set to
  the economy prior loadout (Appendix B.6); team equip recomputed.
- **P7.1.9 `apply(state_pre, state_post, members: set[action])`:** builds `s0 + S`:
  start from `state_post` and undo every window member **not** in `S`, in reverse
  tick order. Returns the edited state and an `edit_hash`. **Verify:** `apply(pre,
  post, all_members)` equals `post` exactly; `apply(pre, post, ∅)` equals `pre` on
  every field the members can touch (fields they cannot touch, like the clock, are
  compared to `post` and documented).

**Test loop P7.1:** every intervention's three cases; `apply` identities; derived-field
consistency checker `state.validate()` (alive counts equal the sum of alive bits, equip
sums match, pairwise symmetric, dead players have `None` equipment) passes after every
intervention on the synthetic stories.

### P7.2 The sanity gate (on the real model, held-out pro rounds)

Sample `N = 10,000` instantaneous events and `10,000` drift intervals from `test`
rounds. For each, apply the intervention and compare `W`. Thresholds are **targets**
and live in config. The gate is `engine sanity --counterfactual`, exit 1 on any failure.

- **P7.2.1 Un-kill:** restoring the victim moves `W` toward the victim's side in
  ≥ 98% of cases; median |ΔW| ≥ 0.03.
- **P7.2.2 Remove enemy smoke:** `W` for the non-owner side changes by ≥ −0.01 in
  ≥ 95% of cases (removing the enemy's smoke should not hurt you), and |ΔW| ≤ 0.25 in
  99% (no wild swings).
- **P7.2.3 Un-flash:** the blinded side's `W` is non-decreasing (≥ −0.005) in ≥ 95%.
- **P7.2.4 Un-plant:** T side `W` decreases in ≥ 95%.
- **P7.2.5 Freeze position:** |ΔW| ≤ 0.15 in 99% of intervals; mean ΔW within ±0.005
  (movement edits should not have a systematic sign).
- **P7.2.6 Order independence of `apply`:** for windows with ≥ 2 members, applying
  members in two different orders yields identical states (bitwise) and identical `W`.
- **P7.2.7 Off-distribution rate:** `ood.py` computes the distance from each edited
  state's `h_t` to its 10th nearest training embedding (FAISS, pro+FACEIT index built in
  P9.3 or a temporary one here); an edit is flagged if the distance exceeds the 99th
  percentile of factual test states. Flag rate ≤ 5% per intervention type.
- **P7.2.8 Report:** `eval/sanity_cf_wp_v<N>.md` with every rate, the worst 20 cases
  per test (match, round, tick, before/after `W`, a radar PNG), so failures can be
  eyeballed.

### P7.3 Failure loop

If any P7.2 test fails: (1) inspect the worst cases; (2) choose one remedy per
iteration: increase the matching augmentation rate in P6.3.3, add training data, add a
feature that makes the edited state representable, or lower `L_set`/`d_model` if the
model is memorizing; (3) retrain; (4) rerun P6.5–P6.6 and P7.2. Log each iteration in
`eval/DECISION_wp_v<N>.md`. Maximum five iterations before escalating to Faig with the
numbers. P8 does not start until the gate is green.

**Definition of done P7:** intervention library fully tested; sanity gate green on the
adopted model; report written.
**Effort:** 3 days for the library and tests; 1–5 days of gate iterations.

---

## P8 — Attribution engine (realized value V)

Purpose: partition every round's win-probability path completely and exactly among
actions, enablers, movement, and the team clock, so that condition C1 holds by
construction.

### P8.1 Partition of the round timeline

- **P8.1.1 Do:** `partition.breakpoints(states, windows)`: the sorted set of grid
  sample times and every window's `[start_tick − 1, end_tick + 1]` bounds. The timeline
  becomes a sequence of intervals, each labelled `event` (inside a window) or `drift`
  (between). Grid samples that fall inside a window are dropped as breakpoints (the
  window's own pre/post boundary samples are used). **Verify:**
  `test/engine/test_partition.py`: intervals tile `[freeze_end, terminal]` with no gaps or
  overlaps; every window is exactly one `event` interval; synthetic stories produce the
  expected interval list.
- **P8.1.2 Do:** `W` at every breakpoint from `Predictor` with factual history
  (one batched call per round). **Verify:** count equals breakpoints.
- **P8.1.3 Do:** total change `ΔW_round = W(terminal) − W(freeze_end)` = `outcome −
  W_0` after calibration pinning. Every interval's `ΔW_i = W(end_i) − W(start_i)`.
  `Σ ΔW_i = ΔW_round` by telescoping. **Verify:** exact to floating error.

### P8.2 Event windows: Shapley over coalitions

For a window with member set `A` (instantaneous actions and pop/fade pseudo-actions),
pre-state `s_pre`, post-state `s_post`, define the coalition value
`v(S) = W(apply(s_pre, s_post, S))` for `S ⊆ A`.

```
φ_a = Σ_{S ⊆ A\{a}}  |S|! (|A|−|S|−1)! / |A|!  · [ v(S ∪ {a}) − v(S) ]
```

- **P8.2.1 Do:** `shapley.exact(members, v)`: enumerate all `2^|A|` coalitions once,
  cache `v(S)` by `edit_hash`, compute every `φ_a`. Used when `|A| ≤ shapley_exact_max
  (6)`. **Verify:** `test/engine/test_shapley.py` with a hand-computable `v` on 3
  members: matches the textbook values; efficiency `Σ φ_a = v(A) − v(∅)` to 1e-12;
  symmetry (two members with identical marginals get equal `φ`); null member gets 0.
- **P8.2.2 Do:** `shapley.sampled(members, v, n_perm = shapley_samples)`: average
  marginal contributions over `n_perm` random permutations with antithetic pairs
  (each permutation and its reverse); then **enforce efficiency** by distributing the
  residual `v(A) − v(∅) − Σ φ_a` proportionally to `|φ_a|` and recording the residual
  magnitude in the action's `residual` field. Used when `|A| > 6`. **Verify:** on a
  7-member additive `v`, sampled values within 1e-3 of exact; residual < 1e-3 before
  enforcement; exactly 0 after; `n_perm` convergence test (doubling halves the error).
- **P8.2.3 Do:** `v(S)` evaluation batches all coalitions of a window into one
  `Predictor` call (cache hits on the shared history). **Verify:** bench shows ≤ 1
  `Predictor` call per window.
- **P8.2.4 Faig's C1 scenario, as a test on the real model:** take any real kill
  window with exactly one kill; add a synthetic no-effect member (a decoy pop 2,000
  units away from everyone, which `remove_utility` can undo); assert `φ_decoy` is `0 ±
  1e-9` **when the model is exactly indifferent** — and, since a real model is never
  exactly indifferent, the production test asserts `|φ_decoy| ≤ 0.002` across 200 such
  windows and `|φ_kill − ΔW| ≤ 0.002`, while the exact-zero property is asserted with
  a stubbed `W` that ignores decoys (`test_attribution_shapley.py::null_action_same_tick`
  runs both variants). This is the explicit acceptance test for C1.
- **P8.2.5 Do:** `φ` sign: computed CT-positive; stored raw and converted to actor-sign
  (`+φ` for CT actor, `−φ` for T) in `V`. Pseudo-actions (`smoke_pop`, `smoke_fade`,
  `fire_start`, `fire_end`) belong to the thrower. **Verify:** sign test on the
  synthetic stories.

### P8.3 Durative actions: pop, fade, and enabled events

A durative action's realized value is the sum of (a) its `pop` share (the WP jump when
its token appears), (b) its `fade` share (the WP change when its token disappears,
usually opposite in sign), (c) its enabler shares from every instantaneous event it
raised the likelihood of (P8.5), and (d) nothing from drift (its influence on movement
during its life is credited to the movers; documented approximation).

- **P8.3.1 Do:** `fade` pseudo-actions are members of the window at `end_tick`; if no
  other member exists there the window has one member and `φ_fade = ΔW` of that
  interval. **Verify:** synthetic smoke story: `V_smoke = φ_pop + φ_fade + enabler
  shares`; with a stubbed `W` that values the smoke at `+0.05` while present,
  `φ_pop = +0.05`, `φ_fade = −0.05`, net zero when nothing happened, which is the
  intended behaviour (a smoke that produced nothing is worth nothing in realized terms;
  its expected value `Q` still captures what it usually produces).
- **P8.3.2 Do:** movement actions (rotation, hold, peek, save) take their value from
  P8.4 movement shares over their windows plus `φ` at any `event` window where the
  mover's position change is itself a member (the peek: the LOS onset tick is a window
  with the peek as a member, undone by `freeze_position` to the previous sample).
  **Verify:** synthetic walking story: B's rotation value equals the sum of B's movement
  shares over its window.

### P8.4 Drift intervals: movement and clock

For a drift interval `[t0, t1]` with `ΔW = W(s1) − W(s0)`:

- **P8.4.1 Do:** movers `M` = alive players whose displacement over the interval is
  ≥ `movement_min_units` (32). If `|ΔW| < drift_threshold_wp` (0.02): leave-one-out,
  `φ_p = W(s1) − W(freeze_position(s1, p, s0))` for each mover, then efficiency-enforce
  as in P8.2.2. If `|ΔW| ≥ 0.02`: exact Shapley over `M` with `v(S) = W(s1 with every
  mover not in S frozen)` when `|M| ≤ 6`, sampled otherwise. **Verify:**
  `test/engine/test_movement.py` on the walking story: B gets the movement share; a
  stationary A gets none; efficiency holds.
- **P8.4.2 Do:** the remainder `ΔW − Σ_p φ_p` is the **clock** share (CT-positive,
  signed), never assigned to a player. Per round, `clock_ct` accumulates the positive
  remainders and `clock_t` accumulates the magnitude of the negative ones, so both
  buckets are non-negative and read as "value the clock gave this team". **Verify:**
  per-interval identity `Σ φ_p + clock = ΔW` to 1e-9; per round
  `clock_ct − clock_t = Σ_intervals clock`.
- **P8.4.3 Do:** movement-credit variance metric for the P6.7 rate ablation: the
  standard deviation of per-player movement shares per round on the ablation subset.
  **Verify:** metric computed and written.

### P8.5 Enabler splits via hazard heads

For an instantaneous event `e` with actor `k`, victim/target `v`, direct share `φ_e`,
and enabler set `E(e)` from P4.5, with hazard `q_v(s)` the head matching the event type
(death hazard for kills and damage, `plant15` for plants, `defuse15` for defuses),
evaluated at `s_pre`:

```
raw_i   = max(0, q_v(s_pre) − q_v(remove_i(s_pre))) / q_v(s_pre)     for i ∈ E(e)
share_i = clamp(raw_i, band_i)                                       band_trade = [0.05, 0.40]
total   = min(Σ share_i, enabler_total_cap = 0.5), shares rescaled proportionally if capped
V_i    += share_i · φ_e            (signed for the enabler's side; the dead teammate in a trade
                                   is on the actor's side, so same sign)
V_k    += (1 − total) · φ_e
```

- **P8.5.1 Do:** `remove_i` is the matching intervention (`unflash`,
  `remove_utility`, `undo_kill` of the teammate for `trade_setup`, `undo_damage` for
  `he_damage`, `remove_utility` for `fire_pressure`). For `info` enablers with no
  action link, the share goes to the team clock bucket. **Verify:**
  `test/engine/test_enabler_split.py` with a stubbed hazard head: known `q` values →
  known shares; clamps and the cap behave; shares sum with the actor's remainder to
  `φ_e` exactly.
- **P8.5.2 Do:** when `q_v(s_pre) < 0.02` (the event was very unlikely), raw shares are
  unstable; use the clamped lower bound for enablers present and log the case.
  **Verify:** test with `q = 0.01`.
- **P8.5.3 Do:** fallback shares (used only if P6.7.2 found the hazard head too weak):
  `flash 0.20`, `smoke_cover 0.10`, `trade_setup 0.20`, `he_damage 0.10`,
  `fire_pressure 0.10`, cap `0.5`; selected by config `enabler_mode: hazard|fixed`, and
  the report states which mode produced the numbers. **Verify:** both modes tested.

### P8.6 Round assembly and conservation

- **P8.6.1 Do:** `engine.attribute_round(states, actions, windows, predictor) ->
  list[ScoredAction], clock_ct, clock_t`: runs P8.1–P8.5, sums each action's direct,
  enabler, and movement components into `V`, records `method`
  (`exact|sampled|loo`), `n_coalition`, `residual`. **Verify:**
  `test/engine/test_attribution_round.py` on every synthetic story with a stubbed `W`:
  the conservation identity

```
Σ_actions V_raw(CT-positive) + clock_ct − clock_t  =  W(terminal) − W(freeze_end)
```

  to 1e-9 (exact windows) and 1e-6 (any sampled window); the two-kills-same-tick story
  gives both killers non-zero, opposite-sign direct shares that sum to the window's ΔW;
  the no-effect decoy gets 0 with the stub.
- **P8.6.2 Do:** `V` identity per action: `V = V_direct + V_enabler + V_movement`
  stored separately. **Verify:** test.
- **P8.6.3 Do:** real-model conservation on 1,000 sampled test rounds: max
  `|residual|` ≤ 1e-6 for rounds with only exact windows, ≤ 1e-3 otherwise; written to
  `eval/attribution_wp_v<N>.md` with the distribution of `V` by action type, the share
  of |ΔW| attributed to players vs clock (**target** ≥ 60% to players; informational).
  **Verify:** `engine eval --attribution` exits 1 on residual violations.
- **P8.6.4 Do:** determinism: same inputs → bitwise identical `V` (seeded permutation
  sampling keyed by `action_id`). **Verify:** run twice, compare checksums.
- **P8.6.5 Do:** `bench/bench_attribution.py`: **target** ≤ 15 s per map on GPU, ≤ 90 s
  CPU; counts `Predictor` calls and coalition evaluations. If over budget: raise
  `drift_threshold_wp` first, then lower `shapley_samples`, then `shapley_exact_max`
  to 5; each change recorded with its effect on the residual distribution.

**Test loop P8:** all P8 tests green with stubs; real-model conservation and bench
within budget; `run_all.py --suite all` green.
**Definition of done P8:** `actions_valued.parquet` for the golden demo with every
action's `V` and components; conservation holds; C1 test passes in both variants;
report written.
**Risks:** combinatorial blow-up in busy windows (sampling + cap on members by merging
sub-tick duplicates of the same actor); model non-smoothness under edits (the P7 gate
exists for this); clock share too large (informational target; if players get < 40%,
revisit movement thresholds, not the conservation).
**Effort:** 6–8 days.

---

## P9 — Expected value, decision and execution

Purpose: for every action, what it is usually worth here (`Q`), what the usual choice
here is worth (`B`), the best known alternative (`R`), and therefore the decision score
`D` and execution score `X`, with neighbour panels and shrinkage.

### P9.1 Action descriptors

- **P9.1.1 Do:** `descriptors.encode(action, state_pre) -> dict`: `type`, `subtype`,
  `weapon_class`, `from_callout`, `to_callout` / `land_callout`, `timing_bucket`
  (0–15 s, 15–45 s, 45–115 s, post-plant early ≤ 20 s, post-plant late),
  `alive_diff_bucket` (−4..+4 clipped to ±3), `has_los_to_enemy` (for peeks/holds),
  `n_enemies_affected` (flash/HE/molotov), `loadout_class` (buys), `site` (bomb). Each
  categorical with its own vocabulary saved alongside P5.3.2 vocabularies. **Verify:**
  `test/engine/test_descriptors.py`: every action type yields a descriptor; unknown
  callout → `UNK`; buckets' boundaries exact.
- **P9.1.2 Do:** the similarity granularity is the descriptor plus the encoder
  embedding `h_pre` of `s_pre`; both are stored per valued action in
  `actions_valued.parquet` (embedding as a float16 list). **Verify:** column present.

### P9.2 Q and B networks

- **P9.2.1 Do:** `q_model.QNet`: input `h_pre` (frozen from the adopted `wp_v<N>`
  encoder; config `q_finetune_encoder: false`) concatenated with descriptor embeddings
  and the tier embedding → MLP `[512, 256, 1]` → `Q̂`. Target: actor-signed `V` from P8.
  Loss: Huber (`δ = 0.05`). Trained on all tiers with the tier token, `train` split,
  early stopping on `val`. **Verify:** `test/engine/test_q_model.py`: shapes; overfit
  tiny; checkpoint round-trip.
- **P9.2.2 Do:** `q_model.BNet`: same input without the descriptor → `B̂(s)` = expected
  actor-signed `V` of whatever action was taken at states like `s`. Trained on the same
  rows. Because `V` is actor-signed, `B` is estimated per side: input includes the
  acting side bit. **Verify:** same tests.
- **P9.2.3 Do:** evaluation on `test` pro: mean `Q̂` vs mean `V` in 20 bins of `Q̂`
  (calibration of the regression) within ±0.01 per bin with `n ≥ 200`; Spearman between
  `Q̂` and `V` ≥ 0.3 (**target**, informational; the realized `V` is noisy by nature).
  Same for `B̂`. **Verify:** `engine eval --value` writes `eval/q_v<N>.md` and exits 1 on
  the calibration target.
- **P9.2.4 Do:** per-type `Q` sanity: mean `Q̂` of kills > 0; of `team_kill` < 0; of
  `defuse_complete` > 0 for CT; of `plant_complete` > 0 for T. **Verify:** in the same
  eval.

### P9.3 Neighbours and the panel

- **P9.3.1 Do:** `neighbours.build_index(split='train', tier='pro')`: FAISS
  `IndexHNSWFlat` (or `IndexFlatIP` if the corpus is < 2 M) over `h_pre` of every
  valued pro action in `train`, with a side-table of `(match_key, round, action_id,
  descriptor, V)`. Saved under `models/neighbours_v<N>/`. **Verify:**
  `test/engine/test_neighbours.py` on a synthetic index: a query equal to a stored
  vector returns it first; results never include the query's own match (filtered by
  `match_key`).
- **P9.3.2 Do:** `neighbours.panel(h_pre, side, k = k_neighbours (200))`: among the `k`
  nearest pro states (same side, same map version), count actions by descriptor
  `type+subtype+location`, list the top 20 with frequency, mean `V`, mean `Q̂`, and `n`.
  **Verify:** panel sums to `k`; ties ordered deterministically.
- **P9.3.3 Do:** regret: `R = Q̂(s,a) − max{ Q̂(s,a') : a' in panel with n ≥
  n_min_regret (30) }` where `Q̂(s,a')` is evaluated at the **current** state `s` with the
  alternative's descriptor. `R ≤ 0` by construction when `a` itself is in the panel with
  support; if `a` has support < `n_min`, it is still included in the max (its own `Q̂`
  counts), so `R ≤ 0` always. **Verify:** test asserts `R ≤ 0` on 10,000 actions and
  that an alternative with `n < 30` never sets the max.

### P9.4 Scores, verdicts, shrinkage

```
D = Q̂(s,a) − B̂(s)
X = V − Q̂(s,a)
V = B̂(s) + D + X                      (identity, exact by construction)
verdict_decision  = good if D > +δ_D, bad if D < −δ_D, neutral otherwise   (δ_D = 0.01)
verdict_execution = good if X > +δ_X, bad if X < −δ_X, neutral otherwise   (δ_X = 0.02)
```

- **P9.4.1 Do:** `scores.compute(valued_action, Q̂, B̂, panel) -> ScoredAction` adding
  `Q, B, D, R, X`, verdicts, panel reference. **Verify:**
  `test/engine/test_decision_split.py`: identity to 1e-9; verdict thresholds inclusive
  as documented; C2 acceptance: every valued action has all fields and a panel for
  ≥ 95% of pro actions on the golden demo.
- **P9.4.2 Shrinkage for per-player aggregates.** For player `p` with `n` actions of
  type class `c` and mean execution `X̄`:

```
X̃_p,c = X̄_p,c · n / (n + k_c)
k_c   = σ²_noise,c / σ²_skill,c
```

  `σ²_noise,c` = within-player variance of `X` on `train`; `σ²_skill,c` = between-player
  variance of player means minus `σ²_noise,c / n̄` (method of moments; floor at a small
  positive value). `k_c` saved per class in `models/q_v<N>/shrinkage.json`. Same for
  `D`. Report shows `X̃` with a 90% interval `± 1.645 · sqrt(σ²_noise,c / (n + k_c))`.
  **Verify:** `test/engine/test_shrinkage.py`: synthetic players with planted true skill
  recovered within the interval for 90% ± 3% of players; shrinkage factor monotone in
  `n`; `k_c` positive.
- **P9.4.3 Planted-signal test:** inject into a synthetic corpus one player whose `V`
  is shifted by `+0.02` on kills; the pipeline must recover `X̃ ≥ 0.01` for that player
  and `|X̃| < 0.005` for the others. **Verify:** `test/engine/test_planted_signal.py`.

### P9.5 Output

- **P9.5.1 Do:** `actions_scored.parquet` per match and rows in
  `engine_action_scores` (P5.2) keyed by `run_id`. **Verify:** schema tests; round-trip.
- **P9.5.2 Do:** `bench/bench_value.py`: Q/B inference + panel for one map **target**
  ≤ 5 s GPU, ≤ 20 s CPU.

**Test loop P9:** descriptor, Q/B, neighbours, decision split, shrinkage, planted
signal tests green; `engine eval --value` passes; bench within budget.
**Definition of done P9:** every action of the golden demo carries `V, Q, B, D, R, X`,
verdicts, and a panel; identity holds; evaluation report written.
**Risks:** `Q` under-determined for rare descriptors (falls back toward `B` through the
MLP's smoothing; report flags `n < 30` descriptors as low confidence); panel dominated
by one map phase (same-phase filter is in the descriptor via `timing_bucket`).
**Effort:** 5–7 days.

---

## P10 — CLI and report

Purpose: one command turns a demo into a report; every other operation is a
subcommand; everything is timed against the C3 budget.

### P10.1 Commands

- **P10.1.1 `engine analyze <demo.dem> [--tier pro] [--out DIR] [--model wp_vN]
  [--cpu]`:** parse → states → actions → predict → attribute → value → report. Writes
  `DIR/` with `report.json`, `report.md`, `charts/`, `actions_scored.parquet`,
  `manifest.json`, `timings.json`. Refuses demos that fail P1.4 quarantine rules with
  a clear message. **Verify:** `test/engine/test_cli_analyze.py` runs it on the golden
  demo with a stubbed `Predictor` (fast) and checks every output exists and validates
  against the report schema (P10.2); a second run produces byte-identical
  `report.json` except timestamps.
- **P10.1.2 `engine parse <dir|demo> [--workers N] [--force]`:** P2.6 worker.
- **P10.1.3 `engine build-states <match_key|--all>`, `engine build-actions ...`,
  `engine build-dataset --split splits_vN.json [--sample-hz 2] [--assert-qa]`.**
- **P10.1.4 `engine train-wp --config ... --name wp_vN`, `engine train-q --wp wp_vN
  --name q_vN`, `engine calibrate wp_vN`, `engine build-neighbours q_vN`.**
- **P10.1.5 `engine eval --model wp_vN [--calibration|--attribution|--value|--all]`,
  `engine sanity --model wp_vN [--counterfactual]`, `engine bench [analyze|parse|states|
  predict|attribution|value]`, `engine inventory [--assert-targets|--rebuild-catalog]`.**
- **P10.1.6 Common:** every command prints its `run_id`, `config_hash`, device, and
  stage timings; writes a manifest; exits non-zero on failure with the redacted error;
  `--help` on every command. **Verify:** `test/engine/test_cli_help.py` runs `--help`
  for every subcommand; `test/engine/test_cli_errors.py` checks a missing file gives
  exit 2 and no traceback in normal mode.
- **P10.1.7 Do:** legacy `src/main.py` is unchanged except a one-line pointer in its
  `--help` epilog: "engine commands: `.venv-engine\Scripts\python.exe -m engine.cli`".

### P10.2 Report content and schema

- **P10.2.1 `report.json`** (JSON Schema in `engine/report/schema.json`, versioned):
  `meta` (match, map_version, tier, teams, date, model versions, config_hash,
  run_id, timings), `rounds[]` each with `wp_curve` (breakpoint times and `W`),
  `windows[]`, `actions[]` (every ScoredAction with `V, V_direct, V_enabler,
  V_movement, Q, B, D, R, X`, verdicts, enablers with their shares, panel top-20,
  method, residual, flags), `clock_ct`, `clock_t`, `conservation_residual`;
  `players[]` with per-type sums of `V`, `D̃`, `X̃` with intervals and `n`;
  `teams[]` with clock totals and buy valuations; `warnings[]` (OOD-flagged edits,
  sampled windows, low-support descriptors, fixed-share fallback in use). **Verify:**
  `test/engine/test_report_schema.py` validates the golden report and a synthetic one
  with every optional field absent.
- **P10.2.2 `report.md`:** human-readable: match header; per-round section with the WP
  chart, the top 5 |V| actions as "action cards" (who, what, where, when, `V`, `Q`,
  `B`, `D`, `X`, verdicts, enabler lines like "flash by X: 31% of this kill", and the
  panel's top 5 lines "smoke to CT (41%), mean V +0.021"); player table; team table;
  warnings. **Verify:** golden `report.md` compared after stripping timestamps.
- **P10.2.3 Charts (`charts/`):** `round_<n>_wp.png` (WP curve with action markers
  coloured by side and sized by |V|), `player_<id>_summary.png` (D̃ vs X̃ scatter per
  type with intervals), `team_clock.png`. matplotlib, no seaborn. **Verify:** files
  exist with non-zero size; a pixel-hash golden is not used (fonts differ by machine);
  the data behind each chart is in `report.json`.
- **P10.2.4 Redaction:** report never contains file-system paths outside the output
  directory, SteamIDs appear hashed unless `--show-ids`, no URLs. **Verify:**
  `test/engine/test_report_redaction.py`.

### P10.3 Time budget

- **P10.3.1 Do:** `bench/budget.yaml` initial budgets (GPU / CPU seconds): parse 15/15,
  states 8/8, actions 3/3, predict 5/30, attribution 15/90, value 5/20, report 4/4,
  total 55/170, hard ceiling 120/300. **Verify:** `engine bench analyze --assert`
  on the golden demo passes; the result is pasted into `eval/bench_<date>.md`.
- **P10.3.2 Do:** when over budget, the order of knobs: `los_cache`, `drift_threshold_wp`,
  `shapley_samples`, `k_neighbours`, chart resolution; each recorded with its effect.
  `sample_hz` is **not** a latency knob (decided: 2 Hz is the design value; changing
  it is an ablation decision in P6.7.2).

**Test loop P10:** CLI tests, report schema, redaction, help/errors green; bench
within budget; `run_all.py --suite all` green.
**Definition of done P10:** `engine analyze` on the golden demo produces a complete,
schema-valid, reproducible report within budget; every subcommand has a test.
**Effort:** 4–5 days.

---

## P11 — Validation against the world, and the review loop

Purpose: evidence that the numbers mean what they claim, in the form a buyer will ask
for, plus the loop in which Faig reviews real rounds and the engine is corrected.

### P11.1 External anchors (informational, with bands)

Published anchors are approximate and come from different models; they are sanity
bands, not targets to fit.

- **P11.1.1 Opening kill in a 5v5, balanced equipment:** mean `|V_direct|` of the
  first kill in rounds with equipment ratio within 0.8–1.25 should fall in **0.12–0.30**
  (HLTV's secondary sources quote ~0.20; Szmida & Toka show +0.15 to +0.30 across
  contexts). Written to `eval/anchors_wp_v<N>.md` with the distribution by map.
- **P11.1.2 Duel difficulty:** rifle-vs-rifle kills should carry a larger mean `V`
  than rifle-vs-pistol kills in the same alive-difference bucket (HLTV's eco adjustment
  direction). Spearman between weapon-class disadvantage and mean `V` positive.
- **P11.1.3 Man-advantage ordering:** mean `W` of CT in 3v2 > 2v3 > 1v2 at matched
  time and bomb state; `|ΔW|` of a kill that creates 1v1 from 2v1 larger than from 5v4
  (Leetify's stated ordering). Checked on test pro states.
- **P11.1.4 Clutch and defuse:** a `defuse_start` in a 1vX with < 10 s should carry
  positive `V` for CT and the abort negative (the Sloan paper's +25% example is the
  shape, not the number).
- **P11.1.5 Player-level correlation:** on test pro matches, Spearman between players'
  summed `V` per map and HLTV's published per-map rating for the same matches (manually
  collected for ≥ 50 matches into `data/engine/eval/hltv_ratings.csv`, gitignored; the tracked fixture is a header-only template) ≥ **0.6**
  (informational: HLTV's rating is kill-centric; perfect agreement would mean we add
  nothing). The report lists the biggest disagreements for the review loop.
- **P11.1.6 Do:** `engine eval --anchors` writes the report; thresholds in config are
  bands; results outside a band are flagged, not failed, and discussed in the review.

### P11.2 Analyst review loop (Faig)

- **P11.2.1 Do:** generate reports for 20 rounds chosen to cover: opening kills,
  trades, flash-assisted kills, smoke-covered plants, clutches, saves, a double kill on
  one tick, a round decided by the clock, an eco win, a failed defuse. Save under
  `eval/review_round_<i>/`. **Verify:** 20 directories exist.
- **P11.2.2 Do:** Faig reviews each round's `report.md` and charts and records, per
  round, in `eval/review_log.md`: "agree / disagree / unsure" for the top-3 actions'
  `V`, `D`, `X` and the panel, with a sentence. The callout images from P2.3.3 are
  reviewed in the same session.
- **P11.2.3 Do:** every "disagree" becomes a numbered item; each is traced to a stage
  (parse, state, action detection, model, attribution, value) and fixed or
  explicitly accepted with a reason; fixes come with a regression test in
  `test/engine/test_review_items.py` reproducing the case from the stored states.
  **Verify:** the test file grows with one function per item.
- **P11.2.4 Do:** repeat P11.2.1–P11.2.3 with 20 new rounds until a cycle has ≤ 2
  disagreements out of 60 judgements (**target**), minimum two cycles.

### P11.3 Robustness

- **P11.3.1 Out-of-map check:** train once with one active-duty map held out (P1.5.2),
  evaluate log loss on that map vs the default model; the gap is reported (**estimate**:
  large, since positions are map-specific; this documents that a new map needs data
  before the engine trusts itself on it, and the CLI warns when `map_version` has <
  100 training matches).
- **P11.3.2 Patch drift:** calibration on the newest month only; if ECE > 0.03 the
  report warns "model predates recent patches".
- **P11.3.3 Adversarial demos:** a FACEIT demo analysed with `--tier pro` must print a
  warning that the tier token was overridden; a 64-tick demo is refused (P1.4 rule) with
  a clear message. **Verify:** `test/engine/test_cli_guards.py`.

**Definition of done P11:** anchor report written with every band's result; at least
two review cycles done with the log and regression tests; robustness checks reported.
**Effort:** 4–6 days plus Faig's review time.

---

## P12 — Operations, retraining, documentation, hand-off

### P12.1 Model registry and versions

- `models/<kind>_v<N>/manifest.json` is the registry; `engine inventory --models`
  lists them with their eval numbers. The report's `meta` names every model version
  used. A model is **adopted** only via a line in `docs/ENGINE_MODELS.md` (date,
  version, eval numbers, decision rule satisfied). **Verify:** `engine analyze` refuses
  a model not listed as adopted unless `--allow-unadopted`.

### P12.2 Retraining cadence

- Monthly: `engine inventory`, new splits file `v<N+1>` rolling the test months
  forward, retrain `wp` and `q`, calibrate, run the full eval and sanity gates, update
  the registry if the decision rules pass. A checklist in `docs/ENGINE_OPS.md`.
- On a map rework or removal from the active pool: add the version to
  `engine/maps/versions.yaml`, rebuild assets and callouts for the new version, mark
  old-version matches as `map_version` old, retrain. The CLI warns on a map version with
  too few matches (P11.3.1).
- On a game patch that changes mechanics (timers, smoke duration, economy): update
  config constants with the patch date; snapshots before the date keep the old
  constants through `game_version`.

### P12.3 Monitoring

- Every `engine analyze` appends one line to `data/engine/eval/usage.jsonl`: run_id,
  map_version, tier, timings, warnings count, conservation residual. A weekly script
  summarizes it. Any residual > 1e-3 or timing over the ceiling is listed.

### P12.4 Documentation to write (all in the repo)

- `docs/ENGINE_PLAN.md` (this), `docs/ENGINE_ENV.md` (P0.1.5, P2.1), `docs/
  DATA_SOURCES.md` (P1.1.1), `docs/ENGINE_MODELS.md` (P12.1), `docs/ENGINE_OPS.md`
  (P12.2), `engine/state/SCHEMA_CHANGELOG.md`, `test/engine/golden/CHANGELOG.md`.
- `CLAUDE.md`: add an "Engine" section: venv, commands, module map, the C1–C3
  acceptance tests, the decision rules of P6.7.2, the gate rule of P7, "never add a
  feature without a `LIVE_SAFE` justification", "never change `sample_hz` for latency".
- `README.md`: a short "Engine v2" section pointing to the plan; reference-run numbers
  stay as they are (per the docs-accuracy rule).

### P12.5 Hand-off checklist for a buyer demo

- Golden demo report, five pro match reports, the anchors report, the calibration
  report, the ablation table, the sanity-gate report, the review log, the bench table,
  and a one-page method summary (the "explain like I'm five" version from today, plus
  the formula appendix). All produced by commands in this plan; nothing hand-edited.

**Effort:** 2–3 days spread across the project.

---

## Effort summary and order

| Phase | Working days (estimate) | Depends on |
|---|---|---|
| P0 Environment, scaffolding | 2–3 | — |
| P1 Data acquisition | 3–4 (+ background downloads) | P0 |
| P2 Parsing, map assets, callouts | 4–6 | P0 |
| P3 State builder | 5–7 | P2 |
| P4 Action extractor | 5–6 | P3 |
| P5 Storage, datasets, QA | 3–4 | P3, P4, P1 |
| P6 Win model (incl. baseline, ablations) | 8–12 | P5 |
| P7 Interventions, sanity gate | 4–8 | P6 |
| P8 Attribution | 6–8 | P7 |
| P9 Expected value, decision split | 5–7 | P8 |
| P10 CLI and report | 4–5 | P9 |
| P11 Validation and review loops | 4–6 (+ Faig's time) | P10 |
| P12 Operations and docs | 2–3 | all |
| **Total** | **55–79 working days** | |

P1 downloads and P2 callouts can run in parallel with P3–P4 development on the golden
demo. Training time in P6 is machine time, not working time.

---

## Appendix A — Schemas

### A.1 Raw tables (awpy 3 parse, polars)

| Table | Columns (dtype) |
|---|---|
| `meta` | `map (str)`, `tick_rate (i32)`, `game_version (str)`, `server (str)`, `demo_sha256 (str)` |
| `rounds` | `round_num (i32)`, `start_tick`, `freeze_end_tick`, `end_tick`, `official_end_tick (i64)`, `winner_side (str: ct/t)`, `end_reason (str)`, `bomb_planted (bool)`, `ct_score_before`, `t_score_before (i32)` |
| `ticks` | `tick (i64)`, `round_num`, `steamid (u64)`, `name (str)`, `side (str)`, `x,y,z (f32)`, `yaw,pitch (f32)`, `vx,vy,vz (f32)`, `health (i16)`, `armor (i16)`, `has_helmet (bool)`, `has_defuser (bool)`, `has_c4 (bool)`, `is_alive (bool)`, `active_weapon (str)`, `inventory (list[str])`, `money (i32)`, `equipment_value (i32)`, `is_scoped`, `is_ducking`, `is_walking`, `is_reloading`, `is_defusing`, `is_planting (bool, nullable)`, `flash_duration (f32)`, `spotted_by (list[u64])`, `kills_round`, `damage_round (i32)` |
| `kills` | `tick`, `round_num`, `attacker_steamid`, `victim_steamid`, `assister_steamid (nullable)`, `flash_assister_steamid (nullable)`, `weapon (str)`, `headshot`, `penetrated (i16)`, `through_smoke`, `attacker_blind`, `victim_blind (bool)`, `noscope (bool)`, `distance (f32)`, `attacker_x,y,z`, `victim_x,y,z (f32)`, `team_kill (bool)` |
| `damages` | `tick`, `round_num`, `attacker_steamid (nullable)`, `victim_steamid`, `weapon`, `dmg_health`, `dmg_armor (i16)`, `hitgroup (str)` |
| `grenades` | `entity_id (i64)`, `thrower_steamid`, `type (str: smoke/flash/he/molotov/incendiary/decoy)`, `throw_tick`, `detonate_tick`, `expire_tick (nullable)`, `x,y,z` at detonate |
| `smokes` | `entity_id`, `start_tick`, `end_tick`, `x,y,z`, `thrower_steamid` |
| `infernos` | `entity_id`, `start_tick`, `end_tick`, `x,y,z`, `extent_radius (f32)`, `thrower_steamid` |
| `flashes` | `entity_id`, `detonate_tick`, `player_steamid`, `blind_duration (f32)` (one row per blinded player) |
| `bomb` | `tick`, `round_num`, `event (str: plant_start/plant_complete/defuse_start/defuse_complete/defuse_abort/drop/pickup/explode)`, `steamid (nullable)`, `site (str, nullable)`, `has_kit (bool, nullable)`, `x,y,z` |
| `weapon_fire` | `tick`, `steamid`, `weapon` |
| `footsteps` | `tick`, `steamid`, `x,y,z` (if exposed) |
| `items` | `tick`, `steamid`, `event (drop/pickup)`, `item (str)` |

### A.2 Player token (per snapshot, ×10)

| Field | dtype | Null policy | Live-safe justification |
|---|---|---|---|
| `side` | cat {ct,t} | never null | known |
| `alive` | 0/1 | never null | known |
| `x,y,z` | f32 normalized to map bounds | null if never seen this round | current position |
| `nav_area` | cat (per map version) | null off-mesh | current |
| `callout` | cat (per map version) | null off-map | current |
| `grid_cell` | cat | null off-map | current |
| `yaw,pitch` | f32 (sin/cos encoded) | null when dead | current |
| `speed` | f32 | null when dead | current |
| `health`, `armor` | f32 | 0 when dead (legitimate) | current |
| `has_helmet`, `has_defuser`, `has_c4` | 0/1 | null when dead | current |
| `weapon_class` | cat | null when dead/unknown | current |
| `n_smoke`, `n_flash`, `n_he`, `n_molly`, `n_decoy` | i8 | null when dead | current |
| `is_scoped`, `is_ducking`, `is_walking`, `is_reloading`, `is_defusing`, `is_planting` | 0/1 | null if parser lacks the prop | current |
| `flash_duration_remaining` | f32 s | 0 when not blind | current |
| `in_smoke`, `on_fire` | 0/1 | never null when alive | derived from current tokens |
| `money`, `equipment_value` | f32 | null when dead | current |
| `seconds_since_last_seen_by_enemy` | f32 | null if never seen | past only |
| `seconds_since_last_shot` | f32 | null if never shot | past only |
| `spotted_by_count` | i8 | null when dead | current |
| `heard_enemy` | 0/1 | null when dead | past window only |
| `kills_round`, `damage_round` | f32 | never null | past only |
| `player_id` | cat (UNK for unseen) | never null | identity |
| `tier` | cat | never null | metadata |

### A.3 Utility token (≤ 16 per snapshot)

`kind (smoke/fire/decoy)`, `x,y,z (normalized)`, `time_left_s (f32)`, `owner_side`,
`owner_id (cat)`, `radius_units (f32)`.

### A.4 Context token

`map_version (cat)`, `tier (cat)`, `time_remaining_s (f32, live clock)`,
`time_into_round_s`, `bomb_planted (0/1)`, `bomb_site (cat A/B/null)`,
`time_since_plant_s (null pre-plant)`, `defuse_in_progress`, `defuse_with_kit (null
when no defuse)`, `bomb_carrier_alive (null post-plant)`, `alive_ct`, `alive_t`,
`equip_ct`, `equip_t`, `money_ct`, `money_t`, `loss_streak_ct`, `loss_streak_t`,
`score_ct`, `score_t`, `round_num`, `half`, `is_overtime`, `n_active_smokes`,
`n_active_fires`.

### A.5 Action types and params

| type | kind | params keys |
|---|---|---|
| `kill` | instant | weapon, weapon_class, headshot, through_smoke, attacker_blind, victim_blind, wallbang, noscope, distance_units, attacker_callout, victim_callout, victim_health_before, assister_id, flash_assister_id, victim_id |
| `team_kill` | instant | as kill |
| `damage` | instant (window metadata) | damage_total, hits, weapon_class, armor_damage, victim_id |
| `flash` | instant | blinded_enemies[{id,duration}], blinded_teammates[...], blinded_self, land_callout, throw_callout, pop_delay_s |
| `he` | instant | enemies_damaged[{id,dmg}], teammates_damaged, damage_total, land_callout |
| `peek` | instant | from_callout, target_id, target_callout, target_spotted_me_first |
| `plant_start`, `plant_complete` | instant | site, planter_callout, ct_alive, t_alive |
| `defuse_start`, `defuse_complete`, `defuse_abort` | instant | kit, ct_alive, t_alive |
| `bomb_drop`, `bomb_pickup` | instant | callout |
| `smoke` | durative | land_callout, throw_callout, throw_tick, pop_tick, expire_tick, blocks_pairs_at_pop |
| `molotov` (incl. incendiary) | durative | land_callout, enemies_damaged, enemies_displaced |
| `decoy` | durative | land_callout |
| `rotation` | durative | from_callout, to_callout, nav_dist_units, duration_s, toward_bomb |
| `hold` | durative | callout, duration_s, facing_callout |
| `save` | durative | from_callout, to_callout, teammates_alive |
| `buy` | round | loadout_class, equip_value, money_before, money_after, has_armor, has_helmet, has_kit, utility_count, primary_class |
| `team_buy` | round | team_class, total_equip |
| `drop` | instant | weapon_class, receiver_id |
| pseudo: `smoke_pop`, `smoke_fade`, `fire_start`, `fire_end`, `decoy_start`, `decoy_end` | instant | parent_action_id |

### A.6 SQLite additions (migration v3, additive)

```
engine_matches(match_key TEXT PK, source, event, event_tier INT, date TEXT, team_ct_start,
  team_t_start, map, map_version, demo_sha256 TEXT UNIQUE, file_bytes INT, game_version,
  tick_rate INT, tier, status, quarantine_reason, created_at)
engine_runs(run_id TEXT PK, kind, started_at, finished_at, config_hash, git_sha,
  manifest_path, status)
engine_action_scores(action_id TEXT, match_key, round_num INT, run_id, V REAL, V_direct REAL,
  V_enabler REAL, V_movement REAL, Q REAL, B REAL, D REAL, R REAL, X REAL, method,
  n_coalition INT, residual REAL, flags TEXT, PRIMARY KEY(action_id, run_id))
idx_eas_match(match_key, round_num); idx_eas_run(run_id)
```

---

## Appendix B — Formulas

### B.1 Win probability and conservation

```
W(s_t)            CT round-win probability, calibrated, W(terminal) ∈ {0, 1} after pinning
ΔW_i = W(end_i) − W(start_i)                     per interval i of the partition
Σ_i ΔW_i = W(terminal) − W(freeze_end)           telescoping
```

### B.2 Shapley over coalitions (event windows)

```
v(S) = W( apply(s_pre, s_post, S) )              S ⊆ A, A = window members
φ_a  = Σ_{S ⊆ A\{a}} |S|!(|A|−|S|−1)!/|A|! · [v(S∪{a}) − v(S)]
Σ_a φ_a = v(A) − v(∅) = ΔW_window                efficiency (conservation)
φ_a = 0 if v(S∪{a}) = v(S) ∀S                    null action (condition C1)
```

Sampled estimator for |A| > 6: `φ̂_a = (1/n) Σ_π [v(Pre_π(a) ∪ {a}) − v(Pre_π(a))]` over `n`
permutations with antithetic pairs; residual `ΔW − Σ φ̂_a` redistributed ∝ |φ̂_a|.

### B.3 Movement and clock (drift intervals)

```
movers M = { p alive : ‖pos_p(t1) − pos_p(t0)‖ ≥ movement_min_units }
v(S) = W( s1 with every p ∈ M\S frozen to pos_p(t0) )
φ_p  = Shapley over M (exact ≤ 6, sampled otherwise), or leave-one-out
       φ_p = W(s1) − W(freeze(s1, p, s0)) with efficiency enforcement when |ΔW| < 0.02
clock = ΔW − Σ_p φ_p    (CT-positive, signed)
clock_ct = Σ max(0, clock) over the round ;  clock_t = Σ max(0, −clock)   (both ≥ 0)
```

### B.4 Enabler split (hazard-based)

```
q_v(s)    = hazard head for the event type (death within 5 s / plant within 15 s / defuse within 15 s)
raw_i     = max(0, q_v(s_pre) − q_v(remove_i(s_pre))) / q_v(s_pre)
share_i   = clamp(raw_i, band_i)        band_trade = [0.05, 0.40]; other bands [0, 0.6]
total     = min(Σ_i share_i, 0.5); if capped, share_i ← share_i · 0.5 / Σ share
V_i      += share_i · φ_e ;   V_actor += (1 − total) · φ_e
```

Fixed fallback (only if hazard AUC < 0.75): flash 0.20, smoke_cover 0.10, trade 0.20,
he 0.10, fire 0.10, cap 0.5.

### B.5 Realized value

```
V_raw(a) = Σ_windows φ_a  +  Σ_events share_a · φ_e  +  Σ_intervals φ_a^move
V(a)     = +V_raw if actor is CT, −V_raw if actor is T
Σ_a V_raw(a) + clock_ct − clock_t = W(terminal) − W(freeze_end)      (conservation identity, CT-positive)
```

### B.6 Buy valuation

```
s_prior   = freeze-end snapshot with every player's equipment replaced by the economy prior:
            the median loadout class observed in training for (money bucket, side, round type)
V_team_buy = W(s_freeze_end) − W(s_prior)          (CT-positive, signed per side)
V_buy(p)   = Shapley over players' loadouts with v(S) = W(s_prior with S's real loadouts)
             (exact, |S| ≤ 5)
```

### B.7 Expected value, decision, execution

```
Q(s,a) = E[V | s, a]        learned regression on (h_pre, descriptor(a), tier, side)
B(s)   = E[V | s]           learned regression on (h_pre, tier, side)
R(s,a) = Q(s,a) − max{ Q(s,a') : a' ∈ panel(s), n(a') ≥ n_min }  ≤ 0
D      = Q(s,a) − B(s)
X      = V − Q(s,a)
V      = B(s) + D + X
```

### B.8 Shrinkage

```
X̃_p,c = X̄_p,c · n_p,c / (n_p,c + k_c),   k_c = σ²_noise,c / σ²_skill,c
σ²_skill,c = max(ε, Var_p(X̄_p,c) − σ²_noise,c / n̄_c)
CI_90   = X̃ ± 1.645 · sqrt( σ²_noise,c / (n_p,c + k_c) )
```

### B.9 Calibration and drift checks

```
Temperature: p = σ(logit / T), T = argmin NLL on val (pro)
ECE = Σ_b (n_b / N) · |mean_pred_b − obs_b|,  20 bins;  target ≤ 0.02 on test pro
Martingale: E[W(t+k) − W(t) | W(t) ∈ bin] within ±0.01 for k ∈ {1, 5, 10} s
```

---

## Appendix C — Test index (every planned test script)

Legacy suite (unchanged, must stay green): `test/test_invariants.py`, `test_db.py`,
`test_attribution.py`, `test_feature_extractor.py`, `test_state_sampler.py`,
`test_score_impact.py`, `test_pipeline.py`, `test_download.py`,
`test_attribution_windows.py`, `test_build_state.py`, `test_win_probability.py`,
`test_attribution_integration.py`, `test_attribution_compare.py`.

Engine suite (`test/engine/`), grouped by phase. L1 = unit (seconds, synthetic),
L2 = integration on the golden demo (minutes), L3 = evaluation on real splits (hours,
run by `engine eval/sanity`), L4 = human review.

| Phase | Script | Level | What it proves |
|---|---|---|---|
| P0 | `test_env.py` | L1 | venv, torch, device, locked versions |
| P0 | `test_log_redaction.py` | L1 | no signed URL reaches a log |
| P0 | `test_paths.py`, `test_config.py`, `test_manifest.py` | L1 | layout, config validation + hash, manifests |
| P0 | `test_golden_present.py` | L2 | golden demo hash |
| P1 | `test_hltv_source.py`, `test_hf_archive.py` | L1 | downloaders against local servers, redaction |
| P1 | `test_matches_meta.py`, `test_inventory.py`, `test_quarantine.py`, `test_splits.py` | L1 | metadata, counts, quarantine rules, split integrity |
| P2 | `test_parse_parity.py` | L2 | awpy 2 vs awpy 3 zero-diff |
| P2 | `test_map_assets.py`, `test_los_fixtures.py`, `test_nav_coverage.py`, `test_callouts.py` | L1/L2 | assets pinned, LOS fixtures, nav coverage, callout polygons |
| P2 | `test_raw_schema.py`, `test_raw_golden.py`, `test_parse_worker.py` | L1/L2 | raw tables, golden counts, worker idempotency |
| P3 | `test_sampler.py`, `test_player_token.py`, `test_utility_token.py`, `test_context.py`, `test_pairwise.py` | L1 | each state component |
| P3 | `test_leakage.py` | L1 | outcome-flip invariance, LIVE_SAFE list, no future rows |
| P3 | `test_labels.py`, `test_states_schema.py`, `test_states_golden.py`, `test_synthetic.py` | L1/L2 | labels, schema, golden counts, fixtures |
| P4 | `test_action_types.py`, `test_actions_kills.py`, `test_actions_utility.py`, `test_actions_movement.py`, `test_actions_bomb.py`, `test_actions_economy.py` | L1 | each type, positive and negative |
| P4 | `test_enablers.py`, `test_windows.py`, `test_actions_schema.py`, `test_actions_golden.py`, `test_actions_invariants.py` | L1/L2 | enablers, grouping, schema, golden, invariants |
| P5 | `test_db.py` (extended), `test_invariants.py` (extended), `test_layout.py`, `test_dataset.py`, `test_qa.py` | L1 | migration v3, layout, shards/vocab/norm, QA |
| P6 | `test_baseline.py`, `test_tokens.py`, `test_set_encoder.py`, `test_temporal.py`, `test_heads.py`, `test_wp_model.py` | L1 | baseline, invariance, causality, shapes, gradients |
| P6 | `test_overfit_tiny.py`, `test_train_loop.py`, `test_augment.py`, `test_predict.py`, `test_calibrate.py` | L1 | learning, checkpoints, augmentation, caching, temperature |
| P6 | `engine eval --calibration`, `engine sanity --model` | L3 | ECE, reliability, pinning, monotone edits, martingale |
| P7 | `test_interventions.py`, `test_apply.py`, `test_state_validate.py` | L1 | every intervention, apply identities, derived consistency |
| P7 | `engine sanity --counterfactual` | L3 | the gate (un-kill, smoke, flash, plant, freeze, order, OOD) |
| P8 | `test_partition.py`, `test_shapley.py`, `test_attribution_shapley.py` (C1), `test_movement.py`, `test_enabler_split.py`, `test_attribution_round.py` | L1 | tiling, Shapley axioms, C1 both variants, movement, splits, conservation |
| P8 | `engine eval --attribution` | L3 | residuals on 1,000 real rounds, player vs clock share |
| P9 | `test_descriptors.py`, `test_q_model.py`, `test_neighbours.py`, `test_decision_split.py` (C2), `test_shrinkage.py`, `test_planted_signal.py` | L1 | descriptors, Q/B, panel, identity + C2, shrinkage, signal recovery |
| P9 | `engine eval --value` | L3 | Q/B calibration, per-type sanity |
| P10 | `test_cli_analyze.py`, `test_cli_help.py`, `test_cli_errors.py`, `test_cli_guards.py`, `test_report_schema.py`, `test_report_redaction.py` | L1/L2 | CLI, report schema, reproducibility, redaction |
| P10 | `bench_analyze.py --assert` (C3) | L2 | time budget |
| P11 | `engine eval --anchors`, `test_review_items.py` | L3/L4 | external bands, regression tests from Faig's review |

Bench scripts: `bench_gpu.py`, `bench_parse.py`, `bench_states.py`, `bench_actions.py`,
`bench_predict.py`, `bench_attribution.py`, `bench_value.py`, `bench_analyze.py`.

Global test loops: (1) after every step, the subphase's L1 tests; (2) at every subphase
end, `run_all.py --suite all`; (3) at every phase end, L2 on the golden demo plus the
bench for that stage; (4) at P6, P7, P8, P9 end, the L3 evaluations; (5) at P11, L4
review cycles until the target is met; (6) nightly during P6–P9: `run_all.py --suite
all` and `bench_analyze.py --assert` on a scheduled task, failures listed next morning.

---

## Appendix D — Config keys (`engine/config.yaml`)

```
sampling:   sample_hz 2 | boundary_offset_ticks 1 | max_utility_tokens 16 | los_cache true
clock:      round_time_s 115 | bomb_timer_s 40 | freeze_time_s 20 (unused by features)
actions:    simultaneity_ticks 2 | damage_gap_s 2 | movement_min_units 32 | rotation_min_units 600
            hold_min_s 5 | hold_max_speed 20 | save_min_units 800 | save_window_s 5
            trade_window_s 5 | trade_max_units 1200 | enabler_recent_s 3
            buy_bands {pistol 1000, eco 2000, half 3500, force 4500}
model:      d_model 128 | d_pool 256 | L_set 3 | L_time 2 | n_heads 4 | temporal_window_s 10
            hazard_horizon_s 5 | plant_horizon_s 15 | defuse_horizon_s 15
            loss_weights {win 1.0, hazard 0.5, plant 0.2, defuse 0.2, tte 0.1} | terminal_weight 2.0
train:      lr 3e-4 | wd 0.01 | warmup_steps 1000 | batch_rounds 256 | max_epochs 30 | patience 3
            pro_oversample 4 | pro_finetune_epochs 2 | pro_finetune_lr 3e-5 | seed 1337
augment:    util_drop 0.1 | pos_jitter_units 16 | unflash 0.05 | los_flip 0.02 | dead_token_drop 0.5
calib:      ece_target 0.02 | bin_tol 0.03 | min_bin_n 200 | martingale_tol 0.01
sanity:     thresholds for P6.6 and P7.2 as listed there | ood_percentile 99 | ood_max_rate 0.05
attribution: shapley_exact_max 6 | shapley_samples 200 | drift_threshold_wp 0.02
            enabler_mode hazard | enabler_total_cap 0.5 | trade_share_band [0.05, 0.40]
            other_share_band [0.0, 0.6] | min_hazard 0.02
value:      k_neighbours 200 | n_min_regret 30 | delta_D 0.01 | delta_X 0.02 | q_finetune_encoder false
bench:      budgets per stage (gpu/cpu) | hard_ceiling_gpu 120 | hard_ceiling_cpu 300
data:       awpy_data_release <pin> | targets {pro_maps 3000, pro_per_map 200, faceit_maps 50000}
qa:         max_null_pct per family | max_outliers
```

---

## Appendix E — Dependencies for `.venv-engine` (ask-before list)

| Package | Why | Notes |
|---|---|---|
| `torch` (AMD ROCm Windows build, 2.9.x) | the models | from AMD's index; fallback WSL2 or CPU wheel |
| `awpy` 3.x | demo parse (Rust), VisibilityChecker, NavMesh | replaces awpy 2 for the engine only |
| `polars` 2.x | awpy 3 output frames | required by awpy 3 |
| `pyarrow` | parquet read/write | |
| `numpy`, `pandas` | numerics, legacy interop | pandas only where legacy code is imported |
| `lightgbm` | yardstick baseline, optional ensemble | same version as legacy lock |
| `scikit-learn` | metrics, calibration helpers | |
| `faiss-cpu` | neighbour index | if no 3.12 wheel, `hnswlib` or `pynndescent` |
| `matplotlib` | charts | no seaborn |
| `pyyaml` | config | |
| `jsonschema` | report schema validation | |
| `zstandard` | demo decompression (reuse) | already in legacy lock |
| `requests` | downloads (reuse of `scraper.download_demo`) | already in legacy lock |

No pytest (decided). No seaborn, no plotly, no notebooks in the repo.

---

## Appendix F — Risk register

| # | Risk | Likelihood | Impact | Mitigation / fallback | Where |
|---|---|---|---|---|---|
| R1 | AMD Windows PyTorch unstable | medium | high | WSL2 ROCm; CPU training | P0.1.5 |
| R2 | Pro demo volume too low | medium | high | FACEIT with tier token; report it honestly | A4, P1 |
| R3 | awpy 3 parity differences | medium | medium | per-field decision table; workarounds | P2.1.3 |
| R4 | Callouts missing for a map | high | medium | hand-drawn polygons; review images | P2.3 |
| R5 | Pairwise LOS too slow | medium | medium | per-match cache; area-to-area table | P3.5 |
| R6 | A leaky feature slips in | low | very high | automated outcome-flip audit + LIVE_SAFE list | P3.6 |
| R7 | NN does not beat the baseline | medium | high | decision rule; back to features before attribution | P6.7.2 |
| R8 | Hazard head too weak for enabler splits | medium | medium | fixed-share fallback, declared in reports | P6.7.2, P8.5.3 |
| R9 | Counterfactual states off-distribution | medium | high | augmentation, OOD detector, the P7 gate, five-iteration loop | P6.3.3, P7 |
| R10 | Busy windows explode combinatorially | low | medium | sampling with efficiency enforcement, member merging | P8.2.2 |
| R11 | Clock absorbs too much credit | medium | low | informational target; movement thresholds | P8.4 |
| R12 | Rare descriptors give unstable Q | high | low | smoothing via MLP; low-confidence flags | P9.2 |
| R13 | Reports disagree with analyst intuition | high | medium | review loop with regression tests | P11.2 |
| R14 | Map rework mid-project | medium | medium | map_version keys; retrain checklist | P2.4.2, P12.2 |
| R15 | Secret or path leaks into a report or log | low | high | redaction hooks and tests; security review before commits | P0.2.4, P10.2.4 |

---

## Appendix G — Decision log (today)

All decisions in Section 2.1 were made on 2026-10-19 in conversation with Faig. The
rationale for the ones most likely to be questioned later:

- **NN over LightGBM:** not for the architecture's sake; because the input changes from
  32 team aggregates to per-player positional state with pairwise sight, which trees
  cannot use well, and the published CS2 result (Szmida & Toka 2025) shows the gain
  once positions are in. The LightGBM baseline stays as the yardstick and the decision
  rule in P6.7.2 can still reject the NN.
- **Shapley with state editing over grid bracketing:** the only construction that
  gives exact zero to a no-effect action on a shared tick (null-action axiom) and exact
  conservation (efficiency axiom).
- **Hazard-based enabler splits over fixed percentages:** situation-aware; bands and a
  cap protect against a weak head; a fixed fallback exists and is declared.
- **Regret from seen actions only:** extrapolated Q values are the most likely place
  for a confident wrong number; unacceptable for a buyer demo.
- **2 Hz:** actions are valued at their exact ticks regardless; the rate only affects
  movement credit; 4 Hz is an ablation with a decision rule, not a latency knob.
- **FACEIT in training, pro in evaluation:** geometry and mechanics are tier-independent
  and need volume; the tier token keeps expectations separate; the ablation decides.
- **CLI first:** the engine must be right before it is pretty.

---

## Appendix H — Open questions for Faig (answer at review, none block the plan)

*[q1 answered 2026-10-19: no HLTV automation (Amendment 2, D18). q6 answered: Amendment 2, D21. The rest are open.]*

1. **Data sources (P1.1.1):** are you comfortable with lightly automated HLTV demo
   downloads (rate-limited, browser-like), or should pro demos come only from the
   public archive and manual downloads?
2. **Callouts (P2.3):** will you review the per-map callout images, or should I pick
   a community callout set and we accept it as is?
3. **Golden demo (P0.7.1):** any preference for the match (team, event, map)? I would
   otherwise take a recent tier-one Mirage or Inferno BO1.
4. **Review cycles (P11.2):** two cycles of 20 rounds each is the minimum in the plan.
   Is that review time acceptable for you?
5. **Disk (A5):** which drive should `data/engine/` live on, and how much space can it
   use?
6. **Tier buckets (D7):** `pro / faceit_10 / faceit_8_9 / other` is my proposal. Keep,
   or collapse to `pro / faceit`?
7. **Score in the context token (A.4):** I include match score and overtime. HLTV and
   Leetify exclude score. Keep (it is live information and affects play) or drop?
8. **Shrink target for D̃ (B.8):** D = Q − B has a non-zero class mean (kills about
   +0.07), so shrinking toward 0 would reward volume. **Decided 2026-10-19 (Faig):
   shrink toward the class mean**; `engine/value/shrinkage.py` does so.
9. **Round floor (P1.4.1):** "≥ 16 rounds" refused legitimate MR12 maps ending 13-0 to
   13-2. **Decided 2026-10-19 (Faig): the floor is 13.**

---

## Appendix I — Plain-language explanation (the version for a non-technical reader)

**The win meter.** A meter on the wall shows how likely the CT team is to win the round,
0 to 100. The neural net is the kid who reads the meter. Every half second, and right
before and right after anything happens, the kid looks at the whole map and says a
number.

**Who moved the needle.** When the needle jumps, we want to know who did it. The trick is
"take it back and look again". Two things happen on the same tick: player A throws a
smoke that lands nowhere useful, player B dies. The needle drops 50. We rewind, pretend
the smoke never happened but B still died: still down 50, so the smoke moved it by 0.
Pretend B did not die but the smoke still landed: no move, so B's death moved it by 50.
A gets 0, B gets minus 50. That is the Shapley formula: try every order of "take this one
back, now that one", average what each action moved on its own. Two promises: the pieces
always add up to the whole jump, and anything that changed nothing gets exactly zero.

**Helpers.** If B was killed because they were flashed, the flasher deserves a cut. We
ask the kid: "how likely was B to die in the next five seconds?" Then take back the flash
and ask again. If the chance went from 60 to 20, the flash was responsible for two thirds
of that kill, so the flasher gets two thirds of the credit and the shooter keeps the rest.
No fixed percentages, the situation decides (with a safety band for trades).

**Good idea or good hands.** Every action gets three numbers. **V**, what actually
happened because of you. **Q**, what usually happens when someone does this kind of thing
in this kind of spot, averaged over thousands of cases. **B**, what people usually do here
and what that is usually worth, the par score for the situation. **Decision** is Q minus
B: was the choice better than the usual choice. **Execution** is V minus Q: did you carry
it out better than most. **Regret** is Q minus the best known choice here; zero means
nothing we have seen anyone do here would have been better. Par plus decision plus
execution equals V, always.

**Luck.** One lucky shot does not make a good shooter. Execution scores are pulled toward
zero until a player has done enough actions to earn the number.

**Speed.** The kid reads the meter about forty thousand times for one map. Reading is
fast, so the whole thing is done in about a minute.

---

## Appendix J — Sources consulted (2026-10-19)

- HLTV, "Introducing Rating 3.0" and the October 2026 tweaks (Round Swing, eco
  adjustment): hltv.org/news/42485; cs2news.com coverage of the hotfix.
- Leetify, "How does Leetify Rating actually work?" and the CS2 recalibration posts.
- Xenopoulos, Doraiswamy, Silva, "Valuing Player Actions in Counter-Strike: Global
  Offensive", IEEE Big Data 2020 (WPA, XGBoost).
- Xenopoulos et al., "ESTA: An Esports Trajectory and Action Dataset", 2022 (set
  representation of game state noted as unexplored for win probability).
- Szmida, Toka, "Evaluating Player Actions in Professional Counter Strike using Temporal
  Heterogeneous Graph Neural Networks", MIT Sloan Sports Analytics Conference 2025
  (114 Inferno matches, 4 Hz, THGNN log loss 0.4587 vs XGBoost 0.5353, SHAP-based action
  attribution with a ridge surrogate).
- Independent CS2 Inferno win-probability study (220 tier-one matches): logistic
  regression, tree ensembles, temporal CNN, transformer, graph attention statistically
  indistinguishable; external player-skill prior collapsed out-of-time.
- Liu, Schulte, "Deep Reinforcement Learning in Ice Hockey for Context-Aware Player
  Evaluation", IJCAI 2018 (Q-function impact, GIM).
- Jutamulia, Hosoi, "Leveraging expected action value to measure on-court
  decision-making skills in basketball", 2025 (strategy vs execution split).
- Shapley value in sports: generalized Shapley for basketball player importance; Sloan
  paper on TD learning with Shapley attribution in the NBA.
- Empirical Bayes shrinkage: Efron–Morris; PGA Tour skill-vs-noise analysis (2025).
- Guo et al., "On Calibration of Modern Neural Networks", ICML 2017 (temperature scaling).
- Zaheer et al., "Deep Sets", NeurIPS 2017; Lee et al., "Set Transformer", ICML 2019.
- awpy 3.x (PyPI): `VisibilityChecker`, `NavMesh.find_area/find_path`, Polars output;
  awpy issue #390 on CS2 callouts via `env_cs_place`.
- demoparser2 / community pipelines: ~10–15 s per demo parse on a desktop.
- AMD, "PyTorch on Windows" release notes (ROCm 7.2.x; RX 9060 XT listed; Python 3.12;
  only PyTorch supported on Windows).

---

*End of plan.*

---

## Amendment 1 (2026-10-19, night) — clean break, confirmed by Faig

Faig confirmed deleting the old engine instead of building beside it. This amendment
overrides the matching parts of the plan above; everything not mentioned stands.

### Deleted (git rm)
`src/score_impact.py`, `src/win_probability.py`, `src/attribution.py`,
`src/state_sampler.py`, `src/feature_extractor.py`, `src/extract.py`, `src/main.py`,
`src/pipeline.py`, `src/db.py` (legacy schema), their tests under `test/`,
`test/_synthetic.py`, `test/_harness.py` (re-created under `test/engine/`),
`docs/design/impact-attribution.md`, `requirements.txt`, `requirements.lock`. The
legacy `.venv` (3.11) is renamed `.venv-old311` and can be removed by Faig later.

### Moved, logic unchanged
`src/scraper.py` → `engine/sources/scraper.py`, `src/faceit_api.py` →
`engine/sources/faceit_api.py`, `test/test_download.py` → `test/engine/test_download.py`,
`test/test_scraper.py` → `test/engine/test_scraper.py`. The only edits are import
paths (`import db` → `from engine import db`) because the module moved; the download
validation, redaction, allowlist and API code are byte-for-byte otherwise.

### Consequences
- **One venv**, Python 3.12 at `.venv`, requirements in `requirements-engine.txt` +
  `requirements-engine.lock`. `pyproject.toml` reads deps from the new file. D11 no
  longer needs a second venv.
- **P2.1 parity is void** (no awpy 2 side). Correctness of the parser rests on the
  synthetic `RawTables` fixtures, the raw-schema validator, and the golden demo once
  one exists.
- **P5.2 migration v3 is void.** The engine owns a fresh database
  `data/engine/engine.db` with `SCHEMA_VERSION = 1`: tables `matches` (Appendix A.6
  `engine_matches` columns), `runs`, `action_scores`, and `processed` (`match_id` PK,
  `map`, `demo_url`, `processed_at`) which serves `is_processed()` for both id forms so
  the moved sources keep their dedup invariant. Additive migrations from v1 onward, same
  mechanism as before (`PRAGMA user_version`). `data/crosshair.db` on the Mac is an
  archive and is never opened by the engine.
- **The seam** becomes `engine.ingest.register_demo(match_id, demo_path, demo_url)`:
  copies the `.dem` under `data/engine/demos/<tier>/`, inserts `matches` + `processed`,
  and nothing else. Parsing, states and actions are separate CLI stages, so a
  downloaded demo is never half-processed.
- **CLI:** `python -m engine.cli <command>`; `pyproject.toml` registers
  `crosshair = engine.cli:main`. Legacy `main.py` commands `status|scrape|demo` are
  re-provided as `engine.cli status|scrape|ingest`.
- **Test harness:** `test/engine/_harness.py` is the one harness; `test/run_all.py`
  runs `test/engine/test_*.py` only.

### Contracts fixed for the agents (so parallel work composes)
- `engine/parse/schema.py`: `RawTables` dataclass with one polars DataFrame per
  Appendix A.1 table plus `meta`; `RawTables.validate()` checks columns and dtypes;
  `RawTables.slice_until(tick)` returns the causal slice used by the state builder.
- `engine/state/schema.py`: `PlayerToken`, `UtilityToken`, `ContextToken`,
  `Snapshot` dataclasses (fields exactly Appendix A.2–A.4; `pair: np.ndarray[10,10,5]`;
  `meta: match_key, round_num, tick, t_s, kinds`; `labels: dict | None`), with
  `Snapshot.to_row()` / `from_row()` for parquet and `Snapshot.validate()` (the derived
  consistency checker of P7.1). Interventions edit `Snapshot` objects, never tensors.
- `engine/actions/types.py::Action` exactly as P4.1.1.
- `engine/model/predict.py::Predictor.predict(snapshots, histories) -> dict(win,
  hazard, plant15, defuse15)`; a `StubPredictor` with pluggable rules lives in
  `test/engine/_stubs.py` and is what every attribution/value test uses.
- Batch tensor dict keys exactly as P5.3.1.
- Config keys exactly Appendix D; `engine/config.yaml` is created in P0 with every key
  and later phases only read it.

---

## Amendment 2 (2026-10-19, evening) — FACEIT-first: very good players, not pros

Faig decided on 2026-10-19, after the data research in
`reports/CS2 organizer demo packs.md` and `docs/DATA_SOURCES.md`, to re-aim the
product: **sell it to FACEIT (ESL FACEIT Group) and let ordinary players use it on
their own matches**, with "very good FACEIT players" as the reference population
instead of professionals. This amendment overrides the matching parts of the plan
above; everything not mentioned stands. Amendment 1 stands in full. No code was
changed with this amendment; the code consequences are listed at the end as work
items.

### Why

- HLTV.org's terms (section 2.2, confirmed by Faig in a browser) forbid data mining
  and scraping, commercial exploitation of any content, and building a competing
  product. Automated HLTV downloads are out; manual downloads cannot reach volume, and
  the "commercially exploit" clause still hangs over a sold product trained on them.
- No tournament organizer (ESL/EFG, BLAST, PGL, Valve, ~20 tier-two organizers)
  publishes demo packs. Public archives hold ~2,450 pro maps, 2026 only; nothing
  covers October 2023 to February 2026. No vendor sells `.dem` files. Assumption A4 is
  false.
- FACEIT is the one source where data, terms and buyer point the same way: a
  documented Downloads API, demos of the players who would use the product, and a
  company that already runs an in-house demo analyser and could buy a better one.

### 1. Goal, audience, acceptance tests (replaces §1.1–1.3 where they say "pro")

- **Goal (1.1):** the same engine (realized value, expected value, decision score,
  execution score from a calibrated neural win model, under two minutes per map), at
  a quality that can be sold to ESL FACEIT Group or offered directly to FACEIT players
  as "analyse my match".
- **Audience (1.2):** (a) FACEIT players who want to know which of their actions won
  or lost rounds and what very good players do in the same spot; (b) FACEIT/EFG as a
  buyer of the engine or its author. Every number shown is computed and evaluated on
  the **baseline tier** (below). Professional demos are a transfer test, informational.
- **Acceptance tests (1.3):** C1–C3 unchanged in substance; every "pro" in their
  wording becomes "baseline tier": C1's conservation runs on every round of the golden
  demo and on 1,000 sampled baseline-tier rounds; C2's neighbour-panel coverage is
  ≥ 95% of baseline-tier actions; C3's golden demo is a baseline-tier demo.
- **Non-goals (1.4):** unchanged for v1: CLI and report files only. A web upload page
  or a FACEIT-integrated app is the obvious v2 for the "regular people" half of the
  aim and is **not decided**; nothing in v1 may make it harder (reports stay
  self-contained JSON + Markdown + PNG).

### 2. Tiers (replaces D7 and glossary "Tier"; answers Appendix H q6)

| token | definition | role |
|---|---|---|
| `faceit_10` | FACEIT match in which all ten players are level 10 (≥ 2,001 elo) | **baseline tier**: training, validation, test, calibration, neighbour panel, Q/B, anchors, review loop, golden demo |
| `semi_pro` | FACEIT championship / hub rooms of organised competition: ESEA Advanced and Main, ESL Challenger League, ESL/IEM and EWC open qualifiers | training with its own token; reported separately in eval |
| `pro` | tournament demos from the public archives (or hand-downloaded) | **transfer test only**: never trains the adopted model in v1, never sets the baseline |
| `faceit_8_9` | FACEIT levels 8–9 | reserved; not collected in v1 (quota) |
| `other` | anything else at inference | analysed with a report warning "tier below the trained tiers" |

Per-match metadata records `elo_min` and `elo_avg` of the roster (additive columns,
ask-before, see work items). An **elite** flag (`elo_min ≥ 3,000`, the elo band where
professionals pug) is derived from them and reported, but is not a token in v1; "baseline
= elite only" is an ablation in P6.7 (row (i)), adopted only if it beats `faceit_10` on
the same test months by ≥ 0.003 log loss **and** keeps ECE ≤ 0.02 on the elite test
rows.

**D7 (replaced):** train on `faceit_10` + `semi_pro` with the tier token (and `pro` as
an ablation row only); evaluate and calibrate on `faceit_10` test months; never feed
per-player external ratings. Config keys `train.pro_oversample`, `train.pro_finetune_*`
are read as applying to the **baseline tier**; the oversample default becomes 1 (the
baseline is already the majority); the fine-tune stage stays, on `faceit_10` only.

### 3. Data acquisition (replaces P1.1, P1.2, A4, R2)

- **P1.1 (replaced): professional demos are optional and transfer-only.** Source: the
  Hugging Face repos found on 2026-10-19 (`blanchon/opencs2_dataset_demo`, 1,988 maps,
  128-tick by header on every row, 2026-03-23 to 2026-04-20; `cs2povarchive/cs2-demos`,
  283 maps incl. 219 IEM Cologne Major 2026; `aldigobbler/cs2demos`, 177 maps), via the
  existing `hf_archive.py` (listing, waves, sha256). Header-check tick rate before
  ingest; keep HLTV match id and sha256 per file; ingest tier-one events first and stop
  at about 300–400 tier-one maps unless disk and time are free. HLTV automation
  (`hltv.py`) is **dormant**: not run, not extended, not deleted (deletion is
  ask-before). Event-organizer dumps: closed, none exist. The HLTV licence letter is
  **deferred**, not planned.
- **P1.2 (replaced): FACEIT is the primary source.**
  - **P1.2.0 Application.** Faig applies for the FACEIT Downloads API free tier
    ("up to 1,500 requests or 200 GB per month") with the purpose stated plainly:
    per-action impact analytics for FACEIT matches, demos stored locally for model
    training, never redistributed. Faig reads the FACEIT Developer Terms and Platform
    Terms in a browser before the first bulk run (no fetcher could read them) and
    records the clauses on storage, ML training and commercial use in
    `docs/DATA_SOURCES.md`. Keep the approval e-mail with the key.
  - **P1.2.1 Discovery by player** (exists): high-elo players by region, their match
    history, `--min-level 10` so all ten are level 10. Add `--min-elo 3000` runs for
    the elite subset once the base is collected.
  - **P1.2.2 Discovery by competition** (new, additive to `faceit_api.py`): organizer
    search, then published championships, then `type=past` match pages, then the same
    `demo_url` and the same Downloads exchange. Targets ESEA Advanced/Main, ESL
    Challenger League, ESL/IEM and EWC open qualifiers; tier `semi_pro`. Every
    endpoint and field is verified against one real response before use (rule: never
    invent FACEIT fields). Unverified today: whether these seasons are exposed as
    public championships and whether their `demo_url` is populated.
  - **P1.2.3 Continuous collection.** FACEIT keeps demos about 30 days; there is no
    back-catalogue. The collector runs at least weekly (`scrape --source api
    --min-level 10 --max N`), sized to the monthly quota: with compressed FACEIT demos
    the 200 GB cap is **estimated** at about 1,000 maps/month (measure in month one;
    the 1,500-request cap binds first if the mean file is under about 135 MB). A quota
    ledger (count and bytes per calendar month, additive table, ask-before) stops the
    run at the cap; `engine inventory` prints month-to-date usage.
  - **P1.2.4 Targets (replace P1.1.6 / P1.2.3 and `data.targets`):**
    first training when `faceit_10` ≥ 1,500 maps with ≥ 150 per active-duty map and
    ≥ 300 per month over the last three months; growth ≥ 800 maps/month; `semi_pro`
    ≥ 300 maps before its token is trusted (else it is merged into `other` for that
    training run and the report says so); `pro` ≥ 300 maps for the transfer test.
    The 50,000-map FACEIT target is withdrawn (it needs the enterprise tier).
    `inventory --assert-targets` reports shortfalls as before.
- **P1.3–P1.5 unchanged** (inventory, quarantine at 128-tick, frozen month splits;
  all tiers share the month boundaries).
- **A4 (replaced):** "The FACEIT free Downloads tier is granted and yields ≥ 800
  `faceit_10` maps/month." If refused: the Playwright path on the Mac for Faig's own
  account (his own matches and friends', far lower volume) while the enterprise
  conversation runs; if the quota is lower than estimated: train on what exists and
  say so in the evaluation report.
- **R2 (replaced):** "FACEIT key revoked or quota cut": likelihood medium, impact
  high; mitigation: disclosed purpose on the application, provenance per demo, data
  already on disk keeps its value, enterprise tier as the paid fallback.
- **Provenance (new rule):** every demo row records `source`, `source_url` (redacted),
  `downloaded_at` and `licence_basis` (`faceit_downloads_api`, `hf_<repo>`,
  `manual`), so any block can be purged together with the models trained on it
  (additive columns, ask-before).

### 4. Model, gates and evaluation (word changes, same rules)

- **P5.3 / P6.3.2:** tier sampling draws from `faceit_10` + `semi_pro`; `pro` shards
  are built but excluded from training unless an ablation row asks for them.
- **P6.5:** temperature on `faceit_10` val; reliability and ECE ≤ 0.02 on `faceit_10`
  test; the same report is also produced on the `pro` transfer set, **informational**.
- **P6.6, P7.2:** sanity and the counterfactual gate run on held-out `faceit_10`
  rounds (thresholds unchanged).
- **P6.7.1 ablation rows:** (a) LightGBM; (b) NN `faceit_10` only; (c) NN
  `faceit_10` + `semi_pro` with tier token (default); (d)–(g) as before on (c);
  **(h)** (c) + `pro` in training; **(i)** baseline = elite only. **P6.7.2** rules keep
  their numbers with "pro test" read as "`faceit_10` test"; (h) is kept only if it
  improves `faceit_10` test log loss by ≥ 0.003 **and** the pro transfer log loss (then
  `pro` becomes a training tier by a note in `docs/ENGINE_MODELS.md`).
- **P9.3:** the neighbour index and Q/B are built on `faceit_10` states (plus
  `semi_pro` if adopted in training); the panel reads "what very good players do here".
- **P11.1 anchors:** the published bands came from professional play; they stay as
  informational bands with a note that pug numbers may sit outside them. **P11.1.5**
  (HLTV rating correlation) is replaced by a correlation against the per-player match
  stats FACEIT returns for the same match rooms (fields to be verified against a real
  response; ≥ 50 test matches; Spearman ≥ 0.6 informational).
- **P11.4 (new) Pro transfer report:** `engine eval --transfer` runs log loss, Brier,
  ECE, the anchors and the conservation check on the `pro` test set; results are
  reported, never gated; the report is part of the hand-off (it answers "does it hold
  up on pro play?").
- **P12.5 hand-off:** "five pro match reports" becomes five `faceit_10` match reports
  plus one `semi_pro` and one `pro` report; add the data-provenance statement (sources,
  terms read, dates) and the FACEIT Downloads approval.
- **Golden demo (P0.7):** a `faceit_10` map from the first collected batch: full
  match, ≥ 16 rounds, standard map, no forfeit, 128-tick. Pinned by sha256 as before.

### 5. Decisions added (Appendix G)

| ID | Decision | Consequence |
|---|---|---|
| D17 | Product aim: FACEIT/EFG as buyer, FACEIT players as users; baseline tier `faceit_10`. | This amendment. |
| D18 | No automated access to hltv.org under its current terms; `hltv.py` dormant. | P1.1 replaced; Appendix H q1 answered "no". |
| D19 | Professional demos are a transfer test, not training data, in v1. | P6.7 row (h) is the only way back in. |
| D20 | Data provenance per demo is mandatory. | Additive schema change, ask-before. |
| D21 | Tier buckets `faceit_10 / semi_pro / pro / faceit_8_9 / other`; elite is a flag, not a token. | Appendix H q6 answered. |

### 6. Open questions for Faig (none block the data work)

1. Elite threshold: `elo_min ≥ 3,000` as proposed, or `elo_avg`? (affects only a flag.)
2. Should `semi_pro` rooms be collected from month one, or after the `faceit_10`
   base reaches its target? (They compete for the same monthly quota.)
3. v2 delivery for players: web upload, a FACEIT-integrated app, or a Discord bot?
   Not needed now; it decides whether reports gain a machine-readable summary block.
4. Playwright fallback on the Mac for your own account if the Downloads application
   is refused: acceptable?

### 7. Work items implied (not done; code is untouched by this amendment)

- `engine/config.yaml` `data.targets`: replace with the P1.2.4 numbers; add
  `data.tiers` thresholds and `data.quota` (requests, bytes per month).
- `engine/sources/faceit_api.py`: additive discovery by competition (P1.2.2); elo
  metadata per match; quota ledger hook.
- `engine/db.py`: additive migration v2: `matches.elo_min`, `matches.elo_avg`,
  `matches.licence_basis`, `matches.source_url`, `matches.downloaded_at`; a `quota`
  table. **Ask-before, per the schema rule.**
- `engine/state/schema.py` tier vocabulary and `engine/cli.py --tier` choices:
  `semi_pro` added; `golden` kept.
- `engine/evaluation/registry.py` / `docs/ENGINE_MODELS.md`: "test pro" wording to
  "test `faceit_10`"; a `transfer` column for the pro numbers.
- `engine/cli.py eval --transfer` (P11.4); `engine/evaluation/anchors.py` P11.1.5
  replacement.
- `test/engine/test_cli_guards.py`: the "FACEIT demo analysed with `--tier pro`"
  warning stays; add the `other`-tier warning.
- `docs/ENGINE_OPS.md`: weekly collector step and quota check in the checklist.
