# Engine CLI

`python -m engine.cli <command> [options]` (Windows: `.venv\Scripts\python.exe -m engine.cli ...`;
`pyproject.toml` also registers the `crosshair` console script). Plan reference: P10.1,
Amendment 1. Implementation: `engine/cli.py`, `engine/analyze.py`, `engine/report/`.

## Common behaviour

Every command:

- prints its `run_id`, `config_hash` and device on the first log line, and the stage
  timings at the end, through the redacting run logger (`engine.log`): console lines
  go to stderr as `[LEVEL] message key=value`, and the same records go to
  `<log-dir>/<run_id>.jsonl`;
- writes a manifest: `analyze` writes `<out>/manifest.json`; every other command (and a
  failed `analyze`) writes `<log-dir>/<run_id>/manifest.json` with its argv and exit code;
- answers `--help`;
- checks free space at start (P0.3.2): below 100 GB it warns; below 20 GB it refuses
  batch work (`parse` on a directory, `scrape`) with exit 2.

Common options (every command):

| option | meaning |
|---|---|
| `--config PATH` | engine config YAML (default `engine/config.yaml`) |
| `--log-dir DIR` | run logs and manifests (default `data/engine/logs`) |
| `--debug` | also print the traceback of an unexpected error |

### Exit codes

| code | meaning |
|---|---|
| 0 | done |
| 1 | the command ran and failed (failed parses, a failed check, an unexpected error) |
| 2 | usage error (no command at all prints the overview and exits 0) or missing prerequisite: missing file, bad argument, bad config, no model / model not adopted, unfinished module, refused demo (quarantine, not 128-tick), not enough disk space |
| 130 | interrupted |

Errors print one redacted line and no traceback (`--debug` adds it; the log file always has it).

### Environment variables

| variable | meaning |
|---|---|
| `CROSSHAIR_ENGINE_DB` | engine SQLite file (default `data/engine/engine.db`); tests point it at a temp file |
| `CROSSHAIR_STUB_PREDICTOR=1` | **test only**: `analyze` uses a rule stub (`test/engine/_cf_states.RuleStub(good_win)`) instead of a trained model |
| `FACEIT_API_KEY` | `scrape --source api`; `scrape` loads the repo's `.env` itself |

### Delegated commands

`build-dataset`, `train-wp`, `calibrate`, `eval`, `sanity` and `review` run another
module's `main(argv)`, imported only when the command runs. Every
argument the CLI does not know is passed through. If the module does not exist yet,
has no `main`, or raises `NotImplementedError`, the command exits 2 with a message. To
see the module's own options put them after `--`:
`python -m engine.cli train-wp -- --help`.

`train-q` and `build-neighbours` are different: `engine.value.q_model` and
`engine.value.neighbours` have no `main()`, so small adapters in `engine/cli.py` call
`q_model.train_q` and `neighbours.build_index` directly, and exit 2 with a message when
their inputs (the valued-actions parquet, its `h_pre` embeddings, the Q model dir) are
absent.

---

## Commands

### `analyze`

Demo to report (P10.1.1): parse, states, actions, predict, attribute, value, report.

```
python -m engine.cli analyze <demo.dem> [--tier pro|faceit_10|faceit_8_9|other] [--out DIR]
        [--model wp_vN | --model-dir DIR] [--q-model q_vN --neighbours neighbours_vN]
        [--cpu] [--allow-unadopted] [--show-ids] [--no-db]
```

| option | meaning |
|---|---|
| `--tier` | tier token fed to the model (default `pro`) |
| `--out DIR` | output directory (default `data/engine/reports/<demo stem>`) |
| `--model wp_vN` / `--model-dir DIR` | the win model (`data/engine/models/wp_vN` or any directory) |
| `--q-model`, `--neighbours` | enable the value stage (both or neither); without them `Q, B, D, R, X` are null and the report warns `value_stage_skipped` |
| `--cpu` | run the model on the CPU |
| `--allow-unadopted` | use a model not adopted in `docs/ENGINE_MODELS.md` (warns `model_unadopted`) |
| `--show-ids` | put SteamIDs and player names in the report (default: hashed `p_<hex>` labels only) |
| `--no-db` | do not write `action_scores` rows |
| `--raw-tables-story STORY` | **test only**: skip parsing and analyse a synthetic story from `test/engine/_synthetic.py` (`base`, `flash`, `double_kill_same_tick`, `trade`, `retreat`, `drop`); the demo argument only names the match |
| `--raw-tables-tick-rate N` | **test only**: tick rate of that story (64 exercises the refusal) |

Guards:

- quarantine rules (P1.4.1) run on the header before parsing; a demo that is not
  128-tick is refused: `refused: demo tick rate is 64, the engine analyses only
  128-tick demos (quarantine rule tick_rate_not_128)`, exit 2;
- the model directory must exist and be adopted (P12.1), else exit 2;
- `--tier` different from the demo's source tier (its `matches` row by sha256, else a
  FACEIT server name) prints `tier token overridden: ...` and adds `tier_overridden`
  to the report (P11.3.3);
- a map version with fewer than 100 registered matches warns
  `map_version_low_support` (P11.3.1); without an engine DB the check is skipped with
  an info line.

Outputs in `--out`:

| file | content |
|---|---|
| `report.json` | the report, validated against `engine/report/schema.json` (`report_schema_version: 1`) and redacted; byte-identical across runs except `meta.run_id`, `meta.created_at`, `meta.timings` |
| `report.md` | match header; per round the WP chart and the top 5 actions by abs(V) as cards (who, what, where, when, V, Q, B, D, X, verdicts, enabler lines, the panel's top 5); player table; team table; warnings (grouped) |
| `charts/round_<n>_wp.png` | CT win probability over the round, actions marked by side and sized by abs(V) |
| `charts/player_<p_hex>_summary.png` | shrunk D vs X per type class with 90% intervals (value stage), else V per action type |
| `charts/team_clock.png` | team clock per round |
| `actions_scored.parquet` | one row per scored action, the columns of `engine.value.scores.SCORED_SCHEMA` |
| `timings.json` | seconds per stage + `total` (`meta.timings` in the report covers the stages up to `value`) |
| `manifest.json` | run manifest (`engine.manifest`): inputs with sha256, config hash, git sha, timings, model versions |

DB: one `runs` row (kind `analyze`) and the `action_scores` rows under the run_id.

Usage log (P12.3): a successful run that writes the DB also appends one line through
`engine.evaluation.usage.append_usage` (run_id, match_key, map_version, tier, device,
timings, warnings count, max conservation residual) to `data/engine/eval/usage.jsonl`,
or to `usage.jsonl` next to the database when `CROSSHAIR_ENGINE_DB` is set. A failed
append only logs a warning. `--no-db` runs are not logged.

Redaction (P10.2.4): the report holds no URL (`<url>`), no absolute path outside the
output directory (`<path>`; paths inside become relative), and without `--show-ids` no
SteamID (players are `p_<10 hex>` = sha256 of the engine's player id hash; this is a
pseudonym, not anonymity).

Examples:

```
python -m engine.cli analyze data\engine\demos\pro\m1.dem --model wp_v1
python -m engine.cli analyze m1.dem --model wp_v1 --q-model q_v1 --neighbours neighbours_v1 --out reports\m1
set CROSSHAIR_STUB_PREDICTOR=1 && python -m engine.cli analyze syn.dem --raw-tables-story base --out %TEMP%\syn   (test hook)
```

### `parse`

P2.6 worker on one demo or a directory (`data/engine/parsed/<match_key>/`, one parquet
per table + `meta.parquet` + `manifest.json`).

```
python -m engine.cli parse <dir|demo.dem> [--workers N] [--force] [--smoke] [--out DIR]
```

The quarantine rules (P1.4.1) run on the header before every parse: a demo that fails
one is reported as `status: quarantined` with the `reason`, nothing is parsed or
written, and when the match has a `matches` row the file is moved under
`data/engine/demos/quarantine/<reason>/` and the row set to `quarantined` (`moved: true`
in the result; without a row the file stays, `moved: false`). `build-states` and
`build-actions` refuse a quarantined match and never overwrite its status.
`--smoke` runs only the header + quarantine check and writes nothing (no move either).
Prints one JSON result line per demo on stdout. Exit 1 when any demo failed (a
quarantined demo is not a failure); 2 when the path is missing or (directory) free
space is below 20 GB.

### `build-states`

Parsed match to `states/<map_version>/<yyyy-mm>/<match_key>.parquet` (P5.1 layout) with
its directory manifest; sets the match status to `states`.

```
python -m engine.cli build-states <match_key> | --all [--tier T] [--month yyyy-mm]
```

Tier and month come from the match's DB row; `--tier` / `--month` fill them when the
row lacks them. States include the boundary samples of every action (the extractor
runs too). `--all` processes every parsed match and exits 1 if any failed.

### `build-actions`

Same inputs as `build-states`; writes `actions/...` and `windows/...` parquet in the
P5.1 layout and sets the status to `actions`.

```
python -m engine.cli build-actions <match_key> | --all [--tier T] [--month yyyy-mm]
```

### `build-dataset`

Delegates to `engine.data.dataset.main` (P5.3). `--split` is accepted as the plan's
spelling of `--split-file`.

```
python -m engine.cli build-dataset --split data\engine\datasets\splits_v1.json [--sample-hz 2] [--assert-qa]
```

### `train-wp`

Delegates to `engine.model.train.main` (P6.3).

```
python -m engine.cli train-wp --vocab datasets\vocab_v1.json --norm datasets\norm_v1.json [--out DIR] [--cpu]
```

### `train-q`

Trains the Q and B networks (P9.2) by calling `engine.value.q_model.train_q` on a
valued-actions parquet. The descriptor vocabulary is fitted on the same frame; with
`--split-file`, rows are assigned to train / val by `match_key` (others ignored),
without it every row is train.

```
python -m engine.cli train-q --valued data\engine\eval\actions_valued_train.parquet [--split-file splits_v1.json] [--out DIR] [--cpu]
```

The parquet needs `h_pre` (embeddings from the adopted win model), `V`, `actor_side`,
`tier`, `type`, `match_key` and the `desc_*` columns of `engine.value.descriptors`.
Exit 2 when `--valued` is missing, the file is absent, `h_pre` is absent or unusable,
or a column is missing. Output: `data/engine/models/q_v<N+1>/` (or `--out`).

### `calibrate`

Delegates to `engine.model.calibrate.main` (P6.5).

```
python -m engine.cli calibrate --model-dir data\engine\models\wp_v1 [--assert]
```

### `build-neighbours`

Builds the FAISS neighbour index (P9.3) by calling `engine.value.neighbours.build_index`
and `save_index`. Keeps rows of `--tier` (default `pro`) and, with `--split-file`, only
train matches; `desc_ids` come from the Q model's `vocab.json`.

```
python -m engine.cli build-neighbours --valued actions_valued_train.parquet --q-model q_v1 [--split-file splits_v1.json] [--tier pro] [--out DIR]
```

The parquet needs `h_pre`, `match_key`, `round_num`, `action_id`, `descriptor_key`,
`V`, `actor_side`, `map_version`, `tier` and the `desc_*` columns. Exit 2 when an
input is absent (parquet, `h_pre`, `--q-model` directory or its `vocab.json`) or no
row is left after filtering. Output: `data/engine/models/neighbours_v<N+1>/` (or `--out`).

### `eval`

Runs one or more evaluations (P10.1.5); `--model` is passed on as `--model-dir`
(a bare name resolves under `data/engine/models`).

```
python -m engine.cli eval --model wp_v1 [--calibration] [--value] [--attribution] [--anchors] [--robustness] [--all]
```

| target | module | model flag passed |
|---|---|---|
| `--calibration` | `engine.model.calibrate` | `--model-dir` |
| `--value` | `engine.value.q_model` (no `main` yet: exit 2) | `--model-dir` |
| `--attribution` | `engine.attribution.evaluate` (not written yet: exit 2) | `--model-dir` |
| `--anchors` | `engine.evaluation.anchors` (P11.1) | `--model-dir` |
| `--robustness` | `engine.evaluation.robustness` (P11.3) | none |

Module options follow after `--`, e.g.
`eval --anchors --model wp_v1 -- --scored s.parquet --actions a.parquet`.

With several targets the exit code is the worst one. No target: exit 2.

### `sanity`

```
python -m engine.cli sanity --model wp_v1 [--counterfactual]
```

`--counterfactual` runs the P7.2 gate (`engine.counterfactual.sanity`); without it the
P6.6 model sanity (`engine.model.sanity`, not written yet: exit 2).

### `review`

Delegates to `engine.evaluation.review.main` (P11.2): picks the review rounds, writes
`eval/review_round_<i>/`, keeps `eval/review_log.md`, and lists the disagree items.

```
python -m engine.cli review --scored actions_scored.parquet [--actions a.parquet] [--states s.parquet] [--n 20] [--seed 1337] [--cycle 1]
python -m engine.cli review --parse data\engine\eval\review_log.md
```

### `bench`

Runs `bench/bench_<which>.py` (default `analyze`) with the remaining arguments.

```
python -m engine.cli bench [analyze|parse|states|actions|predict|attribution|value|gpu] [args]
python -m engine.cli bench analyze -- --synthetic --assert
```

`bench_analyze.py`: golden mode needs the pinned golden demo plus `--model-dir DIR` or
`--stub` (exit 2 otherwise); `--synthetic` times the `base` story with the rule stub;
`--assert` exits 1 when a measured stage is over 1.5x its budget or the total is over
the hard ceiling (n/a stages are not judged).

### `inventory`

P1.3 counts by tier x map x month x status, disk use, failures, quarantine reasons;
written to `data/engine/eval/inventory_<date>.md`.

```
python -m engine.cli inventory [--assert-targets] [--rebuild-catalog] [--models]
```

`--models` lists model directories and whether each is adopted. `--assert-targets`
exits 1 when a data target is missed.

### `status`

Engine DB counts (matches, processed ids, runs, action scores, by status) and free space.

```
python -m engine.cli status
```

### `scrape`

Discover, download and register demos from a source (`engine.ingest.run`).

```
python -m engine.cli scrape [--source api|playwright|hltv|hf] [--max N] [--tier T] [--min-level 10] [--min-elo 2500] [--region EU]
```

Refused below 20 GB free. Exit 1 when any match failed.

### `ingest`

Register one local demo through the seam (`engine.ingest.register_demo`): copy under
`data/engine/demos/<tier>/`, `matches` + `processed` rows.

```
python -m engine.cli ingest <demo.dem> [--match-id ID] [--tier pro|faceit_10|golden] [--source local]
```

---

## Model adoption (`docs/ENGINE_MODELS.md`)

`analyze` refuses a model that is not adopted (P12.1) unless `--allow-unadopted`. This
applies to the win model (`--model` / `--model-dir`) and to the value models
(`--q-model` and `--neighbours`) alike; with `--allow-unadopted` each unadopted one
adds a `model_unadopted` warning to the report. A model counts as adopted when the
adoption table of `docs/ENGINE_MODELS.md` has a row whose model cell is its directory
name (for example `wp_v3`) and whose status cell is `adopted`. Rows are added with
`python -m engine.evaluation.registry adopt ...`, never by hand; the table header is

```
| date | model | kind | log loss | Brier | ECE | baseline log loss | hazard AUC | decision rule satisfied | status |
```

The table is empty today, so no model is adopted.

## Report schema

`engine/report/schema.json` (JSON Schema draft 2020-12). Top level: `report_schema_version`
(1), `meta`, `rounds[]` (`wp_curve`, `windows`, `actions` with `V, V_direct, V_enabler,
V_movement, Q, B, D, R, X`, verdicts, enablers with shares, panel top 20, method,
residual, flags; `clock_ct`, `clock_t`, `conservation_residual`), `players[]` (per-type
V sums, shrunk D and X with 90% intervals and n), `teams[]` (by side: clock total,
V total, buys), `warnings[]` (`code`, `message`, optional round / action). Unknown
numbers are `null`, never 0.
