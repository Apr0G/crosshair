# Engine operations

Plan reference: P12.2 (retraining), P12.3 (monitoring), P12.5 (buyer hand-off), with
the P11 validation steps they depend on. Command reference: `docs/ENGINE_CLI.md`.
Model adoption: `docs/ENGINE_MODELS.md`. Data sources: `docs/DATA_SOURCES.md`.

Every command below runs from the repo root with the project venv:
`.venv\Scripts\python.exe` (Python 3.12). Written against `engine/cli.py` as it
stood on 2026-10-19; **"missing"** marks a step whose command does not exist yet (the
list at the end says what is needed).

## Status today

No demo, no dataset and no trained model exist yet, so none of the checklists
below has been run. They are the procedure for when data arrives. Since Amendment 2
(2026-10-19) the baseline tier is `faceit_10`; every "pro" below reads `faceit_10`
unless it says "transfer". **Weekly, not monthly:** FACEIT keeps demos about 30 days
and the free Downloads tier is 1,500 requests or 200 GB per month, so step 2's
`scrape` runs at least weekly with `--max` sized to the remaining monthly quota
(`inventory` prints it once the quota ledger exists; until then count by hand).

## Monthly retraining checklist (P12.2)

1. **Take stock.** `.venv\Scripts\python.exe -m engine.cli status` and
   `.venv\Scripts\python.exe -m engine.cli inventory --assert-targets` (shortfalls are
   reminders, not a gate). Check free space: batch work is refused below 20 GB.
2. **New demos.** Download (`... -m engine.cli scrape --source api --min-level 10`
   for FACEIT; optional pro transfer set per `docs/DATA_SOURCES.md`), then
   `... -m engine.cli parse data\engine\demos\pro` (and `...\faceit`),
   `... -m engine.cli build-states --all`, `... -m engine.cli build-actions --all`,
   `... -m engine.cli inventory --rebuild-catalog`.
3. **New split file `splits_v<N+1>.json`** rolling the test months forward (never edit
   an old one). **Missing as a command**: from Python,
   `from engine import db; from engine.data import splits; splits.make_splits(db.list_matches())`.
4. **Datasets.** `... -m engine.cli build-dataset --split data\engine\datasets\splits_v<N+1>.json --assert-qa`.
5. **Baseline yardstick.** `... -m engine.model.baseline_lgbm` (no CLI subcommand;
   writes `models/baseline_v<N>`).
6. **Win model.** `... -m engine.cli train-wp -- --vocab data\engine\datasets\vocab_v<N>.json --norm data\engine\datasets\norm_v<N>.json`
   (ablations: `--no-temporal`, `--no-pairwise`, `--no-aux-heads`, `--tiers pro`).
7. **Calibrate.** `... -m engine.cli calibrate -- --model-dir data\engine\models\wp_v<N> --assert`
   (also reachable as `eval --calibration --model wp_v<N>`).
8. **Sanity gates.** `... -m engine.cli sanity --model wp_v<N> --counterfactual [-- --states <root> --max-matches K]`
   (P7.2; samples events, order windows and drift intervals from every stored match
   under `data/engine/{states,actions,windows}` with the map's real geometry; exit 0
   green, 1 a failed rate, 2 missing model / calibration / states / map assets). It
   writes `data\engine\eval\sanity_cf_wp_v<N>.md` and `models/wp_v<N>/sanity_cf.json`; `registry
   adopt` refuses a wp model without a green record. The model-level gate
   (`sanity --model wp_v<N>`, P6.6) is **missing** (`engine/model/sanity.py` does not exist yet). On a failure
   follow the P7.3 loop: one remedy per iteration, retrain, rerun P6.5–P6.6 and P7.2,
   log it in `data\engine\eval\DECISION_wp_v<N>.md`, at most five iterations, then escalate to Faig.
9. **Value models.** From the valued actions (with `h_pre` from the adopted win model):
   `... -m engine.cli train-q --valued <actions_valued.parquet> --split-file data\engine\datasets\splits_v<N+1>.json [--wp-model wp_v<N>]`,
   then `... -m engine.cli build-neighbours --valued <same> --q-model q_v<N> --split-file <same>`.
   Both exit 2 while an input is missing. `analyze --q-model / --neighbours` refuses
   them until they are adopted like the win model (step 11).
10. **Evaluate.** `... -m engine.cli eval --model wp_v<N> --calibration`; attribution
    eval (`--attribution`) is **missing**; anchors: see P11 below.
11. **Decide.** Apply the P6.7.2 rules in `docs/ENGINE_MODELS.md`. If they pass:
    `... -m engine.evaluation.registry adopt wp_v<N> --logloss ... --brier ... --ece ... --baseline-logloss ... --hazard-auc ... --rule "..."`.
    The registry refuses numbers that fail the rule. Retire the old row by editing its
    status cell in place.
12. **Bench.** `... -m engine.cli bench analyze` on the golden demo; paste the result
    into `data\engine\eval\bench_<date>.md`. Over budget: turn the knobs in the P10.3.2 order
    (`los_cache`, `drift_threshold_wp`, `shapley_samples`, `k_neighbours`, chart
    resolution). Never `sample_hz`.
13. **Record** what changed in `docs/ENGINE_PROGRESS.md` (Faig maintains it).

## Map rework or removal from the active pool (P12.2)

1. Add the new version to `engine/maps/versions.yaml` (map, version label, first game
   build), so `engine.maps.version.detect` tags new demos `<map>@<version>`.
2. Rebuild map assets and callouts for the new version (`engine/maps/assets.py`,
   `engine/maps/callouts.py`; callouts per the P2.3 decision Faig still has to make).
   No CLI command does this yet.
3. Matches of the old layout keep their old `map_version`; re-run `build-states` and
   `build-actions` only for the new-version matches.
4. Retrain (monthly checklist from step 3). Until the version has 100 matches the
   engine is less reliable there: check with
   `... -m engine.evaluation.robustness --map-version <map>@<version>`. `engine
   analyze` itself warns `map_version_low_support` below 100 registered matches
   (P11.3.1, same count: non-quarantined `matches` rows of that version).
5. A map leaving the pool keeps its data; it just stops growing.

## Game patch that changes mechanics (P12.2)

1. Note the patch date and game build.
2. Update the affected constants in `engine/config.yaml` (round and bomb timers, smoke
   duration, economy bands) with a comment naming the patch date. Every later
   artifact gets a new `config_hash`.
3. Snapshots before the patch must keep the old constants through `game_version`;
   the per-build switch is not implemented yet (the config holds one value per key).
   Until it is, a mechanics patch means: keep the old config for old data, and treat
   post-patch data as a new dataset.
4. Watch patch drift: newest-month calibration through
   `... -m engine.evaluation.robustness --reliability <newest_month.json>`; ECE above
   0.03 warns "model predates recent patches" (P11.3.2). Retrain when it does.

## Monitoring (P12.3)

- Every successful `engine analyze` run that writes to the DB (not `--no-db`) appends
  one line to `data\engine\eval\usage.jsonl` (run_id, match_key, map_version, tier,
  device, timings, warnings count, max conservation residual) through
  `engine.evaluation.usage.append_usage` (`analyze.append_usage_record`); with
  `CROSSHAIR_ENGINE_DB` set the file sits next to that DB instead. A failed append is
  logged, never raised.
- Weekly: `... -m engine.evaluation.usage` prints runs by tier and map version, every
  run with |residual| > 1e-3, and every run over the time ceiling (120 s GPU,
  300 s CPU from `bench.hard_ceiling_*`). Investigate any listed run before the next
  release; a residual > 1e-3 breaks condition C1.

## Validation and review loop (P11)

- **Anchors (P11.1):**
  `... -m engine.evaluation.anchors --model-dir data\engine\models\wp_v<N> --scored <actions_scored.parquet> --actions <actions.parquet> --states <states.parquet>`
  writes `data\engine\eval\anchors_wp_v<N>.md`. Bands are informational; out-of-band
  results go into the review. Player-level correlation needs
  `data\engine\eval\hltv_ratings.csv` (gitignored; copy the header-only template
  `test\engine\fixtures\hltv_ratings.csv`, never fill the template itself: SteamIDs
  and nicknames must not enter the public repo) filled by hand for ≥ 50 test matches
  (columns `match_key,map,steamid64,player_name,hltv_rating`, `match_key` as the
  engine names the match). Man advantage needs a `W` column on the states.
  `... -m engine.cli eval --anchors --model wp_v<N> -- --scored <...> --actions <...> --states <...>`
  runs the same module (`engine.evaluation.anchors`, `--model` passed as `--model-dir`).
- **Review cycle (P11.2):**
  `... -m engine.evaluation.review --scored <...> --actions <...> --states <...> --cycle 1`
  writes `data\engine\eval\review_round_<i>\case.json` for 20 rounds covering the ten
  categories, and `review_log.md`. Run `engine analyze` on each case's match and put
  that round's `report.md` and charts in its folder. Faig fills the log; then
  `... -m engine.evaluation.review --parse data\engine\eval\review_log.md` lists every
  disagreement as a numbered item and says whether the cycle met the target (≤ 2).
  Each fixed item gets one function in `test\engine\test_review_items.py`. Repeat with
  new rounds (a new `--out-root`) until the target is met, minimum two cycles.
- **Robustness (P11.3):** out-of-map gap after training once with one map held out
  (`splits.make_splits(..., holdout_maps=("de_<map>",))`), then
  `... -m engine.evaluation.robustness --metrics-default a.json --metrics-holdout b.json`.
  FACEIT-as-pro warning and 64-tick refusal: `test\engine\test_cli_guards.py`.

## Buyer hand-off checklist (P12.5)

Everything produced by commands, nothing hand-edited:

- [ ] golden demo report (`engine analyze` on the golden demo, adopted models only)
- [ ] five `faceit_10` match reports, one `semi_pro`, one `pro` (transfer) report
- [ ] data-provenance statement (sources, terms read, dates) and the FACEIT Downloads approval
- [ ] anchors report (`anchors_wp_v<N>.md`)
- [ ] calibration report (`eval --calibration`)
- [ ] ablation table (`data\engine\eval\ablations_v<N>.md`, P6.7.1)
- [ ] counterfactual sanity-gate report (`sanity --counterfactual`)
- [ ] review log with at least two cycles and the regression tests
- [ ] bench table (`data\engine\eval\bench_<date>.md`) within the C3 ceiling
- [ ] one-page method summary: the plain-language explanation (plan Appendix I) plus
      the formula appendix (plan Appendix B)
- [ ] `docs/ENGINE_MODELS.md` naming every model the reports used

## Commands missing on 2026-10-19

| needed for | missing | where it would live |
|---|---|---|
| new split file | a CLI command around `splits.make_splits` | `engine/cli.py` |
| HLTV downloads | dormant since Amendment 2 (terms forbid); not to be built | — |
| FACEIT competition discovery | organizer → championships → past matches (Amendment 2 P1.2.2) | `engine/sources/faceit_api.py` |
| monthly quota ledger | requests + bytes per month, stop at the cap (Amendment 2 P1.2.3) | `engine/db.py` (ask-before), `engine/ingest.py` |
| pro transfer report | `eval --transfer` (Amendment 2 P11.4) | `engine/cli.py` |
| model sanity gate | `sanity --model` | `engine/model/sanity.py` |
| attribution eval | `eval --attribution` | `engine/attribution/evaluate.py` |
| map assets / callouts rebuild | no command | `engine/maps/` |
| per-patch constants | config keyed by game build | `engine/config.py` |
