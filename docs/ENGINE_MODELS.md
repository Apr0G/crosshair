# Engine models: registry and adoption

Plan reference: P12.1, P6.7.2. Code: `engine/evaluation/registry.py`.

Every trained model lives in `data/engine/models/<kind>_v<N>/` with a `manifest.json`
(what produced it: config hash, git sha, inputs, package versions). Existing on disk
does **not** mean usable. A model is **adopted** only when it has a row with status
`adopted` in the table below. `engine analyze` refuses a model without that row unless
`--allow-unadopted` is passed (and then the report carries a `model_unadopted` warning).

## Rules for this file

- Add rows only with the registry, never by hand:
  `.venv\Scripts\python.exe -m engine.evaluation.registry adopt <kind>_v<N> --logloss ... --brier ... --ece ... --baseline-logloss ... [--hazard-auc ...] --rule "..."`.
  It refuses a model with no manifest, a second row for the same model, a `wp` model
  whose numbers fail P6.7.2, a `wp` model without `calibration.json` (R6) and a `wp`
  model without a green `sanity_cf.json` from `engine sanity --counterfactual` (P7).
- One row per model. To take a model out of use, change the **status** cell of its row
  in place to `retired`, `superseded`, `revoked` or `rejected`. Do not add a second row.
- Only the **model cell** of a row with status `adopted` adopts a model.
  `engine.analyze` reads the table through the registry (`adoption_rows`), so a
  version named in prose or in another row's rule text adopts nothing; the registry
  still refuses rule text that names another model version, to keep rows unambiguous.
- All numbers are on the **test months of the baseline tier `faceit_10`** of the split
  file the model was trained with (never val). Since Amendment 2 (2026-10-19) the
  baseline is FACEIT level 10, not pro; pro numbers are a transfer test and go in the
  rule cell as `transfer: logloss=... ece=...` until the registry gains a column. Every
  "pro" in the decision rules below reads "`faceit_10`".
- `docs/ENGINE_PROGRESS.md` is the place for narrative; this file is the record.

## Decision rules (P6.7.2, written before any result so they are not negotiated later)

- **Adopt the neural win model** only if the pro+FACEIT model (c) or the pro-only
  model (b) beats the LightGBM baseline (a) on pro test log loss by **≥ 0.01** and has
  **ECE ≤ 0.02** (`calib.ece_target`). Otherwise stop, write it up, and go back to
  features (P3) before any attribution work. The registry enforces this for `wp` rows.
- **Keep FACEIT in training** only if (c) beats (b) on pro test log loss; otherwise
  train pro-only.
- **Keep the temporal encoder** only if (c) beats (d, no temporal encoder) by
  **≥ 0.003** log loss or improves the martingale drift check (P6.6.2). Same rule for
  pairwise bias against (e).
- **Hazard head** (needed by attribution regardless): AUC on test pro **≥ 0.80** is the
  target (`sanity.hazard_auc_target`). Between 0.75 and 0.80 the hazard enabler mode
  is used and every report warns "hazard head below target". Below 0.75
  (`sanity.hazard_auc_floor`) enabler splits fall back to the fixed shares
  (`attribution.fixed_shares`, P8.5.3) and the report says so.
- **4 Hz sampling** only if the 4 Hz run (g) reduces movement-credit variance by
  **≥ 20%** at **≤ 1.6×** the state-building time. `sample_hz` is never changed to
  save time (P10.3.2).
- **Ensemble** `W = α·NN + (1−α)·LGBM` only if test log loss improves by **≥ 0.002**.
- **Counterfactual gate (P7):** nothing is attributed with a model until
  `engine sanity --counterfactual` passes on it. At most five failure-loop iterations
  (P7.3) before the numbers go to Faig.
- **Q / B value models** have no numeric rule in the plan; their row records the
  identity check (C2) and panel coverage in the rule cell.

## Adoption table

| date | model | kind | log loss | Brier | ECE | baseline log loss | hazard AUC | decision rule satisfied | status |
|---|---|---|---:|---:|---:|---:|---:|---|---|

No models yet (2026-10-19): no training data exists, so nothing has been trained or
adopted.
