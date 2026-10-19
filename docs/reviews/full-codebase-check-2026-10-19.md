# Full codebase check — Crosshair

**Date:** 2026-10-19
**Repo:** github.com/Apr0G/crosshair (public)
**Commit:** `9f7b2bb` on `main`
**Working tree:** clean (no untracked, no modified tracked files); 7 source files under `engine/data/` are on disk but ignored by git (see CRIT-01)
**Scope:** 252 files in scope · 251 read in full · 0 unread · 1 deliberately excluded (the previous audit report)
**Findings:** 1 CRITICAL · 6 HIGH · 43 MEDIUM · 79 LOW

## Summary

The engine v2 tree is a one-night rewrite that has never seen a real demo, a trained model or a network call, and the audit should be read in that light: the modules exist and are internally tested on synthetic data, but several of the gates the plan relies on are either not wired or wired in a way that cannot pass. It is a prototype with sharp edges, not something to hand to a buyer.

The single most urgent thing is CRIT-01: the commit pushed to the public repository on 2026-10-19 is missing the whole `engine/data` package, because the `.gitignore` rule `data/` also matches `engine/data/`. A fresh clone cannot build states, datasets or run ten of the test scripts. Nothing works outside this PC until the rule is anchored and the package is committed.

Below that, the HIGHs cluster into three habits rather than six accidents. First, gates that exist on paper but not in the pipeline: the quarantine rules run only in `analyze`, the P7 sanity gate always exits 2, adoption is a token scan that reads prose, and value models are never checked at all. Second, the counterfactual editor disagrees with the state builder about what a state is (team money, eye height, smoke rays), so a zero-effect action will pick up a nonzero Shapley value on the real model and condition C1 cannot pass; the test fixtures copy the editor's conventions and therefore cannot see it. Third, unknowns emitted as legitimate values (`on_fire = 0` for every player on every real demo, because awpy gives no fire extents), which the tests hide by injecting values production never produces.

Two acceptance claims in the progress log are weaker than they read. The C2 identity V = B + D + X is true by definition (X is the residual) and the "both variants" of C1 are both stubs. The one habit that shows up in a dozen places is a test fixture that is more convenient than the producer's real output: extent radius set, decoy expiry set, spotted-by filled, money summed alive-only, players_ct keys the parser never emits. Each instance is MEDIUM; together they mean the 89-green suite proves less than it looks.

No secret, signed URL, session cookie or player identifier was found in the tree or in git history, and every tracked command in CLAUDE.md parses. The redaction and dedup controls hold for the cases they were written for; the gaps found are edge cases (a path with a space, a non-.dem signed URL, a second series between the same teams on one day).

## Findings at a glance

| ID | Sev | Title | File | Rule |
|---|---|---|---|---|
| CRIT-01 | CRITICAL | The pushed commit is missing the engine/data package: the `data/` gitignore rule also matches `engine/data/` | `.gitignore:25` | — |
| HIGH-01 | HIGH | The quarantine gate never runs on the batch path: parse, build-states and build-dataset accept 64-tick, short and non-5v5 demos | `engine/parse/worker.py:113` | R10 |
| HIGH-02 | HIGH | Model adoption is read by scanning for `<kind>_vN` tokens, so a model named in another model's rule text counts as adopted | `engine/analyze.py:289` | R19 |
| HIGH-03 | HIGH | The Q/B value model and the neighbour index are loaded without any adoption check | `engine/analyze.py:181` | R19 |
| HIGH-04 | HIGH | On real demos every player's on_fire is 0 and the fire_pressure enabler never fires: the fire extent is always null and unknown is emitted as 0 | `engine/state/players.py:282` | R7 |
| HIGH-05 | HIGH | Counterfactual edits rewrite team money as an alive-only sum while the builder sums every account, so a zero-effect action gets a nonzero Shapley value (C1 broken) | `engine/counterfactual/interventions.py:304` | R21 |
| HIGH-06 | HIGH | Counterfactual line-of-sight and smoke checks are cast from feet positions while the builder uses eye points | `engine/counterfactual/interventions.py:210` | — |
| MED-01 | MED | Two quarantine rule sets disagree, and the stored-corpus one reads keys the parser never emits | `engine/data/quarantine.py:50` | — |
| MED-02 | MED | A model directory is accepted as adopted by basename alone, from any location | `engine/analyze.py:305` | R19 |
| MED-03 | MED | Test hooks (CROSSHAIR_STUB_PREDICTOR, --raw-tables-story) skip the adoption gate and write to the real engine.db and reports dir by default | `engine/cli.py:258` | R19 |
| MED-04 | MED | `or` used as a numeric null-coalesce on the 128-tick gate and in the parser's quarantine check | `engine/analyze.py:220` | R8 |
| MED-05 | MED | scrape exits 1 instead of 2 when FACEIT_API_KEY is missing, and a YAML syntax error in the config also exits 1 | `engine/cli.py:608` | R22 |
| MED-06 | MED | Re-registering an existing match_id with a different demo reports success and leaves a file the DB does not describe | `engine/db.py:362` | R1 |
| MED-07 | MED | The analyze output folder's manifest.json records the absolute demo path and the hostname | `engine/manifest.py:111` | R14 |
| MED-08 | MED | Decoys never expire in the state and start when they end: expire_tick is always null and detonate_tick is the last trajectory tick | `engine/state/builder.py:147` | — |
| MED-09 | MED | kills_round counts self-kills and kills with unknown team_kill, unlike the parser's ticks.kills_round | `engine/state/players.py:234` | — |
| MED-10 | MED | Knife-skin display names map to weapon class 'unknown', not 'knife' | `engine/state/players.py:126` | — |
| MED-11 | MED | A Rust panic or a dead worker aborts the whole parse batch before any failure is logged | `engine/parse/worker.py:118` | — |
| MED-12 | MED | Grenade trajectory segments split on an 8-tick gap keep the same entity_id, producing spurious mid-flight flash/HE/decoy actions and mislinked clouds | `engine/parse/awpy3_parser.py:618` | — |
| MED-13 | MED | A null snapshot health becomes health 0 / dead / equipment 0 / empty inventory | `engine/parse/awpy3_parser.py:477` | R7 |
| MED-14 | MED | Utility parent ids are re-salted after their pop/fade pseudo-actions copied the old id, so a second same-tick smoke's credit goes to the first | `engine/actions/utility.py:281` | — |
| MED-15 | MED | A drop/give is valued only at the drop tick; the teammate's pickup jump has no owner | `engine/actions/windows.py:59` | — |
| MED-16 | MED | The P7 sanity gate cannot go green from the CLI and certifies a different intervention path than analyze runs | `engine/counterfactual/sanity.py:395` | R19 |
| MED-17 | MED | The Q model, neighbour index and WP encoder record no link to each other, so a mismatched pair silently gives wrong alternative Q and regret | `engine/value/neighbours.py:195` | — |
| MED-18 | MED | train-wp and calibrate never check that --vocab/--norm match the shards, so a wrong vocab silently scrambles every categorical | `engine/model/train.py:510` | — |
| MED-19 | MED | An empty or pro-less val/test split crashes with KeyError after a full first epoch, and calibrate overwrites calibration.json before crashing | `engine/model/train.py:145` | R22 |
| MED-20 | MED | The Predictor silently serves uncalibrated probabilities when calibration.json is missing, and adoption does not require it | `engine/model/checkpoints.py:101` | R6 |
| MED-21 | MED | read_shards loads an entire split into RAM, which cannot fit at the configured corpus targets | `engine/data/dataset.py:571` | — |
| MED-22 | MED | Path redaction in reports stops at the first space, and find_leaks shares the blind spot | `engine/report/writer.py:41` | R14 |
| MED-23 | MED | find_leaks puts the raw matched string, cut to 80 characters, into its messages | `engine/report/writer.py:119` | R9 |
| MED-24 | MED | The exhausted-429 error from the FACEIT Data API is swallowed, so a throttled scrape reports 'no new demos' and exits 0 | `engine/sources/scraper.py:237` | — |
| MED-25 | MED | zstd decompression stops after the first frame, so a multi-frame .dem.zst is silently truncated and marked processed | `engine/sources/scraper.py:468` | — |
| MED-26 | MED | HLTV match ids are built from event, date, teams and map only, so two series between the same teams on one day collide and the later one is silently skipped | `engine/sources/hltv.py:170` | R1 |
| MED-27 | MED | HF free-space check runs once per wave on compressed sizes, and a re-run over compressed entries aborts | `engine/sources/hf_archive.py:234` | — |
| MED-28 | MED | The value stage is never timed by the golden benchmark, so the C3 total always omits it | `bench/bench_analyze.py:85` | — |
| MED-29 | MED | A second human-review cycle with the same seed re-selects the same rounds | `engine/evaluation/review.py:389` | — |
| MED-30 | MED | Grid cell ids are not pinned: the per-map grid test holds by construction | `test/engine/test_grid.py:45` | — |
| MED-31 | MED | The leakage flip audit never shifts end_tick, the source of the historical time_remaining leak | `engine/state/leakage.py:40` | R5 |
| MED-32 | MED | test_states_golden and test_actions_golden treat parse()'s tuple as RawTables and swallow the error as 'prerequisite missing' | `test/engine/test_states_golden.py:66` | R22 |
| MED-33 | MED | C2's identity V = B + D + X is definitional and every test of it is vacuous | `engine/value/scores.py:159` | R21 |
| MED-34 | MED | Nothing produces shrinkage.json, so real reports never carry shrunk per-player D/X | `engine/value/shrinkage.py:109` | — |
| MED-35 | MED | Knife rounds cannot be excluded: awpy 3.1 has no is_knife_round column, so the knife plumbing is dead and any knife round in the rounds table is sampled, labelled and scored | `engine/parse/awpy3_parser.py:345` | — |
| MED-36 | MED | The anchors report falls back to the raw SteamID64 when a ratings row has no player_name | `engine/evaluation/anchors.py:490` | R14 |
| MED-37 | MED | hltv_ratings.csv is a tracked fixture that the docs tell Faig to fill with real SteamIDs and nicknames | `test/engine/fixtures/hltv_ratings.csv:1` | — |
| MED-38 | MED | train-q and build-neighbours accept input with no split and no warning, so Q/B and the panel can learn from test months silently | `engine/cli.py:463` | R4 |
| MED-39 | MED | P6.7.2's hazard-head rule is not implemented: no 'below target' warning, no automatic fixed-share fallback, hazard AUC optional at adoption | `engine/evaluation/registry.py:101` | R19 |
| MED-40 | MED | ENGINE_PROGRESS marks P6, P8 and P9 done while three gated commands and the valued-actions producer do not exist | `docs/ENGINE_PROGRESS.md:11` | — |
| MED-41 | MED | The C1 bound is stated two ways and the real-model variant does not exist | `docs/ENGINE_PLAN.md:58` | R21 |
| MED-42 | MED | The quarantine rule '>= 16 rounds' refuses legitimate MR12 maps that end 13-0, 13-1 or 13-2 | `engine/parse/awpy3_parser.py:125` | — |
| MED-43 | MED | B.8 shrinks D toward 0 although D's per-class mean is not 0, and the open question is not in Appendix H | `engine/value/shrinkage.py:19` | — |
| LOW-01 | LOW | A Downloads-API 403 (key without Downloads scope) is caught per match, so the run keeps POSTing for every remaining match | `engine/ingest.py:246` | — |
| LOW-02 | LOW | redact_text is narrower than the legacy blanket '?query' strip: scheme-less non-.dem paths and upper-case schemes keep their query | `engine/log.py:33` | — |
| LOW-03 | LOW | --debug prints the traceback without redaction | `engine/cli.py:701` | — |
| LOW-04 | LOW | Exit code 130 on Ctrl-C and raw bench/delegate return codes contradict the 0/1/2 contract (CLAUDE.md:'Exit codes everywhere') | `engine/cli.py:690` | — |
| LOW-05 | LOW | Generic integer validation rejects 0 for qa.max_outliers / train.seed; a 0 s martingale horizon is accepted (vacuous check); test_qa asserts max_outliers=0 via dataclasses.replace | `engine/config.py:374` | — |
| LOW-06 | LOW | Duplicate YAML keys are silently last-wins | `engine/config.py:442` | — |
| LOW-07 | LOW | build-states/build-actions --all drop the CLIError message and only count failures | `engine/cli.py:390` | — |
| LOW-08 | LOW | The error handler imports engine.analyze and can raise again, printing a raw chained traceback | `engine/cli.py:692` | — |
| LOW-09 | LOW | min_level issues a second GET /matches/{id} per candidate and re-queries rejected matches every run | `engine/sources/faceit_api.py:136` | — |
| LOW-10 | LOW | HLTV rows store a temp file:// URI (with the Windows user name) as processed.demo_url; the real HLTV URL is lost | `engine/sources/hltv.py:282` | — |
| LOW-11 | LOW | hf match_id_for is not injective ('a/b' vs 'a-b', 'x.dem' vs 'x.dem.zst'); the second file is silently skipped | `engine/sources/hf_archive.py:215` | — |
| LOW-12 | LOW | faceit_10 demos live in demos/faceit while CLAUDE.md, DATA_SOURCES.md and ENGINE_CLI.md say demos/<tier> | `engine/paths.py:42` | — |
| LOW-13 | LOW | Persistent context user_data_dir is the real Chrome Default profile (legacy, unchanged, macOS paths) | `engine/sources/scraper.py:154` | — |
| LOW-14 | LOW | _scores skips the win increment for a round with no roster, so later score_before values are confidently off by one | `engine/parse/awpy3_parser.py:863` | — |
| LOW-15 | LOW | Blind-to-flash linking picks the thrower's latest throw before the blind, so a quick double flash mislinks (needs-domain-review) | `engine/parse/awpy3_parser.py:644` | — |
| LOW-16 | LOW | _SEGMENT_GAP (8 ticks) and the defuse-complete tolerance (4 ticks) are tick literals rather than seconds | `engine/parse/awpy3_parser.py:135` | — |
| LOW-17 | LOW | int(round(raw)) sits outside the try in _tick_rate_from_header, so a NaN playback_time raises instead of returning None | `engine/parse/awpy3_parser.py:227` | — |
| LOW-18 | LOW | Demo-header map name is joined into asset file paths unvalidated on the worker path (read-only, fixed suffixes) | `engine/maps/assets.py:67` | — |
| LOW-19 | LOW | engine/maps/plot.py has no test script and its __main__ exits 1 with a traceback on missing assets | `engine/maps/plot.py:62` | — |
| LOW-20 | LOW | MIN_ROUNDS 16 refuses legitimate MR12 13-0..13-2 results (plan P1.4.1 as written; Faig's decision) | `engine/parse/awpy3_parser.py:125` | — |
| LOW-21 | LOW | Score fallback assumes a side swap at the start of overtime (needs-domain-review; only when parser scores are null) | `engine/state/context.py:97` | — |
| LOW-22 | LOW | MapGeometry.is_visible / find_area swallow every exception, so an awpy API mismatch silently yields all-NaN LOS | `engine/state/geometry.py:108` | — |
| LOW-23 | LOW | decoy_end pseudo-action is emitted at a nominal end while the state keeps the token alive (covered by the decoy MEDIUM) | `engine/actions/utility.py:240` | — |
| LOW-24 | LOW | Window bounds hard-code ±1 instead of sampling.boundary_offset_ticks (config allows >= 1) | `engine/actions/windows.py:78` | — |
| LOW-25 | LOW | Loadout is read at freeze_end, so buys during remaining buy time are missed (plan P4.4.1 as written; needs-domain-review) | `engine/actions/economy.py:79` | — |
| LOW-26 | LOW | remove_utility's decrement of the capped active-utility count is overwritten by recompute_derived (>16 tokens only) | `engine/counterfactual/interventions.py:517` | — |
| LOW-27 | LOW | enablers.split divides by q_pre with no zero guard; config allows attribution.min_hazard = 0 | `engine/attribution/enablers.py:154` | — |
| LOW-28 | LOW | In a {damage by A, kill by B} window apply(empty) leaves A's damage_round inflated (undo order) | `engine/counterfactual/interventions.py:356` | — |
| LOW-29 | LOW | Calibration group keys differ between fit (vocab inverse '<unk>'/'<null>') and Predictor (raw string / 'None') in group mode | `engine/model/predict.py:181` | — |
| LOW-30 | LOW | LightGBM yardstick early-stops on all-tier val while the NN uses val pro, skewing the P6.7.2 comparison | `engine/model/baseline_lgbm.py:143` | — |
| LOW-31 | LOW | Baseline team sums turn an alive player's unknown equipment/helmet/defuser into 0 (R7, yardstick only) | `engine/model/baseline_lgbm.py:86` | — |
| LOW-32 | LOW | baseline_lgbm and neighbours.save_index import torch-bearing modules only for next_version_dir | `engine/model/baseline_lgbm.py:224` | — |
| LOW-33 | LOW | dataset.build writes vocab_vN before the shard existence check and saves norm_vN without exclusive create | `engine/data/dataset.py:637` | — |
| LOW-34 | LOW | write_dir_manifest re-hashes every parquet in the month directory on each single-match write (quadratic) | `engine/data/layout.py:102` | — |
| LOW-35 | LOW | splits.assign duplicate check is O(n^2); make_splits has no CLI entry point | `engine/data/splits.py:54` | — |
| LOW-36 | LOW | train keeps every step's loss in memory for the whole run | `engine/model/train.py:297` | — |
| LOW-37 | LOW | The markdown player table does not escape '|' in names (--show-ids runs; needs-domain-review) | `engine/report/writer.py:228` | — |
| LOW-38 | LOW | GPU bench scripts end with sys.exit instead of the documented HIP-teardown workaround; result files are date-only and overwrite same-day runs | `bench/bench_gpu.py:111` | — |
| LOW-39 | LOW | Opening-kill anchor counts kills with unknown alive counts as 5v5 | `engine/evaluation/anchors.py:284` | — |
| LOW-40 | LOW | Adoption rewrites docs/ENGINE_MODELS.md in place without an atomic write | `engine/evaluation/registry.py:157` | — |
| LOW-41 | LOW | usage.py docstring and ENGINE_OPS.md say analyze does not call append_usage; analyze.py does | `engine/evaluation/usage.py:20` | — |
| LOW-42 | LOW | run_all does not set CROSSHAIR_ENGINE_DB for child tests and forwards a relative demo path unresolved; SKIP column shows the tally line | `test/run_all.py:63` | — |
| LOW-43 | LOW | Repo-root eval/ is not gitignored but sanity's default --out and ENGINE_OPS paths write there | `engine/counterfactual/sanity.py:384` | — |
| LOW-44 | LOW | scripts/scrape_watchdog.sh is dead: it greps for and relaunches the deleted src/main.py with flags engine.cli lacks | `scripts/scrape_watchdog.sh:27` | — |
| LOW-45 | LOW | README Setup / Getting started / Full CLI / layout describe the deleted src/ pipeline; 'MIT' with no LICENSE file | `README.md:113` | — |
| LOW-46 | LOW | pyproject package-data lists only config.yaml, so a non-editable install ships no schema.json / vocab_weapons.yaml / versions.yaml / callouts.json; pyflakes is a runtime dep; the console script is not installed | `pyproject.toml:26` | — |
| LOW-47 | LOW | requirements-engine.lock pins Windows+AMD cp312 torch/rocm URLs, so it is not installable elsewhere; CLAUDE.md says torch is not in the requirements | `requirements-engine.lock:59` | — |
| LOW-48 | LOW | .env.example asks for FACEIT_EMAIL / FACEIT_PASSWORD that nothing reads | `.env.example:7` | — |
| LOW-49 | LOW | CLAUDE.md and DATA_SOURCES.md promise is_processed_any checks _m1, _m2, ...; db.id_forms checks only _m1 for a bare id, by design | `CLAUDE.md:190` | — |
| LOW-50 | LOW | ENGINE_OPS.md 'missing' table and notes contradict the code: anchors path is right, train-q/build-neighbours exist, append_usage is wired | `docs/ENGINE_OPS.md:150` | — |
| LOW-51 | LOW | ENGINE_PROGRESS.md gives two commands that do not parse (scrape --engine-copy; analyze without the demo positional) | `docs/ENGINE_PROGRESS.md:19` | — |
| LOW-52 | LOW | ENGINE_MODELS.md says only a status 'adopted' row counts; ENGINE_CLI.md says the file does not exist and suggests a 4-column table | `docs/ENGINE_CLI.md:337` | — |
| LOW-53 | LOW | ENGINE_PROGRESS.md stale facts: 'Nothing is committed', '16 commands'; CLAUDE.md links a renamed section | `docs/ENGINE_PROGRESS.md:13` | — |
| LOW-54 | LOW | DATA_SOURCES.md lists HF with tier '—' but ingest maps hf -> pro; scrape --source hf cannot pass a repo and exits 1 | `docs/DATA_SOURCES.md:19` | — |
| LOW-55 | LOW | Audit skill files still describe the legacy src/ layout and do not block data/engine/engine.db, data/engine/demos or .venv-old311 | `.claude/skills/full-codebase-check/SKILL.md:41` | — |
| LOW-56 | LOW | test_registry asserts the real ENGINE_MODELS.md has no adoption rows (fails the first legitimate adoption) and has a literal check(True) | `test/engine/test_registry.py:53` | — |
| LOW-57 | LOW | test_review_items placeholder only checks that a function is callable | `test/engine/test_review_items.py:23` | — |
| LOW-58 | LOW | test_report_redaction prints raw leaked strings in failure details | `test/engine/test_report_redaction.py:53` | — |
| LOW-59 | LOW | test_anchors never has a blank player_name, so the raw-SteamID fallback is untested | `test/engine/test_anchors.py:178` | — |
| LOW-60 | LOW | test_decision_split hard-codes V=0.02 against delta_X read from config | `test/engine/test_decision_split.py:44` | — |
| LOW-61 | LOW | Torch/faiss-importing test scripts crash with ImportError (ERROR) instead of exiting 2 when the platform torch build is absent (10 scripts) | `test/engine/test_augment.py:11` | — |
| LOW-62 | LOW | test_cli_errors refusal cases omit --out, so a guard regression would write data/engine/reports/syn | `test/engine/test_cli_errors.py:62` | — |
| LOW-63 | LOW | _synthetic always fills spotted_by (parser emits null) and three STORY_FACTS extras are never read; test_synthetic hard-codes the same values | `test/engine/_synthetic.py:117` | — |
| LOW-64 | LOW | _fake_states pair/context fixtures (smoke_between 0.0 for dead pairs, money alive-only) and test_state_schema's pair table differ from the builder | `test/engine/_fake_states.py:208` | — |
| LOW-65 | LOW | test_map_assets 'hashes stable across calls' is vacuous (cache); test_parse_worker '--force rewrote the manifest' uses >= so an unchanged manifest passes | `test/engine/test_map_assets.py:38` | — |
| LOW-66 | LOW | test_raw_schema docstring points to a nonexistent test_raw_golden.py | `test/engine/test_raw_schema.py:4` | — |
| LOW-67 | LOW | test_awpy3_parser.check_rawtables turns AttributeError from RawTables.from_parsed into a skip (exit 0) | `test/engine/test_awpy3_parser.py:191` | — |
| LOW-68 | LOW | test_temporal 'editing the next snapshot leaves the prediction unchanged' never gathers the edited row | `test/engine/test_temporal.py:99` | — |
| LOW-69 | LOW | test_attribution_round's V identity and V_raw sign checks are definitional; w_start is never pinned to an independent W(freeze_end) | `test/engine/test_attribution_round.py:98` | — |
| LOW-70 | LOW | Vacuous range checks on sigmoid/booster outputs and a loop-bounded step count (test_baseline, test_predict, test_heads, test_overfit_tiny) | `test/engine/test_baseline.py:49` | — |
| LOW-71 | LOW | test_actions_economy re-implements the team_class tie-break inside the test | `test/engine/test_actions_economy.py:72` | — |
| LOW-72 | LOW | test_dataset 'callouts keyed per map version' has an `or k1 == UNK` escape and a conditional check that silently disappears | `test/engine/test_dataset.py:122` | — |
| LOW-73 | LOW | test_download size-cap test is satisfied by the Content-Length pre-check; rejection cases accept any Exception; stale src/ sys.path insert | `test/engine/test_download.py:174` | — |
| LOW-74 | LOW | test_log_redaction covers only .dem-suffixed scheme-less paths and lower-case schemes | `test/engine/test_log_redaction.py:36` | — |
| LOW-75 | LOW | test_config never loads malformed or duplicate-key YAML through load() | `test/engine/test_config.py:127` | — |
| LOW-76 | LOW | test_scraper.py imports `scraper` from the deleted src/ path, so the manual Playwright verification tool cannot run | `test/engine/test_scraper.py:21` | — |
| LOW-77 | LOW | test_hf_archive uses the real 20 GB free-space floor on TMP (false FAIL on a full disk); mkdtemp dirs in ten tests are never removed | `test/engine/test_hf_archive.py:174` | — |
| LOW-78 | LOW | test_dataset, test_qa and test_labels run all checks at module level; test_matches_meta's UNIQUE check is a whitespace snapshot; test_env only checks Python >= 3.12 | `test/engine/test_matches_meta.py:56` | — |
| LOW-79 | LOW | test_predict's bf16 tolerance check swings 40x with GPU load (1.1e-3 alone vs 4.1e-2 under contention): unpinned ROCm nondeterminism; test_train_loop's target_failures check only asserts a list type | `test/engine/test_predict.py:126` | — |

## Findings

### CRITICAL

#### CRIT-01 — The pushed commit is missing the engine/data package: the `data/` gitignore rule also matches `engine/data/`

**File:** `.gitignore:25` · **Category:** L. Repo hygiene / packaging · **Confidence:** high · **Rule:** — · **Also:** `engine/cli.py:52`, `test/engine/_fake_states.py:268`

**What's wrong**  
`.gitignore` line 25 is the bare pattern `data/`, which git applies to any directory named `data` at any depth. `git ls-files engine/data` is empty while seven modules (`__init__`, `dataset`, `inventory`, `layout`, `qa`, `quarantine`, `splits`; 1,473 lines) exist on disk, and 32 files under engine/, test/ and bench/ import `engine.data`. `git check-ignore -v engine/data/dataset.py` reports `.gitignore:25:data/`. The 2026-10-19 commit was pushed to the public repo in this state.

**Evidence**  
```
# .gitignore:25
data/
```

**How it fails**  
Verified by extracting `git archive HEAD` into a temp directory and running it with the repo venv: `build-states`, `build-actions`, `inventory`, `calibrate`, `eval --calibration`, `bench predict` and `scrape --source hltv` exit 1 with `ModuleNotFoundError: engine.data`; `build-dataset` and `train-wp` exit 2 with the misleading message that the module 'does not exist yet'; `analyze` with a real model fails inside `load_predictor` (engine.model.predict imports engine.data); `train-q` / `build-neighbours` fail with `--split-file`. Eighteen test scripts error at import, and `test/run_all.py` in the clone reports 71 passed, 0 failed, 18 errored, 3 skipped, against 89/0/0/3 on this PC. Only the synthetic stub path survives.

**Blast radius**  
The published repository cannot run its own pipeline or most of its test suite. Any collaborator, CI run or buyer evaluation fails immediately, and the commit message's '89 passed' is true only for this working tree.

**Suggested fix**  
Change the rule to the anchored form `/data/` (root only) and commit `engine/data/`. Then `git status --ignored` and `git check-ignore -v` on every source path to confirm nothing else is swallowed. Add a test that imports every `engine.*` module from `git ls-files` output so an ignored package fails the suite.

### HIGH

#### HIGH-01 — The quarantine gate never runs on the batch path: parse, build-states and build-dataset accept 64-tick, short and non-5v5 demos

**File:** `engine/parse/worker.py:113-115` · **Category:** F. Data-pipeline correctness · **Confidence:** high · **Rule:** R10 · **Also:** `engine/cli.py:342`, `engine/data/quarantine.py:70`, `docs/ENGINE_PLAN.md:474`, `CLAUDE.md:212`

**What's wrong**  
`worker.parse_one` parses, validates and writes every demo; `RawTables.validate` checks only that `tick_rate` is a positive integer and accepts zero-row tables. `cli._build_one` builds states and actions with no `check_tick_rate` / `quarantine_check` and no look at `matches.status`, and under `--all` calls `db.set_status(key, 'states')`, which would overwrite a `quarantined` status and clear its reason. `engine/data/quarantine.py` (`check`, `check_file`, `quarantine`) has no production caller; nothing ever writes `status = quarantined`, so `inventory`'s quarantine counts are always empty and its 'non-quarantined only' target filter is a no-op. Only `analyze` and `parse --smoke` (which writes nothing) run `quarantine_check`. `ingest.SOURCES['hf']` registers the (64-tick) HF archive as tier `pro`.

**Evidence**  
```python
# engine/parse/worker.py:113-115
            raw.validate(strict=True)            # SchemaError -> retried, then logged
            timings = {"parse": time.perf_counter() - t0}
            _write(path, out_dir / key, raw.frames(), meta, timings)
```

**How it fails**  
Scrape or ingest a 64-tick, 3-round or 4v5 demo, then `parse`, `build-states --all`, `build-dataset`: it lands in states, actions and training shards. CLAUDE.md and plan P1.4.1 say it is refused. Normal batch operation, local.

**Blast radius**  
Off-spec matches in the WP training corpus; the 128-tick invariant is enforced for `analyze` only; quarantine records can be silently reset.

**Suggested fix**  
Run the P1.4.1 check at `register_demo` or in `parse_one` before writing, with one rule table shared by worker, analyze and `engine/data/quarantine.py`, and move failures with `quarantine.quarantine()`. Have `_build_one` refuse `status = quarantined` and never overwrite it. Add a worker test with a fake 64-tick parser.

#### HIGH-02 — Model adoption is read by scanning for `<kind>_vN` tokens, so a model named in another model's rule text counts as adopted

**File:** `engine/analyze.py:289` · **Category:** B. Gate logic · **Confidence:** high · **Rule:** R19 · **Also:** `engine/evaluation/registry.py:140`, `docs/ENGINE_MODELS.md:7`

**What's wrong**  
`adopted_models()` collects every `[a-z]+_v\d+` token on any ENGINE_MODELS.md line that does not contain reject/retire/supersede/revoke, and never reads the status column. `registry.adopt()` refuses those four words in `--rule` text but not other model tokens, although the module docstring promises both readers agree. `registry.is_adopted` does read the status cell, and `analyze` does not use it. `inventory --models` uses the same loose reader.

**Evidence**  
```python
# engine/analyze.py:289
        out.update(re.findall(r"\b[a-z]+_v\d+\b", line))
```

**How it fails**  
`adopt wp_v4 --rule "beats wp_v3 by 0.012"` (or 'beats lgbm_v1', or a q model rule naming wp_v1): `analyze --model wp_v3` now passes `check_model` without `--allow-unadopted` and with no warning, even if wp_v3 was retired or never adopted. A hand-edited status such as `candidate` is also read as adopted. Reachable the moment the first adoption row is written.

**Blast radius**  
Reports and `action_scores` rows from a model that never passed P6.7.2 or the P7 gate, labelled as fully adopted.

**Suggested fix**  
Make `adopted_models` / `check_model` use `registry.is_adopted` (model cell plus `status == adopted`), and have `adopt()` refuse rule text containing any model token other than the one being adopted. Extend test_registry's agreement check to compare whole sets and to include a rule naming a second model.

#### HIGH-03 — The Q/B value model and the neighbour index are loaded without any adoption check

**File:** `engine/analyze.py:181-185` · **Category:** B. Missing gate · **Confidence:** high · **Rule:** R19 · **Also:** `engine/cli.py:250`, `docs/ENGINE_MODELS.md:51`

**What's wrong**  
`ValueModels.load` checks only that both directories exist. `cmd_analyze` builds the dirs from `paths.MODELS_DIR / <name>` and `analyze_demo` loads them with no `check_model` call, no `--allow-unadopted` consultation and no `model_unadopted` warning. ENGINE_MODELS.md says Q/B models get registry rows, and the registry adopts `q_v<N>` names.

**Evidence**  
```python
# engine/analyze.py:181-185
    def load(cls, q_dir: Path, neighbours_dir: Path, device: str = "cpu") -> "ValueModels":
        q_dir, neighbours_dir = Path(q_dir), Path(neighbours_dir)
        for d in (q_dir, neighbours_dir):
            if not d.is_dir():
                raise ModelUnavailable(f"value model directory not found: {d.name}")
```

**How it fails**  
`analyze x.dem --model wp_v1 --q-model q_v9 --neighbours neighbours_v9` with no row for either: Q, B, D, R, X, the verdicts and the neighbour panels in the report and in `action_scores` come from untested artifacts. test_cli_guards never covers it.

**Blast radius**  
Every value-stage number in every report.

**Suggested fix**  
Call `check_model(q_dir, allow_unadopted)` and `check_model(neighbours_dir, allow_unadopted)` before `ValueModels.load` on both the demo and the story path, and emit the warning on the allow path.

#### HIGH-04 — On real demos every player's on_fire is 0 and the fire_pressure enabler never fires: the fire extent is always null and unknown is emitted as 0

**File:** `engine/state/players.py:282` · **Category:** F. None -> 0 sentinel · **Confidence:** high · **Rule:** R7 · **Also:** `engine/parse/awpy3_parser.py:680`, `engine/state/utility.py:78`, `engine/actions/enablers.py:189`, `test/engine/test_utility_token.py:89`

**What's wrong**  
awpy 3.1 gives no flame extents, so the parser hard-codes `extent_radius` to null (documented in ENGINE_PROGRESS and ENGINE_ENV). `utility.inside()` returns False for a None radius, so `on_fire = int(False) = 0` for every alive player on every tick, including one standing in a molotov; the schema makes `on_fire` non-null when alive, so None is not even allowed. `enablers._fire` skips any fire with a None radius and the plan's 'victim on fire' branch does not exist, so `fire_pressure` is never produced and `enemies_displaced` is always None. The counterfactual `_in_token` returns False the same way, so `remove_utility` of a fire never changes anything. Tests pass only because they inject `extent_radius=150.0`.

**Evidence**  
```python
# engine/state/players.py:282
            on_fire  = int(any(inside(u, *pos) for u in utility if u.kind == "fire"))
```

**How it fails**  
Every real demo: the model learns that fire never touches anyone, and every molotov's enabler share goes back to the killer. ENGINE_PROGRESS suggests a nominal radius 'would be safer'; nothing was decided or implemented.

**Blast radius**  
A constant-0 feature trained into the WP model; molotov credit understated in every report; `fire_pressure` is dead code.

**Suggested fix**  
Faig's choice: a nominal fire radius flagged as nominal (state token, action params and LIVE_SAFE text), or make `on_fire` nullable (schema, validate, LIVE_SAFE, dataset layout) and emit None when the extent is unknown; derive 'victim on fire' from inferno damage rows for the enabler. Add tests with `extent_radius=None`, the shape production emits.

#### HIGH-05 — Counterfactual edits rewrite team money as an alive-only sum while the builder sums every account, so a zero-effect action gets a nonzero Shapley value (C1 broken)

**File:** `engine/counterfactual/interventions.py:304-305` · **Category:** F. Train/serve mismatch; R21 C1 · **Confidence:** high · **Rule:** R21 · **Also:** `engine/state/context.py:148`, `engine/state/schema.py:477`, `test/engine/_cf_states.py:109`, `test/engine/test_context.py:77`

**What's wrong**  
Every intervention ends in `recompute_derived`, which sets `ctx.money_ct/money_t` from alive tokens only (a dead token's money is None). The builder's `context.money()` sums every roster account, dead included; the LIVE_SAFE text says 'current sum of accounts' and test_context asserts dead accounts are included. Reproduced on all six synthetic stories through the real builder: `recompute_derived(s.copy())` on an unedited snapshot changes only money (base story, round 1: money_t 700 → 300; 47 of the story's snapshots affected). `Snapshot.validate` checks equip sums but never money. The C1 fixtures (`_cf_states`) build money alive-only, so test_interventions, test_apply and test_attribution_shapley cannot see it.

**Evidence**  
```python
# engine/counterfactual/interventions.py:304-305
    c.money_ct = _sum_alive(ps, "ct", "money")
    c.money_t  = _sum_alive(ps, "t", "money")
```

**How it fails**  
In `apply`, v(full) is the factual W(s_post) while every other coalition is rebuilt by undoing members and recomputing. With any player dead (true after the first kill of every round), v(full) and v(full minus a null member) differ by the money rewrite, and `money_ct/t` are model inputs (CTX_LAYOUT). So φ(null decoy) = ½[W(money with dead) − W(money alive-only)] ≠ 0 on the real model in nearly every kill window, and the same artifact enters movement credit and enabler hazard shares. Conservation still holds because the residual lands in the clock, so the identity check cannot detect it.

**Blast radius**  
Every attributed window with two or more members, every drift interval after a death, every hazard split; the acceptance condition C1 cannot pass on a real model.

**Suggested fix**  
Leave money untouched in `recompute_derived` (accounts do not change when a kill is undone), or recompute exactly as the builder does; make `_cf_states` use the builder's definition; add a test that passes real builder snapshots through `recompute_derived` and `apply(s, s, members, set())` and asserts equality. Faig should confirm the intended definition (all-roster is the documented one).

#### HIGH-06 — Counterfactual line-of-sight and smoke checks are cast from feet positions while the builder uses eye points

**File:** `engine/counterfactual/interventions.py:210` · **Category:** G. Train/serve feature mismatch · **Confidence:** high · **Rule:** — · **Also:** `engine/counterfactual/interventions.py:227`, `engine/state/pairwise.py:33`, `engine/analyze.py:723`

**What's wrong**  
`_los_value` and `_smoke_value` pass `_pos(player)` (raw z) to the geometry, while `pairwise.table` uses `_eye(p) = (x, y, z + EYE_STAND or EYE_DUCK)`. Confirmed with a recording geometry: builder call at z = 64, intervention call at z = 0 with ducking ignored. The ref-copy shortcut needs exact position equality, so after any movement `undo_kill` and `freeze_position` recompute the revived or frozen player's whole LOS row at feet height, and `analyze` passes the real `MapGeometry`. For smoke, feet and eye segments disagree in a band around the smoke edge (wider with height differences), and `remove_utility` revisits only pairs whose feet segment crosses the removed smoke, so a pair whose eye segment crossed it keeps a stale `smoke_between = 1`. Dead-pair smoke values are also written as 0/1 instead of NaN, but both models mask dead pairs, so that part is inert.

**Evidence**  
```python
# engine/counterfactual/interventions.py:210
        vis = geometry.is_visible(pa, pb)
```

**How it fails**  
Every kill window and every movement coalition on a real map: a feet-level ray is blocked by floor geometry and waist-high cover that the eye ray clears, so counterfactual LOS reads 'blocked' where the factual definition reads 'visible'. The difference is credited to the kill or the movement and offset in the clock gap, so conservation hides it.

**Blast radius**  
Systematic, not random, bias in kill and movement V on real maps; the magnitude is unmeasured (no assets or demo here) but the mechanism is exact.

**Suggested fix**  
Factor the eye-point helper out of `pairwise.py` and use it in `_los_value` and `_smoke_value`; when a smoke token is added or removed, recompute every alive pair; return NaN for pairs with a dead player. Add a test that recomputes a builder snapshot's pair rows through `_recompute_rows` with the same geometry and asserts they are unchanged.

### MEDIUM

#### MED-01 — Two quarantine rule sets disagree, and the stored-corpus one reads keys the parser never emits

**File:** `engine/data/quarantine.py:50-51` · **Category:** F. Data-pipeline correctness · **Confidence:** high · **Rule:** — · **Also:** `engine/parse/awpy3_parser.py:127`, `test/engine/test_quarantine.py:22`

**What's wrong**  
engine/data/quarantine.py `check()` requires `players_ct` / `players_t`, but awpy3_parser.header_info and parse meta emit `side_counts: {ct, t}`. Its thresholds (tick exactly 128, 8 min to 6 h, reasons `tick_rate` / `few_rounds` / `duration`) differ from awpy3_parser.quarantine_check (±0.5 tick, 15 min to 5 h, `tick_rate_not_128` / `too_few_rounds` / `duration_implausible`), and the parser's reason names are not in `quarantine.REASONS`, so passing them to `quarantine.quarantine()` raises ValueError. test_quarantine.py hand-builds meta in the wrong shape and locks the wrong contract in.

**Evidence**  
```python
# engine/data/quarantine.py:50-51
    if meta.get("players_ct") != TEAM_SIZE or meta.get("players_t") != TEAM_SIZE:
        return "not_5v5"
```

**How it fails**  
Wire the stored-corpus stage as its docstring says (header_info -> check) and every real demo is quarantined as `not_5v5`. Wire the parser's check instead and `quarantine()` raises on the reason string. Latent only because neither is wired today (HIGH-01).

**Blast radius**  
Would quarantine 100% of demos the moment P1.4 is wired; the two gates give different verdicts on the same demo.

**Suggested fix**  
Keep one rule table. Have `engine.data.quarantine.check` delegate to `awpy3_parser.quarantine_check` (or the reverse) and share the reason vocabulary. Rebuild test_quarantine's GOOD meta from a real `header_info` result.

#### MED-02 — A model directory is accepted as adopted by basename alone, from any location

**File:** `engine/analyze.py:305-306` · **Category:** B. Gate logic · **Confidence:** high · **Rule:** R19 · **Also:** `test/engine/test_cli_guards.py:104`

**What's wrong**  
`check_model` compares only `model_dir.name` with the adopted set. Nothing ties an adopted row to the files under `data/engine/models/<name>` (no manifest hash, no path check). test_cli_guards locks this in: an empty temp folder named `wp_v3` is asserted to pass.

**Evidence**  
```python
# engine/analyze.py:305-306
    if not allow_unadopted and model_dir.name not in adopted_models(doc):
        raise ModelUnavailable(f"model {model_dir.name} is not adopted in docs/ENGINE_MODELS.md; "
```

**How it fails**  
`analyze x.dem --model-dir D:\scratch\wp_v2` where that folder is a different, unadopted retrain that shares the name. It passes the gate and the report says `model_versions.wp = wp_v2`.

**Blast radius**  
Wrong model silently used under an adopted name; reports mislabelled.

**Suggested fix**  
With `--model-dir`, require the path to resolve under `paths.MODELS_DIR/<name>` or compare the directory manifest's config_hash/git_sha with the values recorded at adoption; otherwise require `--allow-unadopted`.

#### MED-03 — Test hooks (CROSSHAIR_STUB_PREDICTOR, --raw-tables-story) skip the adoption gate and write to the real engine.db and reports dir by default

**File:** `engine/cli.py:258` · **Category:** B. Test hook reachable in production · **Confidence:** high · **Rule:** R19

**What's wrong**  
When the env var is set, `cmd_analyze` uses `RuleStub` and `analyze_demo` skips `check_model` (it checks only when `predictor is None`). `write_db` defaults to True and `out_dir` to `data/engine/reports/<stem>`. `--raw-tables-story` on a real demo name does the same with synthetic tables.

**Evidence**  
```python
# engine/cli.py:258
        lg.warning("TEST HOOK: CROSSHAIR_STUB_PREDICTOR is set; using a rule stub instead of the model")
```

**How it fails**  
A shell still carrying `CROSSHAIR_STUB_PREDICTOR=1` after a manual test run, then `analyze real.dem`: stub W values go into `action_scores` under the real match key with no report warning beyond `model_versions.wp = RuleStub`.

**Blast radius**  
Stub or synthetic scores mixed into the production `action_scores` table under real match keys.

**Suggested fix**  
When either hook is active, force `write_db=False`, refuse a default out_dir under `data/engine/reports`, and add a `test_hook` report warning.

#### MED-04 — `or` used as a numeric null-coalesce on the 128-tick gate and in the parser's quarantine check

**File:** `engine/analyze.py:220-221` · **Category:** F. `or` over numerics · **Confidence:** high · **Rule:** R8 · **Also:** `engine/parse/awpy3_parser.py:307`

**What's wrong**  
Both sites use the pattern CLAUDE.md forbids. A `tick_rate_raw` of 0.0 falls through to the rounded value; a NaN is truthy, and `abs(nan - 128) > 0.5` is False, so a NaN rate passes `check_tick_rate`.

**Evidence**  
```python
# engine/analyze.py:220-221
    rate = meta.get("tick_rate_raw") or meta.get("tick_rate")
    if rate is None or abs(float(rate) - EXPECTED_TICKRATE) > TICKRATE_TOL:
```

**How it fails**  
Not reachable from today's parser (`_tick_rate_from_header` maps raw <= 0 to None and raises on NaN at `round()`), so the effect is latent. It is the literal R8 pattern on the R10 gate.

**Blast radius**  
A future meta source emitting 0.0 or NaN bypasses the tick-rate refusal.

**Suggested fix**  
Use `engine.state.players._first_valid` and refuse non-finite values explicitly.

#### MED-05 — scrape exits 1 instead of 2 when FACEIT_API_KEY is missing, and a YAML syntax error in the config also exits 1

**File:** `engine/cli.py:608` · **Category:** J. Exit codes · **Confidence:** high · **Rule:** R22 · **Also:** `engine/cli.py:679`, `engine/config.py:442`

**What's wrong**  
A missing or empty key raises `EnvironmentError` inside the source generator, outside `ingest.run`'s per-match try; `main()` maps only `FileNotFoundError` to 2. `config.load` calls `yaml.safe_load` unguarded, so `yaml.YAMLError` (a tab indent, an unclosed bracket) is neither OSError nor ConfigError and reaches the generic handler as exit 1. No literal `Bearer None` header is ever sent. Neither case has a test.

**Evidence**  
```python
# engine/cli.py:608
    failed = ingest.run(a.source, a.max, tier=a.tier, **kw)
```

**How it fails**  
`scrape --source api` with no `.env` key, or `status --config broken.yaml` with a syntax error: both report 'ran and failed'.

**Blast radius**  
Scripts and the runbook cannot tell a configuration error from a failed run.

**Suggested fix**  
Check the key after `_load_dotenv` and raise `CLIError(..., 2)`; catch `yaml.YAMLError` in `config.load` and re-raise as `ConfigError` with line and column. Add both cases to test_cli_errors.

#### MED-06 — Re-registering an existing match_id with a different demo reports success and leaves a file the DB does not describe

**File:** `engine/db.py:362-363` · **Category:** E. Data integrity · **Confidence:** high · **Rule:** R1 · **Also:** `engine/ingest.py:169`

**What's wrong**  
`register_match` prints a notice and returns the existing key when the sha differs. In `register_demo`, `key == match_id`, so the freshly copied `.dem` is kept and the CLI prints 'registered as <id>' with exit 0. The row's `demo_sha256`, date and tier describe the earlier file.

**Evidence**  
```python
# engine/db.py:362-363
                print(f"[db] {key}: already registered with a different sha256; keeping the earlier row")
            result = key
```

**How it fails**  
`ingest other.dem --match-id K` (or a same-stem demo in a different tier) when K is registered and its original file was moved or deleted: `demos/<tier>/K.dem` now holds different bytes than the row; a later `parse` writes `parsed/K` from the wrong demo. Neither test_ingest nor test_db covers the case.

**Blast radius**  
One match's states and actions built from a different demo under that match's metadata, silently.

**Suggested fix**  
Raise (or return a distinguishable sentinel) from `register_match` when the key exists with a different non-null sha, so `register_demo` deletes the copy and exits non-zero. Add the case to test_ingest and test_db.

#### MED-07 — The analyze output folder's manifest.json records the absolute demo path and the hostname

**File:** `engine/manifest.py:111` · **Category:** H. Report hygiene · **Confidence:** medium · **Rule:** R14 · **Also:** `engine/analyze.py:761`, `test/engine/test_cli_analyze.py:72`

**What's wrong**  
`manifest.build` always writes `hostname` and `_describe_input` writes `str(p)` for `inputs=[demo_path]`. `report.json` / `report.md` go through `writer.redact`, but `manifest.json` next to them does not, and no test checks it (test_cli_analyze reads the manifest and checks only `extra`).

**Evidence**  
```python
# engine/manifest.py:111
    return {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
```

**How it fails**  
Share the output folder (the thing a buyer receives) and it carries `C:\Users\<user>\...` and the machine name.

**Blast radius**  
Local username and machine name leave with every shared report folder. Whether manifest.json is inside R14's scope is Faig's call.

**Suggested fix**  
Record inputs by file name plus sha256 (or relative to `paths.ROOT`), drop or hash the hostname in analyze manifests, and run `find_leaks` over every text file in out_dir in the redaction test.

#### MED-08 — Decoys never expire in the state and start when they end: expire_tick is always null and detonate_tick is the last trajectory tick

**File:** `engine/state/builder.py:147-149` · **Category:** F. Null end -> active forever · **Confidence:** medium · **Rule:** — · **Also:** `engine/state/utility.py:30`, `engine/actions/utility.py:229`

**What's wrong**  
The parser never links decoys to a cloud, so `expire_tick` is null and `detonate_tick = coalesce(null, last_tick)`. `utility._active` keeps a row with a null end active to round end; `builder.until` cannot null a missing end. Actions, by contrast, end the decoy at a nominal lifetime, so `decoy_end` lands on a tick where the state does not change.

**Evidence**  
```python
# engine/state/builder.py:147-149
        dec = raw.grenades.filter((pl.col("type") == "decoy") & (pl.col("throw_tick") >= start))
        v.decoys = [{"entity_id": g["entity_id"], "start_tick": g["detonate_tick"],
                     "end_tick": g["expire_tick"], "throw_tick": g["throw_tick"],
```

**How it fails**  
Any round with a decoy: a `decoy` token appears (probably after the decoy already finished) and stays in the utility set for the rest of the round, counting toward the 16-token cap.

**Blast radius**  
Fake utility tokens in the model input; busy rounds can push real smokes or fires out of the capped list.

**Suggested fix**  
One end-of-life rule in both modules: a nominal decoy duration from the start tick, and a start taken from the landing event rather than the last tracked tick (or drop the token when the start is unknown). Add a test with a null decoy expire_tick.

#### MED-09 — kills_round counts self-kills and kills with unknown team_kill, unlike the parser's ticks.kills_round

**File:** `engine/state/players.py:234` · **Category:** F. Wrong filter · **Confidence:** high · **Rule:** —

**What's wrong**  
The parser sets `team_kill = (a_side == v_side) & (att != vic)`, so a suicide has `team_kill=False` and an unknown side gives `None`. Both pass `_nan(t) is not True` here, while the parser's own `kills_round` excludes both. The docstring says team kills are excluded.

**Evidence**  
```python
# engine/state/players.py:234
        kills = float(sum(1 for a, t in zip(att, tk) if _nan(a) == steamid and _nan(t) is not True))
```

**How it fails**  
A player dies to their own HE or molotov: their `kills_round` rises by one and disagrees with `ticks.kills_round` for the same tick.

**Blast radius**  
One player feature wrong after a self-kill.

**Suggested fix**  
Count a kill only when the attacker is this player, the victim is a different player, and `team_kill` is False (or require the victim in `enemies`, as the damage sum does).

#### MED-10 — Knife-skin display names map to weapon class 'unknown', not 'knife'

**File:** `engine/state/players.py:126-127` · **Category:** F. Vocab lookup · **Confidence:** medium · **Rule:** —

**What's wrong**  
vocab_weapons.yaml carries display-name aliases for every gun but none for knife skins. `weapon_class('Karambit')`, `'Butterfly Knife'`, `'M9 Bayonet'`, `'Shadow Daggers'` all return `unknown` (checked with a no-write probe); `weapon_knife_karambit` maps correctly.

**Evidence**  
```python
# engine/state/players.py:126-127
    if bare.startswith(("knife", "bayonet")):
        return "knife"
```

**How it fails**  
If awpy 3.1 reports `active_weapon` as display names, every player running with a skinned knife out gets `weapon_class = unknown`.

**Blast radius**  
The weapon_class feature is wrong whenever a skinned knife is out, on demos that report display names.

**Suggested fix**  
Lower-case and match on 'knife', 'bayonet', 'daggers' substrings, or add the knife display names as aliases. First confirm which form awpy 3.1 reports.

#### MED-11 — A Rust panic or a dead worker aborts the whole parse batch before any failure is logged

**File:** `engine/parse/worker.py:118-119` · **Category:** D/I. Malformed .dem handling · **Confidence:** medium · **Rule:** — · **Also:** `engine/parse/worker.py:168`

**What's wrong**  
`parse_one` claims it never raises, but a pyo3 `PanicException` derives from `BaseException` and escapes `except Exception`. With `procs > 1`, a hard worker death raises `BrokenProcessPool` from `list(pool.map(...))`, which fails every pending future, and `run()` raises before `_append_failures`.

**Evidence**  
```python
# engine/parse/worker.py:118-119
        except Exception as exc:                  # retried once, then reported
            last = exc
```

**How it fails**  
One corrupt demo in a scrape-sized directory: every rerun dies at the same file, `_failures.jsonl` gets nothing, and the bad demo is never named.

**Blast radius**  
Whole parse batches; nothing already written is lost.

**Suggested fix**  
Catch `BaseException` minus KeyboardInterrupt/SystemExit in parse_one; use `submit` + `as_completed` and convert a per-future failure into a 'failed' result (recreating a broken pool) so `_append_failures` always runs.

#### MED-12 — Grenade trajectory segments split on an 8-tick gap keep the same entity_id, producing spurious mid-flight flash/HE/decoy actions and mislinked clouds

**File:** `engine/parse/awpy3_parser.py:618` · **Category:** F. Identity / join keys · **Confidence:** medium · **Rule:** — · **Also:** `engine/actions/utility.py:147`, `docs/ENGINE_ENV.md:108`

**What's wrong**  
`_collapse_grenades` starts a new row at a >8-tick gap but every segment keeps `_key` as `entity_id` and gets `detonate_tick = its own last_tick`. The smoke/fire link joins by that id, so a second grenade can inherit the first cloud's timing and position (detonate before throw). `extract_flashes` / `extract_he` / `extract_decoys` treat each row as a grenade, so segment 1 becomes a flash or HE action at a mid-flight tick with empty victim lists; the ids differ by tick so `ctx.unique` keeps both. ENGINE_ENV.md implies the segments are distinct entities.

**Evidence**  
```python
# engine/parse/awpy3_parser.py:618
        pl.col("_key").first().alias("entity_id"),
```

**How it fails**  
Any entity without a serial that is reused, or any trajectory with a >8-tick hole. Whether awpy 3.1 trajectories have such gaps is unverified.

**Blast radius**  
Extra actions that each own a window and absorb ΔW; wrong pop/fade timing for the affected utility; inflated per-type counts.

**Suggested fix**  
Derive `entity_id` from (key, segment ordinal) so segments stay distinct, and collapse rows sharing (entity_id, thrower) in the utility extractors. Add a no-serial reuse case to test_awpy3_parser.

#### MED-13 — A null snapshot health becomes health 0 / dead / equipment 0 / empty inventory

**File:** `engine/parse/awpy3_parser.py:477` · **Category:** F. None -> 0 sentinel · **Confidence:** medium · **Rule:** R7 · **Also:** `engine/parse/awpy3_parser.py:448`

**What's wrong**  
A row whose `health` is null (no pawn decoded at that tick, or the column absent so `_col` yields an all-null literal) is rewritten as a dead player with legitimate values, and x/y/z forward-filled. `is_alive` is NOT NULL so validation passes.

**Evidence**  
```python
# engine/parse/awpy3_parser.py:477
        (health > 0).fill_null(False).alias("is_alive"),
```

**How it fails**  
If the column name differs in awpy 3.1, every player is dead at every tick and the parse still validates. Trigger unverified; the code path is certain.

**Blast radius**  
Alive counts, the core WP feature, on any affected tick; a whole match if the column is missing.

**Suggested fix**  
Keep health/equipment null when health is null; fail validation when health is entirely null on a non-empty ticks table, or carry awpy's own alive flag.

#### MED-14 — Utility parent ids are re-salted after their pop/fade pseudo-actions copied the old id, so a second same-tick smoke's credit goes to the first

**File:** `engine/actions/utility.py:281` · **Category:** C. Stale reference after mutation · **Confidence:** high · **Rule:** — · **Also:** `engine/attribution/engine.py:315`

**What's wrong**  
`_durative` builds both parents with the same default action_id and each pop/fade stores it in `parent_action_id`; `Context.unique` then salts the second parent but not its pseudo-actions. Confirmed with a no-write probe: both `smoke_pop` rows point at one parent. attribution credits pseudo φ to `acts[parent]`.

**Evidence**  
```python
# engine/actions/utility.py:281
    return ctx.unique([a for a in out if a is not None], lambda a: a.params.get("entity_id"))
```

**How it fails**  
Two smoke (or inferno, or decoy) rows with the same thrower and start tick but different entity_ids, which needs duplicate area rows from the parser (unverified on real demos).

**Blast radius**  
Per-action V for the affected utility is wrong; player totals unchanged (same thrower).

**Suggested fix**  
Salt the parent before building its pseudo-actions, or rewrite `parent_action_id` after `unique()`; better, dedupe identical area rows. Add a check to test_review_items.

#### MED-15 — A drop/give is valued only at the drop tick; the teammate's pickup jump has no owner

**File:** `engine/actions/windows.py:59` · **Category:** C. Spec intent · **Confidence:** medium · **Rule:** — · **Also:** `engine/actions/economy.py:145`

**What's wrong**  
`extract_drops` sets `state_post_tick = pickup + 1` and hands the pickup tick to the builder as a boundary, but `windows.group` uses only `a.tick`, so the drop's window is `[drop-1, drop+1]`, where the dropper loses the weapon. The pickup lies in a drift interval and its WP jump goes to movement or the clock. Nothing outside actions reads `state_post_tick`.

**Evidence**  
```python
# engine/actions/windows.py:59
        if a.kind != "instant" or a.tick is None:
```

**How it fails**  
Every give after freeze end.

**Blast radius**  
Mid-round gives get ~0 or negative V; the receiver's gain is spread over drift actors (plan P4.6.1 says every jump has an owner).

**Suggested fix**  
Add a pickup pseudo-action credited to the drop (the pop/fade pattern), or document `state_post_tick` as metadata only.

#### MED-16 — The P7 sanity gate cannot go green from the CLI and certifies a different intervention path than analyze runs

**File:** `engine/counterfactual/sanity.py:395-397` · **Category:** J. Gate · **Confidence:** high · **Rule:** R19 · **Also:** `engine/counterfactual/sanity.py:129`, `engine/cli.py:67`, `test/engine/test_sanity_gate.py:143`

**What's wrong**  
`main()` returns 2 unconditionally once `--model` and `--states` exist (sampling 'not wired yet'), and the CLI never forwards `--states`, so `sanity --counterfactual` always exits 2. `run_gate` / `_edit` also build every edited state with `geometry=None`, while `analyze` passes the real `MapGeometry`, so even a hand-run gate judges states whose LOS rows are NaN rather than the recomputed 0/1 rows attribution scores. test_sanity_gate only asserts the exit-2 paths; ENGINE_OPS.md:37 and ENGINE_PROGRESS.md:63 say the command exits 1 on a failed rate.

**Evidence**  
```python
# engine/counterfactual/sanity.py:395-397
          "grouping and the P5 dataset reader, which are not wired yet; call run_gate() with "
          "samples directly")
    return 2
```

**How it fails**  
Run the documented P7 command with a real model: exit 2 every time. `analyze` never consults a gate result, so attribution is produced regardless.

**Blast radius**  
The hard gate CLAUDE.md puts in front of all attribution is unreachable and, when run by hand, does not test the production path.

**Suggested fix**  
Wire sampling into `main`, forward `--states`, add a `geometry` argument to `run_gate`/`_edit`/`apply`, have `analyze` require a green gate record for the model, and correct the two docs. Make the test expect 0/1 with real inputs.

#### MED-17 — The Q model, neighbour index and WP encoder record no link to each other, so a mismatched pair silently gives wrong alternative Q and regret

**File:** `engine/value/neighbours.py:195` · **Category:** G. Artifact provenance · **Confidence:** high · **Rule:** — · **Also:** `engine/analyze.py:181`

**What's wrong**  
`build-neighbours` encodes stored `desc_ids` with the vocab of the `--q-model` given at build time and stores `h_pre` from whatever WP encoder valued the frame; neither manifest records those dependencies, and `ValueModels.load` checks only the vector dimension. `DescriptorVocab.fit_frame` re-sorts tokens, so ids shift on every refit.

**Evidence**  
```python
# engine/value/neighbours.py:195
    manifest.write(out, "neighbours", outputs=outputs, extra={"kind": nidx.kind, "n": nidx.n, "d": nidx.d,
```

**How it fails**  
`neighbours_v1` built with `q_v1`, then `analyze --q-model q_v2 --neighbours neighbours_v1`: alternatives' Q come from the wrong embeddings; R stays <= 0 so nothing fires. An old id past q_v2's embedding size raises an index error and aborts the value stage.

**Blast radius**  
R, panel alternatives, verdicts and C2 panels for every analyzed match until both are rebuilt together.

**Suggested fix**  
Record the Q vocab hash and the WP model version in the neighbour and Q manifests; refuse or warn in `ValueModels.load` when they disagree.

#### MED-18 — train-wp and calibrate never check that --vocab/--norm match the shards, so a wrong vocab silently scrambles every categorical

**File:** `engine/model/train.py:510` · **Category:** G. Feature/vocab mismatch · **Confidence:** medium · **Rule:** — · **Also:** `engine/data/dataset.py:640`

**What's wrong**  
Each shard index records the vocab name but `iter_shards` checks only `layout_hash`; `Vocab.from_dict` / `NormStats.from_dict` ignore the layout hash they store. A larger vocab from another build raises no IndexError.

**Evidence**  
```python
# engine/model/train.py:510
            vocab=Vocab.load(a.vocab), norm=NormStats.load(a.norm),
```

**How it fails**  
Two `build-dataset` runs (say an ablation suffix), then `train-wp --train train --vocab vocab_v2.json`: weapon, callout, grid, player and map-version ids point at different categories; the model trains on scrambled inputs and the model dir copies the wrong vocab for the Predictor.

**Blast radius**  
A silently degraded model that may still pass gates; the LightGBM comparison would be unfair.

**Suggested fix**  
Store a content hash of vocab/norm in the shard index and compare it in fit/calibrate/baseline, or default the flags to the names recorded in the train shard index and refuse a mismatch.

#### MED-19 — An empty or pro-less val/test split crashes with KeyError after a full first epoch, and calibrate overwrites calibration.json before crashing

**File:** `engine/model/train.py:145` · **Category:** I. Error handling · **Confidence:** high · **Rule:** R22 · **Also:** `engine/model/calibrate.py:301`, `engine/data/dataset.py:571`

**What's wrong**  
`predict_logits` returns `{}` for n == 0 and every caller indexes `['win']`. `read_shards` on a missing or mistyped split name globs a non-existent dir and returns empty arrays. `--tiers pro` with no pro rows in val, or month splits whose last months hold no pro matches, do the same.

**Evidence**  
```python
# engine/model/train.py:145
    out: dict[str, np.ndarray] = {}
```

**How it fails**  
`train-wp` runs one full epoch, then `_val_metrics` raises KeyError before `last.pt` is saved; exit 1 with a traceback. `calibrate` with empty test pro saves a new calibration.json and then crashes in `evaluate`.

**Blast radius**  
Hours of GPU time per attempt; a calibration file replaced by a run that then fails.

**Suggested fix**  
Check up front that each split exists and has labelled (pro, where required) rows and exit 2 with the reason; make `read_shards` raise on a missing directory.

#### MED-20 — The Predictor silently serves uncalibrated probabilities when calibration.json is missing, and adoption does not require it

**File:** `engine/model/checkpoints.py:101-102` · **Category:** G. Calibration · **Confidence:** high · **Rule:** R6 · **Also:** `engine/evaluation/registry.py:134`

**What's wrong**  
A model directory without calibration.json yields `cal = None`, and `Predictor.predict` then applies a plain sigmoid. `registry.adopt` needs only manifest.json; the ECE it checks is a number typed on the CLI. calibration.json also does not record which checkpoint it was fitted for.

**Evidence**  
```python
# engine/model/checkpoints.py:101-102
    cal_p   = run_dir / "calibration.json"
    cal     = json.loads(cal_p.read_text(encoding="utf-8")) if cal_p.is_file() else None
```

**How it fails**  
Adopt a model whose calibrate run never finished, or point `Predictor` at `best.pt` after calibration was fitted on `final.pt`: every V = W(after) - W(before) is on a scale never checked against ECE <= 0.02, with no warning.

**Blast radius**  
Every attribution number from that model.

**Suggested fix**  
Refuse (or require an explicit flag) in `Predictor`/`analyze` when calibration.json is absent; record the checkpoint `state_hash` in calibration.json and compare on load; have `adopt` read the numbers from calibration.json/eval.json.

#### MED-21 — read_shards loads an entire split into RAM, which cannot fit at the configured corpus targets

**File:** `engine/data/dataset.py:571-573` · **Category:** H. Resource lifecycle · **Confidence:** medium · **Rule:** —

**What's wrong**  
`train.main`, `calibrate.main` and the baseline call `read_shards` on full splits; `concat` keeps the shard list and the concatenated copy alive together, and `fit()` copies again for tier filtering. The pair tensor alone is about 2 KB per snapshot.

**Evidence**  
```python
# engine/data/dataset.py:571-573
def read_shards(split: str | Path, base: Path | None = None) -> dict[str, np.ndarray]:
    """Every shard of a split, concatenated (a split name under `base`, or a directory)."""
    return concat(list(iter_shards(split, base)))
```

**How it fails**  
At the config targets (3,000 pro plus 50,000 FACEIT maps, 2 Hz plus boundary samples) the pair block is on the order of hundreds of GB; the process is OOM-killed at load time on the first full-scale run.

**Blast radius**  
train-wp, calibrate and baseline cannot run on the target corpus; tests never see it.

**Suggested fix**  
Memory-map the shards (uncompressed `.npy` per key with `mmap_mode`) or stream batches shard by shard with a `WindowIndex` built from round keys and times only.

#### MED-22 — Path redaction in reports stops at the first space, and find_leaks shares the blind spot

**File:** `engine/report/writer.py:41-42` · **Category:** B/I. Redaction · **Confidence:** high · **Rule:** R14 · **Also:** `test/engine/test_report_redaction.py:32`

**What's wrong**  
The path regexes exclude whitespace, so `C:\Users\Some User\Desktop\x.pt` becomes `<path> User\Desktop\x.pt` (verified by running `scrub_text`), and `find_leaks` on that output returns nothing. The POSIX pattern fails the same way, and the out_dir rewrite breaks under an out_dir containing a space. Free-text sources are `window_failed` warnings built from `str(err)` and enabler flags. The redaction test uses `Path.home()` (no space on this PC) and only space-free literals.

**Evidence**  
```python
# engine/report/writer.py:41-42
_WIN_PATH_RE = re.compile(r"(?:\\\\[^\s'\"<>|]+|[A-Za-z]:[\\/][^\s'\"<>|]*)")
_POSIX_RE    = re.compile(r"(?:(?<=^)|(?<=[\s'\"(=]))/(?:[^\s/'\"<>]+/)+[^\s'\"<>]*")
```

**How it fails**  
A report string holding an absolute path with a space, common on Windows (`Program Files`, user names with spaces).

**Blast radius**  
Part of the username and the directory tree in report.json/report.md; the control's own checker cannot see it.

**Suggested fix**  
Anchor path redaction on known roots (home, cwd, `paths.DATA_DIR`, out_dir) and replace prefixes literally, or redact from the drive letter to the next quote or line end; make `find_leaks` check those prefixes too; add fixed spaced-path, relative-path and scheme-less URL cases to the test.

#### MED-23 — find_leaks puts the raw matched string, cut to 80 characters, into its messages

**File:** `engine/report/writer.py:119` · **Category:** B. Credential handling · **Confidence:** high · **Rule:** R9

**What's wrong**  
The diagnostic message contains the unredacted string truncated by character count, the exact pattern R9 forbids. Today only test_report_redaction prints it (on synthetic data).

**Evidence**  
```python
# engine/report/writer.py:119
            out.append(f"url in {key}: {s[:80]}")
```

**How it fails**  
Any future caller that logs `find_leaks` output (for example a pre-write assertion in analyze) would print the signed URL it just detected, depending on where the signature falls in the first 80 characters.

**Blast radius**  
Test stdout today; a log line tomorrow.

**Suggested fix**  
Report the key and the leak kind (or a `redact_text`-processed excerpt), never the matched text.

#### MED-24 — The exhausted-429 error from the FACEIT Data API is swallowed, so a throttled scrape reports 'no new demos' and exits 0

**File:** `engine/sources/scraper.py:237-238` · **Category:** I. Error handling / C. request pacing · **Confidence:** high · **Rule:** — · **Also:** `engine/sources/scraper.py:94`

**What's wrong**  
`_api_get` caps each Retry-After wait at 60 s and raises `RuntimeError` after four attempts. `get_player_match_ids` catches it as a generic Exception and returns `[]`; `faceit_api.get_match_demo_urls` and `get_match_min_level` do the same. The run then moves to the next player, sending four more early requests each.

**Evidence**  
```python
# engine/sources/scraper.py:237-238
        print(f"    lookup failed for {nickname}: {e}")
        return []
```

**How it fails**  
FACEIT returns 429 with Retry-After > 60 s or keeps returning 429: nothing is attempted, `ingest.run` prints 'no new demos found', `cmd_scrape` exits 0, and the key keeps being hammered against the server's window.

**Blast radius**  
A whole scrape run silently produces nothing and reports success; risk of the key being restricted. scraper.py is ask-before-touch (R18).

**Suggested fix**  
Let the rate-limit error propagate (a dedicated subclass) through the per-player and per-match lookups the way `PermissionError` does, and honour Retry-After without the cap or abort when it is larger. Needs Faig's approval to touch scraper.py.

#### MED-25 — zstd decompression stops after the first frame, so a multi-frame .dem.zst is silently truncated and marked processed

**File:** `engine/sources/scraper.py:468` · **Category:** D. Archives · **Confidence:** medium · **Rule:** — · **Also:** `test/engine/test_download.py:63`

**What's wrong**  
python-zstandard 0.25's `stream_reader` defaults to `read_across_frames=False`, so `read()` returns `b''` at the end of frame 1. The output is above `MIN_DEMO_BYTES`, so `download_demo` succeeds and the demo is registered. The content-length check covers only the compressed transfer. test_download compresses with a single `compress()` call, so it cannot see it.

**Evidence**  
```python
# engine/sources/scraper.py:468
                reader = dctx.stream_reader(compressed)
```

**How it fails**  
A `.dem.zst` written by pzstd or by concatenating frames. Whether FACEIT, HLTV or HF archives are multi-frame is unverified.

**Blast radius**  
Silent per-demo data loss; the match is marked processed and never re-downloaded.

**Suggested fix**  
Pass `read_across_frames=True` (or use `copy_stream` with the existing cap) and add a two-frame fixture to test_download. scraper.py is ask-before-touch (R18).

#### MED-26 — HLTV match ids are built from event, date, teams and map only, so two series between the same teams on one day collide and the later one is silently skipped

**File:** `engine/sources/hltv.py:170-171` · **Category:** F. Idempotency · **Confidence:** high · **Rule:** R1 · **Also:** `engine/sources/hltv.py:267`, `test/engine/test_hltv_source.py:194`

**What's wrong**  
The HLTV numeric id is never part of the key. With `skip_seen` the second series is skipped as already in the DB; without it, each colliding map file is unlinked as 'already in db'. If the (unverified) page selectors return None, every match becomes `unknown_unknown_unknown-unknown_<map>` and every match after the first is skipped. Per-match errors are also printed and swallowed inside the generator, so `ingest.run` reports zero failures for a corrupt or 404 archive. None of this is tested.

**Evidence**  
```python
# engine/sources/hltv.py:170-171
def demo_name(info: HltvMatch, map_name: str) -> str:
    return f"{slug(info.event)}_{info.date or 'unknown'}_{slug(info.team1)}-{slug(info.team2)}_{map_name}.dem"
```

**How it fails**  
A GSL opener and decider, or an upper final then grand final, between the same teams on one UTC date.

**Blast radius**  
Whole series or whole runs silently dropped from the pro corpus; failed downloads reported as clean.

**Suggested fix**  
Put the HLTV numeric match id in the stem (`hltv-<id>_<map>`), refuse to name a match when event, date or teams are missing, and surface per-match failures through `ingest.run`'s failed count. Add collision and failure cases to test_hltv_source.

#### MED-27 — HF free-space check runs once per wave on compressed sizes, and a re-run over compressed entries aborts

**File:** `engine/sources/hf_archive.py:234-235` · **Category:** D/H. Resources · **Confidence:** high · **Rule:** — · **Also:** `engine/sources/hf_archive.py:200`, `test/engine/test_hf_archive.py:40`

**What's wrong**  
Waves are planned from `f.size` (compressed for `.dem.zst` LFS entries) and the 20 GB floor is checked only before each wave; compressed and decompressed copies coexist in the system temp dir and ingest copies rather than moves. Separately, `download_waves` compares the listing sha (of the compressed bytes) with the decompressed file on disk and raises `FileExistsError` on any second run over a compressed archive. test_hf_archive uses raw `.dem` entries only.

**Evidence**  
```python
# engine/sources/hf_archive.py:234-235
        if not _space_ok(space_dir):
            return
```

**How it fails**  
`scrape --source hf` starting with 20 to 40 GB free fills the system drive; a resumed `download_waves` over `.dem.zst` entries fails at the first existing file.

**Blast radius**  
System drive full mid-batch; wave downloads cannot be resumed without deleting files by hand.

**Suggested fix**  
Re-check space before each file using an estimated decompressed size; compare the sha only for raw `.dem` entries (as `fetch()` already does). Add a compressed entry and a re-run to the test.

#### MED-28 — The value stage is never timed by the golden benchmark, so the C3 total always omits it

**File:** `bench/bench_analyze.py:85-88` · **Category:** L. Acceptance measurement · **Confidence:** high · **Rule:** —

**What's wrong**  
`analyze_demo` enables the value stage only when `value_dirs` is passed; `measure_golden` never passes it and the bench CLI has no `--q-model` / `--neighbours` options. The value stage is budgeted at 5 s GPU / 20 s CPU and the written JSON does not record that it was skipped.

**Evidence**  
```python
# bench/bench_analyze.py:85-88
    pred = analyze.stub_predictor() if stub else None
    with tempfile.TemporaryDirectory() as tmp:
        res = analyze.analyze_demo(demo, tier="pro", out_dir=Path(tmp) / "report", model_dir=model_dir,
                                   device="cuda" if device == "gpu" else "cpu", cfg=config.load(),
```

**How it fails**  
Every golden run with `--assert` can report a C3 pass on a pipeline that is over the hard ceiling once value runs.

**Blast radius**  
The C3 acceptance number reads low.

**Suggested fix**  
Add the two options, pass them as `value_dirs`, and mark the total 'partial' in the JSON and the verdict when value was not run.

#### MED-29 — A second human-review cycle with the same seed re-selects the same rounds

**File:** `engine/evaluation/review.py:389` · **Category:** B. Logic · **Confidence:** high · **Rule:** —

**What's wrong**  
`main()` takes `--cycle` but never builds `exclude` from the earlier cycle's case files (`read_cases` exists and is unused), and selection is deterministic for a seed. P11.2.4 requires 20 new rounds per cycle.

**Evidence**  
```python
# engine/evaluation/review.py:389
    sel = select_review_rounds(frame, a.n, a.seed)
```

**How it fails**  
`review --cycle 2 --out-root <new>` writes exactly the cycle-1 rounds; into the old root it exits 1 with `FileExistsError`. Only `select_review_rounds(exclude=...)` is tested, never `main` with cycle 2.

**Blast radius**  
The second review cycle silently repeats the first.

**Suggested fix**  
Add `--exclude-from` (or derive it from `--cycle` and the previous root) that loads `read_cases()` and passes the pairs as `exclude`; test `main` across two cycles.

#### MED-30 — Grid cell ids are not pinned: the per-map grid test holds by construction

**File:** `test/engine/test_grid.py:45` · **Category:** K. Vacuous check · **Confidence:** high · **Rule:** —

**What's wrong**  
`grid_for` builds the grid from the same centroid min/max plus a margin, so 'every centroid inside' and 'grid is stable' cannot fail. No concrete `(x0, y0, n_cols, n_rows)` or known `(x, y) -> cell` is recorded per map.

**Evidence**  
```python
# test/engine/test_grid.py:45
        check(f"{m}: every centroid in [0, {n})", ok)
```

**How it fails**  
A change to `CELL_SIZE`, `MARGIN` or a new nav file in the pinned release silently renumbers every `grid_cell` (a model vocabulary input) while the test stays green.

**Blast radius**  
Silent grid-id drift between trained models and new states.

**Suggested fix**  
Record per-map grid parameters for release 2000933 in a fixture and compare, or assert a few known cell ids.

#### MED-31 — The leakage flip audit never shifts end_tick, the source of the historical time_remaining leak

**File:** `engine/state/leakage.py:40` · **Category:** G. Leakage audit strength · **Confidence:** high · **Rule:** R5 · **Also:** `test/engine/test_leakage.py:53`

**What's wrong**  
`flip_outcome` flips winner and end_reason and shifts only `official_end_tick`; `poison()` never touches rounds, grenades or items. The negative-control probe in test_leakage leaks through `official_end_tick`, so it only proves that case is caught. A feature derived from `end_tick` would be identical in both builds and pass.

**Evidence**  
```python
# engine/state/leakage.py:40
        pl.when(is_r).then(pl.col("official_end_tick") + int(round(shift_s * tr)))
```

**How it fails**  
A future feature that reads `rounds.end_tick` directly (the 2026-08-01 bug class) passes the R5 gate. Today the causal slice nulls `end_tick`, so there is no live leak.

**Blast radius**  
The automated audit misses the one leak class the project has already been bitten by.

**Suggested fix**  
Shift `end_tick` too (and the explode-at-end bomb rows), and add a second negative control that derives a feature from `end_tick`.

#### MED-32 — test_states_golden and test_actions_golden treat parse()'s tuple as RawTables and swallow the error as 'prerequisite missing'

**File:** `test/engine/test_states_golden.py:66-71` · **Category:** K. Exit-code semantics · **Confidence:** high · **Rule:** R22 · **Also:** `test/engine/test_actions_golden.py:56`, `engine/parse/awpy3_parser.py:905`

**What's wrong**  
`awpy3_parser.parse()` returns `(tables, meta)`; both scripts then read `raw.meta`, and the blanket `except Exception` turns the AttributeError into `return 2`. They also call `map_version.detect(raw.meta, ...)` without the `build_num` remap analyze.py applies, so the golden map_version would be `<map>@unknown`. Today both correctly exit 2 earlier because GOLDEN.json is unpinned.

**Evidence**  
```python
# test/engine/test_states_golden.py:66-71
        raw = parse(demo)
        a   = map_assets.ensure(raw.meta["map"])
        geo = MapGeometry.from_assets(a, map_callouts.load(raw.meta["map"]),
                                      map_version.detect(raw.meta, raw.meta["map"]))
    except Exception as e:  # missing assets / callouts are prerequisites, not failures
        print(f"  prerequisite missing: {type(e).__name__}: {e}")
```

**How it fails**  
Pin a golden demo: both gates exit 2 forever, run_all shows SKIP, `--write` never runs, and a builder or extractor regression on real data is never caught.

**Blast radius**  
The P3.8.2 / P4.8 golden regression gates are disabled for good, silently.

**Suggested fix**  
Unpack the tuple into `RawTables.from_parsed(tables, meta)`, remap `game_version` to `build_num`, and narrow the except to `MapAssetError` / `FileNotFoundError` so code bugs surface as exit 1.

#### MED-33 — C2's identity V = B + D + X is definitional and every test of it is vacuous

**File:** `engine/value/scores.py:159-161` · **Category:** K. Vacuous check · **Confidence:** high · **Rule:** R21 · **Also:** `test/engine/test_decision_split.py:109`, `test/engine/test_analyze_e2e.py:237`, `CLAUDE.md:155`

**What's wrong**  
`D = Q - B` and `X = V - Q`, so `B + D + X == V` algebraically for any Q and B, including garbage from an untrained net. test_decision_split, its DB copy and the e2e test all assert it; ENGINE_PROGRESS cites 'within 7e-18' as C2 acceptance evidence. The informative halves of C2 (R <= 0 via an independent formula, >= 95% panel coverage) are genuinely tested.

**Evidence**  
```python
# engine/value/scores.py:159-161
    X = V - Q
    if abs(B + D + X - V) > IDENTITY_TOL:
        raise ArithmeticError(f"identity V = B + D + X off by {B + D + X - V}")
```

**How it fails**  
Every run: green regardless of Q/B quality.

**Blast radius**  
False confidence that the value stage is consistent end to end.

**Suggested fix**  
Label the identity a plumbing guard in CLAUDE.md and the tests, and add a check that can fail (Q/B against planted truth, or calibration of Q against realized V).

#### MED-34 — Nothing produces shrinkage.json, so real reports never carry shrunk per-player D/X

**File:** `engine/value/shrinkage.py:109` · **Category:** L. Unwired stage · **Confidence:** high · **Rule:** — · **Also:** `engine/analyze.py:192`, `docs/ENGINE_PROGRESS.md:72`

**What's wrong**  
`fit_k` / `save` are called only from tests. `train-q` and every other command never write `q_v<N>/shrinkage.json`, so `analyze` always has `k_map = None`, `players[].by_class` is empty, the markdown D~/X~ columns show n/a and the player chart falls back. ENGINE_PROGRESS and ENGINE_CLI present the output as delivered.

**Evidence**  
```python
# engine/value/shrinkage.py:109
def fit_k(frame: pl.DataFrame, by: str = "type_class", *, values=VALUES, eps: float = EPS) -> dict:
```

**How it fails**  
Every production report.

**Blast radius**  
P9.4.2 per-player estimates never appear; a reader of the progress log believes P9.5 is end to end.

**Suggested fix**  
Fit and save shrinkage in `train-q` (on the train split) or add a command, and list it in ENGINE_OPS until then.

#### MED-35 — Knife rounds cannot be excluded: awpy 3.1 has no is_knife_round column, so the knife plumbing is dead and any knife round in the rounds table is sampled, labelled and scored

**File:** `engine/parse/awpy3_parser.py:345` · **Category:** J. CS2 domain / F. round scope · **Confidence:** needs-domain-review · **Rule:** — · **Also:** `engine/state/builder.py:333`, `docs/ENGINE_ENV.md:91`

**What's wrong**  
The installed awpy 3.1 exposes no `is_knife_round` (its rounds frame is reconstructed from game-rules state), so `_col` yields an all-null column, `meta['knife_rounds']` is always empty and the `_scores` knife skip never runs. `conform()` drops `is_knife` and nothing downstream reads it; the builder iterates every round and `half_of` / `is_overtime` use the raw round number. ENGINE_ENV.md says knife rounds score nothing, which is not true under awpy 3.1.

**Evidence**  
```python
# engine/parse/awpy3_parser.py:345
        _col(r, "is_knife_round", dtype=BOOL).alias("is_knife"),
```

**How it fails**  
If awpy's rounds table includes a knife round (common in pro/HLTV demos; unverifiable without a demo), its kills get WP deltas and training rows, and if it is numbered round 1 the half and overtime flags shift by one for every later round. Raise to HIGH if a real pro demo shows a numbered knife round.

**Blast radius**  
Pro-tier datasets and reports.

**Suggested fix**  
Detect knife rounds from the data (a round where both teams hold only knives at freeze end, or the round before the first full-buy restart), drop them from `rounds` in `parse()`, renumber, and keep the list in meta. Add a fake-demo test. Correct ENGINE_ENV.md.

#### MED-36 — The anchors report falls back to the raw SteamID64 when a ratings row has no player_name

**File:** `engine/evaluation/anchors.py:490` · **Category:** I. Privacy / report hygiene · **Confidence:** high · **Rule:** R14 · **Also:** `test/engine/test_anchors.py:178`

**What's wrong**  
`report_markdown` writes the `player` field into the 'Biggest disagreements' table of `data/engine/eval/anchors_<model>.md`; the CSV loader drops only all-blank rows; there is no `--show-ids` gate on this path. The join key is hashed, the display value is not.

**Evidence**  
```python
# engine/evaluation/anchors.py:490
            pairs.append({"match_key": key[0], "player": r.get("player_name") or str(r["steamid64"]),
```

**How it fails**  
Fill hltv_ratings.csv by hand and leave one name blank: the raw ID is printed into a file P12.5 lists in the buyer hand-off. Local and gitignored today, and the CSV already carries the ID, hence MEDIUM.

**Blast radius**  
Raw SteamIDs in a shareable report.

**Suggested fix**  
Fall back to `player_id_hash(...)` (the `p_<hex>` form review.py uses), never the raw ID; escape `|` in names. Add a blank-name row to test_anchors.

#### MED-37 — hltv_ratings.csv is a tracked fixture that the docs tell Faig to fill with real SteamIDs and nicknames

**File:** `test/engine/fixtures/hltv_ratings.csv:1` · **Category:** A. PII headed for the public repo · **Confidence:** high · **Rule:** — · **Also:** `docs/ENGINE_OPS.md:103`, `docs/ENGINE_PLAN.md:1684`, `engine/evaluation/anchors.py:587`

**What's wrong**  
The file is in `git ls-files` (header only today), not gitignored, and is the default `--hltv-csv`. ENGINE_OPS and ENGINE_PLAN P11.1.5 say to fill it by hand for 50 or more test matches. The `.gitignore` rationale for `data/` is precisely to keep player names and SteamIDs out of this public repo.

**Evidence**  
```
# test/engine/fixtures/hltv_ratings.csv:1
match_key,map,steamid64,player_name,hltv_rating
```

**How it fails**  
Follow the documented procedure, then `git add -A`: SteamID64s and nicknames of every player in 50+ pro matches land in public history. Nothing has leaked yet, and the pre-commit security review is a backstop, hence MEDIUM not CRITICAL.

**Blast radius**  
Player PII in public git history, irreversible.

**Suggested fix**  
Default the path to `data/engine/eval/hltv_ratings.csv` (gitignored) and keep only a header-only template under test/, or key the CSV by `player_id_hash` with no names. Update ENGINE_OPS and the plan.

#### MED-38 — train-q and build-neighbours accept input with no split and no warning, so Q/B and the panel can learn from test months silently

**File:** `engine/cli.py:463` · **Category:** G. Split hygiene · **Confidence:** high · **Rule:** R4 · **Also:** `engine/cli.py:467`, `docs/ENGINE_CLI.md:194`

**What's wrong**  
`--split-file` is optional and ENGINE_CLI.md documents 'without it every row is train', trusting the caller to pass a train-only parquet (the example is named `*_train.parquet`), but no command produces such a file and nothing records in the manifest that no split was applied. `train_q` does not re-split and, with no val rows, early-stops on training loss. `DescriptorVocab.fit_frame` always runs on the whole frame before the split (no label leak; held-out-only tokens get untrained embedding rows instead of UNK).

**Evidence**  
```python
# engine/cli.py:463
            return ["train"] * f.height
```

**How it fails**  
Pass a valued parquet that includes test months (the natural thing to do, since nothing writes a train-only one): the P9.2.3 evaluation and every regret number on held-out matches are contaminated with no message.

**Blast radius**  
The decision/execution split's held-out numbers.

**Suggested fix**  
Require `--split-file` (or default to the latest `splits_v<N>.json` and refuse when absent), or warn loudly and write `split: null` into the manifest; fit the vocab on train rows only.

#### MED-39 — P6.7.2's hazard-head rule is not implemented: no 'below target' warning, no automatic fixed-share fallback, hazard AUC optional at adoption

**File:** `engine/evaluation/registry.py:101` · **Category:** Decision rules · **Confidence:** high · **Rule:** R19 · **Also:** `engine/analyze.py:579`, `docs/ENGINE_PLAN.md:1183`

**What's wrong**  
`sanity.hazard_auc_target` / `hazard_auc_floor` are read only by config validation. Nothing compares a model's hazard AUC with them; analyze warns only when `enabler_mode` was set to fixed by hand; `rule_problems` checks the log-loss margin and ECE only, and `hazard_auc` is optional.

**Evidence**  
```python
# engine/evaluation/registry.py:101
def rule_problems(name: str, numbers: dict, ece_max: float = ECE_MAX) -> list[str]:
```

**How it fails**  
A wp model with hazard AUC 0.6 is adopted and analyze uses hazard-mode enabler splits with no warning.

**Blast radius**  
Every enabler share (flash, smoke, trade, HE, fire) in reports from a weak hazard head.

**Suggested fix**  
Store hazard AUC with the model (train.py computes it), have analyze warn between floor and target and force fixed mode below the floor, and require `hazard_auc` for wp adoption.

#### MED-40 — ENGINE_PROGRESS marks P6, P8 and P9 done while three gated commands and the valued-actions producer do not exist

**File:** `docs/ENGINE_PROGRESS.md:11` · **Category:** L. Plan steps not delivered · **Confidence:** high · **Rule:** — · **Also:** `engine/cli.py:66`, `engine/cli.py:61`, `engine/attribution/engine.py:440`

**What's wrong**  
`engine/model/sanity.py` (P6.6.1 `sanity --model`), `engine/attribution/evaluate.py` (P8.6.3 `eval --attribution`, the 1,000-round conservation check that is half of C1) and a `main()` for `eval --value` (P9.2.3) do not exist; `delegate` exits 2. Nothing writes `actions_valued.parquet` with `h_pre` for a corpus (the engine declares the column 'filled by P9; None here'), yet `train-q` and `build-neighbours` require that file, and ENGINE_PROGRESS's next steps present them as runnable. ENGINE_OPS does list the missing commands; the phase status lines and the summary ('every phase has its code') do not.

**Evidence**  
```
# docs/ENGINE_PROGRESS.md:11
- **Every phase P0–P12 has its code and tests.** Full suite: `test/run_all.py` → **89 passed, 0 failed, 3 skipped** (the three golden-demo tests), 258 s; pyflakes clean across `engine/`, `test/`, `bench/`.
```

**How it fails**  
Follow the next-steps list: steps 6 and 7 cannot run, and the P6/P8/P9 definitions of done cannot be met.

**Blast radius**  
Misdirected planning; the C1 conservation half has no implementation.

**Suggested fix**  
Mark P6.6.1, P8.6.3, P9.1.2, P9.2.3 and P9.4.2 as not built in the phase table, or build them: a corpus-level valued-actions writer, the three eval/sanity modules, and a shrinkage fit in train-q.

#### MED-41 — The C1 bound is stated two ways and the real-model variant does not exist

**File:** `docs/ENGINE_PLAN.md:58` · **Category:** J. Acceptance test · **Confidence:** high · **Rule:** R21 · **Also:** `docs/ENGINE_PLAN.md:1344`, `test/engine/test_attribution_shapley.py:34`

**What's wrong**  
Plan 1.3 and CLAUDE.md demand `V_A == 0 ± 1e-9` on the real model; P8.2.4 says the real-model bound is 0.002 and exact zero holds only with a stub. test_attribution_shapley's `--real` path is a hard-coded exit 2 that never loads a model, the default run passes on two stubs, and ENGINE_PROGRESS reports 'C1 both variants' as done.

**Evidence**  
```
# docs/ENGINE_PLAN.md:58
| C1 | No faulty result caused by two actions at the same tick. An action with zero effect must get exactly zero even if a teammate's −0.5 action lands on the same tick. | `test/engine/test_attribution_shapley.py::null_action_same_tick` builds the exact A/B scenario on the real model and asserts `V_A == 0 ± 1e-9` and `V_B == ΔW ± 1e-9`. Also: conservation per round `|Σ V − (W_end − W_start)| < 1e-6` on every round of the golden demo and on 1,000 sampled pro rounds. |
```

**How it fails**  
A green run_all reads as C1 met; a future real-model run would check a bound two million times looser than CLAUDE.md states, if it ran at all.

**Blast radius**  
C1 can be declared met without real-model evidence. With HIGH-03 unfixed, even the 0.002 bound will fail.

**Suggested fix**  
Settle one bound in plan 1.3 / CLAUDE.md, implement `--real` through the registry (exit 2 only when no adopted model exists), and reword the progress line.

#### MED-42 — The quarantine rule '>= 16 rounds' refuses legitimate MR12 maps that end 13-0, 13-1 or 13-2

**File:** `engine/parse/awpy3_parser.py:125` · **Category:** J. CS2 domain · **Confidence:** needs-domain-review · **Rule:** — · **Also:** `docs/ENGINE_PLAN.md:474`

**What's wrong**  
Under MR12 a regulation map ends at 13 wins, so 13-0 through 13-2 maps have 13 to 15 rounds. The code implements the plan literally; Appendix H does not list the question.

**Evidence**  
```python
# engine/parse/awpy3_parser.py:125
MIN_ROUNDS         = 16
```

**How it fails**  
`analyze` refuses such a pro demo with `too_few_rounds`; once the batch gate is wired they are excluded from training too.

**Blast radius**  
One-sided pro maps, often the most informative extremes, cannot be analysed or trained on.

**Suggested fix**  
Accept a completed MR12 map (one side reached 13, or overtime finished) after confirming how awpy counts rounds. Faig's decision.

#### MED-43 — B.8 shrinks D toward 0 although D's per-class mean is not 0, and the open question is not in Appendix H

**File:** `engine/value/shrinkage.py:19` · **Category:** Formula · **Confidence:** high · **Rule:** — · **Also:** `docs/ENGINE_PLAN.md:1557`, `docs/ENGINE_PROGRESS.md:74`

**What's wrong**  
D = Q − B has a nonzero class mean (kills about +0.07 per ENGINE_PROGRESS). The docstring calls 0 'the population mean of X and D by construction', and `shrink()` pulls D toward 0, so a kill-heavy player's D̃ grows with n and measures volume, while σ²_skill is computed around the class mean. ENGINE_PROGRESS tells Faig to answer 'Appendix H ... D̃ shrink target', but Appendix H has no such entry.

**Evidence**  
```python
# engine/value/shrinkage.py:19
Shrinkage is toward 0, the population mean of X and D by construction.
```

**How it fails**  
Once shrinkage is fitted (nothing fits it today), every per-player D̃ is biased by kill count.

**Blast radius**  
Per-player decision estimates in reports.

**Suggested fix**  
Record the question in Appendix H; most likely shrink D toward the class mean. Fix the docstring. Faig's call.

### LOW

| ID | Title | File | Fix |
|---|---|---|---|
| LOW-01 | A Downloads-API 403 (key without Downloads scope) is caught per match, so the run keeps POSTing for every remaining match | `engine/ingest.py:246` | Re-raise auth-class errors from the per-match handler. |
| LOW-02 | redact_text is narrower than the legacy blanket '?query' strip: scheme-less non-.dem paths and upper-case schemes keep their query | `engine/log.py:33` | Add a generic scheme-less path?query rule, make _URL_RE case-insensitive, extend test_log_redaction. |
| LOW-03 | --debug prints the traceback without redaction | `engine/cli.py:701` | Print redact_text(''.join(traceback.format_exception(exc))). |
| LOW-04 | Exit code 130 on Ctrl-C and raw bench/delegate return codes contradict the 0/1/2 contract (CLAUDE.md:'Exit codes everywhere') | `engine/cli.py:690` | Clamp to 0/1/2 or amend the rule and R22. |
| LOW-05 | Generic integer validation rejects 0 for qa.max_outliers / train.seed; a 0 s martingale horizon is accepted (vacuous check); test_qa asserts max_outliers=0 via dataclasses.replace | `engine/config.py:374` | Allow 0 where it is legitimate; require horizons > 0 and distinct; route test_qa through from_dict. |
| LOW-06 | Duplicate YAML keys are silently last-wins | `engine/config.py:442` | Load with a SafeLoader subclass that raises ConfigError on duplicates. |
| LOW-07 | build-states/build-actions --all drop the CLIError message and only count failures | `engine/cli.py:390` | Log the message before counting. |
| LOW-08 | The error handler imports engine.analyze and can raise again, printing a raw chained traceback | `engine/cli.py:692` | Guard the import or move AnalyzeError to a dependency-free module. |
| LOW-09 | min_level issues a second GET /matches/{id} per candidate and re-queries rejected matches every run | `engine/sources/faceit_api.py:136` | Fetch /matches/{id} once and derive both values. |
| LOW-10 | HLTV rows store a temp file:// URI (with the Windows user name) as processed.demo_url; the real HLTV URL is lost | `engine/sources/hltv.py:282` | Store info.url (redacted) via an extra_meta hook. |
| LOW-11 | hf match_id_for is not injective ('a/b' vs 'a-b', 'x.dem' vs 'x.dem.zst'); the second file is silently skipped | `engine/sources/hf_archive.py:215` | Append a short hash of the full path. |
| LOW-12 | faceit_10 demos live in demos/faceit while CLAUDE.md, DATA_SOURCES.md and ENGINE_CLI.md say demos/<tier> | `engine/paths.py:42` | Correct the docs or rename the layout entry with a file migration. |
| LOW-13 | Persistent context user_data_dir is the real Chrome Default profile (legacy, unchanged, macOS paths) | `engine/sources/scraper.py:154` | Use a throwaway profile under data/ after a one-time FACEIT login. Ask-before-touch. |
| LOW-14 | _scores skips the win increment for a round with no roster, so later score_before values are confidently off by one | `engine/parse/awpy3_parser.py:863` | Mark score state unknown from then on or count by the next known roster. |
| LOW-15 | Blind-to-flash linking picks the thrower's latest throw before the blind, so a quick double flash mislinks (needs-domain-review) | `engine/parse/awpy3_parser.py:644` | Link by nearest detonation tick instead of latest throw. |
| LOW-16 | _SEGMENT_GAP (8 ticks) and the defuse-complete tolerance (4 ticks) are tick literals rather than seconds | `engine/parse/awpy3_parser.py:135` | Express in seconds and scale by meta tick_rate as _LINK_TOL_S does. |
| LOW-17 | int(round(raw)) sits outside the try in _tick_rate_from_header, so a NaN playback_time raises instead of returning None | `engine/parse/awpy3_parser.py:227` | Move the rounding inside the guarded block. |
| LOW-18 | Demo-header map name is joined into asset file paths unvalidated on the worker path (read-only, fixed suffixes) | `engine/maps/assets.py:67` | Reject names outside ^[a-z0-9_]+$ or ACTIVE_POOL in ensure(). |
| LOW-19 | engine/maps/plot.py has no test script and its __main__ exits 1 with a traceback on missing assets | `engine/maps/plot.py:62` | Add test_map_plot.py; catch MapAssetError and exit 2. |
| LOW-20 | MIN_ROUNDS 16 refuses legitimate MR12 13-0..13-2 results (plan P1.4.1 as written; Faig's decision) | `engine/parse/awpy3_parser.py:125` | Decide whether 13 is the floor. |
| LOW-21 | Score fallback assumes a side swap at the start of overtime (needs-domain-review; only when parser scores are null) | `engine/state/context.py:97` | Encode the CS2 overtime side rule explicitly. |
| LOW-22 | MapGeometry.is_visible / find_area swallow every exception, so an awpy API mismatch silently yields all-NaN LOS | `engine/state/geometry.py:108` | Catch only the out-of-mesh error, log the first failure per process. |
| LOW-23 | decoy_end pseudo-action is emitted at a nominal end while the state keeps the token alive (covered by the decoy MEDIUM) | `engine/actions/utility.py:240` | One lifetime rule in both modules. |
| LOW-24 | Window bounds hard-code ±1 instead of sampling.boundary_offset_ticks (config allows >= 1) | `engine/actions/windows.py:78` | Pass the offset into group() or pin it to 1 in validation. |
| LOW-25 | Loadout is read at freeze_end, so buys during remaining buy time are missed (plan P4.4.1 as written; needs-domain-review) | `engine/actions/economy.py:79` | Read at freeze_end + buy time clipped to the first kill, or record both. |
| LOW-26 | remove_utility's decrement of the capped active-utility count is overwritten by recompute_derived (>16 tokens only) | `engine/counterfactual/interventions.py:517` | Take the count from the pre-pop list or skip the overwrite when the caller adjusted it. |
| LOW-27 | enablers.split divides by q_pre with no zero guard; config allows attribution.min_hazard = 0 | `engine/attribution/enablers.py:154` | Add min_hazard to _STRICTLY_POSITIVE or treat q_pre <= 0 as the fallback case. |
| LOW-28 | In a {damage by A, kill by B} window apply(empty) leaves A's damage_round inflated (undo order) | `engine/counterfactual/interventions.py:356` | Roll back same-window damage members in undo_kill. |
| LOW-29 | Calibration group keys differ between fit (vocab inverse '<unk>'/'<null>') and Predictor (raw string / 'None') in group mode | `engine/model/predict.py:181` | Derive keys through the vocab in both places. |
| LOW-30 | LightGBM yardstick early-stops on all-tier val while the NN uses val pro, skewing the P6.7.2 comparison | `engine/model/baseline_lgbm.py:143` | Restrict the baseline's val to pro when pro rows exist. |
| LOW-31 | Baseline team sums turn an alive player's unknown equipment/helmet/defuser into 0 (R7, yardstick only) | `engine/model/baseline_lgbm.py:86` | Make the sum NaN when any alive value is unknown. |
| LOW-32 | baseline_lgbm and neighbours.save_index import torch-bearing modules only for next_version_dir | `engine/model/baseline_lgbm.py:224` | Move next_version_dir (and h_matrix) to a torch-free module. |
| LOW-33 | dataset.build writes vocab_vN before the shard existence check and saves norm_vN without exclusive create | `engine/data/dataset.py:637` | Check shard dirs first; take the version as max over vocab/norm; write with mode 'x'. |
| LOW-34 | write_dir_manifest re-hashes every parquet in the month directory on each single-match write (quadratic) | `engine/data/layout.py:102` | Reuse hashes for unchanged files or write once per directory at the end of --all. |
| LOW-35 | splits.assign duplicate check is O(n^2); make_splits has no CLI entry point | `engine/data/splits.py:54` | Use Counter; add a `splits` command (ENGINE_OPS already lists it as missing). |
| LOW-36 | train keeps every step's loss in memory for the whole run | `engine/model/train.py:297` | Keep only under max_steps or cap it. |
| LOW-37 | The markdown player table does not escape '|' in names (--show-ids runs; needs-domain-review) | `engine/report/writer.py:228` | Escape '|' and strip newlines in table cells. |
| LOW-38 | GPU bench scripts end with sys.exit instead of the documented HIP-teardown workaround; result files are date-only and overwrite same-day runs | `bench/bench_gpu.py:111` | End with _fake_states.finish(); put device/stub in the file name. |
| LOW-39 | Opening-kill anchor counts kills with unknown alive counts as 5v5 | `engine/evaluation/anchors.py:284` | Skip unknowns and count them. |
| LOW-40 | Adoption rewrites docs/ENGINE_MODELS.md in place without an atomic write | `engine/evaluation/registry.py:157` | Write a temp file and os.replace. |
| LOW-41 | usage.py docstring and ENGINE_OPS.md say analyze does not call append_usage; analyze.py does | `engine/evaluation/usage.py:20` | Fix the docstring and ENGINE_OPS.md:88-91, :150. |
| LOW-42 | run_all does not set CROSSHAIR_ENGINE_DB for child tests and forwards a relative demo path unresolved; SKIP column shows the tally line | `test/run_all.py:63` | Set a per-run temp DB in env; resolve the demo path; show the '(skipped:' line. |
| LOW-43 | Repo-root eval/ is not gitignored but sanity's default --out and ENGINE_OPS paths write there | `engine/counterfactual/sanity.py:384` | Default to paths.EVAL_DIR or ignore eval/. |
| LOW-44 | scripts/scrape_watchdog.sh is dead: it greps for and relaunches the deleted src/main.py with flags engine.cli lacks | `scripts/scrape_watchdog.sh:27` | Delete (ask first) or rewrite for engine.cli scrape. |
| LOW-45 | README Setup / Getting started / Full CLI / layout describe the deleted src/ pipeline; 'MIT' with no LICENSE file | `README.md:113` | Replace with v2 commands; add LICENSE or drop the claim. |
| LOW-46 | pyproject package-data lists only config.yaml, so a non-editable install ships no schema.json / vocab_weapons.yaml / versions.yaml / callouts.json; pyflakes is a runtime dep; the console script is not installed | `pyproject.toml:26` | Declare all data files or document repo-only use; move pyflakes to a dev extra. |
| LOW-47 | requirements-engine.lock pins Windows+AMD cp312 torch/rocm URLs, so it is not installable elsewhere; CLAUDE.md says torch is not in the requirements | `requirements-engine.lock:59` | Document the lock as Windows+AMD only or split a platform lock. |
| LOW-48 | .env.example asks for FACEIT_EMAIL / FACEIT_PASSWORD that nothing reads | `.env.example:7` | Remove the two lines. |
| LOW-49 | CLAUDE.md and DATA_SOURCES.md promise is_processed_any checks _m1, _m2, ...; db.id_forms checks only _m1 for a bare id, by design | `CLAUDE.md:190` | State the real contract (and UNIQUE demo_sha256 as backstop) or change id_forms. |
| LOW-50 | ENGINE_OPS.md 'missing' table and notes contradict the code: anchors path is right, train-q/build-neighbours exist, append_usage is wired | `docs/ENGINE_OPS.md:150` | Remove the stale rows and rewrite :42-43, :88-91, :106-108; same in ENGINE_PROGRESS.md:86. |
| LOW-51 | ENGINE_PROGRESS.md gives two commands that do not parse (scrape --engine-copy; analyze without the demo positional) | `docs/ENGINE_PROGRESS.md:19` | Drop the flag; write the analyze line as ENGINE_CLI.md:134 does. |
| LOW-52 | ENGINE_MODELS.md says only a status 'adopted' row counts; ENGINE_CLI.md says the file does not exist and suggests a 4-column table | `docs/ENGINE_CLI.md:337` | Point at the registry and the 10-column table. |
| LOW-53 | ENGINE_PROGRESS.md stale facts: 'Nothing is committed', '16 commands'; CLAUDE.md links a renamed section | `docs/ENGINE_PROGRESS.md:13` | Update the summary lines. |
| LOW-54 | DATA_SOURCES.md lists HF with tier '—' but ingest maps hf -> pro; scrape --source hf cannot pass a repo and exits 1 | `docs/DATA_SOURCES.md:19` | Give the real default tier and the CLI limitation. |
| LOW-55 | Audit skill files still describe the legacy src/ layout and do not block data/engine/engine.db, data/engine/demos or .venv-old311 | `.claude/skills/full-codebase-check/SKILL.md:41` | Add data/** and .venv-old311/** to the blocked table; scope engine/, test/, bench/; replace the P-table with the R-table. |
| LOW-56 | test_registry asserts the real ENGINE_MODELS.md has no adoption rows (fails the first legitimate adoption) and has a literal check(True) | `test/engine/test_registry.py:53` | Restrict the token check to prose lines; make the allow-unadopted check real. |
| LOW-57 | test_review_items placeholder only checks that a function is callable | `test/engine/test_review_items.py:23` | Parse a tiny fixture log until real items exist. |
| LOW-58 | test_report_redaction prints raw leaked strings in failure details | `test/engine/test_report_redaction.py:53` | Print keys and leak kinds only. |
| LOW-59 | test_anchors never has a blank player_name, so the raw-SteamID fallback is untested | `test/engine/test_anchors.py:178` | Add a blank-name row and assert a hashed label. |
| LOW-60 | test_decision_split hard-codes V=0.02 against delta_X read from config | `test/engine/test_decision_split.py:44` | Build the action with V=dX. |
| LOW-61 | Torch/faiss-importing test scripts crash with ImportError (ERROR) instead of exiting 2 when the platform torch build is absent (10 scripts) | `test/engine/test_augment.py:11` | Guard the import in _fake_states or each script, print why, sys.exit(2). |
| LOW-62 | test_cli_errors refusal cases omit --out, so a guard regression would write data/engine/reports/syn | `test/engine/test_cli_errors.py:62` | Pass --out TMP on every analyze invocation. |
| LOW-63 | _synthetic always fills spotted_by (parser emits null) and three STORY_FACTS extras are never read; test_synthetic hard-codes the same values | `test/engine/_synthetic.py:117` | Add a spotted_by=None variant; assert against STORY_FACTS. |
| LOW-64 | _fake_states pair/context fixtures (smoke_between 0.0 for dead pairs, money alive-only) and test_state_schema's pair table differ from the builder | `test/engine/_fake_states.py:208` | Use NaN for dead pairs and sum all accounts. |
| LOW-65 | test_map_assets 'hashes stable across calls' is vacuous (cache); test_parse_worker '--force rewrote the manifest' uses >= so an unchanged manifest passes | `test/engine/test_map_assets.py:38` | Clear the cache or drop it; assert strict inequality. |
| LOW-66 | test_raw_schema docstring points to a nonexistent test_raw_golden.py | `test/engine/test_raw_schema.py:4` | Add the exit-2 placeholder or reword. |
| LOW-67 | test_awpy3_parser.check_rawtables turns AttributeError from RawTables.from_parsed into a skip (exit 0) | `test/engine/test_awpy3_parser.py:191` | Catch ImportError only; schema.py exists. |
| LOW-68 | test_temporal 'editing the next snapshot leaves the prediction unchanged' never gathers the edited row | `test/engine/test_temporal.py:99` | Put the future snapshot inside the gathered window or delete the check. |
| LOW-69 | test_attribution_round's V identity and V_raw sign checks are definitional; w_start is never pinned to an independent W(freeze_end) | `test/engine/test_attribution_round.py:98` | Assert ra.w_start == round_win(snapshot at freeze_end). |
| LOW-70 | Vacuous range checks on sigmoid/booster outputs and a loop-bounded step count (test_baseline, test_predict, test_heads, test_overfit_tiny) | `test/engine/test_baseline.py:49` | Replace with checks that can fail. |
| LOW-71 | test_actions_economy re-implements the team_class tie-break inside the test | `test/engine/test_actions_economy.py:72` | Build a tied team and assert team_buy params['team_class']. |
| LOW-72 | test_dataset 'callouts keyed per map version' has an `or k1 == UNK` escape and a conditional check that silently disappears | `test/engine/test_dataset.py:122` | Assert on the fitted table; add a skip() else-branch. |
| LOW-73 | test_download size-cap test is satisfied by the Content-Length pre-check; rejection cases accept any Exception; stale src/ sys.path insert | `test/engine/test_download.py:174` | Add no-Content-Length and decompression-bomb routes; catch IOError and assert the message; drop the src insert. |
| LOW-74 | test_log_redaction covers only .dem-suffixed scheme-less paths and lower-case schemes | `test/engine/test_log_redaction.py:36` | Add upper-case scheme and non-.dem path?query cases (they fail until log.py is fixed). |
| LOW-75 | test_config never loads malformed or duplicate-key YAML through load() | `test/engine/test_config.py:127` | Add load() cases on temp files. |
| LOW-76 | test_scraper.py imports `scraper` from the deleted src/ path, so the manual Playwright verification tool cannot run | `test/engine/test_scraper.py:21` | Import engine.sources.scraper via the harness path setup. |
| LOW-77 | test_hf_archive uses the real 20 GB free-space floor on TMP (false FAIL on a full disk); mkdtemp dirs in ten tests are never removed | `test/engine/test_hf_archive.py:174` | Monkeypatch check_space; remove TMP at exit. |
| LOW-78 | test_dataset, test_qa and test_labels run all checks at module level; test_matches_meta's UNIQUE check is a whitespace snapshot; test_env only checks Python >= 3.12 | `test/engine/test_matches_meta.py:56` | Wrap in main(); normalise whitespace; check sys.prefix. |
| LOW-79 | test_predict's bf16 tolerance check swings 40x with GPU load (1.1e-3 alone vs 4.1e-2 under contention): unpinned ROCm nondeterminism; test_train_loop's target_failures check only asserts a list type | `test/engine/test_predict.py:126` | Pin deterministic algorithms and compare a p99 on the logit scale; assert target_failures on ece 0.019 vs 0.021. |

## Rule compliance

Rules are the Engine v2 section of CLAUDE.md, numbered R1–R22 as in the audit brief (the legacy P1–P15 table in the skill describes deleted code).

| Rule | Statement | Status | Evidence |
|---|---|---|---|
| R1 | Every source ends at `register_demo`; `is_processed_any` checks both id forms; never double-insert; UNIQUE `demo_sha256` | ⚠ UNENFORCED | Seam and transaction hold (ingest.py, db.py `BEGIN IMMEDIATE`, UNIQUE sha). Gaps: MED-06 (existing key, different sha returns success and keeps an orphan copy), MED-26 (HLTV id collisions skip matches silently), LOW-49 (docs promise `_m2…` checks the code does not make, by design) |
| R2 | All sources produce identical downstream rows; expose `iter_unprocessed_demos` / `download_demo` | ✓ HOLDS | All four sources expose both and reach `ingest.process_one`; HLTV stores a temp `file://` URI as `demo_url` (LOW-10); no cross-source parity test exists |
| R3 | CT-positive sign convention, flipped once per T actor | ✓ HOLDS | `attribution/engine.py` `_Acc.scored` applies `actor_sign` once; `types.py` `V_raw`; value stage actor-signed throughout; test_attribution_round.py:87-92 and test_attribution_shapley.py:153-168 check it independently |
| R4 | Frozen month splits with match grouping; never a random row split; split files superseded, never edited | ⚠ UNENFORCED | `splits.make_splits` writes with mode `x`, per match_key, latest months test (holds). `train-q` / `build-neighbours` accept no split at all with no warning (MED-38) |
| R5 | Every feature has a LIVE_SAFE justification; leakage audit | ⚠ UNENFORCED | 79 LIVE_SAFE keys, `missing_live_safe() == []`, no feature reads end_tick/winner. The flip audit never shifts `end_tick` and `poison()` skips rounds/grenades/items (MED-31) |
| R6 | Calibrated probabilities, ECE ≤ 0.02 on pro test | ⚠ UNENFORCED | Calibration fits on val pro and reports on test pro (holds). The Predictor serves an uncalibrated sigmoid without `calibration.json` and adoption accepts typed numbers (MED-20, MED-20); no test asserts `target_failures` |
| R7 | Never 0 / "?" for unknown | ✗ VIOLATED | HIGH-04 (`on_fire = 0` for every player on real demos); MED-13 (null health → dead); LOW-31 (baseline team sums). Everywhere else unknowns are None/NULL/NaN |
| R8 | `or` is not a null-coalesce over numerics | ✗ VIOLATED (latent) | MED-04: `analyze.check_tick_rate` and `awpy3_parser.quarantine_check` use `tick_rate_raw or tick_rate`; no reachable effect today. All other sites checked are over None-or-value, strings or sort keys |
| R9 | Credential-bearing URLs redacted before any print/exception/DB write; never truncate by length | ⚠ UNENFORCED | Every source redacts, `from None` suppresses signed-URL context, DB writes are redacted twice. Gaps: `writer.find_leaks` prints `s[:80]` (MED-23), `--debug` traceback unredacted (LOW-03), `log.redact_text` misses upper-case schemes and non-`.dem` scheme-less queries (LOW-02) |
| R10 | Tick rate from the header, never a literal; quarantine refuses non-128 | ✗ VIOLATED | Tick rate is read from the header everywhere (holds). Refusal happens only in `analyze` and `parse --smoke`; the batch path never quarantines (HIGH-01) |
| R11 | No invented FACEIT API fields | ? UNVERIFIABLE | Endpoints and `roster[].game_skill_level` match the FACEIT docs as known; not checked against a live response |
| R12 | Additive migrations keyed by `user_version`; newer file refused | ✓ HOLDS | `db.py` SCHEMA_VERSION 1, `_MIGRATIONS`, newer-file refusal; test_db migrates and refuses |
| R13 | Never change `sample_hz` for latency | ✓ HOLDS | `config.yaml` `sample_hz: 2`; bench and attribution touch only `los_cache` / `drift_threshold_wp` / `shapley_*` / `k_neighbours`; `{1,2,4,8}` allowed for ablations only |
| R14 | Reports never contain outside paths or URLs; SteamIDs hashed unless `--show-ids` | ✗ VIOLATED | `report.json`/`report.md` are scrubbed and `--show-ids` is off by default (holds). `manifest.json` in the same folder carries the absolute demo path and hostname (MED-07); path redaction stops at the first space (MED-22); the anchors report prints a raw SteamID64 on a blank name (MED-36) |
| R15 | Non-training modules never import torch at import time | ✓ HOLDS | torch at module level only in `model/{tokens,set_encoder,temporal,heads,wp_model,checkpoints,augment,train,predict}` and `value/{q_model,scores}`; `analyze`/`cli`/`manifest` import lazily. `baseline_lgbm` and `neighbours.save_index` pull torch in for a path helper (LOW-32) |
| R16 | Plain-script tests, exit 0/1/2, write only to temp dirs, every module has a test | ⚠ UNENFORCED | No test writes under `data/`, every DB test sets `CROSSHAIR_ENGINE_DB` first, `run_all` excludes test_scraper. Gaps: golden tests report a code bug as 'prerequisite missing' (MED-32), ten scripts crash on a missing torch instead of exit 2 (LOW-61), `maps/plot.py`, `report/cards.py`, `report/charts.py`, `attribution/clock.py` have no own script, `test_scraper.py` imports a deleted path (LOW-76) |
| R17 | Secrets via env + `.env`; never hardcoded or committed | ✓ HOLDS | Only `FACEIT_API_KEY` is read, from env; `.env*` ignored with `!.env.example`; placeholders only; no key-shaped string anywhere in the tree or history. `.env.example` asks for two unused credentials (LOW-48) |
| R18 | `scraper.py` logic unchanged; ask before touching | ✓ HOLDS | `git diff 32b1373:src/scraper.py HEAD:engine/sources/scraper.py` is the one import line; `faceit_api.py` differs by imports plus the authorised additive `min_level` (plan P1.2.1), which Amendment 1 under-describes (LOW) |
| R19 | A model is used only if adopted; `analyze` refuses otherwise; P7 gate before attribution | ✗ VIOLATED | HIGH-02 (token-scan adoption), HIGH-03 (value models unchecked), MED-20 (adoption without calibration or gate), MED-02 (basename only), MED-16 (gate always exits 2) |
| R20 | Style: absolute imports, banners, type hints, plan-step docstrings | ✓ HOLDS | Every new module; legacy sources exempt by R18; two stale `src/` sys.path inserts in tests (LOW-73) |
| R21 | Conservation identity; C1 exact zero on a shared tick; C2 V = B + D + X and R ≤ 0 | ✗ VIOLATED | Conservation is measured per round and holds by construction (window gap to clock). C1 cannot pass on a real model (HIGH-05, HIGH-06) and its real-model test is a hard-coded exit 2 (MED-41). The C2 identity is definitional (MED-33); R ≤ 0 is genuinely guaranteed and tested |
| R22 | Exit codes 0 ok / 1 failed / 2 usage or prerequisite | ⚠ UNENFORCED | Most paths correct. Missing API key and YAML syntax errors exit 1 (MED-05); empty splits crash with KeyError (MED-19); 130 on Ctrl-C and raw bench codes (LOW-04); golden tests exit 2 on a code bug (MED-32) |

## Coverage

**In scope:** 252 · **Read in full:** 251 · **Unread:** 0 · **Deliberately excluded:** 1

Scope was re-derived from `git ls-files` (242 tracked files) plus the seven `engine/data/*.py` modules that exist on disk but are git-ignored (CRIT-01) and the three audit-skill files under `.claude/skills/` (tracked). Every file was assigned to exactly one of fifteen shards; each shard agent returned a `files_read` list covering every owned file and no `files_skipped`. The nine `fixtures/los_<map>.json` files are structurally identical but distinct (different md5 and nav areas) and were read; the nine `callouts.json` placeholders were parsed.

| File | Lines | Status | Reason |
|---|---|---|---|
| `docs/reviews/full-codebase-check-2026-08-01.md` | 695 | not audited | Output of the previous audit, about code deleted in Amendment 1; its IDs do not carry over because none of the cited files exist |
| `.env`, `*.env` | — | blocked path | Not present in the tree (`ls .env` → none); `.env.example` was read |
| `.venv/**`, `.venv-old311/**` | — | blocked path | Third-party code; dependencies audited from `requirements-engine.txt` / `.lock` and `pip list --outdated` metadata (`pip-audit` not installed, not installed for the audit) |
| `data/**` (`data/engine/engine.db`, `demos/`, `models/`, …) | — | blocked path | Never opened; the legacy `data/crosshair.db` is not on this machine |
| `demos/*.dem` | — | blocked path | None present |

Line numbers in this report were resolved against the files as they stand on 2026-10-19 at commit `9f7b2bb` by matching the quoted evidence, not copied from agent output (several shard agents reported line numbers beyond file length; every citation here was re-derived).

## Unverifiable

| # | Suspected | Why unverifiable | Check by hand |
|---|---|---|---|
| U-1 | awpy 3.1 column names assumed by the parser (`rounds.freeze_end_tick`, snapshot `health`/`side`/`is_crouched`, `shots.steamid`) may differ; a mismatch becomes a silent all-null column, or every player 'dead' at every tick (MED-13) | No real demo on disk; `.venv` is blocked | Run `test\engine\test_awpy3_parser.py <golden.dem>` and inspect `meta['unavailable']`, `ticks.height`, and null counts of `health` before `_dead_rows` |
| U-2 | awpy's rounds table includes knife rounds (and the parser has no way to drop them, MED-35) | Needs a pro demo with a knife round | Parse one and check whether round 1's loadouts are knives only |
| U-3 | Rows for dead or disconnected players may be absent from awpy ticks, leaving them `alive = 1` at their last position (`Track.at` has no staleness bound) | No demo | Confirm every kill victim has `is_alive = false` rows after the kill tick, and check a mid-round disconnect |
| U-4 | `kills.is_trade` from awpy may mark the traded victim (forward-looking) and is not nulled by `slice_until` | awpy 3 definition not inspectable offline | Check the awpy 3.1 definition; if forward-looking, null it for future rows |
| U-5 | The FACEIT signed-URL path always ends in `.dem.zst`, which `log.redact_text`'s scheme-less rule depends on | No live Downloads API response | Inspect `redact_url` output for one real signed URL (never paste the URL anywhere) |
| U-6 | A CDN `Content-Encoding` (gzip, or zstd, which urllib3 2.8 advertises when zstandard is installed) makes `downloaded != total` raise 'truncated download' on every file | Depends on the CDN headers | `curl -sI` one signed FACEIT URL and one HF resolve URL and read `Content-Encoding` |
| U-7 | Multi-frame `.dem.zst` archives exist in the wild (MED-25) | Depends on how FACEIT/HLTV/HF compress | Count zstd frames in one real download (`zstd -l` or `ZstdDecompressor().decompress(..., read_across_frames=True)` size vs single-frame) |
| U-8 | `VisibilityChecker.is_visible` / `NavMesh.find_area` signatures match; if not, every LOS value becomes NaN silently (LOW-22) | Real map assets exist but no builder run was allowed in a read-only audit | Run `test_states_golden.py` once a golden demo is pinned and check the `los` channel is not all NaN |
| U-9 | Feet-level visibility rays (HIGH-06) may graze the floor mesh systematically | Needs assets and a snapshot | On one builder snapshot compare `is_visible(feet_a, feet_b)` with `is_visible(eye_a, eye_b)` over all alive enemy pairs |
| U-10 | Cross-batch float nondeterminism on ROCm can exceed the C1 1e-9 bound even after HIGH-05 is fixed (v(full) and v(full minus A) come from different batches) | Needs a trained model | Predict one snapshot alone and inside batches of different sizes on the GPU and compare bitwise |
| U-11 | `test_predict` bf16 swing (1.1e-3 alone, 4.1e-2 under load) is hipBLASLt/MIOpen heuristic selection under contention and may also affect the fp32 1e-5 checks | Needs repeated GPU runs under controlled load | Run `test_predict.py` 20× alone and 20× beside a GPU-heavy job, log max and p99, repeat with `torch.use_deterministic_algorithms(True)` |
| U-12 | `review.CLOCK_REASONS` may not match awpy 3 round-end reason strings, so `clock_decided` never fires | Only three reason strings appear in tests | Parse a round lost on time and print `rounds.end_reason` |
| U-13 | Known advisories for pinned packages (pyarrow 22.0.0, torch 2.9.1+rocm, requests, pillow 13.0.0) | `pip-audit` is not installed and must not be installed by the audit | Run `pip-audit -r requirements-engine.lock` from a throwaway venv, or check each pin on osv.dev |
| U-14 | A non-editable `pip install .` wheel omits `schema.json`, `vocab_weapons.yaml`, `versions.yaml`, `callouts.json` (LOW-46) | Building a wheel writes `build/` into the repo | In a temp clone: `pip wheel . -w %TEMP%\w` and list the wheel contents |
| U-15 | FACEIT default team names in demo headers (`team_<captain nickname>`) reach `meta.teams` in the report without `--show-ids` | Depends on real FACEIT headers | Inspect `raw.meta['players'][*]['team']` on a real FACEIT demo |
| U-16 | HLTV terms/robots forbid automated demo download; the source sends a spoofed Chrome UA and has no robots check | Needs the live site terms (plan P1.1.1 says check first) | Read hltv.org/robots.txt and the terms before the first real run |
| U-17 | BO-series maps whose per-map dates straddle a month end land in different splits | Depends on what `date` each source writes per map | After a real scrape, check that all `<mid>_mN` rows of a series share one month |
| U-18 | Windows MAX_PATH overflow for long HF archive paths (`hf-<path>.dem.part`) | Depends on the repo and `LongPathsEnabled` | Check the registry value and the longest listed path |

## Not findings

| Looked like | Why it's fine |
|---|---|
| f-string SQL in `engine/db.py` (lines 176, 178, 212, 251, 369, 407-413, 444) | Interpolates only `int()` casts, `SCHEMA_VERSION`, module column-name constants or `?` placeholders; every value is a bound parameter; `register_match` and `_score_params` reject unknown keys first |
| `is_processed_any` checks only `<mid>_m1` for a bare id, not `_m2`, `_m3` as CLAUDE.md says | Deliberate per `id_forms`'s docstring so a half-landed series is retried; `process_one` re-checks every yielded id and UNIQUE `demo_sha256` blocks a double insert of the same file. The docs are wrong, not the code (LOW-49) |
| Missing `FACEIT_API_KEY` sending `Bearer None` | `_api_headers` raises `EnvironmentError` on a missing or empty key before any request |
| Path traversal via `match_id` in `ingest` or `dest.with_suffix` on dotted ids | `check_match_id` restricts ids to `[A-Za-z0-9._-]` and refuses `.`/`..`; both destinations are checked with `resolve().is_relative_to`; no `with_suffix` anywhere, temp names use `with_name(name + '.part')` |
| Zip-slip in `hltv.unpack` | Member names come only from `slug()`, `KNOWN_MAPS` and `[a-z0-9]` tokens, never the archive path; each member is capped at `MAX_DEMO_BYTES` declared and streamed |
| Decompression bombs | zst and gz outputs are capped at `MAX_DEMO_BYTES` while streaming; the compressed transfer is capped and `content-length` validated (the caps have no test, LOW-73) |
| `requests` calls without timeout | Every call has one: scraper 30 s and (10, 60) with a 3600 s budget, faceit_api 30 s, hltv (10, 30), hf 30 s |
| Authorization forwarded on redirect | Auth is set only on `open.faceit.com` calls; `requests` strips `Authorization` on cross-host redirects; the download session carries no auth |
| API key or signed URL in exceptions, logs or the DB | No headers dict is printed; `_BEARER_RE` scrubs tokens; `raise IOError(...) from None` suppresses the original context so `format_exception` omits the URL; `process_one` passes `redact_url(demo_url)` and `register_match` redacts again |
| Chrome left running on exceptions | `get_demo_urls_from_page` closes the page in `finally`, `ingest.run` calls `close_browser` in `finally`, `atexit` is a backstop |
| Race between `is_processed` and insert | Single-threaded, one match at a time; `BEGIN IMMEDIATE` plus UNIQUE sha |
| Side casing mismatch between awpy and comparisons | `_side()` lower-cases and maps `t`/`terrorist`/`2`, `ct`/`counter-terrorist`/`3` before any comparison; unknown → null |
| Halftime/overtime sides fixed per match | Sides come from the per-tick snapshot and per-row attacker/victim side; `builder._roster` rebuilds sides each round; scores track teams by roster overlap |
| `official_end_tick or end_tick` (R8) in the parser | `pl.coalesce`, null-based, not truthiness |
| Tick on a round boundary counted twice | `_assign_round` uses a backward asof on the start plus `tick <= end`, so at most one round; `partition.check_tiling` raises on gaps or overlaps |
| Literal `64`/`128` tick rate | `tick_rate = round(playback_ticks / playback_time)` from the header; `Demo.tick_rate` deliberately unused; state, actions and labels scale by `meta.tick_rate` (every synthetic story runs at 64 and 128) |
| Known awpy 3 gaps emitted as 0/False/"?" | `spotted_by`, `expire_tick`, `is_reloading`, `flash_assister_steamid`, `footsteps` are typed nulls or empty frames; `interrupt_plant` and `purchase` rows are dropped and counted. The exceptions are the fire extent (HIGH-04) and null health (MED-13) |
| `ctx.unique` silently drops grenade actions with a colliding `entity_id` (parser shard's note) | `Context.unique` never drops: it re-salts the second duplicate and raises on a third; the real consequence is spurious extra actions (MED-12) |
| `action_id` collisions (kill and flash by one actor on one tick) | `type` is part of the sha1 key; same-type collisions are salted by victim/entity/target/receiver/side/window_end; `assert_unique_ids` catches the rest |
| `is_trade` looks forward (leakage) | `kills.is_trade` is never read in `engine/actions`; `trade_setup` only looks back; all enablers read tick-1 and earlier |
| `time_remaining_s` leaking the round end | Derived from the round clock or the bomb timer from the plant tick; never from `end_tick` |
| Shared-reference mutation in interventions | `Snapshot.copy()` is `copy.deepcopy`; every intervention edits a copy; `apply(full)` returns `s_post.copy()` |
| Exact Shapley weights / efficiency enforcement onto the last member | `r!(n-r-1)!/n!` over memoised coalitions; a member whose every term is 0.0 gets exactly 0.0 via `fsum`; the residual is spread in proportion to `|φ|` and antithetic permutations telescope exactly |
| Sign flip applied twice | Applied once in `_Acc.scored`; conservation sums raw values; `V_raw` re-inverts |
| Enabler shares break conservation | `distribute` returns `phi − fsum(parts)` to the actor; every part goes to an action, a synthetic trade row or the clock |
| Conservation assumed, not measured | `attribution/engine.py` measures the residual with `fsum`, stores it, flags `conservation_ok` and `analyze` warns on failure |
| Dead-pair `smoke_between` written as 0/1 by interventions (part of H-09) | Both models mask pairs with a dead player (`PairBias` `present = ~isnan & both`, baseline `ea = enemy & alive_i & alive_j`), so it has no effect on predictions |
| Omitting `--split-file` on `train-q` is a silent random split | It is a documented 'caller supplies the training set' mode (ENGINE_CLI.md) and no label leaks through the vocab fit; the remaining footgun is MED-38 |
| Vocab/NormStats leakage | `dataset.build` fits both on train rows only; unseen categories map to UNK, None to NULL |
| Post-decision rows kept in training (legacy rule) | Intentional in v2 (plan P6.3.1): kept with normal weight, terminal snapshots weighted ×2 |
| Side-swap augmentation not flipping the label | No side-swap augmentation exists; the five augmentations never touch `y_win` or side |
| Softmax over fully masked rows | The set-encoder key mask always includes the ctx token; the temporal mask ORs in the identity |
| Temporal causal leakage | Windows hold same-round snapshots with `t_j <= t_i`, target last, index-causal mask; `Predictor.window` matches training |
| `torch.load` untrusted pickle | `weights_only=True`; `np.load` without `allow_pickle` |
| Calibration fitted on test | Fits on val pro, reports on test pro; early stopping and fine-tune selection use val pro only |
| LightGBM categorical map mismatch between train and predict | Integer vocab codes from the same build's vocab in a fixed feature order, not per-call pandas categories |
| bf16 on by default for attribution | Training uses bf16 autocast; `Predictor` defaults to `bf16=False` per the decision |
| Regret positive when the realized action is not in the panel | `regret` initialises `best = q_self`, so `R ≤ 0` always; alternatives below `n_min` or NaN are skipped |
| FAISS `-1` ids / normalisation mismatch | Negative ids dropped; index and query use the same `normalize()` with inner product |
| Q self-inclusion | Q is a regression net fit on train rows, not a lookup; panels exclude the query's own match |
| Match or action ids in neighbour panels reaching the report | `Panel.to_dict` omits ids; `ref` is a hashed digest; `_value_fields` copies only counts and means |
| `config_hash` nondeterminism / `awpy_data_release` typing | `yaml.safe_dump(sort_keys=True)` over normalised values; the release is a quoted string and `_coerce` rejects an int |
| `manifest` importing torch | `gpu_name` imports torch only if already in `sys.modules` or `probe=True` |
| `--side` style injection through CLI into SQL | No CLI argument reaches SQL text in v2 |
| Build `--month` broadcasting to undated matches causing split leakage | The month only picks the layout partition; `splits.assign` uses `matches.date` and puts undated matches in no split |
| `action_scores` PK collisions across matches | `action_id = sha1(match_key|round|type|actor|tick)` |
| Tests writing to the production DB or demo store | Every DB-touching test sets `CROSSHAIR_ENGINE_DB` to a temp path before importing engine; demo dirs are repointed to temp; local HTTP servers bind `127.0.0.1:0`; `test_scraper.py` is excluded from `run_all` and launches nothing on import |
| `fixtures/los_<map>.json` ×9 being copies of one template or containing PII | Nine distinct seeded files holding only nav area ids and coordinates |
| `GOLDEN.json` placeholder (`demo_sha256: null`) | Intentional; `test_golden_present` and the golden tests exit 2 with a reason |
| Assistant attribution in git history | `git log --all` grep for co-authored/anthropic/generated: zero hits beyond the filename `CLAUDE.md`; authors are Faig's two identities only |
| Every command block in CLAUDE.md parsing | All 14 parse against `build_parser()` (verified with `parse_known_args`) |
| Amendment 1 deletions and moves | Every listed `src/`, `test/` and `docs/design` file is absent from disk and `git ls-files`; `scraper.py` differs from `32b1373` by one import line |
| Other source directories swallowed by broad `.gitignore` patterns | `git status --ignored` shows only `engine/data/*.py` outside `.venv*`, `data/`, `__pycache__`; no `build`, `dist` or `demos` directory exists under engine/test/bench |

## Cost

18 Opus 5.5 agents (15 read-only shards in batches of three, 3 adversarial verifiers), about 3.5 M agent tokens plus coordinator overhead; roughly 2 h 15 m wall clock. The review hit all 252 files; a rerun could skip the nine LOS fixtures and merge the three test shards' helper files into their owning module shards.
