# Engine v2 — progress log

Started 2026-10-19 (night). Faig asleep; green light given for implementing every step
of `docs/ENGINE_PLAN.md` (with Amendment 1) that needs no data and no Faig. Coding by
Opus 5.5 agents (≤ 3 concurrent), engineering and verification by Fable 5.1.

Legend: ✅ done and verified · 🟡 done, needs data/model/Faig to finish · ⛔ deferred
(reason) · 🙋 needs Faig.

## Summary (end of night, 2026-10-20 early morning)
- **Every phase P0–P12 has its code and tests.** Full suite: `test/run_all.py` → **89 passed, 0 failed, 3 skipped** (the three golden-demo tests), 258 s; pyflakes clean across `engine/`, `test/`, `bench/`.
- **Nothing has touched a real demo, a trained model, or the network.** All verification is on the six synthetic stories, hand-built snapshots, and local HTTP servers.
- **Nothing is committed.** `git status`: legacy files deleted/moved (staged), ~150 new files untracked. Faig commits when he wants, after a security review.
- **End-to-end works on synthetic input:** `python -m engine.cli analyze --raw-tables-story base --allow-unadopted` (stub predictor) produces a schema-valid report with exact conservation.

## Next steps when Faig is back (in order)
1. Answer the items under "Things Faig must do" and Appendix H of the plan (golden demo, pro source, callouts, disk, tier buckets, score in context, D̃ shrink target).
2. Put one pro demo in `data/engine/demos/golden/`, set its sha in `test/engine/golden/GOLDEN.json`, run `python -m engine.cli parse` on it, then `test_states_golden.py --write`, `test_actions_golden.py --write`, `bench/bench_parse.py`, `bench/bench_states.py` (P2.5.2, P2.6.2, P3.8.2–3, P4.7.2). Fix whatever the first real demo reveals (awpy columns, dead-player rows, `spotted_by`).
3. Data: decide HLTV path (terms + `.rar`), start FACEIT 10 via `python -m engine.cli scrape --source api --min-level 10 --engine-copy`, `inventory`, `splits`.
4. Build datasets (`build-dataset --assert-qa`), set `qa.*` thresholds from the first report, train the LightGBM baseline, train `wp_v1`, calibrate, `eval`, `sanity --model`, ablations → decision per P6.7.2 in `eval/DECISION_wp_v1.md`.
5. Sanity gate `sanity --counterfactual` (P7.2); loop per P7.3 until green.
6. Real-model attribution checks (P8.6.3, C1 bound), train Q/B (`train-q`), `build-neighbours`, `eval --value`, `eval --anchors`, review loop (`review`).

## Status by phase

### P0 — environment and scaffolding ✅ (2026-10-19 night)
- P0.1 ✅ Python 3.12.10 installed per-user; `.venv` (3.12) created; old 3.11 venv renamed `.venv-old311` (gitignored, Faig may delete). AMD ROCm 7.2.1 + torch 2.9.1+rocm7.2.1 installed from AMD wheels; GPU detected; training smoke test fp32/bf16 OK (see `docs/ENGINE_ENV.md`). `requirements-engine.txt` + `requirements-engine.lock` written. `bench/bench_gpu.py` ⛔ not written as a script yet (the smoke test was run inline; P0.1.4 script to add in a later wave).
- P0.2–P0.6 ✅ package tree, config (+hash, validation), paths, manifests, JSON-lines logging with redaction, harness, `test/run_all.py`, bench harness + budgets. 6 test scripts green, pyflakes clean.
- P0.7 🟡 golden demo: `GOLDEN.json` placeholder, `test_golden_present.py` exits 2. 🙋 Needs a pro demo (Faig, Appendix H q3).
- Map assets ✅ awpy-data 2000933 pinned; mesh+nav cached for the 9 competitive maps.

### Contracts (Amendment 1) ✅
- `engine/parse/schema.py` (RawTables, causal `slice_until`), `engine/state/schema.py` (tokens, Snapshot, validate, LIVE_SAFE), `engine/actions/types.py` (Action, registry), `test/engine/_synthetic.py` (6 stories, STORY_FACTS). 613 checks green.

### P2 — parser and map assets ✅ (code) / 🟡 (never run on a real demo)
- `engine/parse/awpy3_parser.py` (A.1 tables from awpy 3.1.0; derived defuse start/abort, explode, is_planting, flash time left, kill distance, dead-player rows), `worker.py` (multiprocess, retries, `_failures.jsonl`, `--smoke`), `engine/maps/{assets,grid,version,callouts,plot}.py`, `versions.yaml`, empty `callouts.json` ×9. 318 checks green.
- 🟡 Not obtainable from awpy (emitted null): `ticks.spotted_by`, `infernos.extent_radius`, flash/HE/decoy `expire_tick`, last round `official_end_tick`, `is_reloading` where absent, `flash_assister_steamid` (unconfirmed column), `footsteps` (unconfirmed). `spotted_by_count`/`seconds_since_last_seen_by_enemy` therefore are null until a real demo shows whether `m_bSpottedByMask` is decodable.
- 🟡 Dropped rows (counted in `meta["dropped_rows"]`): bomb `interrupt_plant`, item `purchase` (A.1 has no names; adding `plant_abort`/`purchase` is a schema decision for Faig).
- ⛔ Real-demo checks (P2.2.3 kill-position coverage, P2.5.2 golden counts, P2.6.2 bench) need a demo. Whether awpy keeps rows for dead players is unconfirmed.
- 🟡 LOS fixtures are structural (random area pairs; 8/8 visible, 8/8 blocked on all 9 maps); named callout pairs pending. `versions.yaml` has no reworks yet (`<map>@unknown`). Nav distances are one-way for drop-downs (symmetry not asserted).

### P3 — state builder ✅ (code) / 🟡 (golden + real-demo timing pending)
- `engine/state/{geometry,sampler,players,utility,context,pairwise,builder,leakage}.py`, `vocab_weapons.yaml`, `SCHEMA_CHANGELOG.md`, `bench/bench_states.py`. 423 checks green; fast causal view proven equal to `slice_until`; leakage audit (winner flip + end shift) clean on all stories and catches injected leaks. Synthetic bench 0.94 ms/snapshot.
- 🟡 Real-demo pairwise cost unknown (estimate 3–5 ms/snapshot → may exceed the 8 s target; P3.5.3 fallback ready to engage). Cached-vs-exact LOS agreement (P3.5.2) and golden states need the demo.
- Contract notes: 2v2 synthetic rounds are padded with dead tokens `player_id = -1` (suggest `PAD_PLAYER_ID` constant in schema); `infernos.extent_radius` is one value per fire (future-ish; nominal radius would be safer); `sampling.heard_window_s` added to config (2.0 s); halves assume MR12; label `is_terminal` stored as `y_is_terminal`.

### P4 — action extractor ✅ (code) / 🟡 (golden counts + real-demo bench pending)
- `engine/actions/{kills,utility,movement,bomb,economy,enablers,extractor,windows}.py`, `bench/bench_actions.py`. 610 checks green; byte-identical output across runs; synthetic bench 0.033 s for 39 actions / 12 windows.
- Deviations: trade distance measured to the nearer of the teammate's death spot or their killer (the plan's 1,200 u to the victim failed the trade story); movement types read only grid samples and the builder is rerun once peeks exist; smoke alive at round end gets no fade; `blinded_self` is seconds (0.0 = not blind); extra params beyond A.5 on utility/molotov/drop (`x,y,z,entity_id,throw_tick,radius_units,end_tick,pickup_tick`); plant/defuse enablers only `smoke_cover`.
- Window contract matches what P8 assumed (`window_id, round_num, start_tick, end_tick, member_action_ids`; pseudo-actions carry `parent_action_id`).
- ⛔ Golden counts (`test_actions_golden.py --write`) and the 3 s budget need the golden demo.

### P5.3/P5.4 + P6 — dataset, QA, win model ✅ (code) / ⛔ (no training: no data)
- `engine/data/{dataset,qa}.py`, `engine/model/{tokens,set_encoder,temporal,heads,wp_model,augment,batching,train,calibrate,predict,checkpoints,baseline_lgbm}.py`, `bench/bench_gpu.py`, `bench/bench_predict.py`, `test/engine/_fake_states.py`. ~400 checks green on GPU and CPU: permutation invariance, causality, gradient flow, determinism, overfit-tiny (loss < 0.05 at step 117 in 9 s on GPU), checkpoint/resume, augmentation isolation, predictor cache + batch invariance, temperature recovery, baseline, dataset shapes/UNK/no split leakage, QA, sampler share. Default model 2,720,917 params (+128 per real player id).
- Bench: training step 33.6 ms fp32 / 8.9 ms bf16 (512×24×128). Predictor 4,060 snaps/s fp32 GPU (target 5,000 missed), 7,550 bf16 (met; bf16 off by default, logits differ ≤ 1e-2), CPU 102/s (target 300 missed).
- ⚠ Environment issues: (1) AMD's Windows torch uses a slow CPU BLAS (~20× slower than numpy) → CPU-only path is far off budget; options: bf16 GPU default for inference after a correctness check, or a separate CPU torch wheel in a second venv for CPU deployments. (2) GPU scripts can hang at interpreter exit inside the HIP runtime; GPU tests end via `_fake_states.finish()` (skips the shutdown step). Other GPU scripts may need the same.
- Deviations: normalization inside the model (shards keep real units); `nav_area` not fed (redundant with grid/callout/position); categorical NULL=2 reserved id; dead players keep a minimal token; horizontal jitter only; fifth calibration phase `75_plus`; `final.pt` = best of best.pt vs fine-tune epochs; baseline lacks distance-to-bomb (schema has no bomb position; consider adding `bomb_x/y/z` to ContextToken); resumed training matches within 1e-3, not bitwise.
- ⛔ P6.1 baseline run, P6.3 training, P6.5 calibration, P6.6 sanity, P6.7 ablations: all need data. Ablation switches exist (`--no-temporal`, `--no-pairwise`, `--no-aux-heads`, `--tiers pro`).

### P7 — interventions and sanity gate ✅ (code) / ⛔ (gate needs a trained model + real states)
- `engine/counterfactual/{interventions,apply,sanity,ood}.py`, `test/engine/_cf_states.py`. 328 checks green: every intervention (positive/no-op/invalid), `apply` identities and order independence, far-away decoy contributes 0 in all 32 coalitions, gate passes a sensible stub and fails a backwards one, OOD detector flags planted points.
- Deviations: interventions take a reference state `fn(state, action, ref, *, geometry=None)`; `reflash` restores from `s_post`; extra interventions (HE, plant_start, defuse_abort, bomb drop/pickup, weapon drop, team buy); `revert_buy` takes the prior loadout from the caller (Appendix B.6 median table not built yet); smoke/fire checks duplicated from `engine/state/utility.py` (keep in step).
- ⛔ `sanity.main()` exits 2 until there are states + a model; no radar PNGs for worst cases (P7.2.8).

### P8 — attribution engine ✅ (code, on stubs) / ⛔ (real-model checks need a trained model)
- `engine/attribution/{types,partition,shapley,movement,clock,enablers,engine}.py`, `test/engine/_attr_states.py`. 215 checks green: conservation residual exactly 0 (exact and sampled), **C1 both variants** (far decoy gets exactly 0, the kill gets exactly ΔW, in 2- and 5-member windows), two kills on one tick → ±0.1 summing to ΔW, determinism, predictor calls = windows + 2 per round.
- Decisions: the 1–2 tick clock gap inside a window is booked to the team clock; terminal W pinned by the engine; movement shares target W(after) − W(all movers frozen), remainder = clock; trade credit goes to a synthetic `trade_setup` row for the dead teammate; `info` enablers → team clock; a window whose undo fails sends its ΔW to the clock and is flagged; unassigned movement → synthetic `positioning` rows.
- ⛔ Real-model C1 bound (|φ_decoy| ≤ 0.002 over 200 windows), conservation on 1,000 real rounds (P8.6.3), bench budget: need the trained model. Movement-variance metric computed, not yet written by P6.7.
- To check when P4 lands: Window fields `window_id, round_num, start_tick (= min tick − 1), end_tick (= max tick + 1), member_action_ids`; pseudo-actions carry `params["parent_action_id"]`.

### P9 — expected value, decision/execution split ✅ (code) / ⛔ (needs trained win model + real data)
- `engine/value/{descriptors,q_model,neighbours,scores,shrinkage}.py`, `bench/bench_value.py`, `test/engine/_value_states.py`. 233 checks green on GPU and CPU: **C2** identity V = B + D + X within 7e-18 on 10,000 synthetic actions, R ≤ 0 always, panels for 100%, shrinkage recovers planted skill for 92% of players, planted +0.02 signal recovered (X̃ 0.016 vs ≤ 0.004 for others). Bench 2.0 s for 5,000 actions (budget 5 s).
- Deviations: neighbour index split by (side, map_version) for speed; one `to_callout` field covers land/victim/target/facing callouts; harmonic mean n̄ in shrinkage; `fit_k` returns full variance records; vocab saved under `q_v<N>/vocab.json`.
- 🙋 **Faig's call:** B.8 shrinks D̃ toward 0, but D's mean within a type class is not 0 (kills ≈ +0.07), so shrinking toward the class average is statistically right. Left as written pending his decision.

### P10 — CLI, analyze pipeline, report ✅ (code, synthetic end-to-end) / ⛔ (real demo + model)
- `engine/cli.py` (16 commands), `engine/analyze.py` (`analyze_tables` / `analyze_demo`), `engine/report/{schema.json,cards,charts,writer}.py`, `bench/bench_analyze.py` (real stages, `--synthetic`), `docs/ENGINE_CLI.md`. 324 checks green: every synthetic story runs end to end (states → actions → attribution → value stage with tiny Q/B + FAISS → report), report validates against the JSON schema, conservation residual 0.0 in every round, reproducible report.json, redaction, guards (64-tick refusal, tier override warning), `--help` for all commands. Synthetic bench 1.05 s total.
- Deviations: teams grouped by side, not identity across halves; report player ids are `p_<10 hex>` hashes (hides from casual readers only); test-only env `CROSSHAIR_STUB_PREDICTOR` and `--raw-tables-story` hook; `analyze` needs `--allow-unadopted` until a model is adopted in `docs/ENGINE_MODELS.md`.
- ⛔ `analyze_demo` on a real demo, map-asset path, real Predictor path, golden bench: need the golden demo + a trained model. `build-states`, `build-actions`, `scrape` have no CLI tests yet.
- Flaky watch: `test_predict.py` bf16-vs-fp32 check failed once (4.1e-2) during a contended full-suite run and passes alone (1.1e-3); `test_wp_model.py` hung once at interpreter exit (HIP shutdown) before `_fake_states.finish()` was added; passes in 5 s now.

### P11 + P12 — evaluation scripts, docs, ops ✅ (code/docs) / ⛔ (no real data)
- `engine/evaluation/{anchors,robustness,review,usage,registry}.py` (170 checks green), `docs/ENGINE_MODELS.md` (adoption table, empty), `docs/ENGINE_OPS.md`, `docs/DATA_SOURCES.md`, `CLAUDE.md` "Engine v2" section (legacy sections annotated obsolete, every surviving rule kept), `README.md` short section.
- Anchor bands are module constants (no `anchors:` config section yet). `anchors.enrich()` joins tick/alive/equipment/weapon class from actions+states.
- ⛔ Not researched: HLTV terms, FACEIT rate limits, organizer demo packs. `ENGINE_OPS.md` lists commands that do not exist yet (split creation CLI, `scrape --source hltv` with a match list, map-assets rebuild, per-patch constants).
- To wire after P10 lands: `cli eval --anchors` must point at `engine.evaluation.anchors`; `analyze.py` must call `evaluation.usage.append_usage`.

### P1 + P5.1/P5.2 — data layer ✅ (code) / 🟡 (needs real sources)
- `engine/db.py` v1 (matches, runs, action_scores, processed; WAL; migrations mechanism), `engine/ingest.py` seam (register_demo, process_one, run), sources moved with import-only edits (scraper.py line 349; faceit_api.py lines 25–26 + additive `min_level`), `hltv.py`, `hf_archive.py`, `data/quarantine.py`, `data/inventory.py`, `data/splits.py`, `data/layout.py`. 244 checks green, downloaders tested against local servers only.
- ⛔ HLTV `.rar` archives refused (needs a new dependency + unrar binary; Faig decides). HLTV page parsing untested against the live site (Cloudflare) and its terms must be read first (P1.1.1).
- ⛔ The Hugging Face archive named in the plan (`Vasy7777/cs2-demo-archive`) is NOT a pro demo archive (19 sample matches, mostly video, 64-tick). P1.1.3 needs a different source; `hf_archive.py` has no default repo.
- 🟡 `faceit_api.min_level` reads `teams.<faction>.roster[].game_skill_level` per the FACEIT Data API v4 swagger; unverified against a live response.

## Deferred items (need data / model / Faig)
(filled in as encountered)

## Things Faig must do
- Choose/provide the golden demo (a recent tier-one 128-tick BO1 map) → `data/engine/demos/golden/`, then set its sha256 in `test/engine/golden/GOLDEN.json`.
- Decide callouts: hand-drawn polygons per map vs a community set (P2.3). Until then positions fall back to `area_<nav id>` names.
- Answer Appendix H of `docs/ENGINE_PLAN.md`.
- Optional: delete `.venv-old311/` (legacy 3.11 venv, no longer used).
- Decide the pro demo source: HLTV automated downloads (needs terms check + `.rar` support = new dependency `rarfile` + unrar/7z binary) vs manual batches. The HF archive from the plan is unusable.

## Deviations from the plan
- Clean break per Amendment 1 (Faig confirmed 2026-10-19): legacy `src/` engine deleted, sources moved to `engine/sources/`, one venv, fresh `engine.db` v1, no awpy-2 parity.
- Budgets exist in both `engine/config.yaml` and `bench/budget.yaml` (the plan named both); `test_config.py` fails if they drift.
- Extra config keys beyond Appendix D were added for thresholds the plan text names (P6.6, P6.7.2, P7.2); `qa.*` thresholds left null until the first QA report.
