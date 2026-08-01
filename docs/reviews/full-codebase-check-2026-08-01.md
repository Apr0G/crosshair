# Full codebase check — Crosshair

**Date:** 2026-08-01
**Repo:** github.com/Apr0G/crosshair (public)
**Commit:** `6d48589` on `main`
**Working tree:** dirty — 3 untracked files (`CLAUDE.md`, `PROMPT_faceit_api_downloader.md`, `pyproject.toml`) plus gitignored `.claude/`
**Scope:** 25 files in scope · 25 read in full · 0 unread
**Findings:** 4 CRITICAL · 16 HIGH · 21 MEDIUM · 17 LOW
**Method:** 9 parallel read-only shards → adversarial verification of every CRITICAL/HIGH against the live database → dedupe. 5 candidate CRITICAL/HIGH findings were **refuted** and moved to *Not findings*.

---

## Summary

The ingestion and storage layers are sound in structure but have a latent double-insert window that has not yet fired; the corpus is currently clean (verified: zero duplicate `(match_id, round_num, tick)` rows in `round_states`). The serious problem is upstream of all of that — **the win-probability model trains on at least two features that encode the future**, so the reported AUC of 0.896 is not a held-out number and every impact score derived from it inherits the distortion. That is the single most urgent thing: `time_remaining_s` is computed from the round's actual end tick, and `ct_spotted_count`/`t_spotted_count` are populated from an unbounded nearest-tick lookup that reaches forward in time. Both were confirmed empirically, not inferred.

One bad habit shows up in roughly ten places and is worth naming on its own: **`0` and `"?"` are used as sentinels for "unknown" in fields where those are legitimate values.** `ct_spread=0.0` for a 1-alive team collides with a genuine stack (149,833 rows vs 3 real ones); `player_side="?"` silently takes the T branch in impact scoring; `damage_after` is `0` because the column name is wrong; `_dist2` returns `0.0` on error, which reads as "enemy is standing on top of you." Each is individually small. Together they mean a meaningful fraction of the feature space encodes "we don't know" as a confident value.

Separately and bluntly: **the repo does not currently run at all.** `.venv` is 108K with no `lib/`, no site-packages and no pip. Only `main.py status` executes, because it touches nothing but stdlib. Every other documented command fails at import, which means no claim in this report about runtime behaviour could be verified by execution — only by reading code and querying the existing database.

---

## Findings at a glance

| ID | Sev | Title | File | Rule |
|---|---|---|---|---|
| CRIT-04 | CRITICAL | Every T-actor `impact` in the DB is sign-flipped — stored data never had the negation applied | `data/crosshair.db` | P5 |
| CRIT-01 | CRITICAL | `time_remaining_s` is derived from the round's end tick — outcome leakage into a core model feature | `src/state_sampler.py:264` | P6 |
| CRIT-02 | CRITICAL | `_info_state` snaps to the nearest cached tick with no bound, pulling future visibility into past states | `src/state_sampler.py:147` | P6 |
| CRIT-03 | CRITICAL | Signed demo URL (credential in query string) reaches stdout and is persisted to the DB | `src/scraper.py:265` | P10 |
| HIGH-01 | HIGH | No transaction across the seam; per-batch commits leave events with no `processed_matches` row | `src/db.py:171` | P1 |
| HIGH-02 | HIGH | `_init_schema` lacks `p_before`/`p_after`/`impact` — a fresh DB cannot run `status` or `score` | `src/db.py:50` | P9 |
| HIGH-03 | HIGH | Live DB lacks 4 columns `insert_round_states` writes — ingestion into it fails today | `src/db.py:188` | P9 |
| HIGH-04 | HIGH | 9.5% of training rows are post-decision states with a side already eliminated | `src/state_sampler.py:248` | P6 |
| HIGH-05 | HIGH | All events in a ~1 s window share one full-window WP delta; 49% of scored events affected | `src/score_impact.py:44` | P5 |
| HIGH-06 | HIGH | `test_local_demo.py` writes rows under `test_local` but marks a different, path-derived id | `test/test_local_demo.py:45` | P1 |
| HIGH-07 | HIGH | `pipeline.run` reports attempted matches as successful; no failure count, always exits 0 | `src/pipeline.py:80` | — |
| HIGH-08 | HIGH | `cmd_player`'s actor filter is dead code — every reported statistic is over the wrong row set | `src/main.py:183` | — |
| HIGH-09 | HIGH | `.gitignore` misses `.env.local` / `.env.backup` on a public repo | `.gitignore:21` | P10 |
| HIGH-10 | HIGH | Permission allowlist auto-approves `pip install *` and arbitrary `python3 -c` | `.claude/settings.json:4` | — |
| HIGH-11 | HIGH | `.venv` has no site-packages — every documented command except `status` fails at import | `requirements.txt:1` | P15 |
| HIGH-12 | HIGH | `get_player_match_ids` returns `[]` on every exception — auth failure looks like "no matches" | `src/scraper.py:124` | — |
| HIGH-13 | HIGH | Response interceptor accepts any URL containing a magic substring; no host allowlist | `src/scraper.py:146` | — |
| HIGH-14 | HIGH | `ct_spread`/`t_spread` return `0.0` for a 1-alive team, colliding with a genuine stack | `src/state_sampler.py:56` | — |
| HIGH-15 | HIGH | `_smoke_impact` counts tick rows instead of unique players — enemy counts inflated up to 65× | `src/feature_extractor.py:317` | — |
| HIGH-16 | HIGH | BO-series `_m1` dedup shortcut permanently abandons maps 2 and 3 of a partially-ingested series | `src/scraper.py:238` | P1 |
| HIGH-17 | HIGH | README's documented getting-started sequence cannot work on a clean checkout | `README.md:88` | — |
| HIGH-18 | HIGH | CLAUDE.md names the test suite as the calibration guard; not one of its 23 checks measures calibration | `test/test_win_probability.py:83` | P7 |
| HIGH-19 | HIGH | 13 of 15 project invariants have no test of any kind | `test/` | P1–P11 |

MEDIUM and LOW findings follow the detailed sections.

---

## CRITICAL

### CRIT-04 — Every T-actor impact value in the database has the wrong sign

**File:** `data/crosshair.db` (stored data, not current code) · **Category:** A. Invariants · **Confidence:** high (verified exhaustively) · **Rule:** P5

**What's wrong**
P5 requires `impact = p_after - p_before` for a CT actor and the negation for a T actor.
The stored data applies **no negation at all** — every row holds the raw CT-perspective
delta regardless of which side acted.

**Evidence**
Sampling 299,779 scored rows with a resolvable side, testing each against both forms
(tolerance 5e-5, since `impact` is stored rounded to 4 decimals):

| | matches `round(p_after − p_before, 4)` | matches `round(−(p_after − p_before), 4)` |
|---|---|---|
| CT rows (159,663) | **159,663** — correct | — |
| T rows (140,116) | **140,116** — CT-perspective, un-negated | 11,124 (only where the delta is ≈0, so both forms coincide) |

A concrete pair from the same match, round and timestamp, both at the top of
`main.py top`:

```
id=2623565  side=ct  p_before=0.019897  p_after=0.967819  impact=+0.9479
id=2623566  side=t   p_before=0.019897  p_after=0.967819  impact=+0.9479   <- must be −0.9479
```

**How it fails**
`main.py top --by negative` is documented as "worst player moments". For any T actor it
returns their *best* moments. `top --by positive` likewise inverts. Any analysis over
the 1,368,519 T-side events reads every T success as a failure and vice versa — silently,
because the numbers are individually plausible.

**Blast radius**
Roughly half of all scored events. This is the product's headline output.

**Important:** the *current* `score_impact.py` negates correctly — verified directly
against the shipped function (a T actor with `p_before=0.60, p_after=0.40` returns
`+0.20`). So this is stale stored data from an earlier scorer, not a live code defect.
The same staleness explains why `rotation` and `buy` are 0% scored (U-1): that older
scorer predated both the negation and those event types.

**Suggested fix**
Re-run `.venv/bin/python src/main.py score` once the venv and the WP model are restored.
`score_match` now clears a match's `p_before`/`p_after`/`impact` before rewriting them,
so the re-score cannot leave a mix of old and new values. Until then, treat every
`impact` value in the DB — and every number derived from it — as unusable for T actors.

---

### CRIT-01 — `time_remaining_s` is derived from the round's actual end tick, leaking the outcome into a core model feature

**File:** `src/state_sampler.py:264` · **Category:** G. Model correctness · **Confidence:** high (verified empirically) · **Rule:** P6

**What's wrong**
`time_remaining_s` is not the CS2 round clock and not the bomb clock. It is *ticks until this round is officially over*, which is only knowable after the round has been decided. It is listed in `win_probability.FEATURES` and is therefore a direct training input.

**Evidence**
```python
# src/state_sampler.py:216
        r_offend  = r.get("official_end") or r.get("end")
# src/state_sampler.py:264
            "time_remaining_s":   round((r_end - tick) / tick_rate, 2),
```
```python
# src/win_probability.py:22-23
    "time_into_round_s",
    "time_remaining_s",
```

**How it fails**
Round duration correlates strongly with outcome — decisive rounds end fast. The model learns "small `time_remaining_s` ⇒ the round is about to end ⇒ whoever is currently ahead wins."

Verified against the live DB at `time_into_round_s = 0.0`, the instant a round starts, when nothing has yet happened:

| `round_won_ct` | avg `time_remaining_s` at t=0 | n |
|---|---|---|
| 0 (T won) | 78.11 | 7,556 |
| 1 (CT won) | 71.96 | 8,095 |

A 6.15-second separation *at t=0* is definitionally future information: no in-round event has occurred to produce it. `MAX(time_remaining_s)` is 160.92 s, well beyond the 115 s CS2 round length, confirming the value tracks `official_end` rather than any game clock.

**Blast radius**
The entire model. AUC 0.896 (README, CLAUDE.md, module docstring) is not a valid held-out estimate. Every `impact` value in the `events` table is a delta between two predictions from this model, so all 530,030 scored events inherit the distortion. At genuine inference time — a live round — the feature cannot be computed at all; only a replayed finished demo can produce it.

**Suggested fix**
Replace with a clock the state actually knows: pre-plant, `max(0, 115.0 - time_into_round_s)`; post-plant, `max(0, 40.0 - (tick - plant_tick)/tick_rate)` using the already-defined `C4_TIMER`. Re-ingest and retrain, and expect the AUC to drop — that drop is the measure of how much leakage there was.

---

### CRIT-02 — `_info_state` snaps to the nearest cached tick with no bound, pulling future visibility into past states

**File:** `src/state_sampler.py:147` · **Category:** G. Model correctness · **Confidence:** high (verified empirically) · **Rule:** P6

**What's wrong**
The visibility lookup picks the globally nearest cached tick, with no maximum distance and no constraint that the tick be in the past. The sibling lookup in the same codebase does this correctly.

**Evidence**
```python
# src/state_sampler.py:147
    nearest = min(cache_ticks, key=lambda t: abs(t - tick))
```
Contrast, same repo:
```python
# src/feature_extractor.py:268-270
    for k in cache:
        if t - window_ticks <= k <= t:
```
`_precompute_visibility` only writes a cache entry when somebody was spotted (`feature_extractor.py:255`), so ticks with no sightings are absent from the dict entirely — which is what makes an unbounded nearest-match reach so far.

**How it fails**
Verified against the live DB — average spotted counts at four different in-round timestamps:

| `time_into_round_s` | avg `ct_spotted_count` | avg `t_spotted_count` | n |
|---|---|---|---|
| 0.0 | 0.923 | 0.906 | 15,651 |
| 1.0 | 0.923 | 0.906 | 15,651 |
| 2.0 | 0.923 | 0.906 | 15,650 |
| 5.0 | 0.923 | 0.906 | 15,649 |

Identical to three decimal places across every timestamp. The value does not vary with time within a round at all — every sample in a round is tagged with the same sighting data, including samples from before first contact happened. At t=0, during freeze-end, players are in spawn and cannot see each other; the feature nonetheless reports ~0.92 enemies spotted.

**Blast radius**
Two of the 32 numeric model features (`ct_spotted_count`, `t_spotted_count`) carry future information on every row of the 1.18 M-row training set. Compounds with CRIT-01.

**Suggested fix**
Mirror `_lookup_spotted`: restrict to `tick - HEARD_WINDOW <= k <= tick`, and return zeros when no cached tick falls in that past window. Re-ingest and retrain.

---

### CRIT-03 — Signed demo URL reaches stdout via an exception message, and is persisted into the database

**File:** `src/scraper.py:265` · **Category:** B. Secrets · **Confidence:** high · **Rule:** P10

**What's wrong**
FACEIT signed download URLs carry their credential in the query string. `requests` builds its `HTTPError` message as `"<code> Client Error: <reason> for url: <full url>"` — including the query string. That exception propagates to a bare `print`. Separately, the Playwright path yields the signed URL itself, which is then written to `processed_matches.demo_url`.

**Evidence**
```python
# src/scraper.py:264-265
    resp = session.get(demo_url, stream=True, timeout=120)
    resp.raise_for_status()
```
```python
# src/pipeline.py:73-74
        except Exception as e:
            print(f"[{match_id}] error: {e}")
```
```python
# src/pipeline.py:47
        db.mark_processed(match_id, map_name, demo_url)
```

**How it fails**
A 403 on the CDN fetch — routine enough that `faceit_api.download_demo` exists specifically to retry it — raises `HTTPError` whose message embeds the complete signed URL. `pipeline.run` catches it and prints it verbatim. Unlike the four deliberate `url[:60]`/`[:80]`/`[:100]` truncations elsewhere, this path has no truncation at all. That stdout is exactly what gets pasted into an issue when a scrape breaks. The database exposure is quieter but permanent: existing `processed_matches` rows hold real (now-expired) signatures.

**Blast radius**
Credential disclosure on a public repo. Signed URLs are short-lived, which bounds it — but the same `print` is the terminal sink for every exception in the pipeline, so the pattern is one API change away from leaking something durable.

**Suggested fix**
Add a URL-redaction helper (`urlsplit`, drop `query`) and route every URL that reaches a print, an exception message, or `mark_processed` through it. Do not slice URLs by character count — truncation length is luck, not a control.

---

## HIGH

### HIGH-01 — No transaction across the seam; per-batch commits can leave events with no `processed_matches` row

**File:** `src/db.py:171` · **Category:** E. Database · **Confidence:** high · **Rule:** P1

`insert_events`, `insert_round_states` and `mark_processed` each open their own connection, and the two bulk inserts commit every 500 rows. Atomicity across the seam is not merely absent — it is impossible in the current shape.

```python
# src/db.py:160-172
    c = _conn()
    try:
        for i in range(0, len(events), batch_size):
            ...
            c.commit()
```
```python
# src/pipeline.py:45-47
        db.insert_events(events)
        db.insert_round_states(states)
        db.mark_processed(match_id, map_name, demo_url)
```

**How it fails** Any exception after the first batch commits — Ctrl-C, disk full, `database is locked` after the 10 s timeout, or the `OperationalError` from HIGH-03 — commits a partial event set with no dedup marker. `pipeline.run` swallows it and continues. The next run's `is_processed` returns False, and the whole match is inserted again. There is no uniqueness constraint on `events` or `round_states` to stop it, and no cleanup path anywhere.

**Blast radius** Silent corpus duplication, which biases WP training toward re-ingested matches. **Verified not yet realised**: `round_states` has zero duplicate `(match_id, round_num, tick)` rows, and `processed_matches`, `events` and `round_states` all report exactly 710 distinct `match_id`s with 0 orphans. This is a live latent bug, not existing damage.

**Suggested fix** One connection for the whole match, one commit after `mark_processed`. Keep batching for memory; drop the per-batch commit.

---

### HIGH-02 — `_init_schema` lacks `p_before`/`p_after`/`impact`; a fresh database cannot run `status` or `score`

**File:** `src/db.py:50` · **Category:** E. Database · **Confidence:** high (reproduced) · **Rule:** P9

The `events` DDL declares 11 columns ending at `round_won`. The live DB has three more, added out of band. `score_impact.py:107` writes them; `main.py:47`, `:144` and `:218` read them.

**How it fails** Clean checkout → `demo demos/mirage.dem` succeeds → `status` dies with `OperationalError: no such column: impact` before printing anything, and `score` dies on the first `UPDATE`. Three of the five steps in README's getting-started block are non-functional on a fresh install.

**Suggested fix** Add `p_before REAL, p_after REAL, impact REAL` to the `events` CREATE TABLE, plus an `ALTER TABLE` migration for existing DBs lacking them.

---

### HIGH-03 — Live DB lacks four columns `insert_round_states` writes; ingestion into it fails today

**File:** `src/db.py:188` · **Category:** E. Database · **Confidence:** high (reproduced) · **Rule:** P9, P1

`insert_round_states` binds 41 columns including `active_smokes_xy`, `active_infernos_xy`, `site_smoked`, `site_on_fire`. `PRAGMA table_info(round_states)` on the live DB returns 38 columns (37 data) and contains none of those four.

**How it fails** `insert_events` completes and commits. `insert_round_states` then raises `OperationalError: table round_states has no column named active_smokes_xy` on its first `executemany`. `mark_processed` never runs. This is HIGH-01's crash window, reachable on the very next ingest, every time. The live database cannot accept a new match at all right now.

**Suggested fix** Additive migration: four `ALTER TABLE round_states ADD COLUMN` statements. Requires the migrate-vs-recreate decision.

---

### HIGH-04 — 9.5% of training rows are post-decision states with a side already eliminated

**File:** `src/state_sampler.py:248` · **Category:** G. Model correctness · **Confidence:** high (verified) · **Rule:** P6

Sampling runs to `official_end` (`round_officially_ended`), which trails the round's decision by the post-round restart delay. The only skip guards are an empty snapshot and *both* sides at zero — a 0-vs-3 post-round state passes both.

**How it fails** Verified: 111,925 of 1,178,871 rows (9.5%) have `alive_ct = 0` or `alive_t = 0`. Of those, 40,454 have `alive_ct = 0` with the label saying T won — trivially separable rows the model is scored on. Combined with CRIT-01's `time_remaining_s → 0` on those same rows, they are free AUC.

**Suggested fix** Filter `WHERE alive_ct > 0 AND alive_t > 0` in `load_training_data`. Do **not** bound the sampler at `end` instead — `score_impact` depends on post-round samples existing to give the round-deciding kill a `p_after`. Filtering at training time decouples the two cleanly.

---

### HIGH-05 — All events inside a ~1 s window receive one identical full-window WP delta

**File:** `src/score_impact.py:44` · **Category:** F. Pipeline correctness · **Confidence:** high (verified) · **Rule:** P5

`_impact_for_event` brackets an event between the two nearest 1 Hz samples, so `p_after - p_before` is the WP change over a whole second, handed identically to every event in that second.

**How it fails** Verified: 244,906 of 489,940 scored non-zero events (**49%**) share an `|impact|` value with at least one other event in the same round. 115,022 of those brackets contain **both** a CT and a T actor — meaning a kill and its trade are recorded as equal-and-opposite full-window swings, so the opening kill of a traded exchange is attributed to whichever side happened to net out ahead over that second.

All 157,074 `buy` events carry a hardcoded `time_into_round = 0.0` (`MIN`/`MAX`/`COUNT(DISTINCT)` all confirm a single value), which is exactly the first sample's timestamp — so every buy in a round would collapse onto one bracket if buys were scored.

**Suggested fix** Attribute at event resolution rather than sample resolution, or split a shared window's delta across its events. Exclude `buy` from impact scoring — it has no instantaneous WP effect.

---

### HIGH-06 — `test_local_demo.py` writes rows under `test_local` but dedups and marks a different id

**File:** `test/test_local_demo.py:45` · **Category:** K. Tests · **Confidence:** high · **Rule:** P1

Events and states are built with a hardcoded `match_id="test_local"` (lines 33, 41); the dedup check and `mark_processed` use `f"local_{demo_path...}"` (lines 45–52). The guard therefore protects an id that appears on zero inserted rows. `src/main.py:95` does this correctly, computing one id and passing it everywhere.

**How it fails** Running the script on two demos merges both into one pseudo-match `test_local`, and creates two `processed_matches` rows referencing zero events. Those orphan markers are then scored as empty, so the real rows keep NULL impact forever. The path-derived key is also not normalised — `demos/x.dem` and `./demos/x.dem` produce different ids for the same file.

**Verified not yet realised**: zero rows with `match_id LIKE 'test%'` or `'local%'` exist in the live DB, and zero orphan `processed_matches` rows. The script has never been run against the production database.

**Suggested fix** Compute `match_id = f"local_{Path(demo_path).stem}"` once and use it for all four calls, mirroring `main.py:95`.

---

### HIGH-07 — `pipeline.run` reports attempted matches as successful; no failure count, always exits 0

**File:** `src/pipeline.py:80` · **Category:** I. Error handling · **Confidence:** high

`i` increments before the `try`; the handler prints and discards. The summary reports `i` — the attempt count. `main()` discards every return value and never calls `sys.exit`.

**How it fails** 50 matches yielded, 40 failing, ends with `done: 50 matches, +<events from the 10 that worked>` and exit code 0. The 40 error lines are buried behind thousands of `stored N/M events` progress lines. No supervisor or `&&` chain can distinguish a fully-successful run from a fully-failed one.

**Suggested fix** Count `n_ok`/`n_fail`, list failed match ids in the summary, return a status, and `sys.exit(1)` when anything failed.

---

### HIGH-08 — `cmd_player`'s actor filter is dead code; every reported statistic is over the wrong row set

**File:** `src/main.py:183` · **Category:** Correctness · **Confidence:** high

The filter loop parses `sit` and never uses it, then unconditionally appends every row. The comment admits the heuristic does not work. `actor_rows` is element-for-element identical to `rows`, which the SQL matched on the name appearing *anywhere* in `situation` or `action` — including as a spotted enemy or a victim.

**How it fails** `kills (as actor)` counts rows where the attacker is frequently somebody else who merely had the searched player in their `situation.enemies_spotted`. `total impact` sums signed impact across other players and both sides, so the signs cancel arbitrarily; `avg impact` divides by that inflated denominator. Every number in the output block is wrong and nothing signals it. Compounded by `LIMIT 5000` with no `ORDER BY`, so "top N by |impact|" is the top N of an arbitrary 5000-row slice.

**Suggested fix** The event schema has no actor column, which is the root cause. Either add one, or relabel the output as "events mentioning `<name>`" so it is not presented as a per-player statistic.

---

### HIGH-09 — `.gitignore` misses every `.env` variant on a public repo

**File:** `.gitignore:21` · **Category:** L. Config · **Confidence:** high · **Rule:** P10

`.env` and `*.env` both require the final path component to *end* in `.env`. Verified with `git check-ignore -v`: `.env.local` and `.env.backup` are **not ignored**.

**How it fails** `cp .env .env.backup` before rotating the key, then `git add -A` — the live FACEIT key lands in a public repo. Irreversible; requires rotation and re-requesting the Downloads grant.

**Verified clean today**: `git ls-files | grep -i env` returns only `.env.example`, and `git log -p --all -S 'FACEIT_API_KEY='` touches only `.env.example` and `README.md`, both placeholders. No key has ever been committed.

**Suggested fix** `.env*` followed by `!.env.example`.

---

### HIGH-10 — Permission allowlist auto-approves arbitrary code execution and package installation

**File:** `.claude/settings.json:4` · **Category:** L. Config · **Confidence:** high

```json
"Bash(python -c ' *)", "Bash(python3 -c ' *)", "Bash(python -)",
"Bash(pip install *)", "Bash(python3 -)"
```
No `deny` block exists. On this machine `python3` and `pip` both resolve to working system interpreters (bare `python` does not).

**How it fails** `python3 -c` and `python3 -` are live, pre-approved arbitrary-code-execution primitives — enough to read `.env` or the Chrome profile with no prompt. `pip install *` executes arbitrary package install-time code as the user, *outside* the venv, directly contradicting CLAUDE.md. The permission dialog is the only control between an agent and the two assets this project's own rubric names as worst-case.

**Suggested fix** Delete those five entries. Nothing in the documented workflow needs them.

---

### HIGH-11 — `.venv` has no site-packages; every documented command except `status` fails at import

**File:** `requirements.txt:1` · **Category:** L. Config · **Confidence:** high (reproduced) · **Rule:** P15

`.venv` is 108K: `bin/`, `include/`, `share/`, `pyvenv.cfg` — no `lib/`. `numpy`, `pandas`, `lightgbm`, `awpy`, `requests` all `ModuleNotFoundError`. `.venv/bin/python -m pip` → `No module named pip`, so the documented repair route is itself blocked. The interpreter is fine (3.11.15 arm64) and orphaned console scripts remain in `bin/`, so packages were installed once and `lib/` was later removed.

**How it fails** `demo`, `train`, `score`, `scrape` and all three test scripts die at import. `status` works only because it touches nothing but stdlib — which is precisely what masks the breakage. Every "verify before declaring done" step in CLAUDE.md is currently impossible.

**Suggested fix** `python3.11 -m venv --clear .venv` then `.venv/bin/python -m pip install -r requirements.txt`. **Pin the dependencies first** (MED-01) — a fresh unpinned resolve will not reproduce the toolchain that parsed the existing 710-match corpus.

---

### HIGH-12 — `get_player_match_ids` returns `[]` on every exception; auth failure is indistinguishable from "no matches"

**File:** `src/scraper.py:124` · **Category:** I. Error handling · **Confidence:** high

The `try` spans both API calls, and `_api_get` has no 429 handling. A 401, 403, 429, connection error and JSON decode error all become an empty list — identical to a player with no CS2 history.

**How it fails** With a revoked key, a full `scrape` walks every player, yields nothing, prints `no new demos found.` and a normal summary, and exits 0. Under `pipeline.loop()` this becomes a silent hourly no-op forever.

**Suggested fix** Narrow the except to genuine skip-this-player errors; let 401/403 propagate. Add 429/`Retry-After` handling inside `_api_get` so both source modules inherit it.

---

### HIGH-13 — Response interceptor accepts any URL containing a magic substring

**File:** `src/scraper.py:146` · **Category:** D. Untrusted input · **Confidence:** high

```python
        if any(ext in url for ext in (".dem.zst", ".dem.gz", "/download/demo", "download?token")):
```
A bare substring search over the whole URL, with no host allowlist and no path anchoring.

**How it fails** A matchroom renders third-party content. Any resource whose URL merely *contains* `.dem.zst` anywhere — including in a query parameter — is captured, yielded, then fetched and written to disk with no host check, no size bound and no content validation. `intercepted` is also unbounded. Related: because `handle_response` accepts any status below 400, a redirect can capture both the redirecting and final URL for one demo, making `len(demo_urls) > 1` true and splitting a BO1 into a fake `_m1`/`_m2` pair — the same demo inserted twice under two ids, which also puts one match on both sides of the `GroupShuffleSplit`.

**Suggested fix** Parse with `urlsplit`; require the netloc to be in a FACEIT/CDN allowlist and the *path* to end in `.dem`/`.dem.zst`/`.dem.gz`. Skip 3xx responses. Cap `intercepted`.

---

### HIGH-14 — `ct_spread`/`t_spread` return `0.0` for a 1-alive team, colliding with a genuine stack

**File:** `src/state_sampler.py:56` · **Category:** F. Pipeline correctness · **Confidence:** high (verified)

`_team_spread` returns `0.0` when fewer than two players contribute. The pairwise arithmetic itself is correct; the sentinel is the defect.

**How it fails** Verified: **all 149,833** rows with `alive_ct = 1` have `ct_spread = 0.0`, versus **3** rows where `alive_ct > 1` and the spread is genuinely zero. So in the training distribution, `spread = 0.0` overwhelmingly means "one player left" while the feature's semantics say "five players stacked tightly together" — the opposite situation. Every 1vX clutch, the highest-leverage state in the game, is encoded as its inverse.

**Suggested fix** Return `None` (SQL NULL); LightGBM handles missing natively and it is distinguishable from a real zero.

---

### HIGH-15 — `_smoke_impact` counts tick rows instead of unique players

**File:** `src/feature_extractor.py:317` · **Category:** F. Pipeline correctness · **Confidence:** high

`_near_count` slices a ±32-tick window and counts rows satisfying a radius test. `r_ticks_df` holds one row per player per tick, so a single enemy standing near the smoke for the whole 1 s window contributes up to 65 rows. `state_sampler.py:252` collapses the identical window with `groupby("name").last()` before counting; `_flash_impact` dedupes via `groupby(["name","side"])`. This one call site does neither.

**How it fails** `enemies_at_pop` and `enemies_mid_smoke` report values like 65 or 130 instead of 1 or 2, in the `outcome` blob of every smoke utility event in the DB. Because the values are unbounded rather than clipped at 5, the error does not look like a clipped count.

**Suggested fix** Dedupe by player before counting, mirroring `state_sampler.py:252`.

---

### HIGH-16 — BO-series `_m1` dedup shortcut permanently abandons maps 2 and 3

**File:** `src/scraper.py:238` · **Category:** A. Invariants · **Confidence:** high · **Rule:** P1

```python
            if is_processed(mid) or is_processed(f"{mid}_m1"):
                print(f"    skipping {mid} (already in db)")
```
The outer guard treats `<mid>_m1` as proof the whole series was ingested.

**How it fails** `pipeline.run` catches per-match exceptions and continues, so a BO3 where map 1 succeeded and map 2 failed leaves `_m1` present and `_m2`/`_m3` absent. On the next run this guard fires *before* `get_demo_urls_from_page` is ever called, and the missing maps are never retried — with a log line reading "already in db". The per-URL check that would handle it correctly is unreachable. `faceit_api.py:83` has the identical hole. The live DB holds 15 `_m<n>`-suffixed rows.

**Suggested fix** Short-circuit only on `is_processed(mid)` and rely on the already-correct per-URL guard.

---

### HIGH-17 — README's documented getting-started sequence cannot work on a clean checkout

**File:** `README.md:88` · **Category:** L. Docs vs reality · **Confidence:** high

Steps 2, 4 and 5 (`status`, `score`, `top`) all fail on a fresh DB because of HIGH-02. CLAUDE.md documents the trap; README does not mention it.

**Suggested fix** Fix HIGH-02, which fixes this. Until then, note the limitation in README.

---

### HIGH-18 — CLAUDE.md names the test suite as the calibration guard; no check measures calibration

**File:** `test/test_win_probability.py:83` · **Category:** K. Tests · **Confidence:** high · **Rule:** P7

CLAUDE.md: *"Impact scoring needs calibrated probabilities... Changes to the model/objective must keep `test/test_win_probability.py` passing."* The 23 checks are 9 tautological range assertions, 13 pure ranking comparisons, and 3 loose magnitude bounds. No Brier score, no log-loss, no reliability binning.

**How it fails** Apply any monotone distortion — `p → 0.5 + 0.9*(p-0.5)` — and all 23 checks still pass while every impact delta is wrong by a state-dependent factor. The suite reports 23/23 for a model whose calibration has been destroyed.

**Suggested fix** Add a reliability check: bin held-out states by predicted probability and assert the observed CT win rate in each bin is within tolerance of the bin centre.

---

### HIGH-19 — 13 of 15 project invariants have no test of any kind

**File:** `test/` · **Category:** K. Tests · **Confidence:** high

P1, P1b, P2, P3, P4, P5, P6, P8, P9, P10, P11 have no test. P7 has a test that does not test the property it claims (HIGH-18). P12/P13/P15 are process rules. Only P14 holds by construction.

The cheap ones would have caught real bugs in this report: a 6-line `PRAGMA table_info` comparison against the INSERT column lists catches HIGH-02 **and** HIGH-03; a 3-line `inspect.signature` comparison covers P4; calling `_impact_for_event` with a synthetic array and asserting the sign flips covers P5 and would have caught the `"?"`-side bug. None need a trained model or a demo.

---

## MEDIUM

| ID | Title | File | Fix |
|---|---|---|---|
| MED-01 | `awpy`, `pandas`, `numpy`, `lightgbm` unpinned with no upper bound and no lockfile; awpy is the sole parser and the version that produced the corpus is recorded nowhere | `requirements.txt:2` | Pin exact versions; record the awpy version that parsed the 710 matches |
| MED-02 | `r.get("official_end") or r.get("end")` — NaN is truthy, so the fallback never fires and the None-guard cannot catch it; round is silently dropped | `state_sampler.py:216`, `feature_extractor.py:749` | NaN-aware `_first_valid` helper |
| MED-03 | `damage_after` reads a `damage` column awpy does not emit — **verified 0 in 40,000/40,000 engagement events**; the same file's `_grenade_damage` already knows the column is `dmg_health` | `feature_extractor.py:1024` | Reuse the column resolution at line 423; emit `None` not `0` |
| MED-04 | `player_side = "?"` takes the T branch in impact scoring — **verified 1,080 such events, 388 already scored** with a flipped sign | `score_impact.py:58` | Explicit ct/t branches; `impact = None` otherwise |
| MED-05 | `predict_batch` raises on any batch spanning >1 map (categorical concat degrades to object dtype) | `win_probability.py:151` | Build one frame for the whole batch |
| MED-06 | Early stopping selects the iteration on the same val set the reported AUC is computed on | `win_probability.py:115` | Three-way split, or stop calling it held-out |
| MED-07 | `tick_rate = 64` hardcoded in 3 files; `extract.py` never captures the demo's real tickrate | `state_sampler.py:187` | Expose `tables["tickrate"]`; thread through |
| MED-08 | `extract.py` swallows smokes/infernos/bomb/footsteps failures with `except Exception: pass` — 6+ model features silently degrade to constants | `extract.py:70` | Catch `AttributeError` only; warn on every miss |
| MED-09 | `cmd_demo` duplicates `process_one` and has already drifted (parses the whole demo *before* the dedup check; misses the `_m1` form) | `main.py:91` | Hoist the check; share one function |
| MED-10 | `--event-type` is unconstrained and interpolated into SQL; `--side`/`--by` are constrained by `choices=` | `main.py:128` | Parameterise |
| MED-11 | Player name interpolated into a `LIKE` pattern: a `'` crashes the query, and `_`/`%` are wildcards that silently match other players | `main.py:171` | Parameterise + `ESCAPE` |
| MED-12 | `--limit 0` means "score every match" (falsy check on a numeric) | `score_impact.py:127`, `main.py:119` | `is not None` |
| MED-13 | `cmd_score` bypasses the DB-existence guard and `sqlite3.connect` creates an empty file, permanently disabling the friendly error | `main.py:120` | Check existence first; open with `mode=rw` |
| MED-14 | `cmd_score` mutates global `sys.argv` and never restores it | `main.py:117` | Pass an argv list |
| MED-15 | Events in a round with no states silently keep impact values from a previous model | `score_impact.py:95` | Clear the match's impact columns before rescoring |
| MED-16 | NULL `time_into_round` is scored as `t=0.0`, which is a real sample timestamp | `score_impact.py:97` | Skip instead |
| MED-17 | `_flash_impact` attributes every flash in a 10 s window to one thrower | `feature_extractor.py:387` | Detect rising edge; narrow the window |
| MED-18 | `trade_kill` breaks on the first spatially-near death, preferring the older one | `feature_extractor.py:948` | Scan all candidates |
| MED-19 | Column-less `pd.DataFrame()` reaches `.sort_values("tick")` → `KeyError` aborts the whole match | `feature_extractor.py:1033` | Preserve columns via `rg.iloc[0:0]` |
| MED-20 | `round(float(nan))` raises on the buy path; `if x` does not catch NaN | `feature_extractor.py:888` | Use `_nan()` guard |
| MED-21 | `demo_url` differs by source (signed CDN URL vs stable resource URL), violating the identical-rows spec | `pipeline.py:47` | Normalise before `mark_processed` |

## LOW

| ID | Title | File | Fix |
|---|---|---|---|
| LOW-01 | 6 stale allowlist entries name files that no longer exist (`src/wp_model.py`, `build_roadmap_pdf.py`, `test/test_extract.py`, `--data-dir`) | `.claude/settings.json:6` | Delete |
| LOW-02 | `.gitignore` misses `data/*.db-wal`, `data/backup.db`, `*.dem.zst`, `*.dem.gz`, unanchored `*.lgb` | `.gitignore:14` | Broaden patterns |
| LOW-03 | `openai` and `playwright-stealth` declared but imported nowhere | `requirements.txt:14` | Drop or move to an extra |
| LOW-04 | `pyproject.toml` console script points at a nonexistent `crosshair` package; omits lightgbm/scikit-learn/requests; no `[build-system]` | `pyproject.toml:17` | Delete or make real — do not half-fix |
| LOW-05 | `.claude/` gitignored, so the audit skill and allowlist are uncommittable and have no history | `.gitignore:32` | `.claude/*` + `!.claude/skills/` if versioning is wanted |
| LOW-06 | `_model` global never invalidated after `train()` — latent for in-process retrain-then-score | `win_probability.py:134` | Assign `_model` at the end of `train()` |
| LOW-07 | `str(row.get("map") or "unknown")` — `np.nan` is truthy, so the fallback yields `'nan'` | `win_probability.py:157` | Explicit NaN check |
| LOW-08 | `predict_batch([])` raises `ValueError` from `pd.concat` | `win_probability.py:151` | Early return |
| LOW-09 | Hand-rolled `bisect_right` handles exactly one duplicate timestamp | `score_impact.py:48` | Use `bisect_right` directly |
| LOW-10 | `wp_all[idx]` is label-indexing used positionally — correct today only because `read_sql_query` returns a RangeIndex | `score_impact.py:81` | `reset_index(drop=True)` to make it explicit |
| LOW-11 | `_dist2`/`_dist3` return `0.0` on error, fabricating a "close" sound cue that passes every range test | `feature_extractor.py:61` | Return `inf` |
| LOW-12 | `_alive_counts_from_snap` returns the strings `"?","?"` — **verified in 210/40,000 events** | `feature_extractor.py:166` | Return `None` |
| LOW-13 | Unknown-side thrower silently becomes CT for every grenade impact calculation | `feature_extractor.py:1103` | Fall back to `name_to_side`; skip if unresolved |
| LOW-14 | `extract.py`'s own documented CLI always crashes — `map_name` is a `str` in a dict iterated as DataFrames | `extract.py:57` | Skip non-frames in the print loop |
| LOW-15 | `_team_spread`/`_min_dist_to_xy` guard on `"X"` then index `"Y"`; `_active_xy` checks neither `end_tick` nor `Y` | `state_sampler.py:84` | Guard the full column set |
| LOW-16 | `groupby("name").last()` composes a row from different ticks (last non-null *per column*) | `state_sampler.py:252` | `drop_duplicates(subset="name", keep="last")` |
| LOW-17 | Test docstrings instruct bare `python`/`python3`, contradicting the `.venv/bin/python` rule | `test_win_probability.py:5` | Use `.venv/bin/python` |

---

## Rule compliance

| Rule | Statement | Status | Evidence |
|---|---|---|---|
| P1 | No path may double-insert a match | ✗ VIOLATED | HIGH-01, HIGH-03, HIGH-06, HIGH-16 — four independent windows. Corpus verified clean today |
| P1b | Uniqueness constraint behind the dedup | ✗ VIOLATED | No UNIQUE on `events`/`round_states`; check-then-act only |
| P2 | Both sources produce identical downstream rows | ✗ VIOLATED | MED-21 — `demo_url` differs by source |
| P3 | All ingestion converges on the seam | ✗ VIOLATED | MED-09 — `cmd_demo` and `test_local_demo.py` both fork it |
| P4 | Source modules expose the two-function interface | ✓ HOLDS | Both accept the same 5 kwargs; `pipeline` calls nothing else |
| P5 | Impact sign: CT `p_after - p_before`, T negated | ⚠ PARTIAL | Casing normalised at `score_impact.py:98`, but MED-04 — `"?"` takes the T branch |
| P6 | `GroupShuffleSplit(groups=match_id)`, no leakage | ✗ VIOLATED | Splitter is correct; CRIT-01, CRIT-02, HIGH-04 leak *inside every row*, which grouping cannot fix |
| P7 | Probabilities stay calibrated | ⚠ UNENFORCED | HIGH-18 — the named guard measures ranking, not calibration |
| P8 | Feature parity across sampler → DB → FEATURES | ✓ HOLDS | All 41 emitted keys match the 41 insert params; all 33 FEATURES exist as columns |
| P9 | Additive-only schema with a migration | ✗ VIOLATED | HIGH-02, HIGH-03; no `ALTER TABLE` anywhere, no `schema_version` |
| P10 | Secrets via env only, never committed | ⚠ PARTIAL | No key ever committed (verified against full history). But CRIT-03 leaks signed URLs and HIGH-09 leaves a gap |
| P11 | No invented FACEIT API surface | ⚠ PARTIAL | Mostly defensive `.get()`; `resp.json()["payload"]["download_url"]` is asserted, not checked |
| P12 | `scraper.py` is ask-before-touch | ✓ HOLDS | Untouched by this audit; findings reported only |
| P13 | `/security-review` before every commit | ⚠ UNENFORCED | No hooks in `.claude/settings.json`; commit `6d48589` — which added the key-handling module — has no review line |
| P14 | Tests stay plain scripts | ✓ HOLDS | No pytest present |
| P15 | `.venv/bin/python`, requirements authoritative | ✗ VIOLATED | HIGH-11 — the venv is empty; LOW-17 — docs say bare `python` |

---

## Coverage

**In scope:** 25 · **Read in full:** 25 · **Unread:** 0

| File | Lines | Status |
|---|---|---|
| `src/feature_extractor.py` | 1351 | read in full |
| `src/state_sampler.py` | 347 | read in full |
| `src/main.py` | 306 | read in full |
| `src/scraper.py` | 302 | read in full |
| `src/db.py` | 230 | read in full |
| `src/win_probability.py` | 176 | read in full |
| `src/score_impact.py` | 147 | read in full |
| `src/faceit_api.py` | 146 | read in full |
| `src/pipeline.py` | 109 | read in full |
| `src/extract.py` | 88 | read in full |
| `src/__init__.py` | 0 | read in full |
| `test/test_win_probability.py` | 216 | read in full |
| `test/test_local_demo.py` | 68 | read in full |
| `test/test_scraper.py` | 13 | read in full |
| `README.md`, `CLAUDE.md`, `PROMPT_faceit_api_downloader.md` | — | read in full |
| `requirements.txt`, `pyproject.toml`, `.gitignore`, `.env.example` | — | read in full |
| `.claude/settings.json`, `.claude/skills/full-codebase-check/**` (3) | — | read in full |

**Deliberately excluded** (Non-negotiable 2, not gaps): `.env`, `.venv/**`, `data/crosshair.db` row contents (schema via `PRAGMA` and aggregate `COUNT`/`AVG` only), `demos/*.dem`, Chrome profile.

---

## Unverifiable

| # | Suspected | Why unverifiable | Check by hand |
|---|---|---|---|
| U-1 | 82% of events (2,271,463) have NULL impact — `rotation` and `buy` are **0% scored** while `engagement`/`utility`/`bomb` are 99–100%. Current `score_impact.py` has no event-type filter. Combined with CRIT-04 (the stored data also predates the T-negation), the stored scoring is near-certainly from an older scorer | Cannot re-run `score` — no lightgbm, no trained model | After rebuilding the venv and training: `main.py score`, then re-check `SELECT event_type, COUNT(*) FROM events WHERE impact IS NULL GROUP BY 1` and re-run the CRIT-04 sign check. If either still fails, it is a live bug |
| U-2 | Whether awpy's `inventory` elements are `list` and `rounds.winner` is lowercase | awpy not importable | `.venv/bin/python -c "from src.extract import extract; t=extract('demos/mirage.dem'); print(type(t['ticks']['inventory'].iloc[0]), t['rounds']['winner'].unique())"` — note the live DB's balanced `round_won_ct` (592,842/586,029) already proves casing is correct *today* |
| U-3 | Real magnitude of the AUC drop once CRIT-01/CRIT-02/HIGH-04 are fixed | Requires a retrain | `main.py train --eval` before and after |
| U-4 | Known CVEs in the resolved dependency set | No pip, no installed distributions, no `pip-audit` | `.venv/bin/python -m pip list --outdated` after the venv is rebuilt |
| U-5 | Whether `.env` contains only the expected keys | Blocked path | `grep -c . .env` and confirm the key count matches `.env.example` — no need to print values |

---

## Not findings

Everything Phase 3 refuted, so the next audit does not re-raise it.

| Looked like | Why it's fine |
|---|---|
| **The `map` categorical is broken at predict time** — single-row `astype("category")` gives every map code 0 | LightGBM persists `pandas_categorical` in the model file and `_data_from_pandas` calls `cat.set_categories(...)`, which is a **label**-keyed reindex. The single-row frame is widened to the full training vocabulary before `.cat.codes` runs. Would only break if the booster were trained from numpy |
| **`post_plant` is always False** — plant ticks collected only from tables with `round_num` | Verified: 6,598/40,000 engagement events have `post_plant: True`, and `round_states` independently agrees at 18.6%. The `bomb` table does carry `round_num` |
| **`round_won_ct` is a constant 0** from a `"CT"` vs `"ct"` casing mismatch | Verified: 592,842 / 586,029 — a clean 50.3/49.7 split. Casing is correct today. (The *absence of normalisation* remains a real latent risk — see MED-02's neighbourhood — but the label is not currently corrupt) |
| **All 8 utility features are constant 0** — `_count_util` rejects non-`list` inventories | Verified: `he_ct` avg 0.922, max 5, 6 distinct values; every util column has a real distribution. Inventory arrives as a `list` and the `he` keywords match |
| **`he_ct`/`he_t` specifically are 0** — keyword list misses the `"he grenade"` spelling | Same verification: `he_ct` is well-distributed. Refuted |
| **`--side`/`--by` are SQL injection vectors** | Both constrained by `argparse choices=`; `--by` additionally passes through a literal dict with a safe `.get` default |
| **`LIMIT {args.n}` is injectable** | `-n` is `type=int`; argparse rejects non-integers before the f-string is built. Real finding is the missing defence-in-depth (LOW), not a live hole |
| **`pipeline.loop` is a silent hot loop** | `time.sleep(interval)` is outside the `try`, so a persistently-failing `run()` still sleeps the full interval. Silent, but not hot |
| **`MODEL_PATH.parent.mkdir(exist_ok=True)` fails if `data/` is absent** | `parents=False` only raises when the *grand*parent is missing. Unreachable anyway — `load_training_data` connects to a DB inside `data/` first |
| **Pre-split `astype("category")` is P6 leakage** | It fits a label→code enumeration only, not a target-derived statistic. No scaler, imputer, or target encoder exists |
| **`with_suffix(".dem.zst")` mishandles ids containing dots** | Verified on 3.11: `Path("…/1-c566fc66-….dem").with_suffix(".dem.zst")` behaves correctly. The defect is the *URL substring* format detection, not the path building |
| **Authorization header leaks on cross-host redirect** | `requests`' `Session.rebuild_auth` strips it. The demo fetch sends no auth header at all — its credential is in the URL, which is CRIT-03 |
| **Playwright selectors are injectable via nicknames** | Built only from constants and an integer loop variable. No `page.evaluate`, no `inner_html` |
| **`_api_headers` can emit a literal `"None"` bearer token** | Both `_api_headers` and `_bearer` raise `EnvironmentError` on a missing key |
| **`_impact_for_event` has an off-by-one at round boundaries** | Hand-walked all five cases (before first, after last, exactly on a sample, single state, empty). All correct. Only the duplicate-timestamp case is imperfect (LOW-09) |
| **Player name→side map is built once per match, breaking at halftime** | `name_to_side` is rebuilt inside the round loop from that round's tick slice. Side swaps handled correctly in both `feature_extractor` and `state_sampler` |
| **538,781 "duplicate event groups" indicate corpus duplication** | Grouping by `(match_id, round_num, time_into_round, event_type, player_side)` collides legitimately — group sizes cluster at 2–5, exactly team size, and 95% are `rotation`/`buy` where teammates share a timestamp. `round_states` has **zero** duplicate `(match_id, round_num, tick)` rows, which is decisive |

---

## Cost

9 shard agents + adversarial verification against the live DB. ~570k subagent tokens, ~10 min wall clock for the parallel phase.

---

## Remediation log — 2026-08-01

Applied in the same session, immediately after this report. The audit itself was
read-only; these are a separate, user-initiated fix pass.

### Fixed

| ID | Change |
|---|---|
| CRIT-01 | `time_remaining_s` now derives from the round clock (`115 − time_into_round`) pre-plant and the C4 timer post-plant. `r_end` bounds only the tick slice and the sample loop |
| CRIT-02 | `_info_state` looks **backwards only**, within `HEARD_WINDOW`; returns zeros when no past sighting exists |
| CRIT-03 | Added `pipeline.redact_url` / `redact_text`; applied to `mark_processed` and to the per-match error print, the two sinks that carried signed URLs |
| CRIT-04 | Re-score required (data fix, not code). `score_match` now NULLs a match's impact columns before rewriting, so a re-score can't leave a mix of scorers |
| HIGH-04 | `load_training_data` excludes `alive_ct = 0 or alive_t = 0`. Deliberately filtered at training, not in the sampler — `score_impact` needs post-round samples for `p_after` |
| HIGH-05 | Partial: `_impact_for_event` now uses a duplicate-safe `bisect_right`. The shared-1 s-bracket attribution is a design change, left for a decision |
| HIGH-06 | One `match_id` (`local_<stem>`) used for rows, dedup and marker; DB write now opt-in behind `--write-db` |
| HIGH-07 | `run()` counts ok/failed, lists failures, returns a count; `main()` and `pipeline` exit non-zero |
| HIGH-08 | Dead actor filter removed; output relabelled "events mentioning `<name>`", `ORDER BY ABS(impact) DESC` added, misleading aggregates dropped |
| HIGH-09 | `.gitignore` → `.env*` + `!.env.example`; verified with `git check-ignore` |
| HIGH-10 | Removed `pip install *`, `python -c`, `python3 -c`, `python -`, `python3 -` and 8 stale entries; scoped to `.venv/bin/python src/main.py <sub>` |
| HIGH-12 | `faceit_api` raises `PermissionError` on 401/403 instead of returning `[]` (scraper side is ask-before) |
| HIGH-14 | `_team_spread` returns `None`, not `0.0`, for <2 alive |
| HIGH-15 | `_near_count` dedupes by player before counting |
| HIGH-16 | `faceit_api` no longer short-circuits on `<mid>_m1` (scraper side is ask-before) |
| HIGH-18 | Added a reliability-binning calibration check to `test_win_probability.py` |
| HIGH-19 | New `test/test_invariants.py` — stdlib-only, covers P8/P9/P4/P1; currently 11/13, failing exactly on HIGH-02 and HIGH-03 |
| MED-02 | `_first_valid` helper in both `state_sampler` and `feature_extractor` |
| MED-03 | `damage_after` resolves `dmg_health` / `dmg_health_real` / `damage`; `None` when absent |
| MED-04 | Unknown side yields `impact = None` instead of silently taking the T branch |
| MED-05 | `predict_batch` builds one frame; mixed-map batches work |
| MED-06 | `train --eval` states that early stopping used the same fold |
| MED-07 | `extract.py` captures `tick_rate`; both consumers read it; warns when ≠ 64 |
| MED-08 | `extract.py` catches `AttributeError` only, and warns |
| MED-09 | `cmd_demo` checks dedup **before** the expensive parse, and checks the `_m1` form |
| MED-10/11 | `cmd_top` and `cmd_player` fully parameterised; LIKE metacharacters escaped |
| MED-12 | `--limit 0` now means zero, in both places |
| MED-13/14 | `cmd_score` passes an argv list, checks DB existence, opens `mode=rw` |
| MED-15/16 | Stale impact cleared before rescoring; NULL `time_into_round` skipped, not scored as `t=0` |
| MED-18/19/20 | `trade_kill` scans all candidates and is `None` for victims; empty grenade frames keep their columns; `round(nan)` guarded |
| MED-21 | `demo_url` normalised via `redact_url` before `mark_processed` |
| LOW-01/02/05 | Allowlist pruned; `.gitignore` broadened; `.claude/skills/` now committable |
| LOW-06/07/08 | `_model` refreshed after `train`; NaN map label; `predict_batch([])` returns empty |
| LOW-09/10 | Duplicate-safe bisect; explicit `reset_index` |
| LOW-11/12/13 | Distances return `inf`; `"?"` sentinels → `None`; unknown thrower side no longer defaults to CT |
| LOW-14 | `extract.py`'s CLI handles scalar entries |
| LOW-15/16 | Full column-set guards; `drop_duplicates` instead of `groupby().last()` |
| LOW-17 | Docs and docstrings use `.venv/bin/python` |
| MED-01 | Upper bounds added to every dependency (`awpy>=2.0.2,<3` etc.) — ceilings, not the corpus's resolution |

### Not fixed — needs a decision (see the questions in the session summary)

| ID | Why |
|---|---|
| HIGH-02, HIGH-03 | DB schema change + migration. CLAUDE.md requires sign-off, and migrate-vs-recreate is a judgement call |
| HIGH-01 | The transactional fix changes `db.py`'s insert API; adjacent to the schema decision |
| HIGH-11 | Rebuilding the venv re-resolves dependencies away from whatever parsed the 710-match corpus |
| HIGH-13, HIGH-16 (scraper side), CRIT-03 (source-side redaction), MED download validation | All in `src/scraper.py` — ask-before-touch under P12. The credential leak was closed at the `pipeline` sinks instead |
| LOW-03, LOW-04 | Removing unused deps and the `pyproject.toml` decision are deletions |

### Verification performed

- `py_compile` on all 15 Python files — clean.
- `test/test_invariants.py` — 11/13, failing precisely on the two unfixed schema items.
- 27/27 assertions against the **shipped** `_impact_for_event`, `redact_url`,
  `redact_text`, `_first_valid` and `_info_state`, extracted by AST so the harness runs
  without numpy/pandas.
- `main.py status`, `top`, `top --event-type`, `top --side` all run; the previously
  crashing `--event-type "O'Neill"` and the clause-subverting `--event-type "x' OR '1'='1"`
  now behave correctly.
- **Not verified:** anything requiring awpy, pandas or lightgbm — the venv is empty
  (HIGH-11). No re-ingest, no retrain, no re-score was possible.
