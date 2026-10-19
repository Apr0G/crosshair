# Audit checklist — Crosshair

## The precision standard

**You are checking code, not vibes.** Every item below resolves to a specific line of
Python that is either right or wrong. What you are hunting:

- an off-by-one in a slice or a `range`
- a `>=` that should be `>` at a round boundary
- an `or` that should be `and` in a guard
- a filter on the wrong column, or a join on the wrong key
- a missing `await` — or in this codebase, a missing `.commit()`
- a variable read before assignment on one branch
- a `None` that becomes `0` and silently changes a mean
- a loop that mutates what it iterates
- an `except: pass` swallowing the error that mattered
- a default argument that is a mutable list
- a comparison against a string that the API actually returns capitalised

Not "this module feels fragile."

**An item that does not apply is different from an item you skipped.** Say which. Write
`N/A — no HTTP server in this repo` when a section genuinely does not apply. Never write
`✓` for something you did not actually look at. The Coverage table in the report exists to
make that distinction visible, and a false `✓` there is worse than an admitted gap.

**Two reads per file.** First pass for what the code is trying to do. Second pass, line by
line, against the sections below. Findings almost always come from the second pass.

**Legacy names.** Section A follows the engine (R1–R22, CLAUDE.md "Engine v2"). Sections
B–L were first written against the legacy `src/` pipeline, deleted in plan Amendment 1;
where an item names a deleted file or function (`main.py`, `pipeline.run`, `store_match`,
`state_sampler`, `win_probability`, `score_impact`, `data/crosshair.db`), apply it to the
engine counterpart in CLAUDE.md's module map (`engine/cli.py`, `engine.ingest.run`,
`engine.ingest.register_demo` / `engine/db.py`, `engine/state/`, `engine/model/`,
`engine/attribution/` + `engine/value/`, `data/engine/engine.db`).

---

## A. Project invariants (highest yield — check these first)

These come from `CLAUDE.md`. A violation is at minimum HIGH, even with no exploit path,
because each one silently corrupts data or invalidates model numbers.

- [ ] **R1 — no double-insert.** Trace every path that reaches `engine.ingest.register_demo`
      / `db.register_match`. Is `db.is_processed_any` checked before *every* download, for
      the id forms `id_forms` documents (bare `<mid>` + `<mid>_m1`; `<mid>_mN` + `<mid>`)?
      Is the insert one transaction, and does the UNIQUE `demo_sha256` back it up? What
      happens when the same key arrives with a different file?
- [ ] **R2 — source parity.** Do all sources (`faceit_api`, `scraper`, `hltv`, `hf_archive`)
      reach `ingest.process_one` and write the same row shape, with a redacted, meaningful
      `demo_url`? Any divergence means two paths write different rows.
- [ ] **R3 — sign convention.** CT-positive internally; a T actor's value is negated exactly
      once. Is the side string normalised (`ct`/`t`) before comparison?
- [ ] **R4 — splits.** Frozen month splits with match grouping (`engine/data/splits.py`),
      latest months test, files written once (mode `x`). Does anything fit vocab / norms /
      models on rows outside train? Does any command accept "no split" silently?
- [ ] **R5 — leakage.** Every feature has a `LIVE_SAFE` entry; nothing reads `end_tick`,
      the winner, or a future snapshot. Does `test_leakage.py` exercise the feature?
- [ ] **R6 — calibration.** ECE ≤ 0.02 on the baseline test tier; does the Predictor ever
      serve an uncalibrated output silently?
- [ ] **R7 / R8 — unknowns.** No `0` / `"?"` sentinel for unknown; no `or` used as a
      null-coalesce over numerics.
- [ ] **R9 — redaction.** Every print, exception, log, file and DB write of a URL goes
      through `redact_url` / `engine.log.redact_text`; no URL truncated by length.
- [ ] **R10 — tick rate.** From the demo header, never a literal; quarantine refuses
      non-128 on every path (batch too, not only `analyze`).
- [ ] **R11 — invented API surface.** Every FACEIT field accessed (`demo_url`, `payload`,
      `download_url`, `items`, `roster[].game_skill_level`, …) — accessed defensively and
      verified against the docs or a real response?
- [ ] **R12 — schema.** Additive migrations in `_MIGRATIONS` keyed by `user_version`;
      `_init_schema` and the migrations agree; a newer file is refused.
- [ ] **R13–R16 —** `sample_hz` untouched for latency; reports carry no outside paths, URLs
      or raw SteamIDs; non-training modules do not import torch at import time; tests are
      plain scripts that write only to temp dirs and exit 0/1/2.
- [ ] **R17–R22 —** secrets only via env / `.env`; `engine/sources/scraper.py` logic
      unchanged; the adoption gate and P7 gate hold; style; conservation, C1 and C2;
      exit codes 0/1/2 everywhere.

---

## B. Secrets and credential handling

- [ ] `os.getenv("FACEIT_API_KEY")` — every read site. Is a missing key a clear error, or
      does it become the literal string `"None"` in an `Authorization` header?
- [ ] Is the key ever included in a **printed** string, an f-string log, an exception
      message, or a `repr()` of a headers dict?
- [ ] `requests` exceptions: does `raise_for_status()` or an unhandled `HTTPError`
      traceback include the request URL with a token query parameter? Signed FACEIT URLs
      carry credentials **in the query string** — printing the URL prints the credential.
- [ ] Truncated URL prints (`url[:60]`, `url[:80]`, `url[:100]`, `url[:120]`) — do they cut
      before or after the signature? Verify against the real URL shape, and treat "it
      happens to truncate early" as luck, not a control.
- [ ] Is `.env` covered by `.gitignore` **and** is the pattern correct (`.env` vs `*.env`
      vs `.env*`)? Would `.env.local` or `.env.backup` be caught?
- [ ] `.env.example` — placeholders only. A real-looking key is a CRITICAL finding.
- [ ] Is the key sent to any host other than `open.faceit.com`? A redirect that preserves
      the `Authorization` header leaks it to the redirect target — check whether
      `allow_redirects` is left at its default on authenticated calls.
- [ ] Does any error path write a request/response dump to disk?

---

## C. The browser-automation boundary (highest-stakes surface here)

`engine/sources/scraper.py` launches Chrome against `~/Library/Application Support/Google/Chrome/Default`
— Faig's **real** profile, with live sessions for every site he uses.

- [ ] Confirm the profile path and whether a throwaway profile would work. **Report the risk;
      do not change it** (R18: ask-before).
- [ ] Can a page the scraper visits be attacker-influenced? Match IDs come from the API, but
      a matchroom renders user-controlled content (nicknames, team names). Any
      `page.evaluate`, `inner_html`, or unsanitised selector interpolation?
- [ ] Selectors built by f-string from non-constant data — a nickname containing a quote
      breaks or redirects a locator.
- [ ] Is the browser context ever closed? `_browser_ctx` is a module global holding a live
      Playwright process — trace the shutdown path. A leaked Chrome process holding the real
      profile after a crash is a real finding.
- [ ] `download.cancel()` and the `response` handler: can a matchroom cause an unbounded
      number of intercepted URLs, or a download that actually lands on disk?
- [ ] Does the `input()` prompt block forever in a non-interactive context (cron, CI, `nohup`)?
- [ ] Request volume and pacing: fixed `time.sleep` values with no jitter, no global rate
      limiter, no `Retry-After` respect. Enough to trip automation detection and get the
      account banned? That is a HIGH under this project's rubric.

---

## D. Untrusted input: network, archives, filesystem paths

Everything arriving from FACEIT or a demo file is untrusted, however friendly the source.

- [ ] **Path traversal on download.** `dest` is built from `match_id`
      (`Path(tmpdir) / f"{match_id}.dem"`). A `match_id` containing `../` or an absolute
      path escapes the temp dir. Is `match_id` validated, or trusted because it came from
      the API? Check `_m{idx}` suffixing too.
- [ ] `dest.with_suffix(".dem.zst")` — verify what `with_suffix` actually does when the
      stem already contains dots. A FACEIT `match_id` looks like `1-c566fc66-...`; confirm
      the resulting filename is what the code assumes.
- [ ] **Decompression bombs.** `zstd.copy_stream` and `gzip` + `shutil.copyfileobj` are
      unbounded. A hostile or corrupt `.dem.zst` fills the disk. Is there a size cap or a
      free-space check?
- [ ] Is the compressed file deleted on the failure path, or only on success?
- [ ] Content-type / magic-byte check before treating a response body as a demo — or does
      an HTML error page get written to `.dem` and handed to the parser?
- [ ] `content-length` is used for the progress display; is the actual downloaded size ever
      validated against it? A truncated download silently parses as a short match.
- [ ] Timeouts on **every** `requests` call. Confirm each one. A missing timeout hangs the
      pipeline forever.
- [ ] `tempfile.TemporaryDirectory()` cleanup on the exception path.
- [ ] Demo parsing: is a malformed `.dem` handled, or does awpy raise something the pipeline
      does not catch?

---

## E. SQL and database correctness

- [ ] **Every** query: parameterised (`?`) or f-string? Find each f-string-built SQL
      statement and name the exact input that reaches it. In `main.py`, `--event-type` and
      `player <name>` flow into SQL text. Note that `--side` and `--by` are constrained by
      `argparse choices=`; `--event-type` and `name` are **not**. Report reachability
      honestly: it is a local CLI, which bounds blast radius but does not make it correct —
      a nickname with a quote breaks the query today.
- [ ] `LIMIT {args.n}` and `LIMIT {int(args.limit)}` — is the cast present in both places?
- [ ] Are `p_before` / `p_after` / `impact` in `_init_schema`? Compare `_init_schema`
      against every column any writer or reader references. A fresh DB that cannot run
      `status` or `score` is a HIGH.
- [ ] Transactions: `store_match` must stay ONE transaction (BEGIN IMMEDIATE … commit)
      covering the deletes, both inserts and the processed marker. Any new write path that
      commits in pieces reopens the partial-match / re-insert window. It is R1.
- [ ] Connection lifecycle: `_conn()` opens a new connection per call and runs the
      migration check every time. Correctness first, performance second.
- [ ] Migrations: every new column must appear in BOTH `_init_schema` and `_MIGRATIONS`
      with `SCHEMA_VERSION` bumped — `CREATE TABLE IF NOT EXISTS` alone never changes an
      existing DB. `test/test_db.py` migrates a v0 database; keep it passing.
- [ ] Indexes vs actual query patterns: `score_impact` filters `round_states` by `match_id`
      and orders by `round_num, time_into_round_s`. Is that index adequate?
- [ ] `sqlite3` `timeout=` values, and whether a `database is locked` error is handled.
- [ ] Type round-tripping: booleans stored as `INTEGER`, `np.bool_` via the custom encoder,
      `None` vs `NULL` vs `NaN` for `min_dist_*_to_bomb`.
- [ ] `PRAGMA` settings: no WAL, no `foreign_keys`. Note if it matters; do not change it.

---

## F. Data-pipeline correctness (off-by-one country)

- [ ] Round boundaries: `freeze_end` and `official_end or end`. What if `official_end` is
      `0` — does `or` fall through to `end` when `0` is a legitimate value? **`or` on a
      numeric field treats `0` as missing.** Check every `a or b` over numerics.
- [ ] Tick slicing: `>=` vs `>` and `<=` vs `<` at both ends. Is a tick exactly on a
      boundary counted in one round, both, or neither?
- [ ] `_impact_for_event`'s `bisect_left` / `bisect_right` pair, and the `i_after += 1`
      adjustment. Walk it by hand for: event before the first state, event after the last,
      event exactly on a sample, single-state round, empty round.
- [ ] `wp_all[idx]` in `score_match` — `idx` comes from `group.index.to_numpy()` after a
      `read_sql_query`. Confirm the index is positional and aligned with `wp_all`, not a
      label index that happens to match. If `states` were ever filtered or re-indexed, this
      silently scores events against the wrong rows. **Verify explicitly.**
- [ ] `groupby(...,  sort=False)` combined with positional indexing — order assumptions.
- [ ] NaN/None propagation: `_safe`, `_nan`, `.get(col, default)`. Does a missing column
      become `0` where `0` is meaningful (e.g. `alive_ct=0` means eliminated)?
- [ ] Integer vs float division; `//` where `/` was meant.
- [ ] `int()` / `float()` casts on possibly-`None` values.
- [ ] Empty-DataFrame guards before `.iloc[0]`, `.max()`, `.mean()`.
- [ ] Player-name to side mapping: rebuilt per round? Players switch sides at halftime —
      a name→side map built once for the match is wrong after the swap. **Check this
      specifically; it would silently mis-sign half of every match.**
- [ ] Bomb-plant-to-round matching by tick range — can one plant match two rounds, or none?
- [ ] Hardcoded `tick_rate = 64`. Is it read from the demo header anywhere? A 128-tick demo
      would make every time-based feature wrong by 2×.

---

## G. Model correctness

- [ ] Leakage beyond the split (P6): any feature computed from the round's outcome, or from
      state after the event being scored.
- [ ] `df["map"].astype("category")` — categories are fitted on whatever data is present.
      At `predict` time `_states_to_frame` builds a **single-row** categorical, so its category
      set differs from training. Does LightGBM map it by code or by label? A mismatch means
      every prediction uses the wrong map. **Verify against LightGBM's actual behaviour, not
      assumption.** This is a plausible CRITICAL-for-correctness.
- [ ] `_score_states` casts `X["map"]` per call — same concern in the batch path.
- [ ] `FEATURES` order vs column order at predict time.
- [ ] `early_stopping(50)` on a single val set, then reporting AUC on that same val set —
      mildly optimistic. Worth a MEDIUM if presented as held-out.
- [ ] Is the model file's provenance checked before `Booster(model_file=...)`?
- [ ] `_model` global cache: never invalidated after retrain within one process.
- [ ] `predict_batch` builds a DataFrame per state and concatenates — correctness over speed,
      but check dtype consistency across the concat.

---

## H. Concurrency, state, and resource lifecycle

- [ ] Module-level mutable globals: `_browser_ctx`, `_model`. Re-entrancy, and cleanup.
- [ ] Generators that hold resources across `yield` (`iter_unprocessed_demos` yields inside
      a loop that owns a browser page).
- [ ] File handles and DB connections on the exception path — `try/finally` present?
- [ ] `seen` sets that grow unbounded across a long run.
- [ ] Signal handling: Ctrl-C mid-insert. Partial state?
- [ ] `pipeline.run()` catches per-match exceptions and continues — does the summary and
      the exit code still reflect every failure?

---

## I. Error handling and observability

- [ ] Every `except Exception` and every bare `except:`. For each: does it swallow an error
      that should stop the run? `pipeline.run` catches per-match and continues — correct.
      `scraper.get_player_match_ids` returns `[]` on **any** exception, making an auth
      failure indistinguishable from a player with no matches. That is a real finding.
- [ ] `except Exception: pass` — locate every one and judge individually.
- [ ] Are failures counted and surfaced in the final summary, or only printed mid-stream?
      A run that silently failed on 40 of 50 matches still prints "done".
- [ ] `print` vs `logging`: no logger in this project. Note it; do not refactor to one.
- [ ] Do error messages leak secrets (§B) or PII (nicknames) into stdout that might be
      pasted into an issue?
- [ ] Exit codes: does `main.py` exit non-zero when the work failed?

---

## J. CS2 domain knowledge (no domain skill exists — check with care)

Flag anything here as `needs-domain-review` rather than asserting it wrong.

- [ ] Round phases: freeze time → live → post-plant → round end → the gap before the next
      freeze end. Which does the sampler actually cover?
- [ ] Bomb timer is 40 s; defuse is 10 s without a kit, 5 s with. Does any feature or label
      assume otherwise?
- [ ] Sides are `ct` / `t` in awpy — verify the exact casing the parser returns.
- [ ] Halftime side swap, and overtime side swaps. `round_won_ct` must mean the CT side of
      **that** round.
- [ ] MR12 (24 rounds + OT) vs MR15 — does anything assume a round count?
- [ ] Tuned constants: `_SMOKE_RADIUS = 144.0`, `SITE_BLOCK_RADIUS = 300.0`,
      `_SOUND_RANGES`, `_SILENCED_WEAPONS`. Are they defensible in CS2 units, and is their
      provenance documented?
- [ ] Equipment value semantics: `current_equip_value` includes or excludes grenades and
      armour? Consistency between the buy event and the state snapshot.
- [ ] Warmup and knife rounds — excluded from `rounds_df`?
- [ ] Coordinate units: `X`/`Y`/`Z` in Hammer units; distance thresholds consistent with that.

---

## K. Tests

- [ ] Which of R1–R22 have **no** test at all? Name each. This is the most valuable output
      of this section.
- [ ] Tests that assert nothing: a script that prints and exits 0 regardless.
      `test/test_local_demo.py` and `test/test_scraper.py` are inspection scripts named
      `test_*` — that naming is itself a finding.
- [ ] Tests with **module-level side effects at import**: `test/test_scraper.py` launches
      Chrome at import time; `test/test_win_probability.py` runs all assertions at import
      and calls `sys.exit`. Any future test runner importing them detonates.
- [ ] Tests that write to the **production** database: `test/test_local_demo.py` calls
      `db.insert_*` against `data/crosshair.db`. That is a HIGH — a test run mutates the
      corpus the model trains on.
- [ ] Vacuous assertions: `between(p, 0.0, 1.0)` on a sigmoid output can never fail. Count
      how many of the 23 checks are tautological.
- [ ] Do the tests depend on a trained model that may not exist, and do they fail with a
      clear message when it does not?
- [ ] Do assertions encode the invariant, or a snapshot of current behaviour?

---

## L. Config, dependencies, agent config, docs-vs-reality

- [ ] `requirements-engine.txt` vs the lock vs `pyproject.toml`: does `pyproject.toml`
      read its deps from the requirements file, does its `[project.scripts]` entry
      (`engine.cli:main`) exist, and does package-data list every data file the engine
      reads at runtime? A broken console-script entry point is a finding.
- [ ] Unpinned ranges (`>=` with no upper bound) on `awpy`, `pandas`, `numpy`, `lightgbm`.
      `awpy` in particular is pre-1.0-stable in API terms; an unpinned major is a real risk
      for a parser this project depends on entirely.
- [ ] Known advisories for the pinned/installed versions — via `pip list` metadata or
      `pip-audit` **if already installed**. Never `pip install` to audit.
- [ ] `.gitignore`: does it cover `.env`, `data/*.db`, `demos/*.dem`, `*.lgb`? Is
      `.claude/` ignored, and is that intended now that a skill lives there?
- [ ] `.claude/settings.json` permission allowlist: entries broader than needed
      (`Bash(pip install *)` permits arbitrary package installation; `Bash(python -c ' *)`
      permits arbitrary Python). Stale entries naming files that no longer exist
      (`src/wp_model.py`, `--data-dir` flags) indicate an allowlist nobody prunes.
- [ ] This skill's own files — does anything here instruct a write outside `docs/reviews/`?
- [ ] **Docs vs reality**, contradiction only:
      - Does every CLI invocation in `README.md`, `CLAUDE.md` and `docs/` actually parse
        against `engine/cli.py`'s `build_parser()`?
      - Does `README.md`'s project layout match the real tree?
      - Are stated numbers (match count, event count, AUC, DB size) presented as current
        when they are from an old reference run?
      - Does `CLAUDE.md` describe invariants the code violates? (Both directions are
        findings: doc wrong, or code wrong.)
      - Do `docs/ENGINE_PLAN.md`, `docs/DATA_SOURCES.md` and `docs/ENGINE_OPS.md` match
        what the code does?
      - Do documented setup steps work on a clean checkout?
