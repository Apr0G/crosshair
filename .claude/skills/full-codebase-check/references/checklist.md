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

---

## A. Project invariants (highest yield — check these first)

These come from `CLAUDE.md`. A violation is at minimum HIGH, even with no exploit path,
because each one silently corrupts data or invalidates model numbers.

- [ ] **P1 — no double-insert.** Trace every path that reaches `db.insert_events` /
      `insert_round_states` / `mark_processed`. Is `is_processed` checked before *every*
      one? Is it checked for **both** `<mid>` and `<mid>_m1` forms? Is there a window
      between the check and the insert where a crash leaves events written but the match
      unmarked? (Look hard at ordering: events inserted → crash → `mark_processed` never
      runs → next run re-inserts everything. This is the single most likely real bug in
      the repo.)
- [ ] **P1b —** is there a uniqueness constraint backing the dedup, or is it check-then-act
      only? What happens if the same `match_id` is inserted twice?
- [ ] **P2 — source parity.** Do `scraper` and `faceit_api` produce the same `match_id`
      shape, the same decompression result, the same `demo_url` recorded? Any divergence
      means the two paths write different rows.
- [ ] **P3 — the seam.** Does anything bypass `pipeline.process_one` to write to the DB?
      (`test/test_local_demo.py` and `main.cmd_demo` both write directly — are they
      consistent with the seam's behaviour, especially dedup and `mark_processed`?)
- [ ] **P4 — source interface.** Do both source modules expose exactly
      `iter_unprocessed_demos(...)` and `download_demo(url, dest)` with compatible
      signatures? Does `pipeline` call anything else on them?
- [ ] **P5 — impact sign.** `p_after - p_before` for CT, negated for T. Verify the actual
      comparison string: is it `"ct"`, `"CT"`, or `"Ct"` that arrives, and is it
      normalised before comparison? A side string that never matches makes every event T-signed.
- [ ] **P6 — split leakage.** Is the split still `GroupShuffleSplit(groups=match_id)`?
      Does any preprocessing (scaling, encoding, imputation, category fitting) happen
      **before** the split, on the full frame? That leaks too.
- [ ] **P7 — calibration.** Any change to objective, class weights, or post-processing that
      would break calibration while leaving AUC fine?
- [ ] **P8 — feature parity.** Cross-check three lists: what `state_sampler` emits, what
      `db.insert_round_states` writes, and `win_probability.FEATURES`. Any feature in
      `FEATURES` that the DB lacks fails at query time; any column the sampler emits that
      the insert omits is silently dropped.
- [ ] **P9 — additive-only schema.** Does `_init_schema` match what the writers insert?
      Can it add a column to an existing DB? (It cannot — `CREATE TABLE IF NOT EXISTS`
      is a no-op on an existing table. Confirm the consequences are documented and handled.)
- [ ] **P10 — secrets.** No key in source, no key in a default argument, no key in a
      committed file, no key in a printed URL.
- [ ] **P11 — invented API surface.** Every FACEIT field accessed (`demo_url`, `payload`,
      `download_url`, `items`, `faceit_elo`, `player_id`, `match_id`) — is it accessed
      defensively, and does the code assume a shape the API does not guarantee?
- [ ] **P12–P15 —** scraper untouched, `/security-review` documented, tests still plain
      scripts, `.venv/bin/python` used in docs and examples.

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

`src/scraper.py` launches Chrome against `~/Library/Application Support/Google/Chrome/Default`
— Faig's **real** profile, with live sessions for every site he uses.

- [ ] Confirm the profile path and whether a throwaway profile would work. **Report the risk;
      do not change it** (P12: ask-before).
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
- [ ] Transactions: `insert_events` commits per batch. If batch 3 of 6 fails, the DB holds
      a partially-inserted match with **no** `processed_matches` row. Next run re-inserts
      batches 1–2. Trace this. It is P1.
- [ ] Connection lifecycle: `_conn()` opens a new connection per call, and `insert_events`
      / `insert_round_states` / `is_processed` / `mark_processed` each open their own.
      Correctness first, performance second.
- [ ] `_conn()` only calls `_init_schema` when the file is **absent** — so a DB created by
      an older version never gains new columns. Confirm and report.
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
      At `predict` time `_state_to_row` builds a **single-row** categorical, so its category
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
- [ ] `pipeline.loop()` runs forever with a bare `except Exception` — does a persistent
      failure become a silent hot loop?

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

- [ ] Which of P1–P15 have **no** test at all? Name each. This is the most valuable output
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

- [ ] `requirements.txt` vs `pyproject.toml`: which deps are in one and not the other?
      `pyproject.toml`'s `[project.scripts]` points at `crosshair.main:main` — does that
      module path exist? A broken console-script entry point is a finding.
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
      - Does every CLI invocation in `README.md` actually parse against `main.py`'s argparse?
      - Does `README.md`'s project layout match the real tree?
      - Are stated numbers (match count, event count, AUC, DB size) presented as current
        when they are from an old reference run?
      - Does `CLAUDE.md` describe invariants the code violates? (Both directions are
        findings: doc wrong, or code wrong.)
      - Does `PROMPT_faceit_api_downloader.md`'s spec match what `faceit_api.py` does?
      - Do documented setup steps work on a clean checkout?
