---
name: full-codebase-check
description: Exhaustive read-only audit of the entire Crosshair tree that produces a dated, severity-ranked findings report at docs/reviews/ and edits no code. Reads every in-scope file end to end, shards the review across parallel subagents, adversarially verifies every CRITICAL and HIGH before it survives, and audits this project's own documented invariants alongside generic vulnerability classes. Use when Faig asks for a full codebase review, a whole-repo audit, a security sweep, a pre-release check, or invokes /full-codebase-check.
---

# Full Codebase Check

An exhaustive, read-only audit of the whole Crosshair tree. It produces one dated,
severity-ranked report and changes nothing else.

This repo is public (`github.com/Apr0G/crosshair`), it holds a FACEIT API key, it drives
Faig's **real Chrome profile**, and its database holds ~2.8 M events keyed to real Faceit
player identities. The severity rubric below is written in those terms, not generic ones.

---

## Non-negotiables

**1. Read-only. The report file is the only write in the entire skill.**

Forbidden for every agent, including you: `Edit`, `Write` (except the one report),
`NotebookEdit`, `sed -i`, `perl -i`, `>` / `>>` redirection into repo files, formatters
(`black`, `ruff format`, `autopep8`), linter `--fix`, `pip install`, and every mutating
git verb — `commit`, `add`, `checkout`, `switch`, `restore`, `stash`, `reset`, `clean`,
`rm`, `mv`, `rebase`, `merge`, `push`. Read-only git (`ls-files`, `status`, `log`, `show`,
`diff`, `rev-parse`, `blame`) is fine.

Even an obviously-wrong one-character typo gets **written down, not fixed**. Fixing is a
separate, later, Faig-initiated task. If you find yourself reaching for Edit, you have
misread this skill.

**2. Never read these paths.** Do not open them, do not `grep` them, do not `cat` them,
do not route around the guard with `python -c`, and do not ask a subagent to do it for you:

| Path | Why |
|---|---|
| `.env`, `*.env` | Live FACEIT API key. Public repo — a key in the report is a key on GitHub. |
| `~/Library/Application Support/Google/Chrome/**` | Faig's real browser profile: cookies and live sessions for every site he's logged into. |
| any browser profile dir | Same, whatever the OS path. |
| `.venv/**` | Third-party code, not ours. Audit deps via metadata (rule below). |
| `data/crosshair.db` | 3.6 GB, real player nicknames and IDs. **Schema only, via `PRAGMA table_info` / `.schema`. Never `SELECT` row contents.** |
| `demos/*.dem` | Binary; embeds player names and Steam IDs. No audit value. |
| `*.key`, `*.pem`, `id_rsa*`, `.netrc`, `~/.aws/**`, `~/.config/gh/**` | Credentials. |

`.env.example` **is** in scope and must be read — it should contain only placeholders, and
a real-looking value there is a top-severity finding under rule 3.

When a finding's truth depends on a blocked path's contents, do not guess. Record it in the
report's **Unverifiable** section as *"unverifiable without operator review"*, with the
exact check Faig should run by hand (e.g. "confirm `.env` has no trailing whitespace in
`FACEIT_API_KEY`" — phrased so he can check without pasting the value anywhere).

**3. Nothing sensitive goes in the report. Quote code, never data.**

No API keys, tokens, signed URLs, cookies, session IDs, player nicknames, Steam IDs,
match IDs tied to real people, or DB rows. Redact to shape: `FACEIT_API_KEY=<redacted>`,
`https://demos-europe-west2.faceit-cdn.net/...<signed>`, `nickname=<player>`.

If a source file, config, fixture, or test contains real-looking sensitive data, **that is
itself a top-severity finding** — report its `file:line` and what class of secret it is,
never its contents.

**4. Read files in full, end to end. No grep-and-skim.**

`grep` is for locating a symbol before you read the file, never a substitute for reading it.
`src/feature_extractor.py` is 1,351 lines and gets read start to finish like everything else;
that is exactly where a `>=`-should-be-`>` hides. A file that is not read in full is reported
as **unreviewed** in the Coverage table with a reason. It is never assumed fine, and never
silently dropped.

**5. No speculation survives into the final report.**

Every finding carries a concrete `file:line` and a concrete failure or exploit path. "Could
be unsafe", "might be a problem", "consider reviewing" are not findings. Anything that fails
Phase 3 verification is dropped, or downgraded and moved to **Not findings** with the reason.
A short report of real findings beats a long one padded with maybes.

---

## Phase 0 — Scope

Establish the file list **first**, before reading anything for content.

```bash
date +%F                                        # the real date, for the report filename
git rev-parse --short HEAD && git branch --show-current
git status --porcelain --untracked-files=all    # untracked + working-tree dirt
git ls-files                                    # tracked
ls -1 .claude/settings.json .claude/settings.local.json 2>/dev/null
find .claude/skills -type f 2>/dev/null
```

**In scope for this repo:**

- **Python source** — `src/**/*.py`, `test/**/*.py`. Every one, in full.
- **Dependency + packaging manifests** — `requirements.txt`, `pyproject.toml`.
- **Config and repo hygiene** — `.gitignore`, `.env.example`.
- **Agent config** — `.claude/settings.json`, `.claude/settings.local.json`, and
  `.claude/skills/**` (including this skill's own files). These are **gitignored**, so
  `git ls-files` misses them — add them explicitly or the audit's own instrument goes
  unreviewed. A permission allowlist that is broader than it needs to be is a real finding.
- **Docs and process claims** — `README.md`, `CLAUDE.md`, `PROMPT_faceit_api_downloader.md`.
  Audited for **contradicting the code**, not for prose quality. A README documenting a CLI
  flag that no longer parses, or a CLAUDE.md invariant the code violates, is a finding.
  Typos and tone are not.

**Out of scope:**

- `.venv/**`, and any vendored or generated code. Audit third-party risk from **metadata
  instead**: read `requirements.txt` / `pyproject.toml`, list every dependency with its
  version constraint, flag unpinned ranges, and check installed versions against known
  advisories via `pip list --outdated` or `pip-audit` **if already installed** — never
  `pip install` a tool to run the audit.
- Build output, `__pycache__/`, `*.pyc`, `dist/`, `build/`, `*.egg-info/`.
- Binaries: `demos/*.dem`, `data/*.lgb`, `.DS_Store`.
- Everything in Non-negotiable 2.

**Record the in-scope file count now.** Phase 4 reconciles files-read against it, and the
two numbers must agree or the gap must be named file by file. As of 2026-08-01 the expected
count is ~25 (18 tracked + 3 untracked + `.claude/settings.json` + this skill's files) —
recompute it, do not trust that number.

---

## Phase 1 — Load the rules being audited against

Read these before dispatching. Generic CWEs are the floor; **violations of Faig's own rules
are the highest-yield findings**, and a generic checklist misses all of them.

1. **`CLAUDE.md`** — the primary rule source. Extract the invariants and carry them into
   every shard prompt. The standing set (re-derive, don't trust this list — the file changes):

   | ID | Invariant |
   |---|---|
   | P1 | No path may double-insert a match. Dedup is `db.is_processed`, checked for **both** `<mid>` and `<mid>_m1` id forms. |
   | P2 | Both ingestion sources produce identical downstream rows; they differ only in how the `.dem` reaches disk. |
   | P3 | All ingestion converges on the seam `pipeline.process_one` → `db.insert_events` / `insert_round_states` / `mark_processed`. No forked pipeline. |
   | P4 | A source module exposes exactly `iter_unprocessed_demos(...)` and `download_demo(url, dest)`. |
   | P5 | Impact sign: `p_after - p_before` for a CT actor, negated for T. |
   | P6 | WP split is `GroupShuffleSplit(groups=match_id)` — never a random row split (leaks future round state). |
   | P7 | WP probabilities must stay **calibrated**, not merely well-ranked; `test/test_win_probability.py` must pass. |
   | P8 | A new WP feature requires all four: `state_sampler` emits it → `db` column + insert → `win_probability.FEATURES` → retrain. |
   | P9 | DB changes additive-only, with a migration. `_init_schema` cannot add columns to an existing DB. |
   | P10 | Secrets via env/`.env` only. Never hardcoded, never committed. |
   | P11 | No invented FACEIT API fields or endpoints. |
   | P12 | `src/scraper.py` (Playwright) is load-bearing and ask-before-touch. |
   | P13 | `/security-review` runs before every commit. |
   | P14 | Tests stay plain scripts, not pytest (decided 2026-08-01). |
   | P15 | `.venv/bin/python` (3.11 arm64); `requirements.txt` is authoritative, `pyproject.toml` is stale. |

2. **`README.md`** — the public claims. Anything it asserts about behaviour is a testable
   claim against the code.
3. **`PROMPT_faceit_api_downloader.md`** — the spec the API path was built to. Its hard
   rules (no Playwright changes, additive-only schema, identical downstream results, dedup
   before download) are auditable requirements, not history.
4. **`references/checklist.md`** — this skill's substantive checklist. Load it in full into
   every shard agent.
5. **Domain knowledge — no skill exists for this, and it is a real gap.** Business-logic
   correctness here is invisible without CS2 knowledge: round phases and freeze time, the
   40 s bomb timer, 5 s/10 s defuse with and without kit, 64-tick demos, `ct`/`t` side
   naming, equipment values, and the tuned constants (`_SMOKE_RADIUS = 144.0`,
   `SITE_BLOCK_RADIUS = 300.0`, `_SOUND_RANGES`). Section J of the checklist carries this
   inline. **Mark any finding that turns on a game-mechanics constant as
   `confidence: needs-domain-review`** rather than asserting it — and note in the report
   that a CS2 domain reference would raise the audit's ceiling.

---

## Phase 2 — Sharded review

One read-only subagent per shard, dispatched in **parallel batches**. Every in-scope file
lands in **exactly one** shard. Before dispatching, re-derive this table from the actual
Phase 0 file list and verify the partition: no file in two shards, no in-scope file in none.
The table below is the layout as of 2026-08-01 — treat it as a starting point, not gospel.

| Shard | Files | Focus |
|---|---|---|
| **S1 — Ingestion & network** | `src/scraper.py`, `src/faceit_api.py` | The secret-handling boundary. API key in headers/logs/tracebacks; the **real Chrome profile** launch; download path construction and decompression (path traversal via `dest`, zip/zstd bombs, partial writes); TLS; 429/403 handling; signed-URL expiry and re-exchange; rate limits vs ToS. Highest-stakes shard. |
| **S2 — Orchestration & CLI** | `src/main.py`, `src/pipeline.py` | SQL assembled with f-strings from CLI arguments; argparse surface; blanket `except Exception` that swallows real failures; exit codes; secrets reaching stdout in error paths; the `--source` seam selection. |
| **S3 — Persistence** | `src/db.py` | P1/P3/P9. Dedup and idempotency; schema vs what writers actually insert; transaction and commit boundaries; connection-per-call behaviour; the JSON encoder; parameterisation of every insert. |
| **S4 — Feature extraction** | `src/feature_extractor.py` | 1,351 lines, read in full. Off-by-ones in tick slicing and windows; side/name mapping; visibility and sound-cue precompute; grenade attribution; `None`/NaN handling; silent `except: pass`. |
| **S5 — Parsing & sampling** | `src/extract.py`, `src/state_sampler.py` | awpy surface assumptions and missing-table fallbacks; round boundary derivation (`freeze_end` / `official_end`); bomb-plant-to-round matching; per-state field completeness vs P8. |
| **S6 — Model & scoring** | `src/win_probability.py`, `src/score_impact.py` | P5/P6/P7/P8. Leakage in the split; `FEATURES` vs DB columns vs sampler output; the `bisect` bracketing in `_impact_for_event` (off-by-one on exact-tick events); impact sign; calibration; model-file trust. |
| **S7 — Tests** | `test/*.py` | Tests that assert the wrong thing, pass vacuously, or assert nothing; **module-level side effects at import**; anything writing to the production DB; and which of P1–P15 have **no** test coverage at all. |
| **S8 — Config, deps, agent config** | `requirements.txt`, `pyproject.toml`, `.gitignore`, `.env.example`, `src/__init__.py`, `.claude/settings.json`, `.claude/skills/**` | Does `.gitignore` actually cover `.env`, the DB, and demos? Unpinned deps and known advisories via metadata; requirements-vs-pyproject divergence; over-broad permission allowlist entries; stale allowlist entries naming files that no longer exist. |
| **S9 — Docs vs reality** | `README.md`, `CLAUDE.md`, `PROMPT_faceit_api_downloader.md` | Every claim tested against code: documented CLI flags that no longer parse, stated invariants the code violates, setup steps that fail, performance/accuracy numbers presented as current. Contradictions only — not prose. |

**Cross-shard reading.** An agent may read outside its shard for **context** — a helper's
definition, a caller's guard, the schema a query targets — and should, because a finding
that ignores a guard one layer up is exactly what Phase 3 refutes. But it may only
**report** on files it owns. If S6 spots a bug in `db.py`, it says so in a `cross_shard_notes`
field and S3 owns the finding. This is what keeps duplicates out.

**Each agent returns, per finding:**

| Field | Requirement |
|---|---|
| `severity` | `CRITICAL` \| `HIGH` \| `MEDIUM` \| `LOW` — per the rubric below |
| `title` | One line, specific. "`cmd_top` interpolates `--event-type` straight into SQL", not "security risk" |
| `file` | Repo-relative path |
| `line` | Real line number, verified against the file as read |
| `category` | Checklist section letter + name |
| `evidence` | The code, **quoted exactly** — no paraphrase, no reflow |
| `failure` | Concrete: *given state X, Y happens*. Preconditions named. Not "could be unsafe" |
| `fix` | Prose. **Not applied.** No diffs |
| `confidence` | `high` \| `medium` \| `needs-domain-review` |
| `rule_violated` | `P1`–`P15` where applicable, else `null` |

**And per agent:** `files_read` (every path, read in full), `files_skipped` (path + reason),
`lines_reviewed`, and `cross_shard_notes`. Phase 4 reconciles coverage from these — an agent
that cannot produce `files_read` has not done the job.

---

## Phase 3 — Adversarial verification

**Every CRITICAL and HIGH gets an independent verifier agent whose job is to refute it.**
The verifier does not see the finder's reasoning beyond the finding record itself. It
re-reads the cited file and its callers and tries to prove the finding wrong.
**Default to refuted when uncertain** — a false CRITICAL costs more than a missed MEDIUM,
because it trains Faig to distrust the report.

Common refutations in *this* codebase, check each explicitly:

- **The guard exists one layer up.** `pipeline.run` wraps `process_one` in `try/except`;
  `main.py` validates before calling. A "crash" that is caught and logged is not a crash.
- **The value is sanitised upstream.** IDs come from the FACEIT API, not user input;
  `argparse` `choices=` already constrains `--source`, `--by`, and `--side` to a fixed set,
  so an "injection" through those is not reachable.
- **The path is unreachable.** Dead branches, `__main__`-only code, and the two manual
  scripts in `test/` that nothing calls in normal operation.
- **The race is serialised.** SQLite's write lock, `PRIMARY KEY` / `INSERT OR IGNORE`, or
  the fact that the pipeline is single-threaded and processes one match at a time.
- **The input is validated by the schema layer on the way in.** A column typed `INTEGER`,
  or a value that `state_sampler` can only ever emit as `0`/`1`.
- **The "missing" column exists in the live DB.** `db.py`'s `_init_schema` and
  `data/crosshair.db` have drifted in **both** directions — confirm which artefact the
  finding is actually about before asserting it.
- **The constant is correct for CS2.** Do not call a game-mechanics number wrong without
  domain grounding; downgrade to `needs-domain-review`.

MEDIUM and LOW get a lighter check: re-read the cited line and confirm it still says what
the finding claims, and that the line number is right after any concurrent edits.

**Then dedupe across shards.** One root cause is **one** finding with three locations, not
three findings. Merge on shared root cause, keep the highest severity, list every
`file:line` under it. Everything refuted moves to **Not findings** with its refutation.

---

## Phase 4 — Report

Write **exactly one** file:

```
docs/reviews/full-codebase-check-YYYY-MM-DD.md
```

`YYYY-MM-DD` is the **real current date** from `date +%F` — never a guess, never a date
carried over from an example. `mkdir -p docs/reviews` is permitted; it creates no repo
content. If a report already exists for today, append `-2`, `-3`, … (`…-2026-08-01-2.md`).
**Never overwrite a prior audit** — the history is the point of dating them.

Follow `references/report-template.md` **exactly**, including the coverage reconciliation
and the rule-compliance table. Both are load-bearing: an audit that quietly skipped six
files is worse than no audit, because it reads as a clean bill of health.

---

## Severity rubric

Calibrated to what Crosshair actually risks. Faig ranked all four of these as real
worst-case outcomes: **Chrome-profile compromise, the API key leaking publicly, player PII
exposure, and a FACEIT ToS ban killing the pipeline.**

**CRITICAL** — a reportable incident if it fires in production:
- The FACEIT API key, a signed URL, or a session token reaching stdout, a log, a traceback,
  a committed file, or a report. The repo is public; exposure is irreversible.
- Anything that writes to, exfiltrates, corrupts, or widens access to the **real Chrome
  profile** beyond reading Faceit — this is Faig's live session for every site he uses.
- Player PII (nicknames, Steam IDs, match identities) leaving the local machine, or landing
  anywhere destined for the public repo.
- Arbitrary code or command execution from a downloaded demo, a signed URL, or an API
  response — including path traversal that writes outside the temp dir during decompression.
- Silent, unrecoverable destruction of `data/crosshair.db` (710 matches; re-scraping is
  days of work and some demos are no longer available).

**HIGH** — exploitable with a precondition, or silent wrongness, or a documented rule broken:
- Any P1–P15 violation without a live exploit path. **A broken invariant is HIGH here even
  when nothing exploits it** — P1 (double-insert) silently poisons the training set, and
  P6 (split leakage) silently invalidates every downstream number.
- Silent data corruption: double-inserted matches, mis-signed impact (P5), events scored
  against the wrong round's states.
- Scraping behaviour likely to trip FACEIT automation detection and get the account banned
  — request volume, missing backoff, absent jitter.
- A crash that loses an in-flight match after the demo has been downloaded and parsed.

**MEDIUM** — a real bug with bounded blast radius:
- Wrong output in a specific, reachable case: an off-by-one at a round boundary, a NaN that
  becomes 0, a rounding error in one feature.
- An unhandled exception that aborts one match but not the run.
- A test that passes vacuously, or asserts the wrong thing.
- Docs that contradict the code in a way that would misdirect a future session.

**LOW** — hygiene, no user-visible failure: dead code, stale allowlist entries, unpinned
dependency with no known advisory, inconsistent naming, a genuinely cosmetic doc error.

**Tiebreaker, in this project's terms:** if it fires, does Faig have to rotate a key,
apologise to a player, re-scrape the corpus, or retrain a model whose numbers are now
known-wrong? CRITICAL or HIGH. If it only means a worse afternoon of debugging? MEDIUM.
If nobody would ever notice at runtime? LOW.

When genuinely torn between two levels, take the **lower** one and say why in the finding.
An inflated report is a report that gets skimmed.

---

## Finish

Report in chat — not a copy of the report, a pointer to it:

1. **Report path.**
2. **Counts by severity** — `3 CRITICAL / 7 HIGH / 12 MEDIUM / 9 LOW`.
3. **The single worst finding, in one sentence**, in plain terms.
4. **Coverage** — `files read in full / in scope`, and name anything unread.
5. **Cost** — rough agent count and token spend, so the next run can be scoped.

Then stop. **Do not fix anything unless Faig asks.** This skill reviews. Offering to fix
the top finding is fine; starting is not.
