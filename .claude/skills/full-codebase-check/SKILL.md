---
name: full-codebase-check
description: Exhaustive read-only audit of the entire Crosshair tree that produces a dated, severity-ranked findings report at docs/reviews/ and edits no code. Reads every in-scope file end to end, shards the review across parallel subagents, adversarially verifies every CRITICAL and HIGH before it survives, and audits this project's own documented invariants alongside generic vulnerability classes. Use when Faig asks for a full codebase review, a whole-repo audit, a security sweep, a pre-release check, or invokes /full-codebase-check.
---

# Full Codebase Check

An exhaustive, read-only audit of the whole Crosshair tree. It produces one dated,
severity-ranked report and changes nothing else.

This repo is public (`github.com/Apr0G/crosshair`), it holds a FACEIT API key, it drives
Faig's **real Chrome profile**, and its data (`data/engine/engine.db`, demos, parsed
tables) is keyed to real Faceit player identities. The severity rubric below is written in
those terms, not generic ones. Since 2026-10-19 the code is the `engine/` package (plan
Amendment 1); the legacy `src/` pipeline is deleted.

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
| `.env`, `.env*` (except `.env.example`), `env` | Live FACEIT API key. Public repo — a key in the report is a key on GitHub. |
| `~/Library/Application Support/Google/Chrome/**` | Faig's real browser profile: cookies and live sessions for every site he's logged into. |
| any browser profile dir | Same, whatever the OS path. |
| `.venv/**`, `.venv-old311/**` | Third-party code, not ours (the second is the dead 3.11 venv). Audit deps via metadata (rule below). |
| `data/**` | `data/engine/engine.db`, `data/engine/demos/`, parsed tables, models, logs, and the legacy `data/crosshair.db`: real player nicknames, Steam IDs and match ids. Never opened. The schema is audited from `engine/db.py`, not the file. |
| `demos/**`, `*.dem`, `*.dem.zst`, `*.dem.gz` | Binary; embeds player names and Steam IDs. No audit value. |
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
`engine/parse/awpy3_parser.py` is long and gets read start to finish like everything else;
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

- **Python source** — `engine/**/*.py`, `test/**/*.py`, `bench/**/*.py`. Every one, in
  full, including git-ignored modules that exist on disk (check `git status --ignored`).
- **Engine data files** — `engine/config.yaml`, `engine/maps/**`, `engine/report/schema.json`,
  `engine/state/*.yaml`, `bench/budget.yaml`, `test/engine/fixtures/**`.
- **Dependency + packaging manifests** — `requirements-engine.txt`,
  `requirements-engine.lock`, `pyproject.toml`.
- **Config and repo hygiene** — `.gitignore`, `.env.example`.
- **Agent config** — `.claude/settings.json`, `.claude/settings.local.json`, and
  `.claude/skills/**` (including this skill's own files). `.claude/settings*.json` are
  **gitignored** (the skills are re-included), so `git ls-files` misses the settings —
  add them explicitly or the audit's own instrument goes
  unreviewed. A permission allowlist that is broader than it needs to be is a real finding.
- **Docs and process claims** — `README.md`, `CLAUDE.md`, `docs/**` (except earlier
  audits in `docs/reviews/`, which are history).
  Audited for **contradicting the code**, not for prose quality. A README documenting a CLI
  flag that no longer parses, or a CLAUDE.md invariant the code violates, is a finding.
  Typos and tone are not.

**Out of scope:**

- `.venv/**`, `.venv-old311/**`, `data/**`, and any vendored or generated code. Audit
  third-party risk from **metadata instead**: read `requirements-engine.txt` / the lock /
  `pyproject.toml`, list every dependency with its
  version constraint, flag unpinned ranges, and check installed versions against known
  advisories via `pip list --outdated` or `pip-audit` **if already installed** — never
  `pip install` a tool to run the audit.
- Build output, `__pycache__/`, `*.pyc`, `dist/`, `build/`, `*.egg-info/`.
- Binaries: `demos/**`, `*.dem*`, `*.lgb`, `.DS_Store`.
- Everything in Non-negotiable 2.

**Record the in-scope file count now.** Phase 4 reconciles files-read against it, and the
two numbers must agree or the gap must be named file by file. On 2026-10-19 the count was
252 (242 tracked + 7 git-ignored `engine/data/*.py` + this skill's 3 files) — recompute
it, do not trust that number.

---

## Phase 1 — Load the rules being audited against

Read these before dispatching. Generic CWEs are the floor; **violations of Faig's own rules
are the highest-yield findings**, and a generic checklist misses all of them.

1. **`CLAUDE.md`** — the primary rule source; its **Engine v2** section binds (sections
   marked obsolete are history). Extract the invariants and carry them into every shard
   prompt. The standing set, as numbered in the 2026-10-19 report (re-derive, don't trust
   this list — the file changes):

   | ID | Invariant |
   |---|---|
   | R1 | Every source ends at `engine.ingest.register_demo`; `db.is_processed_any` checks both id forms; never double-insert; UNIQUE `demo_sha256` backstop. |
   | R2 | All sources produce identical downstream rows and expose `iter_unprocessed_demos` / `download_demo`. |
   | R3 | CT-positive sign convention, flipped once per T actor. |
   | R4 | Frozen month splits with match grouping; never a random row split; split files superseded, never edited. |
   | R5 | Every feature has a `LIVE_SAFE` justification (`engine/state/schema.py`); leakage audit in `test_leakage.py`. |
   | R6 | Calibrated probabilities: ECE ≤ 0.02 on the baseline test tier. |
   | R7 | Never `0` / `"?"` for unknown where they are legitimate values: `None` / NULL / NaN. |
   | R8 | `or` is not a null-coalesce over numerics (`_nan` / `_first_valid`). |
   | R9 | Credential-bearing URLs redacted before any print / exception / file / DB write; never truncate a URL by length. |
   | R10 | Tick rate from the demo header, never a literal; quarantine refuses non-128. |
   | R11 | No invented FACEIT API fields or endpoints. |
   | R12 | Additive DB migrations keyed by `PRAGMA user_version`; a newer file is refused; schema changes ask-first. |
   | R13 | Never change `sample_hz` for latency. |
   | R14 | Reports never contain outside paths or URLs; SteamIDs hashed unless `--show-ids`. |
   | R15 | Modules that do not train or predict never import torch at import time. |
   | R16 | Plain-script tests (never pytest), exit 0/1/2, write only to temp dirs, every module has a test script. |
   | R17 | Secrets via env + `.env`; never hardcoded or committed. |
   | R18 | `engine/sources/scraper.py` logic unchanged; ask before touching. |
   | R19 | A model is used only if adopted in `docs/ENGINE_MODELS.md`; `analyze` refuses otherwise; P7 gate before attribution. |
   | R20 | Style: absolute imports, `# ── section ──` banners, type hints, plan-step docstrings. |
   | R21 | Conservation identity; C1 exact zero on a shared tick; C2 `V = B + D + X`, `R ≤ 0`. |
   | R22 | Exit codes 0 ok / 1 failed / 2 usage or prerequisite, everywhere. |

2. **`README.md`** — the public claims. Anything it asserts about behaviour is a testable
   claim against the code.
3. **`docs/ENGINE_PLAN.md`** (with its amendments) and the other `docs/ENGINE_*.md` —
   the spec the engine is built to. Their hard rules (decision gates, acceptance tests
   C1–C3, the seam, ask-before items) are auditable requirements, not history.
4. **`references/checklist.md`** — this skill's substantive checklist. Load it in full into
   every shard agent.
5. **Domain knowledge — no skill exists for this, and it is a real gap.** Business-logic
   correctness here is invisible without CS2 knowledge: round phases and freeze time, the
   40 s bomb timer, 5 s/10 s defuse with and without kit, 128-tick demos (64-tick refused),
   `ct`/`t` side naming, equipment values, and the tuned constants in `engine/config.yaml`
   (clock, buy bands, trade window, smoke / utility lifetimes). Section J of the checklist carries this
   inline. **Mark any finding that turns on a game-mechanics constant as
   `confidence: needs-domain-review`** rather than asserting it — and note in the report
   that a CS2 domain reference would raise the audit's ceiling.

---

## Phase 2 — Sharded review

One read-only subagent per shard, dispatched in **parallel batches**. Every in-scope file
lands in **exactly one** shard. Before dispatching, re-derive this table from the actual
Phase 0 file list and verify the partition: no file in two shards, no in-scope file in none.
The table below follows the engine module map (CLAUDE.md) as of 2026-10-19 — treat it as a
starting point, not gospel; split a shard that grows past what one agent reads in full.

| Shard | Files | Focus |
|---|---|---|
| **S1 — Sources & network** | `engine/sources/**` | The secret-handling boundary. API key in headers/logs/tracebacks; the **real Chrome profile** launch (R18: report, never change); download path construction and decompression (path traversal, zip/zstd bombs, partial writes); 429/403 handling; signed-URL expiry; rate limits vs ToS. Highest-stakes shard. |
| **S2 — CLI, seam & persistence** | `engine/cli.py`, `engine/ingest.py`, `engine/db.py`, `engine/paths.py`, `engine/manifest.py`, `engine/log.py`, `engine/config.py` (+ `config.yaml`) | R1/R9/R12/R22. Dedup and idempotency; transactions; migrations; argparse surface and exit codes; redaction; config validation. |
| **S3 — Parsing & maps** | `engine/parse/**`, `engine/maps/**` | awpy 3 surface assumptions; tick rate from the header (R10); round boundaries; unknowns as null (R7). |
| **S4 — State** | `engine/state/**` | LIVE_SAFE and leakage (R5); geometry and LOS; per-player tokens; `or` over numerics (R8). |
| **S5 — Actions & data** | `engine/actions/**`, `engine/data/**` | Action windows; splits (R4); dataset fitting on train rows only; quarantine. |
| **S6 — Model** | `engine/model/**` | Calibration (R6); torch imports (R15); training / prediction parity; the LightGBM yardstick. |
| **S7 — Counterfactual & attribution** | `engine/counterfactual/**`, `engine/attribution/**` | Interventions on copies; Shapley exactness; conservation and C1 (R21); sign convention (R3). |
| **S8 — Value, report, evaluation, analyze** | `engine/value/**`, `engine/report/**`, `engine/evaluation/**`, `engine/analyze.py` | C2 (R21); report redaction (R14); adoption gate (R19). |
| **S9 — Tests & bench** | `test/**`, `bench/**` | Vacuous or wrong assertions; module-level side effects; anything writing outside temp dirs; which of R1–R22 have **no** test (R16). |
| **S10 — Config, deps, agent config, docs** | `requirements-engine.*`, `pyproject.toml`, `.gitignore`, `.env.example`, `.claude/skills/**`, `README.md`, `CLAUDE.md`, `docs/**` | `.gitignore` coverage of `.env*`, `data/`, demos; deps via metadata; packaging; docs contradicting the code (CLI lines that no longer parse, invariants the code violates). Contradictions only — not prose. |

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
| `rule_violated` | `R1`–`R22` where applicable, else `null` |

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

- **The guard exists one layer up.** `engine.ingest.run` wraps `process_one` in
  `try/except`; `engine/cli.py` validates before calling. A "crash" that is caught and
  logged is not a crash.
- **The value is sanitised upstream.** IDs come from the FACEIT API, not user input;
  `argparse` `choices=` already constrains `--source`, `--by`, and `--side` to a fixed set,
  so an "injection" through those is not reachable.
- **The path is unreachable.** Dead branches, `__main__`-only code, and the two manual
  scripts in `test/` that nothing calls in normal operation.
- **The race is serialised.** SQLite's write lock, `PRIMARY KEY` / `INSERT OR IGNORE`, or
  the fact that the pipeline is single-threaded and processes one match at a time.
- **The input is validated by the schema layer on the way in.** A column typed `INTEGER`,
  or a value that `state_sampler` can only ever emit as `0`/`1`.
- **The "missing" column exists after migration.** `engine/db.py`'s `_init_schema` and
  `_MIGRATIONS` together define the schema; the DB file itself is a blocked path, so a
  schema finding is about the code, never the file.
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
- Silent, unrecoverable destruction of `data/engine/engine.db` or the demo store
  (FACEIT keeps demos about 30 days; a lost demo cannot be downloaded again).

**HIGH** — exploitable with a precondition, or silent wrongness, or a documented rule broken:
- Any R1–R22 violation without a live exploit path. **A broken invariant is HIGH here even
  when nothing exploits it** — R1 (double-insert) silently poisons the training set, and
  R4 / R5 (split or feature leakage) silently invalidate every downstream number.
- Silent data corruption: double-inserted matches, mis-signed value (R3), events scored
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
