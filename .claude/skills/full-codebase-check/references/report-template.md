# Report template

Write to `docs/reviews/full-codebase-check-YYYY-MM-DD.md` (real date; `-2`, `-3` … if one
already exists for today). Follow this structure exactly.

## Rules for the report itself

1. **No sensitive data.** Quote code, never data. No keys, tokens, signed URLs, cookies,
   player nicknames, Steam IDs, or DB rows. Redact to shape: `FACEIT_API_KEY=<redacted>`,
   `https://…faceit-cdn.net/…<signed>`, `nickname=<player>`. If a file contains real-looking
   sensitive data, report its `file:line` and the class of secret — never the value.
2. **Every finding has a real `file:line`.** Verified against the file as read, not
   remembered. No "somewhere in `db.py`".
3. **Every finding has a concrete failure path**: given state X, Y happens. Name the
   preconditions — what must be true, who must run it, whether it is reachable in normal
   operation. "Could be unsafe" is not a failure path and does not belong in the report.
4. **Stable IDs**: `CRIT-01`, `HIGH-04`, `MED-11`, `LOW-03`. Numbered within severity, in
   report order. These get referenced in follow-up work, so they must not shift meaning
   between audits — a later audit reuses an ID only for the same underlying finding.
5. **No diffs, no patches.** Fixes are prose: what to change and why, enough that someone
   can implement it, without pre-writing the code. This audit does not edit.
6. **Plain, unsoftened language.** "This is wrong and here is what breaks" beats "you may
   wish to consider". No praise padding.

---

## Structure

### Header

```markdown
# Full codebase check — Crosshair

**Date:** 2026-08-01
**Repo:** github.com/Apr0G/crosshair (public)
**Commit:** `6d48589` on `main`
**Working tree:** 3 untracked files (CLAUDE.md, PROMPT_faceit_api_downloader.md, pyproject.toml)
**Scope:** 25 files in scope · 25 read in full · 0 unread
**Findings:** 2 CRITICAL · 6 HIGH · 11 MEDIUM · 8 LOW
```

If the working tree is dirty, say so here — a finding may be about uncommitted code, and
the reader needs to know the audit did not describe `HEAD`.

### Summary

3–6 sentences. Required content:

- Overall state, plainly. Is this deployable, is it a prototype with sharp edges, is
  something actively broken right now?
- **The single most urgent thing**, named in one clause.
- **Whether one bad habit shows up in ten places** — that is more actionable than ten
  separate findings, and it belongs here even when each instance is only a MEDIUM.

No softening. If the honest summary is "the corpus this model trains on is probably
contaminated," write that.

### Findings at a glance

```markdown
| ID | Sev | Title | File | Rule |
|---|---|---|---|---|
| CRIT-01 | CRITICAL | … | `src/scraper.py:284` | P10 |
| HIGH-01 | HIGH | … | `src/db.py:162` | P1 |
```

Every finding, one row, worst first. `Rule` is the `P#` violated or `—`.

### Findings

Per severity, worst first. Each finding:

```markdown
#### CRIT-01 — <specific one-line title>

**File:** `src/scraper.py:284` · **Category:** B. Secrets · **Confidence:** high · **Rule:** P10

**What's wrong**
The precise defect: the wrong condition, the missing check, the incorrect assumption.
One paragraph, no preamble.

**Evidence**
```python
# src/scraper.py:282-285
    print(f"\r  {pct}% ({mb} MB / {total // 1_000_000} MB)", end="", flush=True)
```
Quoted exactly as it appears. Line range in the comment.

**How it fails**
Concrete and preconditioned. Who must run what, in which state, for this to fire.
Name whether it is reachable in normal operation, and whether it is remote or local.

**Blast radius**
What is affected when it does fire — one match, the whole corpus, the API key, the
Chrome profile, a model whose numbers are now known-wrong.

**Suggested fix**
Prose. What to change and why. Not applied. No diff.
```

MEDIUM findings may be terser — **What's wrong**, **Evidence**, **How it fails**,
**Suggested fix** is enough; drop **Blast radius** if it is obvious.

LOW findings may be a single table:

```markdown
| ID | Title | File | Fix |
|---|---|---|---|
| LOW-01 | Stale allowlist entry for deleted `src/wp_model.py` | `.claude/settings.json:14` | Remove the entry |
```

### Rule compliance

**One row per project rule, filled in even when everything passes.** A rule with no finding
is a rule that was checked and held — that is information, and it is why this table is not
optional.

```markdown
| Rule | Statement | Status | Evidence |
|---|---|---|---|
| P1 | No path may double-insert a match | ✗ VIOLATED | HIGH-01 — events commit before `mark_processed` |
| P5 | Impact sign: CT `p_after - p_before`, T negated | ✓ HOLDS | `src/score_impact.py:58` |
| P13 | `/security-review` before every commit | ⚠ UNENFORCED | Documented in `CLAUDE.md`; no hook or CI check |
```

Status is `✓ HOLDS`, `✗ VIOLATED`, `⚠ UNENFORCED`, or `? UNVERIFIABLE`.

### Coverage

```markdown
**In scope:** 25 · **Read in full:** 24 · **Unread:** 1
```

Then a table naming **every** unread or partially-read file with its reason. An audit that
quietly skipped six files is worse than no audit — it reads as a clean bill of health over
a blind spot.

```markdown
| File | Lines | Status | Reason |
|---|---|---|---|
| `data/crosshair.db` | — | not read | Blocked path — schema inspected via PRAGMA only |
| `src/feature_extractor.py` | 1351 | read in full | — |
```

Blocked paths from Non-negotiable 2 are listed here as deliberately excluded, not as gaps.

### Unverifiable

Findings that depend on runtime state, a blocked path, or a live credential. For each: what
is suspected, why it cannot be settled from source, and **the exact check Faig should run by
hand** — phrased so he never has to paste a secret anywhere.

```markdown
| # | Suspected | Why unverifiable | Check by hand |
|---|---|---|---|
| U-1 | `.env` may carry a trailing newline in the key | `.env` is a blocked path | Run `grep -c '.' .env` and confirm the key line has no trailing space |
```

### Not findings

Things that looked wrong and are not — **including everything Phase 3 refuted**, with the
refutation. This section is what stops the next audit re-raising the same six items, so it
is not optional and it is not padding.

```markdown
| Looked like | Why it's fine |
|---|---|
| `--side` interpolated into SQL in `cmd_top` | Constrained by `argparse choices=["ct","t"]`; no other value can reach the string |
| Race between `is_processed` and insert | Pipeline is single-threaded and processes one match at a time; no concurrent writer exists |
```

### Cost

One line: agents dispatched, approximate tokens, wall time. Lets the next run be scoped.
