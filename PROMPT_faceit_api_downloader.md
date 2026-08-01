# Crosshair — Add a FACEIT API Demo Ingestion Path (second downloader)

## Goal
Add a **second** way to download + extract CS2 demo data into the database: pull demos through the **FACEIT Data API + Downloads API** instead of Playwright.

**Hard rules:**
- Do **NOT** remove, disable, or modify the existing Playwright downloader. It stays fully working.
- Do **NOT** change the demo parsing/analysis logic or the DB schema (additive-only if strictly required, with a migration).
- Both paths must produce **identical** downstream results — same parsed data, same rows, same tables.
- The API path plugs into the **same analysis/DB seam** the Playwright path uses. You are adding an alternate *source*, not a parallel pipeline.

---

## Phase 0 — Read & map the project (do this first, write no feature code yet)
Read the entire repo. Then produce a short map (bullet points, no fluff):
- Stack, entrypoints, how the app is run.
- **End-to-end trace of the current Playwright path:** how it discovers matches, downloads the demo, what file format lands on disk (`.dem`, `.dem.zst`, `.dem.gz`), where/how it's decompressed, and where the parsed data is written.
- The parse/analysis module(s) and the exact **DB write layer**.
- DB schema (tables, keys) and the **dedup / idempotency** logic (how it avoids re-inserting a match).
- Config + secrets handling (env vars, `.env`, etc.).
- **THE SEAM:** the exact function/interface where a local demo file (or already-parsed rows) enters analysis + DB write. This is the single plug point the new downloader will feed. Name it explicitly.

**STOP after Phase 0.** Report the map + the seam you'll plug into + your implementation plan. Wait for my go-ahead before Phase 1.

---

## Phase 1 — FACEIT API downloader module
Build a new module that mirrors the Playwright downloader's interface, so it's swappable at the seam.

- **Auth:** `Authorization: Bearer <FACEIT_API_KEY>` from env. Never hardcode the key.
- **Data API base:** `https://open.faceit.com/data/v4/`
- **Match discovery:** player match history → match details. Roughly:
  - `GET /players/{player_id}/history?game=cs2&limit=...` for match IDs.
  - `GET /matches/{match_id}` for details, which include the demo resource URL(s).
- **CRITICAL — demo download is two steps, not one:**
  1. The `demo_url` from match details is a **private resource URL**, not directly downloadable.
  2. Exchange it for a **signed download URL** via the **Downloads API**, then fetch the file from the returned signed URL.
  - The **Downloads API requires separate "Downloads" access** on the key (distinct from base Data API access). **If the key isn't authorized for Downloads, STOP and tell me** — don't try to scrape or work around it.
- **Decompression:** modern CS2 FACEIT demos are `.dem.zst` (zstandard). Handle zstd; keep a fallback for `.gz` if the existing code implies older formats. Reuse existing decompression if the Playwright path already has it.
- **Hand off to the SEAM** identified in Phase 0. Do not reimplement parsing or DB writes.
- **Dedup:** before downloading/inserting, check the DB for the match (by match_id) and skip if present, so API runs and Playwright runs don't collide or double-insert.
- **Robustness:** respect FACEIT rate limits (backoff on 429), handle expired signed URLs, partial downloads, and missing demos (some matches have none). Log clearly which path ran.

---

## Phase 2 — Path selection
- Add a config flag / CLI arg to choose the source: `playwright` (default, unchanged) or `api`.
- Do not change existing default behavior. Playwright stays the default unless the flag says otherwise.

---

## Phase 3 — Verify
- Run the API path on 1–2 matches. Confirm the written rows match the Playwright path's shape/tables exactly.
- Confirm the Playwright path is byte-for-byte untouched (its diff should be empty).
- Report: what changed, any new deps (e.g. a zstd lib), any new env vars, and the exact command to run the API path.

---

## Constraints (apply throughout)
- **Minimal diff.** No refactors of working code beyond what the seam requires.
- **Additive-only** DB changes; if unavoidable, write a migration and call it out.
- Match existing code style, structure, and naming.
- New secrets/config via env + `.env.example` entry, never committed.
- If anything about the seam, schema, or Downloads API access is ambiguous — **stop and ask**, don't guess.
