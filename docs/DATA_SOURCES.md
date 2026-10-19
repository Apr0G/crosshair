# Data sources

Plan reference: P1.1.1 (pro demos), P1.2 (FACEIT level 10). Status as of 2026-10-19.
Code: `engine/sources/`. Every source ends at the seam
`engine.ingest.register_demo(match_id, demo_path, demo_url)`, which copies the `.dem`
under `data/engine/demos/<tier>/` and writes the `matches` and `processed` rows. Dedup
is `db.is_processed_any(match_id)` for both the BO1 id `<mid>` and the series form
`<mid>_m1`, `_m2`, ...; the same file from two sources is caught by the unique
`demo_sha256`.

**Open decision for Faig:** where pro demos come from. Nothing below has been run
against a live site yet, and no pro demo is on this machine.

## Summary

| source | tier | what it offers | state | module |
|---|---|---|---|---|
| HLTV match pages | `pro` | demo archive per match (all maps of a series) | code written, never run live; `.rar` unsupported | `engine/sources/hltv.py` |
| Hugging Face `Vasy7777/cs2-demo-archive` | — | **not a pro archive** (see below) | unusable | `engine/sources/hf_archive.py` (no default repo) |
| Event-organizer demo dumps | `pro` | some organizers publish demos for their events | not researched; nothing written | — |
| FACEIT Data API v4 (+ Downloads API) | `faceit_10` | level 10 matches and their demos | working path from the old pipeline, moved unchanged | `engine/sources/faceit_api.py` |
| FACEIT Playwright session | `faceit_10` | same demos via a logged-in browser | macOS only as written | `engine/sources/scraper.py` (load-bearing, ask before touching) |
| Manual download | any | a demo file you already have | works | `python -m engine.cli ingest <demo.dem> --tier pro` |

## HLTV (pro)

- **Offers:** each match page links a demo archive holding one `.dem` per map. The
  module fetches the archive link, downloads through the existing validated
  `scraper.download_demo` (imported, not modified), unpacks one `.dem` per map and
  names files `<event>_<date>_<team1>-<team2>_<map>.dem`.
- **Terms to check first:** read hltv.org's terms of use before any automated
  download, and stop if they forbid it (P1.1.1). Not done yet.
- **Rate limit we respect:** at least 10 s between requests (enforced in `hltv.py`),
  a browser-like session, one download at a time.
- **Known blockers:**
  - HLTV sits behind Cloudflare. The page parser is written to HLTV's markup as known
    and has not been checked against a live page; re-check it on the first real run.
  - HLTV has historically served `.rar` archives. Unpacking `.rar` needs a new
    dependency (`rarfile`) plus an unrar or 7-Zip binary on the machine, which is on
    the ask-before list. Until Faig decides, a `.rar` raises `UnsupportedArchive` and
    leaves no partial files; `.zip` works with the standard library.
  - `python -m engine.cli scrape --source hltv` has no option to pass the list of
    match URLs that `hltv.py` needs (`engine.ingest.run("hltv", matches=[...])` works
    from Python). Adding it is a CLI change.
- **Manual fallback:** download archives in a browser, unpack them, and register each
  map with `engine.cli ingest <file.dem> --tier pro --source hltv`.

## Hugging Face archive named in the plan (finding: not usable)

The plan (P1.1.1, P1.1.3, assumption A4) names `Vasy7777/cs2-demo-archive` as a public
pro demo archive of about 113 GB. Checked on 2026-10-19: **it is not a pro demo
archive.** Its README describes 19 sample matches (6 complete, 13 casual 3-POV) from a
demo-to-video pipeline; most of the 113 GB is video; file paths are opaque numeric ids
with no map or month; and the replays are 64-tick, which the quarantine rules refuse
(the engine requires 128-tick). `hf_archive.py` still works for any repo that holds
real `.dem` files (listing, waves by map and month, sha256 checks) but has no default
repo. A different pro source is needed; A4 (a few thousand pro maps obtainable) is at
risk until one is chosen.

## Event-organizer dumps

Some tournament organizers publish demo packs for their events. Not researched this
session: for each candidate, record the URL, licence or terms, file format, tick rate
and how many maps since October 2023, here, before writing any downloader.

## FACEIT (level 10, training data only)

- **Offers:** player discovery by elo and region through the FACEIT Data API v4, then
  the two-step Downloads API: `GET /matches/{id}` → `POST /download/v2/demos/download`
  → signed URL, fetched with `scraper.download_demo`.
- **Key:** `FACEIT_API_KEY` in `.env` (loaded by `engine.cli scrape`). The Downloads
  API needs Downloads access granted on that key.
- **Rate limits:** FACEIT's exact limits are not documented in this repo and were not
  checked this session. The code backs off on HTTP 429 (honours `Retry-After`, capped
  at 60 s, 4 attempts) and spaces API calls by 0.1 s. Check FACEIT's developer terms for
  the current limits before large runs.
- **Level filter:** `--min-level 10` keeps a match only if all ten players are at or
  above the level, read from `teams.<faction>.roster[].game_skill_level` per the Data
  API v4 swagger. Not yet verified against a live response.
- **Use:** FACEIT data trains the model with a tier token; it never sets the pro
  baseline, and evaluation and calibration are on pro only (decision D7).
- **Playwright path:** macOS only as written (Chrome profile and binary paths); use
  `--source api` on Windows. `scraper.py` is load-bearing: ask before touching it.

## Target volumes (P1.1.6, P1.2.3, `engine/config.yaml` `data.targets`)

≥ 3,000 pro maps from October 2023 on, ≥ 200 per active-duty map, ≥ 50 per month;
≥ 50,000 FACEIT level 10 maps, ≥ 3,000 per map. `python -m engine.cli inventory
--assert-targets` reports shortfalls (a reminder, not a gate).
