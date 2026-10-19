# Data sources

Plan reference: P1.1.1 (pro demos), P1.2 (FACEIT level 10). Status as of 2026-10-19.
Code: `engine/sources/`. Every source ends at the seam
`engine.ingest.register_demo(match_id, demo_path, demo_url)`, which copies the `.dem`
under its tier's demo directory (`data/engine/demos/pro`, `demos/faceit` for tier
`faceit_10`, `demos/golden`; quarantined files move to `demos/quarantine`) and writes the
`matches` and `processed` rows. Dedup is `db.is_processed_any(match_id)`: a bare id
`<mid>` is checked as `<mid>` and `<mid>_m1`, a series id `<mid>_mN` as itself and
`<mid>` (a stored `_m1` never blocks `_m2`, so a half-landed series is retried); the
unique `demo_sha256` is the backstop for the other maps of a series and catches the
same file from two sources.

**Decision (Faig, 2026-10-19 evening, plan Amendment 2):** FACEIT is the primary and
baseline source (`faceit_10`, all ten players level 10); FACEIT championship rooms give
`semi_pro`; professional demos come only from public archives and are a transfer test,
not training data. No automated access to hltv.org. Research behind this:
`reports/CS2 organizer demo packs.md` (notes in `research_notes/`). Nothing below has
been run against a live site yet, and no demo is on this machine.

## Summary

| source | tier | what it offers | state | module |
|---|---|---|---|---|
| FACEIT Data API v4 + Downloads API, discovery by player | `faceit_10` | level-10 matches, ~30-day retention, free tier 1,500 requests or 200 GB / month | **primary**; working code from v1, moved unchanged; needs the Downloads scope (application) | `engine/sources/faceit_api.py` |
| FACEIT Data API v4, discovery by competition | `semi_pro` | ESEA Advanced/Main, ESL Challenger League, ESL/IEM and EWC open qualifier rooms | planned (Amendment 2 P1.2.2); endpoints documented, `demo_url` presence unverified | `engine/sources/faceit_api.py` (additive) |
| Hugging Face `blanchon/opencs2_dataset_demo` | `pro` | 1,988 HLTV-sourced maps, 2026-03-23 to 04-20, 128-tick by header on every row, 935 GB, mostly tier 2/3 (~68 maps 2+ stars) | transfer set; `hf_archive.py` works on any repo of real `.dem` files | `engine/sources/hf_archive.py` |
| Hugging Face `cs2povarchive/cs2-demos` | `pro` | 283 maps, 121 GB, 219 of them IEM Cologne Major 2026; tick rate unverified | transfer set; header-check first | `engine/sources/hf_archive.py` |
| Hugging Face `aldigobbler/cs2demos` | `pro` | 177 maps, 80 GB, tier 2/3, no licence; tick rate unverified | optional | `engine/sources/hf_archive.py` |
| HLTV match pages | `pro` | one archive per series | **dormant**: terms forbid scraping (checked 2026-10-19); not run, not extended, not deleted | `engine/sources/hltv.py` |
| Hugging Face `Vasy7777/cs2-demo-archive` | — | **not a pro archive** (19 samples, mostly video, 64-tick) | unusable | — |
| Event-organizer demo dumps | — | **none exist** (ESL/EFG, BLAST, PGL, Valve and ~20 tier-two organizers checked 2026-10-19) | closed | — |
| FACEIT Playwright session | own account | the logged-in user's own demos | macOS only as written; fallback if the Downloads application is refused | `engine/sources/scraper.py` (ask before touching) |
| Manual download | any | a demo file you already have | works | `python -m engine.cli ingest <demo.dem> --tier <tier>` |

## HLTV (pro)

- **Offers:** each match page links a demo archive holding one `.dem` per map. The
  module fetches the archive link, downloads through the existing validated
  `scraper.download_demo` (imported, not modified), unpacks one `.dem` per map and
  names files `<event>_<date>_<team1>-<team2>_<map>.dem`.
- **Terms (checked by Faig in a browser, 2026-10-19):** hltv.org/terms section 2.2
  says users may not "conduct, facilitate or organize data mining or web scraping in
  relation to the Website and/or any of the content", nor "commercially exploit the
  Website or any of its content", nor "construct and build a similar or competitive
  website, product, or service". **Verdict: no automated downloads from hltv.org**
  unless HLTV grants permission. hltv.org also answers every non-browser request,
  including `robots.txt`, with a Cloudflare challenge (HTTP 403). `hltv.py` stays in
  the tree but must not be run against the live site. Manual browser downloads
  registered with `engine.cli ingest --tier pro` remain fine.
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
repo: pass one with `engine.cli scrape --source hf --repo owner/name` (tier `pro` by
default, as `engine.ingest.SOURCES` maps `hf`; without `--repo` the command exits 2). A different pro source is needed; A4 (a few thousand pro maps obtainable) is at
risk until one is chosen.

## Event-organizer dumps (closed 2026-10-19)

Surveyed: ESL/EFG (pro.eslgaming.com, esl.gg), BLAST (blast.tv), PGL, Valve (Major
rulebook: demos go to Valve, not the public), CCT, Thunderpick, FISSURE, StarLadder,
Elisa, Skyesports, BetBoom, Perfect World, Galaxy Battle, United21, Roobet, CBCS, EWC.
None hosts demos. Organizers sell the data through GRID and Bayes under exclusive deals
(Elisa/GRID through 2027, EWC/Bayes, EFG/Bayes). The only machine-readable channel
outside HLTV is a FACEIT match room, for events played on FACEIT servers. No vendor
(GRID, Bayes, PandaScore, Abios) sells `.dem` files.

## Public archives (pro transfer set)

Three Hugging Face repos hold raw CS2 tournament demos (table above). The blanchon set
was checked first-hand on 2026-10-19 from its 137 metadata parquets: `tick_rate` 128 on
all 1,988 rows (including 504 maps on FACEIT servers), patch 14141, mean demo 470 MB.
Secondary sources claiming CS2 is 64-tick refer to Valve matchmaking; the quarantine
rule (128 only) stands. Licence tags (CC BY 4.0, Apache-2.0) cover the uploaders'
metadata at most; the demos are organizer/Valve content, and repos can vanish (Reka's
CS2-10k was removed after a Valve takedown). Mirror at ingest; keep HLTV match id and
sha256 per file; `licence_basis = hf_<repo>` in provenance. Nothing public covers
October 2023 to February 2026 as raw demos.

## FACEIT (primary source since Amendment 2: `faceit_10` baseline, `semi_pro` rooms)

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
- **Use (Amendment 2):** `faceit_10` is the baseline tier: training, validation, test,
  calibration, neighbour panel. `semi_pro` (championship rooms) trains with its own
  token. Pro is a transfer test only.
- **Retention:** demos are downloadable for about 30 days after the match (FACEIT
  support article, read via search snippet; Leetify says the same). No back-catalogue:
  the collector must run at least weekly.
- **Downloads scope:** granted by application (Google Form linked from
  docs.faceit.com, expected ~30-day response). The form offers a free tier of "up to
  1,500 requests or 200 GB" per month, or Enterprise (FACEIT moved demo access to a paid
  model in Feb 2024; Leetify quoted ~EUR 270k/yr for its volume). Estimated free-tier
  yield: ~1,000 maps/month with compressed files; measure in month one.
- **Application text (P1.2.0), to adapt:** "Per-action impact analytics for CS2
  matches: a win-probability model attributes round outcomes to individual actions and
  compares each decision with what level-10 players usually do in the same situation.
  Demos are downloaded through the Downloads API, stored locally for model training
  and analysis, and never redistributed. Intended users: FACEIT players analysing their
  own matches; intended partner: FACEIT."
- **Developer Terms:** could not be read by any fetcher (JS-rendered; faceit.com/en/terms
  is bot-gated). **Faig reads them in a browser** before the first bulk run and records
  the storage / ML-training / commercial-use clauses here.
- **Discovery by competition (planned):** `GET /search/championships?name=&game=cs2`,
  `GET /organizers/{id}/championships`, `GET /championships/{id}/matches?type=past&limit=100`;
  each match item carries the same `demo_url` array as `GET /matches/{id}`. The
  `/leagues/*` endpoints are the elo ladder, not ESEA. Verify every field against one
  real response before use.
- **Playwright path:** macOS only as written (Chrome profile and binary paths); use
  `--source api` on Windows. `scraper.py` is load-bearing: ask before touching it.

## Target volumes (Amendment 2 P1.2.4; `engine/config.yaml` `data.targets` still holds the old numbers until the code change)

First training when `faceit_10` ≥ 1,500 maps, ≥ 150 per active-duty map, ≥ 300 per
month over the last three months; growth ≥ 800 maps/month; `semi_pro` ≥ 300 before
its token is trusted; `pro` ≥ 300 for the transfer test. The old targets (≥ 3,000 pro
maps since October 2023; 50,000 FACEIT maps) are withdrawn: the first is not obtainable
legitimately, the second needs FACEIT's enterprise tier. `python -m engine.cli
inventory --assert-targets` reports shortfalls (a reminder, not a gate).
