"""
FACEIT Data + Downloads API demo source.

Alternate to the Playwright scraper: discovers matches with the *same* criteria
(reusing scraper.get_leaderboard_players / get_player_match_ids), then pulls the
demo through the FACEIT Downloads API instead of a browser session.

Mirrors the scraper source interface so pipeline can swap it at the seam:
    iter_unprocessed_demos(...)          -> yields (match_id, resource_url)
    download_demo(resource_url, dest)    -> local .dem Path

Demo download is two steps:
  1. GET /matches/{id} gives a private demo resource URL (not directly fetchable).
  2. POST /download/v2/demos/download exchanges it for a short-lived signed URL,
     which is then fetched + decompressed by the existing scraper downloader.

Required in .env:
    FACEIT_API_KEY   (Data API key; must also be granted Downloads access)
"""

import os
import time
import requests
from pathlib import Path

import scraper
from db import is_processed

DOWNLOADS_URL = "https://open.faceit.com/download/v2/demos/download"


def _bearer() -> dict:
    key = os.getenv("FACEIT_API_KEY")
    if not key:
        raise EnvironmentError("FACEIT_API_KEY not set")
    return {"Authorization": f"Bearer {key}"}


# ── Match discovery (same criteria as the Playwright path) ────────────────────

def get_match_demo_urls(match_id: str) -> list[str]:
    """Demo resource URL(s) for a match via GET /matches/{id}. [] if none."""
    try:
        data = scraper._api_get(f"/matches/{match_id}")
    except Exception as e:
        print(f"    match details failed for {match_id}: {e}")
        return []
    urls = data.get("demo_url") or []
    if isinstance(urls, str):
        urls = [urls]
    return urls


def iter_unprocessed_demos(
    min_elo: int = 2500,
    region: str = "EU",
    max_players: int = 10,
    matches_per_player: int = 20,
    start_offset: int = 0,
):
    """
    Yields (match_id, resource_url), mirroring scraper.iter_unprocessed_demos.
    Identical player/match selection; demo URLs come from the Data API instead
    of a browser interception.
    """
    print(f"Fetching {region} leaderboard (ELO >= {min_elo}, max {max_players} players) ...")
    players = scraper.get_leaderboard_players(
        region=region, min_elo=min_elo, max_players=max_players, start_offset=start_offset
    )
    print(f"Found {len(players)} players.")

    seen = set()

    for i, p in enumerate(players, 1):
        nickname = p["nickname"]
        print(f"  [{i}/{len(players)}] {nickname} (ELO {p['elo']}) ...")
        match_ids = scraper.get_player_match_ids(nickname, limit=matches_per_player)

        for mid in match_ids:
            if not mid or mid in seen:
                continue
            # check both BO1 (mid) and BO3 (mid_m1) formats
            if is_processed(mid) or is_processed(f"{mid}_m1"):
                print(f"    skipping {mid} (already in db)")
                seen.add(mid)
                continue
            seen.add(mid)

            print(f"    getting demo URLs for {mid} ...")
            demo_urls = get_match_demo_urls(mid)
            if not demo_urls:
                print(f"    no demo for {mid}, skipping")
                continue
            for idx, demo_url in enumerate(demo_urls):
                map_match_id = f"{mid}_m{idx + 1}" if len(demo_urls) > 1 else mid
                if is_processed(map_match_id):
                    print(f"    skipping {map_match_id} (already in db)")
                    continue
                print(f"    found: {demo_url[:60]}...")
                yield (map_match_id, demo_url)

            time.sleep(0.1)


# ── Demo download (two-step: exchange → fetch + decompress) ───────────────────

def _signed_url(resource_url: str, max_retries: int = 4) -> str:
    """Exchange a private demo resource URL for a signed download URL."""
    for attempt in range(max_retries):
        resp = requests.post(
            DOWNLOADS_URL,
            headers={**_bearer(), "Content-Type": "application/json"},
            json={"resource_url": resource_url},
            timeout=30,
        )
        if resp.status_code == 429:
            wait = 2 ** attempt
            print(f"    rate limited (429), backing off {wait}s ...")
            time.sleep(wait)
            continue
        if resp.status_code == 403:
            raise PermissionError(
                "FACEIT Downloads API returned 403 — the API key lacks Downloads "
                "access. Grant Downloads to the key and retry."
            )
        resp.raise_for_status()
        return resp.json()["payload"]["download_url"]
    raise RuntimeError(f"Downloads API exchange failed after {max_retries} retries")


def download_demo(resource_url: str, dest: Path) -> Path:
    """
    Two-step FACEIT download: exchange the resource URL for a fresh signed URL,
    then fetch + decompress via the existing scraper downloader (unmodified).
    Signed URLs are short-lived, so re-exchange once if the fetch is denied.
    """
    for attempt in range(2):
        signed = _signed_url(resource_url)
        try:
            return scraper.download_demo(signed, dest)
        except requests.HTTPError as e:
            expired = e.response is not None and e.response.status_code in (401, 403)
            if attempt == 0 and expired:
                print("    signed URL expired/denied, re-exchanging ...")
                continue
            raise
