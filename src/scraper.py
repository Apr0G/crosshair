"""
Faceit demo scraper.

Player discovery uses the public Faceit API (rankings/players/match-history).
Demo URL discovery uses a real Chrome session via Playwright, since the demo
download endpoint requires an authenticated session and isn't exposed publicly.

Required in .env:
    FACEIT_API_KEY   (data API key from developers.faceit.com)

First launch opens a Chrome window — log into Faceit, then press Enter in the
terminal to continue. The session is reused for subsequent matches.
"""

import os
import time
import gzip
import shutil
import requests
from pathlib import Path

FACEIT_BASE = "https://www.faceit.com"
API_BASE    = "https://open.faceit.com/data/v4"


# ── Faceit API (player discovery only) ───────────────────────────────────────

def _api_headers() -> dict:
    key = os.getenv("FACEIT_API_KEY")
    if not key:
        raise EnvironmentError("FACEIT_API_KEY not set")
    return {"Authorization": f"Bearer {key}"}


def _api_get(endpoint: str, params: dict = None) -> dict:
    resp = requests.get(f"{API_BASE}{endpoint}", headers=_api_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_leaderboard_players(region: str = "EU", min_elo: int = 2500, max_players: int = 50, start_offset: int = 0) -> list[dict]:
    players = []
    offset  = 0
    limit   = 100

    while len(players) < start_offset + max_players:
        data  = _api_get(f"/rankings/games/cs2/regions/{region}", {"limit": limit, "offset": offset})
        items = data.get("items", [])
        if not items:
            break
        for item in items:
            elo = item.get("faceit_elo", 0)
            if elo < min_elo:
                return players[start_offset:start_offset + max_players]
            players.append({
                "player_id": item.get("player_id"),
                "nickname":  item.get("nickname"),
                "elo":       elo,
            })
        offset += limit
        if len(items) < limit:
            break
        time.sleep(0.1)

    return players[start_offset:start_offset + max_players]


# ── Playwright browser session ────────────────────────────────────────────────

_browser_ctx = None


def _get_browser_ctx():
    global _browser_ctx
    if _browser_ctx is None:
        from playwright.sync_api import sync_playwright

        pw  = sync_playwright().start()
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(Path.home() / "Library/Application Support/Google/Chrome/Default"),
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--no-first-run",
                "--no-default-browser-check",
            ],
            ignore_default_args=["--enable-automation"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        print("\n>>> Chrome opened. Log into Faceit if needed, then press ENTER to continue ...")
        try:
            page.goto("https://www.faceit.com/en", timeout=15000)
        except Exception:
            pass

        input()

        print("continuing ...")
        page.close()
        _browser_ctx = (pw, None, ctx)

    return _browser_ctx


def _new_page():
    _, _, ctx = _get_browser_ctx()
    return ctx.new_page()


# ── Demo URL discovery ────────────────────────────────────────────────────────

def get_player_match_ids(nickname: str, limit: int = 20) -> list[str]:
    """Gets recent match IDs for a player via the Faceit API."""
    try:
        data = _api_get(f"/players", {"nickname": nickname, "game": "cs2"})
        pid  = data.get("player_id")
        if not pid:
            return []
        matches = _api_get(f"/players/{pid}/history", {"game": "cs2", "limit": limit})
        return [m.get("match_id") or m.get("id") for m in matches.get("items", []) if m.get("match_id") or m.get("id")]
    except Exception:
        return []


def get_demo_urls_from_page(match_id: str) -> list[str]:
    """
    Opens a Faceit matchroom and collects demo URLs.
    Handles BO1 (single URL) and BO3/BO5 (dropdown with one button per map).
    Returns list of demo URLs found.
    """
    page = _new_page()
    intercepted: list[str] = []

    def handle_download(download):
        url = download.url
        if url not in intercepted:
            intercepted.append(url)
            print(f"    intercepted: {url[:100]}")
        download.cancel()

    def handle_response(response):
        url = response.url
        if any(ext in url for ext in (".dem.zst", ".dem.gz", "/download/demo", "download?token")):
            if response.status < 400 and url not in intercepted:
                intercepted.append(url)
                print(f"    intercepted: {url[:100]}")

    page.on("download", handle_download)
    page.on("response", handle_response)

    try:
        page.goto(f"{FACEIT_BASE}/en/cs2/room/{match_id}", wait_until="domcontentloaded", timeout=30000)
        time.sleep(5)

        watch_clicked = False
        for selector in [
            "text=WATCH DEMO", "text=Watch demo", "text=Watch Demo",
            "button:has-text('WATCH DEMO')", "button:has-text('Watch demo')",
            "a:has-text('WATCH DEMO')", "a:has-text('Watch demo')",
        ]:
            try:
                el = page.locator(selector).first
                if el.is_visible(timeout=2000):
                    el.click()
                    watch_clicked = True
                    time.sleep(1.5)
                    break
            except Exception:
                continue

        if watch_clicked:
            time.sleep(5)  # wait for all downloads to fire (BO3/5 can take a moment)

            if not intercepted:
                # no download yet — must be a BO series dropdown, click each map
                for n in range(1, 6):
                    clicked = False
                    for selector in [
                        f"text=Demo {n}",
                        f"button:has-text('Demo {n}')",
                        f"[role='menuitem']:has-text('Demo {n}')",
                        f"[role='option']:has-text('Demo {n}')",
                    ]:
                        try:
                            el = page.locator(selector).first
                            if el.is_visible(timeout=800):
                                el.click()
                                clicked = True
                                time.sleep(2)
                                break
                        except Exception:
                            continue
                    if not clicked:
                        break  # no more maps

        if not intercepted:
            time.sleep(2)

    except Exception as e:
        print(f"  page error for {match_id}: {e}")
    finally:
        page.close()

    return intercepted


def iter_unprocessed_demos(
    min_elo: int = 2500,
    region: str = "EU",
    max_players: int = 10,
    matches_per_player: int = 20,
    start_offset: int = 0,
):
    """
    Generator that yields (match_id, demo_url) one at a time.
    Gets the URL right before yielding so presigned URLs don't expire.
    """
    from db import is_processed

    print(f"Fetching {region} leaderboard (ELO >= {min_elo}, max {max_players} players) ...")
    players = get_leaderboard_players(region=region, min_elo=min_elo, max_players=max_players, start_offset=start_offset)
    print(f"Found {len(players)} players.")

    seen = set()

    for i, p in enumerate(players, 1):
        nickname = p["nickname"]
        print(f"  [{i}/{len(players)}] {nickname} (ELO {p['elo']}) ...")
        match_ids = get_player_match_ids(nickname, limit=matches_per_player)

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
            demo_urls = get_demo_urls_from_page(mid)
            for idx, demo_url in enumerate(demo_urls):
                map_match_id = f"{mid}_m{idx + 1}" if len(demo_urls) > 1 else mid
                if is_processed(map_match_id):
                    print(f"    skipping {map_match_id} (already in db)")
                    continue
                print(f"    found: {demo_url[:60]}...")
                yield (map_match_id, demo_url)

            time.sleep(0.5)


# ── Demo download ────────────────────────────────────────────────────────────

def download_demo(demo_url: str, dest: Path) -> Path:
    """Downloads and decompresses a demo file."""
    print(f"  Downloading {demo_url[:80]} ...")

    session = requests.Session()
    resp = session.get(demo_url, stream=True, timeout=120)
    resp.raise_for_status()

    url_lower = demo_url.lower()
    if ".zst" in url_lower:
        raw_path = dest.with_suffix(".dem.zst")
    elif ".gz" in url_lower:
        raw_path = dest.with_suffix(".dem.gz")
    else:
        raw_path = dest

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(raw_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                mb  = downloaded // 1_000_000
                print(f"\r  {pct}% ({mb} MB / {total // 1_000_000} MB)", end="", flush=True)
    print()

    if raw_path.suffix == ".zst":
        print("  Decompressing (.zst) ...")
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        with open(raw_path, "rb") as compressed, open(dest, "wb") as out:
            dctx.copy_stream(compressed, out)
        raw_path.unlink()

    elif raw_path.suffix == ".gz":
        print("  Decompressing (.gz) ...")
        with gzip.open(raw_path, "rb") as gz, open(dest, "wb") as out:
            shutil.copyfileobj(gz, out)
        raw_path.unlink()

    print(f"  Saved to {dest}  ({dest.stat().st_size // 1_000_000} MB)")
    return dest
