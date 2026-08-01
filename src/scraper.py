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

import atexit
import os
import random
import sys
import time
import gzip
import requests
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

FACEIT_BASE = "https://www.faceit.com"
API_BASE    = "https://open.faceit.com/data/v4"

# Hosts a demo may legitimately come from. The interceptor used to accept ANY URL
# containing ".dem.zst" etc. anywhere in the string — including inside a query
# parameter on a third-party host the matchroom happened to load.
_DEMO_HOSTS     = ("faceit-cdn.net", "faceit.com")
_DEMO_PATH_EXTS = (".dem", ".dem.zst", ".dem.gz")

# Ceiling on a single demo. Real CS2 demos are well under this; zstd's ratio is
# effectively unbounded, so a corrupt or hostile archive could otherwise fill the disk.
MAX_DEMO_BYTES = 4 * 1024 ** 3   # 4 GiB
# Below this a 'demo' is an error page or a stub, not a match.
MIN_DEMO_BYTES = 1000


def redact_url(url: str) -> str:
    """Drop the query string — FACEIT signed URLs carry their credential there.
    Never slice a URL by character count: whether the signature survives a [:80]
    depends on host and path length, which is luck rather than a control."""
    if not url:
        return ""
    try:
        p = urlsplit(str(url))
    except ValueError:
        return "<unparseable url>"
    clean = urlunsplit((p.scheme, p.netloc, p.path, "", ""))
    return f"{clean}?<redacted>" if p.query else clean


def _is_demo_url(url: str) -> bool:
    """Host must be FACEIT's and the PATH must end in a demo extension."""
    try:
        p = urlsplit(url)
    except ValueError:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.hostname or "").lower().rstrip(".")
    # Exact match, or a genuine SUBdomain. A bare `endswith("faceit.com")` also
    # accepts "evilfaceit.com" and "notfaceit.com" — the dot is what makes it a
    # boundary rather than a substring.
    if not any(host == d or host.endswith("." + d) for d in _DEMO_HOSTS):
        return False
    return p.path.lower().endswith(_DEMO_PATH_EXTS)


# ── Faceit API (player discovery only) ───────────────────────────────────────

def _api_headers() -> dict:
    key = os.getenv("FACEIT_API_KEY")
    if not key:
        raise EnvironmentError("FACEIT_API_KEY not set")
    return {"Authorization": f"Bearer {key}"}


def _api_get(endpoint: str, params: dict = None, max_retries: int = 4) -> dict:
    """Both source modules go through here, so 429 handling belongs here too.

    Previously a 429 raised straight through and every caller's blanket except turned
    it into "this player has no matches" — so crossing the rate limit made the scraper
    keep going at full speed while silently producing nothing.
    """
    for attempt in range(max_retries):
        resp = requests.get(f"{API_BASE}{endpoint}", headers=_api_headers(),
                            params=params, timeout=30)
        if resp.status_code == 429:
            hdr = resp.headers.get("Retry-After")
            try:
                wait = min(float(hdr), 60.0) if hdr else min(2 ** attempt, 60.0)
            except (TypeError, ValueError):
                wait = min(2 ** attempt, 60.0)
            wait += random.uniform(0, 0.5)   # jitter: don't re-synchronise across runs
            print(f"    rate limited (429) on {endpoint}, backing off {wait:.1f}s ...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"FACEIT Data API still rate-limited after {max_retries} attempts: {endpoint}")


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
            elo = item.get("faceit_elo")
            # A missing/non-numeric elo is a malformed record, NOT "below threshold".
            # Defaulting it to 0 made one bad row terminate discovery for the whole run.
            if not isinstance(elo, (int, float)):
                print(f"    skipping leaderboard item with no faceit_elo: {item.get('nickname')!r}")
                continue
            if elo < min_elo:
                return players[start_offset:start_offset + max_players]
            if not item.get("nickname") or not item.get("player_id"):
                continue
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
# Backstop: a crash or Ctrl-C must not leave Chrome running on the real profile.
atexit.register(lambda: close_browser())


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
        # Assign BEFORE the prompt: if input() raises, the context is already running
        # and would otherwise be unreachable and unclosable, holding the profile lock.
        _browser_ctx = (pw, None, ctx)

        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            page.goto("https://www.faceit.com/en", timeout=15000)
        except Exception as e:
            print(f">>> could not open faceit.com ({e}) — the profile may be locked "
                  f"by a running Chrome, or there is no network.")

        if sys.stdin.isatty():
            print("\n>>> Chrome opened. Log into Faceit if needed, then press ENTER to continue ...")
            try:
                input()
            except EOFError:
                print(">>> no stdin, continuing without waiting.")
        else:
            # Non-interactive (cron, CI, nohup): input() would either raise EOFError
            # or block forever with no output. Assume the persistent profile is logged in.
            print(">>> non-interactive session, skipping the login prompt.")

        print("continuing ...")
        page.close()

    return _browser_ctx


def close_browser():
    """Tear down the Playwright context and driver. Nothing called this before, so a
    crashed run left Chrome alive holding the real profile."""
    global _browser_ctx
    if _browser_ctx is None:
        return
    pw, _, ctx = _browser_ctx
    _browser_ctx = None
    for closer in (getattr(ctx, "close", None), getattr(pw, "stop", None)):
        try:
            if closer:
                closer()
        except Exception as e:
            print(f"  browser teardown warning: {e}")


def _new_page():
    _, _, ctx = _get_browser_ctx()
    return ctx.new_page()


# ── Demo URL discovery ────────────────────────────────────────────────────────

def get_player_match_ids(nickname: str, limit: int = 20) -> list[str]:
    """Gets recent match IDs for a player via the Faceit API.

    Auth-class failures propagate: returning [] for a 401/403 is indistinguishable
    from "this player has no CS2 history", so a revoked key would walk every player,
    yield nothing, and report a clean run.
    """
    try:
        data = _api_get("/players", {"nickname": nickname, "game": "cs2"})
        pid  = data.get("player_id")
        if not pid:
            return []
        matches = _api_get(f"/players/{pid}/history", {"game": "cs2", "limit": limit})
        return [m.get("match_id") or m.get("id") for m in matches.get("items", []) if m.get("match_id") or m.get("id")]
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status in (401, 403):
            raise PermissionError(
                f"FACEIT Data API returned {status} — the API key is invalid or lacks "
                f"access. Fix FACEIT_API_KEY and retry."
            ) from e
        print(f"    lookup failed for {nickname}: {e}")
        return []
    except Exception as e:
        print(f"    lookup failed for {nickname}: {e}")
        return []


def get_demo_urls_from_page(match_id: str) -> list[str]:
    """
    Opens a Faceit matchroom and collects demo URLs.
    Handles BO1 (single URL) and BO3/BO5 (dropdown with one button per map).
    Returns list of demo URLs found.
    """
    page = _new_page()
    intercepted: list[str] = []

    MAX_INTERCEPTED = 8   # a page can fire arbitrarily many matching requests

    def _remember(url: str):
        if len(intercepted) >= MAX_INTERCEPTED:
            return
        if not _is_demo_url(url):
            return
        # Deduplicate on the PATH, not the full URL: a redirect yields two distinct
        # URLs for one demo, which would make len(demo_urls) > 1 and split a BO1 into
        # a fake _m1/_m2 pair — the same demo inserted twice under two match_ids, and
        # one match landing on both sides of the GroupShuffleSplit.
        key = urlsplit(url).path
        if any(urlsplit(u).path == key for u in intercepted):
            return
        intercepted.append(url)
        print(f"    intercepted: {redact_url(url)}")

    def handle_download(download):
        _remember(download.url)
        download.cancel()

    def handle_response(response):
        # Skip 3xx so only the terminal response is captured.
        if 200 <= response.status < 300:
            _remember(response.url)

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
            # Only short-circuit on the bare id. Treating <mid>_m1 as proof the whole
            # series landed permanently abandons maps 2/3 of a BO where map 1 was
            # stored and a later map failed — the per-URL guard below handles those.
            if is_processed(mid):
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
                print(f"    found: {redact_url(demo_url)}")
                yield (map_match_id, demo_url)

            # Jitter: a perfectly periodic 0.5 s cadence of matchroom loads is a
            # recognisable automation signature against FACEIT's own site.
            time.sleep(0.5 + random.uniform(0, 0.6))


# ── Demo download ────────────────────────────────────────────────────────────

_MAGIC = {b"\x28\xb5\x2f\xfd": "zst", b"\x1f\x8b": "gz"}


def _sniff(path: Path) -> str:
    """Compression format from the file's magic bytes. The old code inferred it from
    a substring of the whole URL — including the signed query string, where a stray
    '.zst' in a token would pick the wrong branch."""
    with open(path, "rb") as f:
        head = f.read(4)
    for magic, kind in _MAGIC.items():
        if head.startswith(magic):
            return kind
    return "raw"


def download_demo(demo_url: str, dest: Path) -> Path:
    """Downloads, validates and decompresses a demo file."""
    safe = redact_url(demo_url)
    print(f"  Downloading {safe} ...")

    tmp = dest.with_name(dest.name + ".part")
    ok  = False
    try:
        # Redact at the point of the request, not at the print site: the signed URL
        # must never escape this function inside an exception. requests puts the full
        # URL in HTTPError, and urllib3 puts "path?query" in every CONNECTION-level
        # error (DNS, TLS, reset, MaxRetryError) — the latter defeats any
        # replace-the-known-URL scrubbing downstream, since the string differs.
        with requests.Session() as session:
            try:
                resp_cm = session.get(demo_url, stream=True, timeout=(10, 60))
            except requests.RequestException as e:
                raise IOError(f"fetching {safe}: {type(e).__name__}") from None
            with resp_cm as resp:
                if resp.status_code >= 400:
                    raise IOError(f"HTTP {resp.status_code} fetching {safe}")

                ctype = (resp.headers.get("content-type") or "").lower()
                if "text/html" in ctype:
                    raise IOError(f"expected a demo, got HTML ({ctype}) from {safe}")

                total = int(resp.headers.get("content-length", 0) or 0)
                if total and total > MAX_DEMO_BYTES:
                    raise IOError(f"demo declares {total / 1024**3:.1f} GiB, over the cap")

                downloaded = 0
                deadline = time.monotonic() + 3600   # total budget, not per-read
                with open(tmp, "wb") as f:
                    try:
                        for chunk in resp.iter_content(chunk_size=65536):
                            if time.monotonic() > deadline:
                                raise IOError("download exceeded its time budget")
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded > MAX_DEMO_BYTES:
                                raise IOError("download exceeded the size cap")
                            if total:
                                print(f"\r  {downloaded * 100 // total}% "
                                      f"({downloaded // 1_000_000} MB / {total // 1_000_000} MB)",
                                      end="", flush=True)
                    except requests.RequestException as e:
                        # A reset mid-transfer carries "path?query" in the message.
                        raise IOError(f"transfer failed for {safe}: {type(e).__name__}") from None
                print()

        # A dropped connection used to produce a short file that parsed as a short
        # match and was then marked processed — silent data loss.
        if total and downloaded != total:
            raise IOError(f"truncated download: got {downloaded} of {total} bytes")

        kind = _sniff(tmp)
        if kind == "zst":
            print("  Decompressing (.zst) ...")
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            written = 0
            with open(tmp, "rb") as compressed, open(dest, "wb") as out:
                reader = dctx.stream_reader(compressed)
                while True:
                    block = reader.read(1024 * 1024)
                    if not block:
                        break
                    written += len(block)
                    if written > MAX_DEMO_BYTES:
                        raise IOError("decompressed demo exceeded the size cap")
                    out.write(block)
        elif kind == "gz":
            print("  Decompressing (.gz) ...")
            written = 0
            with gzip.open(tmp, "rb") as gz, open(dest, "wb") as out:
                while True:
                    block = gz.read(1024 * 1024)
                    if not block:
                        break
                    written += len(block)
                    if written > MAX_DEMO_BYTES:
                        raise IOError("decompressed demo exceeded the size cap")
                    out.write(block)
        else:
            # Validate BEFORE publishing to dest: a rejected download must not leave
            # a file where the caller expects a usable demo.
            if tmp.stat().st_size < MIN_DEMO_BYTES:
                raise IOError(f"demo is implausibly small ({tmp.stat().st_size} bytes)")
            tmp.replace(dest)

        if dest.stat().st_size < MIN_DEMO_BYTES:
            raise IOError(f"demo is implausibly small ({dest.stat().st_size} bytes)")

        ok = True
        print(f"  Saved to {dest}  ({dest.stat().st_size // 1_000_000} MB)")
        return dest
    finally:
        # Clean up on the failure path too — the old code only unlinked on success.
        for leftover in ((tmp,) if ok else (tmp, dest)):
            if leftover.exists():
                try:
                    leftover.unlink()
                except OSError:
                    pass
