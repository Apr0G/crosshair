"""
HLTV pro demo source (P1.1.2): match URLs/ids -> demo archive -> one `.dem` per map,
named `<event>_<date>_<team1>-<team2>_<map>.dem`. Downloads go through
`engine.sources.scraper.download_demo` (imported, never modified). Spacing >= 10 s,
corrupt archives leave no partial files, URLs redacted in every error.

    from engine.sources import hltv
    files = hltv.download_match("2370000", out_dir)          # [Path, Path, ...]
    engine.ingest.run("hltv", matches=["https://www.hltv.org/matches/2370000/x"])

Archives: `.zip` is unpacked with the stdlib. `.rar` (what HLTV has historically
served) raises UnsupportedArchive: unpacking it needs a RAR extractor, which is a new
dependency (ask-before). The match page selectors in `parse_match_page` are written to
HLTV's markup as known, not verified against a live page this session (HLTV sits
behind Cloudflare); re-check them on the first real run. Check the site terms first
(P1.1.1) and stop if they forbid automated download.

Source interface for `engine.ingest.run`: `iter_unprocessed_demos(matches=...)` downloads
and unpacks each match into a work dir and yields `(match_id, file URI)` per map, the
match_id being the file stem; `download_demo(uri, dest)` moves that file to `dest`;
`meta_for(match_id)` gives `event` and `date` for the `matches` row.
"""
import html
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.request import url2pathname

import requests

from engine import db
from engine.data.quarantine import KNOWN_MAPS
from engine.log import redact_text
from engine.sources import scraper
from engine.sources.scraper import redact_url

SOURCE_NAME    = "hltv"
HLTV_BASE      = "https://www.hltv.org"
MIN_INTERVAL_S = 10.0
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/129.0 Safari/537.36"),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_ZIP_MAGIC = b"PK\x03\x04"
_RAR_MAGIC = b"Rar!\x1a\x07"
_MAP_SHORT = {m.removeprefix("de_"): m for m in KNOWN_MAPS}


class UnsupportedArchive(ValueError):
    """The archive format cannot be unpacked here (e.g. .rar)."""


# ── throttle ──────────────────────────────────────────────────────────────────

class Throttle:
    """At least `min_interval_s` between consecutive requests (page or archive)."""

    def __init__(self, min_interval_s: float = MIN_INTERVAL_S, clock=time.monotonic,
                 sleep=time.sleep):
        self.min_interval_s = min_interval_s
        self._clock, self._sleep = clock, sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            gap = self.min_interval_s - (self._clock() - self._last)
            if gap > 0:
                self._sleep(gap)
        self._last = self._clock()


_THROTTLE = Throttle()


# ── match page ────────────────────────────────────────────────────────────────

@dataclass
class HltvMatch:
    match_id: str
    url:      str
    event:    str | None
    date:     str | None          # yyyy-mm-dd (UTC)
    team1:    str | None
    team2:    str | None
    demo_url: str | None


def match_url(match: str, base_url: str = HLTV_BASE) -> tuple[str, str]:
    """(numeric id, page URL) for an HLTV match id or match URL."""
    s = str(match).strip()
    if s.isdigit():
        return s, f"{base_url.rstrip('/')}/matches/{s}/match"
    p = urlsplit(s)
    m = re.search(r"/matches/(\d+)(?:/|$)", p.path)
    if p.scheme not in ("http", "https") or not m:
        raise ValueError(f"not an HLTV match id or URL: {redact_url(s)}")
    return m.group(1), s


def _text(pattern: str, page: str) -> str | None:
    m = re.search(pattern, page, re.S)
    return html.unescape(m.group(1)).strip() if m else None


def parse_match_page(page: str) -> dict:
    """event, date, team1, team2, demo_path from a match page; None where absent."""
    event = (_text(r'<div class="event text-ellipsis">\s*<a[^>]*title="([^"]+)"', page)
             or _text(r'class="event[^"]*"[^>]*>\s*<a[^>]*>([^<]+)</a>', page))
    teams = [html.unescape(t).strip() for t in re.findall(r'<div class="teamName">([^<]+)</div>', page)]
    unix  = _text(r'<div class="date" data-unix="(\d+)"', page)
    date  = (datetime.fromtimestamp(int(unix) / 1000, timezone.utc).strftime("%Y-%m-%d")
             if unix else None)
    demo  = (_text(r'data-demo-link="([^"]+)"', page)
             or _text(r'href="(/download/demo/\d+)"', page))
    return {"event": event, "date": date,
            "team1": teams[0] if len(teams) > 0 else None,
            "team2": teams[1] if len(teams) > 1 else None,
            "demo_path": demo}


def fetch_page(url: str, throttle: Throttle | None = None) -> str:
    (throttle or _THROTTLE).wait()
    safe = redact_url(url)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=(10, 30))
    except requests.RequestException as e:
        raise IOError(f"fetching {safe}: {type(e).__name__}") from None
    if resp.status_code >= 400:
        raise IOError(f"HTTP {resp.status_code} fetching {safe}")
    return resp.text


def resolve(match: str, base_url: str = HLTV_BASE, throttle: Throttle | None = None) -> HltvMatch:
    mid, url = match_url(match, base_url)
    info = parse_match_page(fetch_page(url, throttle))
    demo = urljoin(url, info["demo_path"]) if info["demo_path"] else None
    return HltvMatch(mid, url, info["event"], info["date"], info["team1"], info["team2"], demo)


# ── naming and unpacking ──────────────────────────────────────────────────────

def slug(s: str | None) -> str:
    """Lowercase, runs of anything but [a-z0-9] -> '-'. Never contains '_' (the field
    separator in file names). Missing text becomes 'unknown' (a file-name label only;
    the database keeps NULL)."""
    out = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")
    return out or "unknown"


def map_from_member(name: str) -> str:
    """Map from an archive member name like `navi-vs-faze-m1-mirage.dem`."""
    stem = Path(name).name.lower().removesuffix(".dem")
    for token in re.split(r"[^a-z0-9]+", stem)[::-1]:
        if token in _MAP_SHORT:
            return _MAP_SHORT[token]
    m = re.search(r"de_[a-z0-9]+", stem)
    return m.group(0) if m else slug(stem)


def demo_name(info: HltvMatch, map_name: str) -> str:
    return f"{slug(info.event)}_{info.date or 'unknown'}_{slug(info.team1)}-{slug(info.team2)}_{map_name}.dem"


def unpack(archive: Path, dest_dir: Path, info: HltvMatch) -> list[Path]:
    """Unpack every `.dem` in a zip into `dest_dir` under its P1.1.2 name. All or
    nothing: members are staged first and moved only when every one passed; a corrupt
    archive or a name clash leaves no file behind. Never overwrites."""
    archive, dest_dir = Path(archive), Path(dest_dir)
    with open(archive, "rb") as fh:
        head = fh.read(8)
    if head.startswith(_RAR_MAGIC):
        raise UnsupportedArchive(f"{redact_url(info.url)}: .rar archives are not supported "
                                 f"(needs a RAR extractor dependency); unpack by hand")
    if not head.startswith(_ZIP_MAGIC):
        raise UnsupportedArchive(f"{redact_url(info.url)}: not a zip archive")

    dest_dir.mkdir(parents=True, exist_ok=True)
    staging = dest_dir / f".staging-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    moved: list[Path] = []
    try:
        with zipfile.ZipFile(archive) as z:
            members = [i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(".dem")]
            if not members:
                raise IOError(f"archive from {redact_url(info.url)} holds no .dem file")
            names = [demo_name(info, map_from_member(i.filename)) for i in members]
            if len(set(names)) != len(names):
                raise IOError(f"two maps would share one file name: {sorted(names)}")
            for n in names:
                if (dest_dir / n).exists():
                    raise FileExistsError(f"{dest_dir / n} already exists; not overwriting")
            for info_z, name in zip(members, names):
                if info_z.file_size > scraper.MAX_DEMO_BYTES:
                    raise IOError(f"{info_z.filename}: declares {info_z.file_size} bytes, over the cap")
                written = 0
                # Output names are ours, never the member path: no zip-slip.
                with z.open(info_z) as src, open(staging / name, "wb") as out:
                    while block := src.read(1 << 20):
                        written += len(block)
                        if written > scraper.MAX_DEMO_BYTES:
                            raise IOError(f"{info_z.filename}: over the size cap")
                        out.write(block)
                if written < scraper.MIN_DEMO_BYTES:
                    raise IOError(f"{info_z.filename}: implausibly small ({written} bytes)")
        for name in names:
            (staging / name).replace(dest_dir / name)
            moved.append(dest_dir / name)
        return moved
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError) as e:
        for p in moved:
            p.unlink(missing_ok=True)
        raise IOError(f"corrupt archive from {redact_url(info.url)}: {type(e).__name__}") from None
    except BaseException:
        for p in moved:
            p.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def download_resolved(info: HltvMatch, dest_dir: Path, throttle: Throttle | None = None) -> list[Path]:
    """Download a resolved match's archive and unpack one `.dem` per map."""
    if not info.demo_url:
        raise LookupError(f"no demo link on {redact_url(info.url)}")
    with tempfile.TemporaryDirectory(prefix="hltv_dl_") as tmp:
        (throttle or _THROTTLE).wait()
        archive = scraper.download_demo(info.demo_url, Path(tmp) / "archive.bin")
        return unpack(archive, dest_dir, info)


def download_match(match: str, dest_dir: Path, *, base_url: str = HLTV_BASE,
                   throttle: Throttle | None = None) -> tuple[HltvMatch, list[Path]]:
    """Resolve the page, download the archive, unpack one `.dem` per map."""
    info = resolve(match, base_url, throttle)
    return info, download_resolved(info, dest_dir, throttle)


# ── source interface ──────────────────────────────────────────────────────────

_PENDING: set[Path] = set()           # files this module unpacked and has not handed out
_META:    dict[str, dict] = {}


def iter_unprocessed_demos(matches=(), base_url: str = HLTV_BASE, work_dir: Path | None = None,
                           throttle: Throttle | None = None, skip_seen: bool = True, **_):
    """Yield `(match_id, file URI)` per map of each HLTV match.

    `skip_seen`: skip a match without downloading when any of its maps is already
    processed (by file-name prefix). A series where one map failed is then not retried;
    pass skip_seen=False to download it again (maps already processed are still skipped)."""
    own  = work_dir is None
    work = Path(tempfile.mkdtemp(prefix="hltv_")) if own else Path(work_dir)
    try:
        for match in matches:
            try:
                info = resolve(match, base_url, throttle)
                if skip_seen and db.has_processed_prefix(demo_name(info, "")[:-len(".dem")]):
                    print(f"    skipping {redact_url(info.url)} (already in db)")
                    continue
                files = download_resolved(info, work, throttle)
            except Exception as e:
                print(f"    hltv match {redact_url(str(match))} failed: {redact_text(e, str(match))}")
                continue
            for f in files:
                mid = f.stem
                if db.is_processed_any(mid):
                    print(f"    skipping {mid} (already in db)")
                    f.unlink(missing_ok=True)
                    continue
                _PENDING.add(f.resolve())
                _META[mid] = {"event": info.event, "date": info.date}
                yield mid, f.resolve().as_uri()
    finally:
        if own:
            shutil.rmtree(work, ignore_errors=True)


def _uri_path(uri: str) -> Path:
    return Path(url2pathname(urlsplit(uri).path)).resolve()


def download_demo(url: str, dest: Path) -> Path:
    """A file URI yielded by iter_unprocessed_demos is moved to `dest`; any other URL
    goes through scraper.download_demo."""
    if urlsplit(url).scheme != "file":
        return scraper.download_demo(url, dest)
    src = _uri_path(url)
    if src not in _PENDING:
        raise ValueError("refusing a file URI this source did not produce")
    shutil.move(str(src), str(dest))
    _PENDING.discard(src)
    return Path(dest)


def meta_for(match_id: str) -> dict:
    return dict(_META.get(match_id, {}))
