"""
Hugging Face pro demo archive source (P1.1.3): list the archive, download by map and
month in waves (respecting the P0.3 free-space rules), checksum each file against the
archive listing when available.

    from engine.sources import hf_archive
    files = hf_archive.list_files("owner/repo")                     # [HfFile, ...]
    waves = hf_archive.plan_waves(files, maps={"de_mirage"}, wave_bytes=20e9)
    hf_archive.download_waves(files, out_dir, maps={"de_mirage"})
    engine.ingest.run("hf", repo="owner/repo", maps=["de_mirage"])

Listing: `GET {api}/{datasets|models}/{repo}/tree/{revision}?recursive=true`, following
the `Link: rel="next"` pagination. Only `.dem` / `.dem.zst` / `.dem.gz` files are kept.
`sha256` comes from the entry's `lfs.oid` (the LFS pointer's sha256 of the content);
a non-LFS entry's `oid` is a git blob sha1, so it is not used. Verification runs on raw
`.dem` entries only: `scraper.download_demo` decompresses `.zst`/`.gz` in place, so the
stored bytes the listing hashed are gone by the time we could check them (logged).

Map and month are read from the path (`de_<map>` and `yyyy-mm` tokens) unless a
`classify(path) -> (map, month)` is passed; unknown is None, and a map/month filter
never selects a file whose map/month is unknown.

Note (2026-10-19): the plan's `Vasy7777/cs2-demo-archive` is NOT a pro demo archive:
its README describes 19 sample matches (6 complete, 13 casual 3-POV) from a
demo-to-video pipeline, ~113 GB mostly video, paths are opaque numeric ids with no
map or month, and the replays are 64 Hz. See the agent report; no default repo is set.
"""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import quote

import requests

from engine import db, paths
from engine.sources import scraper
from engine.sources.scraper import redact_url

SOURCE_NAME  = "hf_archive"
HF_API       = "https://huggingface.co/api"
HF_BASE      = "https://huggingface.co"
DEMO_EXTS    = (".dem", ".dem.zst", ".dem.gz")
MAX_WAVE_GB  = 20.0

_MAP_RE   = re.compile(r"(?<![a-z0-9])(de_[a-z0-9]+)")
_MONTH_RE = re.compile(r"(?<!\d)(20\d\d)-(0[1-9]|1[0-2])(?!\d)")
_ID_RE    = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class HfFile:
    path:   str
    size:   int | None
    sha256: str | None
    map:    str | None
    month:  str | None


def classify_path(path: str) -> tuple[str | None, str | None]:
    low = path.lower()
    m   = _MAP_RE.search(low)
    mo  = _MONTH_RE.search(low)
    return (m.group(1) if m else None), (f"{mo.group(1)}-{mo.group(2)}" if mo else None)


def _repo_kind(repo_type: str) -> str:
    if repo_type not in ("datasets", "models"):
        raise ValueError(f"repo_type must be 'datasets' or 'models', got {repo_type!r}")
    return repo_type


def list_files(repo: str, *, repo_type: str = "datasets", revision: str = "main",
               api_base: str = HF_API, classify: Callable | None = None,
               timeout: float = 30) -> list[HfFile]:
    """Every demo file in the repo with size, sha256 (when LFS) and map/month."""
    classify = classify or classify_path
    url: str | None = (f"{api_base.rstrip('/')}/{_repo_kind(repo_type)}/{repo}/tree/"
                       f"{quote(revision, safe='')}?recursive=true")
    out: list[HfFile] = []
    while url:
        safe = redact_url(url)
        try:
            resp = requests.get(url, timeout=timeout)
        except requests.RequestException as e:
            raise IOError(f"listing {safe}: {type(e).__name__}") from None
        if resp.status_code >= 400:
            raise IOError(f"HTTP {resp.status_code} listing {safe}")
        try:
            entries = resp.json()
        except ValueError:
            raise IOError(f"listing {safe}: response is not JSON") from None
        if not isinstance(entries, list):
            raise IOError(f"listing {safe}: expected a JSON list")
        for e in entries:
            path = e.get("path") or ""
            if e.get("type") != "file" or not path.lower().endswith(DEMO_EXTS):
                continue
            lfs = e.get("lfs") or {}
            sha = lfs.get("oid") or lfs.get("sha256")
            sha = sha.lower() if isinstance(sha, str) and re.fullmatch(r"[0-9a-fA-F]{64}", sha) else None
            size = e.get("size")
            mp, mo = classify(path)
            out.append(HfFile(path, size if isinstance(size, int) else None, sha, mp, mo))
        url = resp.links.get("next", {}).get("url")
    return out


def select(files: Iterable[HfFile], maps: Iterable[str] | None = None,
           months: Iterable[str] | None = None) -> list[HfFile]:
    maps   = set(maps) if maps is not None else None
    months = set(months) if months is not None else None
    return [f for f in files
            if (maps is None or f.map in maps) and (months is None or f.month in months)]


def plan_waves(files: Iterable[HfFile], *, maps: Iterable[str] | None = None,
               months: Iterable[str] | None = None,
               wave_bytes: float = MAX_WAVE_GB * 1e9) -> list[list[HfFile]]:
    """Selected files ordered by (month, map, path), one wave per (map, month) group,
    a group larger than `wave_bytes` split across several waves (a single larger file
    gets a wave to itself; unknown size counts 0)."""
    chosen = sorted(select(files, maps, months),
                    key=lambda f: (f.month or "", f.map or "", f.path))
    waves: list[list[HfFile]] = []
    cur: list[HfFile] = []
    cur_bytes = 0
    for f in chosen:
        size = f.size or 0
        if cur and ((cur[0].map, cur[0].month) != (f.map, f.month) or cur_bytes + size > wave_bytes):
            waves.append(cur)
            cur, cur_bytes = [], 0
        cur.append(f)
        cur_bytes += size
    if cur:
        waves.append(cur)
    return waves


def resolve_url(repo: str, path: str, *, repo_type: str = "datasets", revision: str = "main",
                base: str = HF_BASE) -> str:
    prefix = "datasets/" if _repo_kind(repo_type) == "datasets" else ""
    return (f"{base.rstrip('/')}/{prefix}{repo}/resolve/{quote(revision, safe='')}/"
            f"{quote(path)}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, sha256: str | None, stored_path: str) -> Path:
    """scraper.download_demo, then sha256 against the listing for raw `.dem` entries.
    A mismatch deletes the file and raises."""
    out = scraper.download_demo(url, Path(dest))
    if sha256 and stored_path.lower().endswith(".dem"):
        got = _sha256(out)
        if got != sha256:
            out.unlink(missing_ok=True)
            raise IOError(f"sha256 mismatch for {stored_path}: listing {sha256[:12]}…, got {got[:12]}…")
    elif sha256:
        print(f"    {stored_path}: compressed entry, sha256 not verifiable after decompression")
    return out


def _space_ok(dest_dir: Path) -> bool:
    status = paths.check_space(path=dest_dir)
    if status is paths.SpaceStatus.REFUSE:
        print(f"    free space below the floor at {dest_dir}; stopping before the next wave")
        return False
    if status is paths.SpaceStatus.WARN:
        print(f"    free space below the warning line at {dest_dir}; continuing")
    return True


def download_waves(files: Iterable[HfFile], dest_dir: Path, repo: str, *,
                   maps: Iterable[str] | None = None, months: Iterable[str] | None = None,
                   wave_bytes: float = MAX_WAVE_GB * 1e9, repo_type: str = "datasets",
                   revision: str = "main", base: str = HF_BASE) -> list[Path]:
    """Download selected files wave by wave into `dest_dir`; the free-space check runs
    before every wave and a REFUSE stops the run. Existing files with a matching sha256
    are kept, never re-downloaded or overwritten."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    got: list[Path] = []
    for i, wave in enumerate(plan_waves(files, maps=maps, months=months, wave_bytes=wave_bytes), 1):
        if not _space_ok(dest_dir):
            break
        print(f"  wave {i}: {len(wave)} files")
        for f in wave:
            dest = dest_dir / f"{match_id_for(f.path)}.dem"
            if dest.exists():
                if f.sha256 is None or _sha256(dest) == f.sha256:
                    got.append(dest)
                    continue
                raise FileExistsError(f"{dest} exists with different content; not overwriting")
            url = resolve_url(repo, f.path, repo_type=repo_type, revision=revision, base=base)
            got.append(fetch(url, dest, f.sha256, f.path))
    return got


# ── source interface ──────────────────────────────────────────────────────────

def match_id_for(path: str) -> str:
    """`hf-<path without extension>` with every unsafe run replaced by '-'."""
    low = path
    for ext in DEMO_EXTS[::-1]:
        if low.lower().endswith(ext):
            low = low[: -len(ext)]
            break
    return "hf-" + _ID_RE.sub("-", low.replace("/", "-")).strip("-")


_EXPECTED: dict[str, tuple[str | None, str]] = {}      # url -> (sha256, stored path)


def iter_unprocessed_demos(repo: str | None = None, maps: Iterable[str] | None = None,
                           months: Iterable[str] | None = None,
                           wave_bytes: float = MAX_WAVE_GB * 1e9, repo_type: str = "datasets",
                           revision: str = "main", api_base: str = HF_API, base: str = HF_BASE,
                           dest_dir: Path | None = None, **_):
    """Yield `(match_id, resolve URL)` wave by wave; the free-space check runs before
    each wave (on `dest_dir`, default the pro demo dir) and REFUSE ends the generator."""
    if not repo:
        raise ValueError("hf_archive needs repo='owner/name' (no default: see module note)")
    space_dir = Path(dest_dir) if dest_dir is not None else Path(paths.DEMOS_PRO_DIR)
    files = list_files(repo, repo_type=repo_type, revision=revision, api_base=api_base)
    print(f"listed {len(files)} demo files in {repo}")
    for i, wave in enumerate(plan_waves(files, maps=maps, months=months, wave_bytes=wave_bytes), 1):
        if not _space_ok(space_dir):
            return
        print(f"  wave {i}: {len(wave)} files")
        for f in wave:
            mid = match_id_for(f.path)
            if db.is_processed_any(mid):
                print(f"    skipping {mid} (already in db)")
                continue
            url = resolve_url(repo, f.path, repo_type=repo_type, revision=revision, base=base)
            _EXPECTED[url] = (f.sha256, f.path)
            yield mid, url


def download_demo(url: str, dest: Path) -> Path:
    sha, stored = _EXPECTED.get(url, (None, url))
    return fetch(url, dest, sha, stored)
