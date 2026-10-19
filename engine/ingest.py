"""
The seam (Amendment 1): every source that produces a local `.dem` ends here.

`register_demo` copies the demo under `data/engine/demos/<tier>/`, inserts the
`matches` and `processed` rows, and nothing else. Parsing, states and actions are
separate CLI stages, so a downloaded demo is never half-processed.

    from engine import ingest
    key = ingest.register_demo("1-abc", "x.dem", tier="faceit_10", source="faceit_api")
    n_failed = ingest.run("api", max_matches=20, min_level=10)

`process_one` / `run` port the legacy `pipeline.process_one` / `run`: dedup on both id
forms before downloading, download into a temp dir through the source's
`download_demo(url, dest)`, register, redact every URL in output and in the DB.

A source is any module (or object) with `iter_unprocessed_demos(**kwargs)` yielding
`(match_id, url)` and `download_demo(url, dest) -> Path`. Optional: `meta_for(match_id)
-> dict` of extra `matches` columns (event, date, teams) and `close_browser()`.

Header reader contract: `header_reader(path) -> dict` with any of the keys `map`,
`tick_rate`, `game_version`, `map_version`, `date`, `team_ct_start`, `team_t_start`
(others are ignored; a missing key is stored as NULL). Default: the awpy 3 parser's
`engine.parse.awpy3_parser.header_info`, imported lazily.
"""
import hashlib
import importlib
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from engine import db, paths
from engine.log import redact_text
from engine.sources.scraper import redact_url

# FACEIT match ids look like "1-<uuid>", optionally suffixed "_m1" for BO series.
_MATCH_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# tier -> attribute of `engine.paths` holding its demo directory (read at call time,
# so tests can point the attributes at a temp dir)
TIER_DIRS = {
    "pro":       "DEMOS_PRO_DIR",
    "faceit_10": "DEMOS_FACEIT_DIR",
    "golden":    "DEMOS_GOLDEN_DIR",
}

# source name -> (module under engine.sources, default tier)
SOURCES = {
    "playwright": ("scraper",    "faceit_10"),
    "scraper":    ("scraper",    "faceit_10"),
    "api":        ("faceit_api", "faceit_10"),
    "faceit_api": ("faceit_api", "faceit_10"),
    "hltv":       ("hltv",       "pro"),
    "hf":         ("hf_archive", "pro"),
    "hf_archive": ("hf_archive", "pro"),
}

HEADER_KEYS = ("map", "tick_rate", "game_version", "map_version", "date",
               "team_ct_start", "team_t_start")
META_KEYS   = ("event", "event_tier", "date", "team_ct_start", "team_t_start", "map_version")

HeaderReader = Callable[[Path], dict]


# ── helpers ───────────────────────────────────────────────────────────────────

def check_match_id(match_id: str) -> str:
    if not isinstance(match_id, str) or not _MATCH_ID_RE.match(match_id) or match_id in (".", ".."):
        raise ValueError(f"refusing unsafe match_id: {match_id!r}")
    return match_id


def tier_dir(tier: str) -> Path:
    if tier not in TIER_DIRS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(TIER_DIRS)}")
    return Path(getattr(paths, TIER_DIRS[tier]))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _default_header_reader() -> HeaderReader:
    from engine.parse.awpy3_parser import header_info
    return header_info


def _int_or_none(v) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ── the seam ──────────────────────────────────────────────────────────────────

def register_demo(match_id: str, demo_path, demo_url: str | None = None, *, tier: str,
                  source: str, header_reader: HeaderReader | None = None,
                  extra_meta: dict | None = None) -> str:
    """Copy one local `.dem` to `<demos>/<tier>/<match_id>.dem`, read its header, and
    insert its `matches` + `processed` rows. Returns the match_key holding the demo.

    Idempotent: a demo whose sha256 is already registered returns that key without
    copying; an existing destination file is reused only if its sha256 matches (never
    overwritten). `extra_meta` carries source-known columns (META_KEYS)."""
    check_match_id(match_id)
    src = Path(demo_path)
    if not src.is_file():
        raise FileNotFoundError(f"demo not found: {src}")
    bad = set(extra_meta or {}) - set(META_KEYS)
    if bad:
        raise ValueError(f"extra_meta keys not allowed: {sorted(bad)}")
    out_dir = tier_dir(tier)
    digest  = sha256_file(src)

    known = db.match_key_for_sha(digest)
    if known is not None:
        if known != match_id:
            db.register_match({"match_key": match_id, "demo_sha256": digest,
                               "source": source, "tier": tier}, demo_url)  # logs, marks processed
        print(f"[{match_id}] demo already registered as {known}")
        return known

    dest = out_dir / f"{match_id}.dem"
    if not dest.resolve().is_relative_to(out_dir.resolve()):
        raise ValueError(f"match_id escapes the demo dir: {match_id!r}")
    copied = False
    if dest.exists():
        if sha256_file(dest) != digest:
            raise FileExistsError(f"{dest} exists with different content; not overwriting")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        try:
            shutil.copyfile(src, part)
            part.replace(dest)
        finally:
            if part.exists():
                part.unlink()
        copied = True

    try:
        reader = header_reader or _default_header_reader()
        header = reader(dest) or {}
        meta   = {k: header.get(k) for k in HEADER_KEYS}
        meta.update({k: v for k, v in (extra_meta or {}).items() if v is not None})
        meta.update({
            "match_key":   match_id,
            "source":      source,
            "tier":        tier,
            "demo_sha256": digest,
            "file_bytes":  dest.stat().st_size,
            "tick_rate":   _int_or_none(meta.get("tick_rate")),
            "event_tier":  _int_or_none(meta.get("event_tier")),
            "status":      "downloaded",
        })
        key = db.register_match(meta, demo_url)
    except BaseException:
        if copied and dest.exists():
            dest.unlink()            # a demo with no row must not sit in the tier dir
        raise
    if key != match_id and copied and dest.exists():
        dest.unlink()                # lost a race to a duplicate: keep only the earlier file
    return key


# ── download loop (legacy pipeline.process_one / run) ─────────────────────────

def _source_name(source) -> str:
    return str(getattr(source, "SOURCE_NAME", None)
               or getattr(source, "__name__", type(source).__name__)).rsplit(".", 1)[-1]


def process_one(source, match_id: str, demo_url: str, tier: str, *,
                header_reader: HeaderReader | None = None,
                source_name: str | None = None) -> str | None:
    """Download one demo into a temp dir and register it. Returns the match_key, or
    None when the match was already processed (in either id form) and was skipped."""
    check_match_id(match_id)
    tier_dir(tier)                                    # fail on a bad tier before downloading
    if db.is_processed_any(match_id):
        print(f"[{match_id}] already in DB, skipping.")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        demo_path = Path(tmpdir) / f"{match_id}.dem"
        # pathlib's `/` discards the left operand for an absolute right operand.
        if not demo_path.resolve().is_relative_to(Path(tmpdir).resolve()):
            raise ValueError(f"match_id escapes the temp dir: {match_id!r}")

        print(f"[{match_id}] downloading ...")
        source.download_demo(demo_url, demo_path)

        meta_for = getattr(source, "meta_for", None)
        extra    = meta_for(match_id) if callable(meta_for) else None
        key = register_demo(match_id, demo_path, redact_url(demo_url), tier=tier,
                            source=source_name or _source_name(source),
                            header_reader=header_reader, extra_meta=extra)
        print(f"[{match_id}] registered as {key}.")
        return key


def _load_source(name: str):
    if name not in SOURCES:
        raise ValueError(f"unknown source {name!r}; expected one of {sorted(SOURCES)}")
    return importlib.import_module(f"engine.sources.{SOURCES[name][0]}")


def run(source_name, max_matches: int | None = None, *, tier: str | None = None,
        header_reader: HeaderReader | None = None, **source_kwargs) -> int:
    """Discover + download + register. `source_name` is a key of SOURCES (or a source
    object, for tests). `source_kwargs` go to `iter_unprocessed_demos` (e.g.
    `min_level=10` for the API source). Stops after `max_matches` attempts.
    Returns the number of failed matches (0 = clean run)."""
    db.init_db()
    if isinstance(source_name, str):
        src  = _load_source(source_name)
        tier = tier or SOURCES[source_name][1]
        name = SOURCES[source_name][0]
    else:
        src, name = source_name, _source_name(source_name)
        if tier is None:
            raise ValueError("tier is required for a source object")
    print(f"source: {name}  tier: {tier}")

    attempted = 0
    n_ok      = 0
    failed: list[tuple[str, str]] = []
    try:
        items = src.iter_unprocessed_demos(**source_kwargs) if max_matches != 0 else ()
        for match_id, demo_url in items:
            attempted += 1
            print(f"\n── match {attempted}: {match_id} ──")
            try:
                process_one(src, match_id, demo_url, tier, header_reader=header_reader,
                            source_name=name)
                n_ok += 1          # an already-processed match was skipped cleanly: ok
            except Exception as e:
                # `e` can embed the signed URL (requests puts it in every HTTPError).
                msg = redact_text(f"{type(e).__name__}: {e}", demo_url)
                print(f"[{match_id}] error: {msg}")
                failed.append((match_id, msg))
            if max_matches is not None and attempted >= max_matches:
                break              # before pulling the next item: no URL fetched in vain
    finally:
        if hasattr(src, "close_browser"):
            src.close_browser()

    if attempted == 0:
        print("no new demos found.")
    print(f"\ndone: {n_ok} ok, {len(failed)} failed, of {attempted} attempted")
    if failed:
        print("failed matches:")
        for mid, msg in failed:
            print(f"  {mid}: {msg}")
    s = db.stats()
    print(f"db: {s['matches']} matches | {s['processed']} processed ids | by status {s['by_status']}")
    return len(failed)
