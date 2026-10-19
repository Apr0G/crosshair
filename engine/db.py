"""
Engine SQLite database `data/engine/engine.db` (Amendment 1, replaces the voided P5.2 v3 migration).

Fresh schema, SCHEMA_VERSION = 1:
  matches        one row per demo (Appendix A.6 `engine_matches` columns)
  runs           one row per engine run (manifest pointer, status)
  action_scores  per-action scores per run (Appendix A.6 `engine_action_scores`)
  processed      dedup index for the download sources (`match_id` PK, `map`,
                 `demo_url`, `processed_at`), so the moved sources keep their invariant

Additive migrations keyed by `PRAGMA user_version` (`_MIGRATIONS`). A file whose
user_version is newer than this build is refused rather than silently misread.
`is_processed_any` answers for both the BO1 id `<mid>` and the series form `<mid>_m1`
(CLAUDE.md idempotency invariant). `demo_url` is stored only after `redact_url`.

The path is `paths.DB_PATH`, overridable by the env var `CROSSHAIR_ENGINE_DB` (tests
point it at a temp file). Stdlib only at import; numpy is touched lazily by `_Encoder`
and `scraper.redact_url` is imported only when a URL is written.
"""
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from engine import paths

ENV_DB = "CROSSHAIR_ENGINE_DB"

# ── schema ────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = 1

# Additive-only migrations: {target user_version: [statements]}, applied in key order
# to any file whose user_version is behind SCHEMA_VERSION. A new column goes BOTH here
# and in _init_schema (CREATE TABLE IF NOT EXISTS never adds a column to a table).
_MIGRATIONS: dict[int, list[str]] = {}

STATUSES = ("downloaded", "parsed", "states", "actions", "quarantined")

MATCH_COLUMNS = (
    "match_key", "source", "event", "event_tier", "date", "team_ct_start",
    "team_t_start", "map", "map_version", "demo_sha256", "file_bytes", "game_version",
    "tick_rate", "tier", "status", "quarantine_reason", "created_at",
)
RUN_COLUMNS = (
    "run_id", "kind", "started_at", "finished_at", "config_hash", "git_sha",
    "manifest_path", "status",
)
SCORE_COLUMNS = (
    "action_id", "match_key", "round_num", "run_id", "V", "V_direct", "V_enabler",
    "V_movement", "Q", "B", "D", "R", "X", "method", "n_coalition", "residual", "flags",
)

_SERIES_RE = re.compile(r"^(?P<base>.+)_m(?P<n>\d+)$")


class SchemaVersionError(RuntimeError):
    """The database file was written by a newer build (user_version > SCHEMA_VERSION)."""


class _Encoder(json.JSONEncoder):
    """numpy scalars/arrays leak into flag blobs; json can't serialise them."""
    def default(self, obj):
        try:
            import numpy as np
        except ImportError:
            return super().default(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def _dumps(obj) -> str:
    return json.dumps(obj, cls=_Encoder, sort_keys=True)


def _scalar(v):
    """sqlite3 cannot bind numpy integers; unwrap any numpy scalar to its Python value."""
    if type(v).__module__ == "numpy" and hasattr(v, "item"):
        return v.item()
    return v


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _init_schema(c: sqlite3.Connection) -> None:
    c.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            match_key         TEXT PRIMARY KEY,
            source            TEXT,
            event             TEXT,
            event_tier        INTEGER,
            date              TEXT,
            team_ct_start     TEXT,
            team_t_start      TEXT,
            map               TEXT,
            map_version       TEXT,
            demo_sha256       TEXT UNIQUE,
            file_bytes        INTEGER,
            game_version      TEXT,
            tick_rate         INTEGER,
            tier              TEXT,
            status            TEXT,
            quarantine_reason TEXT,
            created_at        TEXT
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id        TEXT PRIMARY KEY,
            kind          TEXT,
            started_at    TEXT,
            finished_at   TEXT,
            config_hash   TEXT,
            git_sha       TEXT,
            manifest_path TEXT,
            status        TEXT
        );

        CREATE TABLE IF NOT EXISTS action_scores (
            action_id   TEXT NOT NULL,
            match_key   TEXT NOT NULL,
            round_num   INTEGER,
            run_id      TEXT NOT NULL,
            V           REAL,
            V_direct    REAL,
            V_enabler   REAL,
            V_movement  REAL,
            Q           REAL,
            B           REAL,
            D           REAL,
            R           REAL,
            X           REAL,
            method      TEXT,
            n_coalition INTEGER,
            residual    REAL,
            flags       TEXT,
            PRIMARY KEY (action_id, run_id)
        );
        CREATE INDEX IF NOT EXISTS idx_eas_match ON action_scores(match_key, round_num);
        CREATE INDEX IF NOT EXISTS idx_eas_run   ON action_scores(run_id);

        CREATE TABLE IF NOT EXISTS processed (
            match_id     TEXT PRIMARY KEY,
            map          TEXT,
            demo_url     TEXT,
            processed_at TEXT
        );
    """)


def _migrate(c: sqlite3.Connection) -> None:
    """Apply any additive migrations this file has not seen; refuse a newer file."""
    version = c.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{db_path()}: schema user_version {version} is newer than this build "
            f"(SCHEMA_VERSION {SCHEMA_VERSION}); refusing to open it. Update the code.")
    if version == SCHEMA_VERSION:
        return
    for target in sorted(_MIGRATIONS):
        if target <= version or target > SCHEMA_VERSION:
            continue
        for stmt in _MIGRATIONS[target]:
            try:
                c.execute(stmt)
            except sqlite3.OperationalError as e:
                # Column already present (added out of band before this ran).
                if "duplicate column" not in str(e).lower():
                    raise
        c.execute(f"PRAGMA user_version = {int(target)}")
        print(f"[db] migrated schema → v{target}")
    c.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")
    c.commit()


# ── connections ───────────────────────────────────────────────────────────────

def db_path() -> Path:
    env = os.environ.get(ENV_DB)
    return Path(env) if env else paths.DB_PATH


def _conn() -> sqlite3.Connection:
    """Open the DB, creating the schema on a fresh file and migrating an old one."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path, timeout=30)
    c.row_factory = sqlite3.Row
    try:
        # busy_timeout MUST be set before anything that takes a lock (journal_mode=WAL
        # needs an exclusive one), or two workers opening at once fail instantly.
        c.execute("PRAGMA busy_timeout=30000")
        try:
            if (c.execute("PRAGMA journal_mode").fetchone()[0] or "").lower() != "wal":
                c.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass                    # another process is mid-switch; it lands in WAL anyway
        # Key off the tables, not the file: a bare sqlite3.connect elsewhere creates an
        # empty file, after which a file-existence check would skip init forever.
        version = c.execute("PRAGMA user_version").fetchone()[0]
        have_tables = c.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='matches'"
        ).fetchone() is not None
        if not have_tables and version == 0:
            _init_schema(c)
            c.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")
            c.commit()
        else:
            _migrate(c)
    except Exception:
        c.close()
        raise
    return c


def init_db() -> None:
    """Create or migrate the schema up front, so parallel workers don't race on it."""
    _conn().close()


# ── reads ─────────────────────────────────────────────────────────────────────

def is_processed(match_id: str) -> bool:
    """Exact-id check against `processed` (what the moved sources call per URL)."""
    c = _conn()
    try:
        return c.execute("SELECT 1 FROM processed WHERE match_id = ?",
                         (match_id,)).fetchone() is not None
    finally:
        c.close()


def id_forms(mid: str) -> tuple[str, ...]:
    """Both id forms of one demo: a bare `<mid>` also checks `<mid>_m1`; a series id
    `<mid>_mN` also checks the bare BO1 form `<mid>`. A stored `_m1` never blocks
    `_m2`/`_m3`, so a series where map 1 landed and map 2 failed is retried."""
    m = _SERIES_RE.match(mid or "")
    return (mid, m["base"]) if m else (mid, f"{mid}_m1")


def is_processed_any(mid: str) -> bool:
    c = _conn()
    try:
        forms = id_forms(mid)
        q = f"SELECT 1 FROM processed WHERE match_id IN ({','.join('?' * len(forms))})"
        return c.execute(q, forms).fetchone() is not None
    finally:
        c.close()


def has_processed_prefix(prefix: str) -> bool:
    """True if any processed id starts with `prefix` (LIKE with escaped wildcards)."""
    esc = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    c = _conn()
    try:
        return c.execute("SELECT 1 FROM processed WHERE match_id LIKE ? ESCAPE '\\' LIMIT 1",
                         (esc + "%",)).fetchone() is not None
    finally:
        c.close()


def get_match(match_key: str) -> dict | None:
    c = _conn()
    try:
        row = c.execute("SELECT * FROM matches WHERE match_key = ?", (match_key,)).fetchone()
        return dict(row) if row else None
    finally:
        c.close()


def match_key_for_sha(demo_sha256: str) -> str | None:
    c = _conn()
    try:
        row = c.execute("SELECT match_key FROM matches WHERE demo_sha256 = ?",
                        (demo_sha256,)).fetchone()
        return row[0] if row else None
    finally:
        c.close()


def list_matches() -> list[dict]:
    c = _conn()
    try:
        return [dict(r) for r in c.execute("SELECT * FROM matches ORDER BY match_key")]
    finally:
        c.close()


def list_runs() -> list[dict]:
    c = _conn()
    try:
        return [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY run_id")]
    finally:
        c.close()


def stats() -> dict:
    c = _conn()
    try:
        return {
            "matches":       c.execute("SELECT COUNT(*) FROM matches").fetchone()[0],
            "processed":     c.execute("SELECT COUNT(*) FROM processed").fetchone()[0],
            "runs":          c.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
            "action_scores": c.execute("SELECT COUNT(*) FROM action_scores").fetchone()[0],
            "by_status":     dict(c.execute(
                "SELECT status, COUNT(*) FROM matches GROUP BY status").fetchall()),
        }
    finally:
        c.close()


# ── writes ────────────────────────────────────────────────────────────────────

def _check_status(status: str) -> None:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; expected one of {STATUSES}")


class MatchConflict(ValueError):
    """`register_match`: the match_key exists and describes a different demo file."""


def register_match(meta: dict, demo_url: str | None = None) -> str:
    """Insert one `matches` row and its `processed` row in ONE transaction; return the
    match_key that holds this demo.

    Idempotent: an existing `match_key` with the same (or no) `demo_sha256` returns that
    key unchanged. An existing key with a DIFFERENT sha256 raises `MatchConflict`
    (MED-06, 2026-10-19): the row describes another file, so nothing is written and the
    caller must not keep its copy. A demo whose `demo_sha256` is already stored under
    another key is the same match (P1.1.5): the earlier row is kept, the duplicate is
    logged, the new id is marked processed (so its source does not download it again)
    and the EXISTING key is returned. `demo_url` is redacted before it is written."""
    unknown = set(meta) - set(MATCH_COLUMNS)
    if unknown:
        raise ValueError(f"unknown matches columns: {sorted(unknown)}")
    key = meta.get("match_key")
    if not key:
        raise ValueError("meta needs a match_key")
    row = {col: _scalar(meta.get(col)) for col in MATCH_COLUMNS}
    row["status"]     = row["status"] or "downloaded"
    row["created_at"] = row["created_at"] or _now()
    _check_status(row["status"])

    safe_url = None
    if demo_url:
        from engine.sources.scraper import redact_url
        safe_url = redact_url(demo_url)

    c = _conn()
    try:
        c.execute("BEGIN IMMEDIATE")
        existing = c.execute("SELECT demo_sha256 FROM matches WHERE match_key = ?", (key,)).fetchone()
        owner    = None
        if existing is None and row["demo_sha256"]:
            hit   = c.execute("SELECT match_key FROM matches WHERE demo_sha256 = ?",
                              (row["demo_sha256"],)).fetchone()
            owner = hit[0] if hit else None

        if existing is not None:
            if row["demo_sha256"] and existing[0] and existing[0] != row["demo_sha256"]:
                raise MatchConflict(f"{key} is already registered with a different demo (sha256 differs); "
                                    f"the earlier row is kept, this file is not registered")
            result = key
        elif owner is not None:
            print(f"[db] duplicate demo: {key} has the same sha256 as {owner}; keeping {owner}")
            result = owner
        else:
            cols = ", ".join(MATCH_COLUMNS)
            c.execute(f"INSERT INTO matches ({cols}) VALUES ({', '.join('?' * len(MATCH_COLUMNS))})",
                      [row[col] for col in MATCH_COLUMNS])
            result = key
        c.execute("INSERT OR IGNORE INTO processed (match_id, map, demo_url, processed_at) "
                  "VALUES (?, ?, ?, ?)", (key, row["map"], safe_url, _now()))
        c.commit()
        return result
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def set_status(match_key: str, status: str, reason: str | None = None) -> None:
    """Set `status` (and `quarantine_reason`, cleared unless given). KeyError if absent."""
    _check_status(status)
    c = _conn()
    try:
        cur = c.execute("UPDATE matches SET status = ?, quarantine_reason = ? WHERE match_key = ?",
                        (status, reason, match_key))
        if cur.rowcount == 0:
            c.rollback()
            raise KeyError(f"no match {match_key!r}")
        c.commit()
    finally:
        c.close()


def upsert_run(run_id: str, kind: str | None = None, *, started_at: str | None = None,
               finished_at: str | None = None, config_hash: str | None = None,
               git_sha: str | None = None, manifest_path: str | None = None,
               status: str | None = None) -> None:
    """Insert a run or update it; a None argument leaves the stored value alone."""
    vals = {"run_id": run_id, "kind": kind, "started_at": started_at,
            "finished_at": finished_at, "config_hash": config_hash, "git_sha": git_sha,
            "manifest_path": None if manifest_path is None else str(manifest_path),
            "status": status}
    sets = ", ".join(f"{col} = COALESCE(excluded.{col}, runs.{col})" for col in RUN_COLUMNS[1:])
    c = _conn()
    try:
        c.execute(f"INSERT INTO runs ({', '.join(RUN_COLUMNS)}) "
                  f"VALUES ({', '.join('?' * len(RUN_COLUMNS))}) "
                  f"ON CONFLICT(run_id) DO UPDATE SET {sets}",
                  [vals[col] for col in RUN_COLUMNS])
        c.commit()
    finally:
        c.close()


def _score_params(r: dict) -> list:
    unknown = set(r) - set(SCORE_COLUMNS)
    if unknown:
        raise ValueError(f"unknown action_scores columns: {sorted(unknown)}")
    for col in ("action_id", "match_key", "run_id"):
        if not r.get(col):
            raise ValueError(f"action_scores row needs {col}: {r!r}")
    flags = r.get("flags")
    if flags is not None and not isinstance(flags, str):
        flags = _dumps(flags)
    return [flags if col == "flags" else _scalar(r.get(col)) for col in SCORE_COLUMNS]


def insert_action_scores(rows: list[dict]) -> int:
    """Write scores in ONE transaction: every `(match_key, run_id)` present in `rows`
    is deleted first, then all rows inserted, so a re-run replaces instead of appending.
    Any bad row rolls the whole batch back. Returns the number of rows written."""
    rows   = list(rows)
    c      = _conn()
    try:
        params = [_score_params(r) for r in rows]
        groups = {(r["match_key"], r["run_id"]) for r in rows}
        c.execute("BEGIN IMMEDIATE")
        for mk, run in groups:
            c.execute("DELETE FROM action_scores WHERE match_key = ? AND run_id = ?", (mk, run))
        c.executemany(f"INSERT INTO action_scores ({', '.join(SCORE_COLUMNS)}) "
                      f"VALUES ({', '.join('?' * len(SCORE_COLUMNS))})", params)
        c.commit()
        return len(params)
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()
